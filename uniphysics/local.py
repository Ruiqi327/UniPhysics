from __future__ import annotations

import multiprocessing
import os
import sys
from dataclasses import dataclass
from queue import Empty
from typing import Any, Iterator

from tqdm import tqdm

try:
    from .thinking import supports_qwen_thinking
except ImportError:  # pragma: no cover - supports direct script-style imports
    from thinking import supports_qwen_thinking


@dataclass(frozen=True)
class LocalConfig:
    model_path: str
    gpus: str = "0"
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = -1
    max_tokens: int = 2048
    batch_size: int = 8
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    trust_remote_code: bool = False
    enable_thinking: bool = True
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    seed: int = 0

    def __post_init__(self):
        if self.batch_size < 1 or self.max_tokens < 1:
            raise ValueError("batch_size and max_tokens must be positive")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if self.max_model_len is not None and self.max_model_len <= self.max_tokens:
            raise ValueError("max_model_len must exceed max_tokens to leave room for the prompt")
        ids = [value.strip() for value in self.gpus.split(",")]
        if any(not value.isdigit() for value in ids) or len(ids) != len(set(ids)):
            raise ValueError("gpus must be unique comma-separated GPU indices")
        if self.temperature < 0 or not 0 < self.top_p <= 1 or self.top_k == 0 or self.top_k < -1:
            raise ValueError("Invalid local sampling parameters")
        if self.repetition_penalty <= 0 or not -2 <= self.presence_penalty <= 2:
            raise ValueError("Invalid penalty parameters")


class LocalGenerator:
    def __init__(self, config: LocalConfig) -> None:
        from vllm import LLM, SamplingParams

        llm_options: dict[str, Any] = {
            "model": config.model_path,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "trust_remote_code": config.trust_remote_code,
            "seed": config.seed,
        }
        if config.max_model_len is not None:
            llm_options["max_model_len"] = config.max_model_len

        self.llm = LLM(**llm_options)
        self.tokenizer = self.llm.get_tokenizer()
        self.enable_thinking = config.enable_thinking
        self.supports_thinking = supports_qwen_thinking(config.model_path)
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            max_tokens=config.max_tokens,
            presence_penalty=config.presence_penalty,
            repetition_penalty=config.repetition_penalty,
            seed=config.seed,
        )

    def _build_prompt(self, prompt: str) -> str:
        if not getattr(self.tokenizer, "chat_template", None):
            return prompt

        messages = [{"role": "user", "content": prompt}]
        template_options: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.supports_thinking:
            template_options["enable_thinking"] = self.enable_thinking

        return self.tokenizer.apply_chat_template(messages, **template_options)

    def generate(self, prompts: list[str]) -> list[str]:
        formatted_prompts = [self._build_prompt(prompt) for prompt in prompts]
        outputs = self.llm.generate(
            formatted_prompts,
            self.sampling_params,
            use_tqdm=False,
        )
        return [output.outputs[0].text if output.outputs else "" for output in outputs]


def _stream_worker(
    worker_index: int,
    gpu_id: str,
    start_index: int,
    prompts: list[str],
    config: LocalConfig,
    result_queue: Any,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    next_offset = 0
    try:
        if not prompts:
            return

        generator = LocalGenerator(config)
        for start in range(0, len(prompts), config.batch_size):
            batch = prompts[start : start + config.batch_size]
            try:
                responses = generator.generate(batch)
            except Exception as error:
                print(f"Local batch failed on GPU {gpu_id}: {type(error).__name__}", file=sys.stderr)
                responses = ["LOCAL_ERROR"] * len(batch)
            result_queue.put(
                ("batch", worker_index, start_index + start, responses)
            )
            next_offset = start + len(batch)
    except Exception as error:
        print(f"Local worker failed on GPU {gpu_id}: {type(error).__name__}", file=sys.stderr)
        remaining = len(prompts) - next_offset
        if remaining > 0:
            result_queue.put(
                (
                    "batch",
                    worker_index,
                    start_index + next_offset,
                    ["LOCAL_ERROR"] * remaining,
                )
            )
    finally:
        result_queue.put(("done", worker_index, 0, []))


def _split_prompts(prompts: list[str], number_of_splits: int) -> list[list[str]]:
    quotient, remainder = divmod(len(prompts), number_of_splits)
    splits: list[list[str]] = []
    for index in range(number_of_splits):
        start = index * quotient + min(index, remainder)
        end = (index + 1) * quotient + min(index + 1, remainder)
        splits.append(prompts[start:end])
    return splits


class CallLocal:
    def __init__(self, prompts: list[str], config: LocalConfig) -> None:
        self.prompts = prompts
        self.config = config
        self.gpu_ids = [
            gpu.strip() for gpu in config.gpus.split(",") if gpu.strip()
        ] or ["0"]

    def iter_results(self) -> Iterator[tuple[int, str]]:
        prompt_slices = _split_prompts(self.prompts, len(self.gpu_ids))
        context = multiprocessing.get_context("spawn")
        result_queue = context.Queue(maxsize=max(1, len(self.gpu_ids) * 2))
        processes: list[multiprocessing.Process] = []
        worker_indices: dict[int, range] = {}
        emitted: set[int] = set()
        slice_start = 0

        for worker_index, (gpu_id, prompt_slice) in enumerate(
            zip(self.gpu_ids, prompt_slices)
        ):
            worker_indices[worker_index] = range(
                slice_start,
                slice_start + len(prompt_slice),
            )
            process = context.Process(
                target=_stream_worker,
                args=(
                    worker_index,
                    gpu_id,
                    slice_start,
                    prompt_slice,
                    self.config,
                    result_queue,
                ),
            )
            process.start()
            processes.append(process)
            slice_start += len(prompt_slice)

        completed_workers: set[int] = set()
        progress = tqdm(
            total=len(self.prompts),
            desc="Generating locally",
            unit="request",
        )
        try:
            while len(completed_workers) < len(processes):
                try:
                    kind, worker_index, batch_start, responses = (
                        result_queue.get(timeout=1.0)
                    )
                except Empty:
                    for worker_index, process in enumerate(processes):
                        if worker_index in completed_workers or process.is_alive():
                            continue
                        for index in worker_indices[worker_index]:
                            if index not in emitted:
                                emitted.add(index)
                                progress.update(1)
                                yield index, "LOCAL_ERROR"
                        completed_workers.add(worker_index)
                    continue

                if kind == "done":
                    completed_workers.add(worker_index)
                    continue

                for offset, response in enumerate(responses):
                    index = batch_start + offset
                    if index in emitted:
                        continue
                    emitted.add(index)
                    progress.update(1)
                    yield index, response
        finally:
            progress.close()
            if len(completed_workers) < len(processes):
                for process in processes:
                    if process.is_alive():
                        process.terminate()
            for process in processes:
                process.join()
            result_queue.close()
            result_queue.join_thread()

    def run(self) -> list[str]:
        responses = ["LOCAL_ERROR"] * len(self.prompts)
        for index, response in self.iter_results():
            responses[index] = response
        return responses


def call_local(prompts: list[str], config: LocalConfig) -> list[str]:
    return CallLocal(prompts, config).run()


def call_local_iter(
    prompts: list[str],
    config: LocalConfig,
) -> Iterator[tuple[int, str]]:
    return CallLocal(prompts, config).iter_results()
