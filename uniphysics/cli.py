"""Command-line interface. Credentials are read only from the environment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
from .api import APIConfig
from .data import DEFAULT_DATASET, dataset_summary, load_dataset
from .thinking import parse_bool


def sampling(parser, local=False, judge=False):
    parser.add_argument("--temperature", type=float, default=0.7 if local else None)
    parser.add_argument("--top-p", "--top_p", dest="top_p", type=float, default=0.9 if local else None)
    parser.add_argument("--max-tokens", "--max_tokens", dest="max_tokens", type=int, default=8192 if judge else 16384)


def api_options(parser):
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint (or UNIPHYSICS_BASE_URL / OPENAI_BASE_URL)")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=4, help="Retries after the first attempt")
    parser.add_argument("--timeout", type=float, default=180)


def dataset_options(parser):
    parser.add_argument("--dataset", "--uniphysics", dest="uniphysics", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--benchmark", default="all", help="all, or comma-separated source names")
    parser.add_argument("--test-mode", "--test_mode", dest="test_mode", choices=("all", "mini"), default="all")
    parser.add_argument("--limit", type=int, help="Use only the first N selected records for a smoke test")


def build_parser():
    parser = argparse.ArgumentParser(prog="uniphysics", description="UniPhysics generation, hybrid judging, and metrics")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("api", "local", "judge"):
        item = sub.add_parser(name)
        item.add_argument("--project-name", "--project_name", dest="project_name", required=True)
        item.add_argument("--output-dir", type=Path, default=Path("results"), help="Output root, relative to current directory")
        item.add_argument("--resume", action="store_true", help="Reuse compatible successful outputs; retry failures")
        sampling(item, local=name == "local", judge=name == "judge")
        if name in {"api", "judge"}:
            api_options(item)
        if name != "judge":
            dataset_options(item)
            item.add_argument("--enable-thinking", "--enable_thinking", dest="enable_thinking", type=parse_bool, default=True)
        if name == "api":
            item.add_argument("--model", required=True)
        elif name == "local":
            item.add_argument("--model-path", required=True, help="Hugging Face model ID or a local model directory")
            item.add_argument("--gpus", default="0", help="One independent model replica per GPU, e.g. 0,1")
            item.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=-1)
            item.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=8)
            item.add_argument("--gpu-memory-utilization", "--gpu_memory_utilization", dest="gpu_memory_utilization", type=float, default=0.9)
            item.add_argument("--max-model-len", "--max_model_len", dest="max_model_len", type=int)
            item.add_argument("--presence-penalty", type=float, default=0.0)
            item.add_argument("--repetition-penalty", type=float, default=1.0)
            item.add_argument("--seed", type=int, default=0)
            item.add_argument("--trust-remote-code", "--trust_remote_code", dest="trust_remote_code", action="store_true")
        else:
            for number in (1, 2, 3):
                item.add_argument(f"--model-{number}", f"--model_{number}", dest=f"model_{number}",
                                  help="Required when responses include open-ended answers with gold answers")
    validate = sub.add_parser("validate", help="Validate dataset schema and print source counts; no model/API needed")
    dataset_options(validate)
    metric = sub.add_parser("metrics", help="Recompute metrics without API calls")
    metric.add_argument("--input", type=Path, required=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            print(json.dumps(dataset_summary(load_dataset(args.uniphysics, args.benchmark, args.test_mode, args.limit)), indent=2))
            return
        if args.command == "metrics":
            from .metrics import run_metric
            print(json.dumps(run_metric(args.input), indent=2))
            return
        from .pipeline import generate, judge
        if args.command == "local":
            from .local import LocalConfig
            config = LocalConfig(**{name: getattr(args, name) for name in LocalConfig.__dataclass_fields__})
            failures = generate(args, config)
        else:
            api_key = os.environ.get("UNIPHYSICS_API_KEY") or os.environ.get("OPENAI_API_KEY")
            base_url = args.base_url or os.environ.get("UNIPHYSICS_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
            if base_url:
                url = urlsplit(base_url)
                if url.scheme not in {"http", "https"} or not url.netloc or url.username or url.password or url.query or url.fragment:
                    raise ValueError("base-url must be an HTTP(S) endpoint without credentials, query or fragment")
            config = APIConfig(model=args.model if args.command == "api" else (args.model_1 or "unused"),
                               temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
                               max_workers=args.num_workers, max_retries=args.max_retries, timeout=args.timeout,
                               mode="eval" if args.command == "api" else "judge",
                               enable_thinking=getattr(args, "enable_thinking", True))
            if args.command == "api":
                if not api_key:
                    raise ValueError("Set UNIPHYSICS_API_KEY or OPENAI_API_KEY in your environment")
                failures = generate(args, config, api_key, base_url)
            else:
                failures = judge(args, config, api_key, base_url)
    except (ValueError, OSError, ImportError) as error:
        parser.error(str(error))
    if failures:
        raise SystemExit(1)
