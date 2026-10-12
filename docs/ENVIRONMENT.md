# Installation from the swift environment

The source evaluation environment is named `swift` and uses Python **3.12.0** on Linux. Its imported backend versions are PyTorch **2.6.0+cu124** (CUDA **12.4**) and vLLM **0.8.4**; Transformers is **4.51.3**. `requirement.txt` pins the nine direct evaluation/backend packages to the versions installed in that environment. The public PyTorch CUDA 12.4 wheel index supplies the `+cu124` builds.

## Fresh evaluation environment

Run these commands from the repository root on Linux x86_64 with an NVIDIA driver compatible with CUDA 12.4:

```bash
conda create -n uniphysics python=3.12.0 pip -y
conda activate uniphysics
python -m pip install --upgrade pip
python -m pip install -r requirement.txt
python -m pip install --no-deps -e .
python -m pip check
python -m uniphysics validate
python -c "import torch, vllm; print(torch.__version__, torch.version.cuda, vllm.__version__)"
```

`--no-deps` on the project installation preserves the versions selected by `requirement.txt`. The pins represent the source environment's evaluation stack; transitive dependencies are resolved by pip and are not a complete lockfile. Installation downloads large GPU wheels. This release verifies the existing environment's backend imports, not a fresh multi-gigabyte installation or full GPU inference.

## Existing swift environment

```bash
conda activate swift
python -m pip install --no-deps -e .
python -m uniphysics validate
bash Eval/run_local_eval.sh --help
```

There is no need to reinstall all packages into the existing environment. Despite its name, the `swift` Conda environment is distinct from the MS-SWIFT training package. UniPhysics evaluation does not import MS-SWIFT.

## Full inventory and known source-environment issues

`docs/swift-environment.txt` inventories all 263 installed distributions, using only package names and versions. It is an audit record, not an installation command. The local editable MS-SWIFT development checkout is recorded as a comment with no private path or source URL; its source is not included or required by the evaluator.

At export time, `pip check` reported these existing environment problems:

- `cuda-python==13.3.1` requires `cuda-bindings`, which is missing.
- `trl==0.24.0` requires `transformers>=4.56.1`, while the installed version is `4.51.3`.
- `omegaconf==2.3.1` requires `antlr4-python3-runtime==4.9.*`, while the installed version is `4.13.2`.

Those unrelated training/tooling packages are not included in the evaluator's `requirement.txt`. The original environment has not been modified to resolve these conflicts. The evaluation pins preserve its functioning PyTorch/vLLM imports without exporting its private editable checkout.

For API-only evaluation on another platform, use a lightweight Python 3.10+ environment and `python -m pip install -e .`; the CUDA requirements file is not needed.
