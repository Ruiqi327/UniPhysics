# UniPhysics

**用于评估大语言模型物理推理能力的统一、质量受控评测框架。**

[English README](README.md) 包含完整参数、框架说明、论文图和引用信息。

## 发布内容

- 清洗集：`Eval/benchmark/uniphysics.jsonl`，5,352 条。
- 剔除集：`Eval/benchmark/UniPhysics_deleted.jsonl`，3,751 条。
- 两份数据逐字节保留，校验和见 `Eval/benchmark/manifest.json`。
- API、本地 vLLM 推理、答案提取、混合评判、分组指标与断点续跑。
- 原始三个 Shell 入口仍保留，核心实现统一放在 `uniphysics/`。

## 安装

原始评测使用 Conda **`swift` 环境**：Python **3.12.0**、PyTorch **2.6.0+cu124**（CUDA 12.4）、vLLM **0.8.4**、Transformers **4.51.3**。已根据该环境导出评测依赖文件 [requirement.txt](requirement.txt)。以下命令在项目根目录执行，本地推理需要 Linux x86_64 和兼容 CUDA 12.4 的 NVIDIA 驱动。

```bash
conda create -n uniphysics python=3.12.0 pip -y
conda activate uniphysics
python -m pip install --upgrade pip
python -m pip install -r requirement.txt
python -m pip install --no-deps -e .
python -m pip check
python -m uniphysics validate
```

如果已经有原来的 `swift` 环境，只需：

```bash
conda activate swift
python -m pip install --no-deps -e .
python -m uniphysics validate
```

`requirement.txt` 锁定当前环境中 9 个评测及推理后端包的版本，间接依赖由 pip 解析；不是全量环境锁文件。完整的 263 项版本清单另存为 [docs/swift-environment.txt](docs/swift-environment.txt)，已去除私人路径，仅用于审计。原环境存在三项依赖问题和一个私有可编辑 MS-SWIFT 安装，详见 [环境说明](docs/ENVIRONMENT.md)；评测不依赖该私有训练源码。

仅使用 API 时，不必安装 CUDA 依赖：在 Python 3.10+ 环境中执行 `python -m pip install -e .` 即可。上面的安装命令会下载较大的 GPU 软件包；目前已确认原 `swift` 环境可导入 PyTorch/vLLM，未重新安装完整 GPU 环境。

## API 评测

凭证只从环境变量读取，不写入代码、参数或运行配置。将模型占位名替换为服务商支持的真实模型 ID。

```bash
read -rsp "API key: " UNIPHYSICS_API_KEY; echo
export UNIPHYSICS_API_KEY
export UNIPHYSICS_BASE_URL="https://api.openai.com/v1"

bash Eval/run_api_eval.sh \
  --project_name api-demo --model YOUR_MODEL_ID --limit 5

bash Eval/judge.sh \
  --project_name api-demo \
  --model_1 FIRST_JUDGE_ID \
  --model_2 SECOND_JUDGE_ID \
  --model_3 ARBITER_ID
```

首轮两个评判模型可以使用相同 ID，但会分别发起独立请求。API 的 `temperature` 和 `top_p` 默认不传；显式设置时会真正传给服务商。所有评判模型共用指定的兼容接口。

## 本地模型

```bash
# 使用上面安装好的环境；已有原环境时改为 conda activate swift。
conda activate uniphysics
bash Eval/run_local_eval.sh \
  --project_name qwen3-4b --model-path Qwen/Qwen3-4B \
  --gpus 0 --temperature 0.7 --top_p 0.8 --top_k 20 \
  --max_tokens 16384 --enable_thinking true
```

模型可使用 Hugging Face ID 或自行准备的模型目录。多 GPU 使用独立副本，每张卡需要能容纳整个模型，不是张量并行。`max_model_len` 必须给输入和输出都留出空间。

## 数据与结果约定

`--benchmark all` 完整读取当前文件；可用 `--benchmark MMLU-Pro,GPQA` 筛选。`--test_mode mini` 按文件顺序取每个来源最多 500 条。`--limit` 用于小规模检查，改跑全量时使用新的项目名。

剔除集保留原始缺陷：所有记录都标注为 `Open-Ended`，9 条没有参考答案，4 条使用 `NBPhO_` 或 `PanPhO_` 前缀。评判时无参考答案的记录会标为 `missing_gold`，不送给模型猜测标准答案。剔除集用于审计和噪声分析，不应与清洗集合并。

输出默认位于当前目录的 `results/<project_name>/`。生成与评判使用同一输出目录。加 `--resume` 可复用匹配的成功结果，失败记录会重试；数据、代码或生成配置变化时会拒绝错误续跑。规则题直接判分，开放题由两个 LLM 独立判断，分歧交给第三次判定。

`metrics.json` 和 `metric.txt` 同时提供微平均、宏平均、物理领域/题型/答案类型分组结果，以及覆盖率、失败数。准确率的分母是已解决的判定数；必须同时报告覆盖率，不能忽略失败和无参考答案的记录。

```bash
python -m uniphysics metrics --input results/api-demo/judge.jsonl
python -m pip install -e '.[dev]'
python -m pytest -q
python tools/check_release.py
```

代码采用 MIT；上游数据和论文配图不因代码许可证而被重新授权，详见 [DATA_LICENSE.md](DATA_LICENSE.md)。发布包不包含原项目历史、模型权重、历史评测结果或私有清洗脚本。
