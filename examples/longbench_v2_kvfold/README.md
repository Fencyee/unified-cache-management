# LongBench V2 KVfold 精度对比

这个目录用于在 Qwen3-32B TP4 上比较 Baseline 与多种 KVfold BF16 codec。目前支持：

| `KVFOLD_CODEC` 规范名称 | 算法 | 主要模式 | 回退模式 |
|---|---|---|---|
| `tunstall_bf16_r160` | 固定 1.60x Tunstall R160 | `high_precision` | `quantized` |
| `tunstall_bf16_r200` | 固定 2.00x Tunstall R200 | `tunstall` | `fp8_fallback` |
| `r160_base_bf16` | Base15 R160 | `base15` | 无回退，空间不足时报错 |

兼容别名 `r160_tunstall`、`r200`、`r160` 和 `r160_base15` 仍可使用，但正式结果建议始终记录上表中的规范名称。其中 `r160` 指向 `r160_base_bf16`，不会与 `tunstall_bf16_r160` 混用。

目录包含三套算法所需的最小原生源码快照、统一 C bridge、算法注册表、vLLM-Ascend monkeypatch、服务启动脚本、LongBench V2 客户端和配对比较脚本。它可以单独复制到另一台机器运行，不读取 UCM 仓库中的其他源码或脚本。

> 这是精度压力测试，不是 UCM Store 性能测试。补丁会在首次 prefill 的当前 attention 之前重建新生成的 K/V，因此比真实 `save -> external load` 路径更激进。测试时不要同时启用 UCM `Cache|Compress|Posix`，否则可能重复处理 K/V。

## 目录与外部依赖

目标机器仍需具备：

- 4 张可用于 TP4 的 Ascend NPU；
- Qwen3-32B BF16 模型和 tokenizer；
- 已能正常运行 Qwen3-32B 的 vLLM、vllm-ascend 和 torch-npu 环境；
- Python 包 `wrapt`；
- C++17 编译器、`sha256sum` 和 `awk`；
- LongBench V2 的 503 条数据文件。

推荐服务端组合为 vLLM 0.19.1 与 vllm-ascend 0.19.1rc1。不要用 LongBench 官方的旧 vLLM 依赖覆盖服务端环境：

```bash
# 服务端仅补充 hook 依赖（若尚未安装）
python3 -m pip install -r requirements-server.txt

# 评测客户端依赖可以装在另一个 Python 环境
python3 -m pip install -r requirements-client.txt
```

复制目录时不要携带为另一种 CPU 指令集构建的 `build/`：

```bash
rsync -a --exclude build --exclude results --exclude logs \
  longbench_v2_kvfold/ user@target:/mnt/longbench_v2_kvfold/
```

后续命令统一假设：

```bash
export TEST_DIR=/mnt/longbench_v2_kvfold
cd "${TEST_DIR}"
```

## 测试链路与压缩单元

```text
Qwen3Attention
  -> QKV projection -> Q/K Norm -> RoPE
  -> AscendAttentionBackendImpl.forward       <-- monkeypatch
       -> 选取 num_actual_tokens 中完整的 128-token 物理 block
       -> 组织为 [K/V, token, local KV head, channel]
       -> D2H -> 所选 codec 压缩 -> 原地解压 -> H2D
       -> 原始 reshape_and_cache
       -> 原始 attention forward
```

Qwen3-32B 有 8 个 KV head，TP4 时每个 rank 有 2 个本地 KV head。一个完整 block 包含：

```text
2 (K/V) × 128 token × 2 local KV heads × 128 channel
= 65,536 BF16
= 131,072 raw bytes
```

三种算法对该 block 的记录大小分别为：

| Codec | 固定记录大小 | 实际 raw/record | 说明 |
|---|---:|---:|---|
| `tunstall_bf16_r160` | 81,920 B | 1.600000x | 名义 5/8 大小向下对齐至 4 KiB；空间不足时整 block 量化回退 |
| `tunstall_bf16_r200` | 65,536 B | 2.000000x | Tunstall 流放不下时使用 FP8 fallback |
| `r160_base_bf16` | 86,016 B | 1.523810x | `5N/4+1` 向上对齐至 4 KiB，余量保存 exception 与可选 M1/M0 |

不足 128 token 的尾块不处理。补丁会严格校验 eager、TP4、BF16、KV cache shape、关闭 prefix cache/speculative decoding、串行调度，以及 `slot_mapping` 中每组 token 确实构成一个完整物理 block；不满足实验口径时请求直接失败。

## 1. 修改服务配置

编辑 `qwen3_32b_longbench.properties`，至少确认：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
model=/mnt/models/Qwen3-32B
server_port=8000
gpu_memory_utilization=0.87
```

Baseline 和所有候选算法必须使用同一份配置。专用配置已固定：

- TP4、`max_model_len=131072`；
- `max_num_batched_tokens=2048`、`max_num_seqs=1`、block size 128；
- 通过 vLLM 0.19.1 的 `--hf-overrides` 注入 Qwen3 YaRN `rope_parameters`；
- `--enforce-eager`；
- 关闭 prefix caching、speculative decoding 和异步调度；
- YaRN factor 4；
- 不启用 UCM Store。

## 2. 构建并运行三算法自测

bridge 只读取本目录 `native/codec/` 中的源码。由于使用 `-march=native`，换机器后必须重新构建：

```bash
cd "${TEST_DIR}"
bash build_bridge.sh
python3 smoke_test_codec.py --codec all
```

也可以单独测试：

```bash
python3 smoke_test_codec.py --codec tunstall_bf16_r160
python3 smoke_test_codec.py --codec tunstall_bf16_r200
python3 smoke_test_codec.py --codec r160_base_bf16
```

自测不仅检查 API 是否能调用，还会分别覆盖：

- Tunstall R160 的 high-precision 和 quantized 模式；
- Tunstall R200 的 Tunstall 和 FP8 fallback 模式；
- Base15 R160 的 M1/M0 恢复规则与 exception 超预算报错；
- 输入不被修改、固定记录大小、批量 block 边界、模式计数和结果确定性。

bridge 会把实际源码内容的 SHA256 写入每个算法的 `source_id`，worker 结果也会记录该值。源码快照来源见 `native/codec/SOURCE_MANIFEST.md`。

## 3. 准备 LongBench V2 数据

```bash
python3 download_dataset.py --output /mnt/longbench-v2/data.json
```

Hugging Face 下载慢时，可在网络较好的机器下载后直接复制 `data.json`。这是固定 revision 的 503 条评测题，不依赖外部 judge。脚本完成后会打印条数、大小和 SHA256。

## 4. 运行 Baseline

终端 A 启动服务：

```bash
cd "${TEST_DIR}"
KVFOLD_ATTN_ENABLE=0 bash start_vllm.sh qwen3_32b_longbench.properties
```

终端 B 运行评测：

```bash
export TEST_DIR=/mnt/longbench_v2_kvfold
export SERVER_CONFIG="${TEST_DIR}/qwen3_32b_longbench.properties"
source "${SERVER_CONFIG}"
cd "${TEST_DIR}"

KVFOLD_ATTN_ENABLE=0 python3 run_longbench.py \
  --base-url "http://127.0.0.1:${server_port}/v1" \
  --served-model-name "${served_model_name}" \
  --run-id baseline-qwen3-001 \
  --server-config "${SERVER_CONFIG}" \
  --tokenizer "${model}" \
  --dataset-json /mnt/longbench-v2/data.json \
  --subset full \
  --output results/baseline.jsonl
```

首次使用建议先把 `--subset full` 和输出文件分别改为 `--subset smoke`、`results/baseline-smoke.jsonl`，确认 12 题都成功后再跑完整 503 题。

## 5. 运行任一候选算法

完整停止 Baseline 服务，为候选算法选择规范名称和全新的 run ID：

```bash
cd "${TEST_DIR}"
export KVFOLD_ATTN_ENABLE=1
export KVFOLD_CODEC=tunstall_bf16_r160
export KVFOLD_RUN_ID=tunstall-r160-qwen3-001
bash start_vllm.sh qwen3_32b_longbench.properties
```

另外两种算法只需替换：

```bash
# 固定 2.0x
export KVFOLD_CODEC=tunstall_bf16_r200
export KVFOLD_RUN_ID=tunstall-r200-qwen3-001

# Base15 R160
export KVFOLD_CODEC=r160_base_bf16
export KVFOLD_RUN_ID=base15-r160-qwen3-001
```

每次切换算法都必须完整重启服务。日志中四个模型 worker 均应打印规范 codec 名、运行时 shape 和各自的 `source_id`。

终端 B 的候选评测命令与 Baseline 相同，但要使用独立 run ID 和结果文件。例如：

```bash
KVFOLD_ATTN_ENABLE=0 python3 run_longbench.py \
  --base-url "http://127.0.0.1:${server_port}/v1" \
  --served-model-name "${served_model_name}" \
  --run-id tunstall-r160-qwen3-001 \
  --server-config "${SERVER_CONFIG}" \
  --tokenizer "${model}" \
  --dataset-json /mnt/longbench-v2/data.json \
  --subset full \
  --output results/tunstall-r160.jsonl
```

候选 worker stats 默认写到：

```text
results/<KVFOLD_RUN_ID>-worker-stats/
```

## 6. 配对比较

候选评测完成后，先停止服务。每个 worker 会在进入 shutdown、释放 NPU 和分布式资源之前写入最终统计；确认四份统计均为 `final=true` 后再比较。例如：

```bash
KVFOLD_ATTN_ENABLE=0 python3 compare_results.py \
  --baseline results/baseline.jsonl \
  --candidate results/tunstall-r160.jsonl \
  --codec tunstall_bf16_r160 \
  --candidate-stats-dir results/tunstall-r160-qwen3-001-worker-stats \
  --expected-workers 4 \
  --json-output results/tunstall-r160-comparison.json \
  | tee results/tunstall-r160-comparison.md
```

比较另外两种算法时，将 `--candidate`、`--codec` 和 stats 目录替换成对应值即可。比较脚本会检查：

- 两边包含完全相同的题目 ID、答案和实际 prompt SHA256；
- 模型、tokenizer、截断、生成参数和服务配置 SHA256 一致；
- 四个 TP worker 都实际处理过 block；
- 四份统计都是 worker 正常退出时写入的 `final=true` 快照；
- 每个 TP rank 均覆盖 Qwen3-32B 的全部 64 层，且各 rank 的工作量一致；
- codec 名、codec ID、run ID、源码 ID 和模式统计一致；
- worker 没有压缩或解压失败。

输出包含 Overall、Easy/Hard、Short/Medium/Long、各 domain 准确率，以及双方都对、仅 Baseline 对、仅候选对、双方都错、答案翻转率和 McNemar 精确检验。

## 7. 后续新增算法

目录采用注册表设计。增加新 codec 时按以下位置扩展，不需要改 attention hook 或 LongBench 客户端：

1. 将最小算法源码放入 `native/codec/`，并更新 `SOURCE_MANIFEST.md`；
2. 在 `native/kvfold_codec_bridge.cc` 增加稳定 codec ID、记录大小函数、roundtrip adapter 和 `CodecOps` 表项；
3. 在 `build_bridge.sh` 加入源码与独立 `source_id`；共享依赖源码只能链接一次；
4. 在 `codec_catalog.py` 注册规范名称、显示名和两种模式标签；
5. 在 `smoke_test_codec.py` 注册算法专属的确定性输入与精度断言；
6. 重新构建，先通过 `--codec all`，再开展完整 LongBench 配对测试。

不要修改已有 codec ID 或让旧别名指向另一套算法，否则历史结果会失去可比性。
当前原生统计 ABI 统一按 `primary/fallback` 两种执行模式表达；若新算法包含三种及以上模式，需同步升级 bridge ABI、Python 统计结构和结果 schema。

## 注意事项

- 完整评测必须是 503/503；请求失败默认算入分母并使比较脚本报错。
- Baseline 与每个候选算法使用不同的结果文件和 run ID，不要混用 `--resume`。
- 当前口径为串行、non-thinking、temperature=0，适合比较压缩带来的相对精度变化，不应直接与其他 CoT、YaRN 或采样配置的公开榜单分数比较。
- CPU codec 会产生 D2H、H2D 和同步开销，请求耗时不能代表 UCM 生产性能。
- `r160_base_bf16` 在 exception 流放不进固定记录时会严格失败；另两种算法会记录各自的回退 block 数。
