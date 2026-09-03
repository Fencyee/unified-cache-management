# LongBench V2 纯 Transformers 精度测试

这是不启动 vLLM、不走 HTTP、也不启用 UCM Store 的简化入口。一个 Python
进程直接加载 Qwen3-32B，Baseline 使用原始 `DynamicCache`；候选测试临时
monkeypatch `DynamicCache.update()`，在 Qwen3 完成 RoPE 后把 K/V 送入所选
KVfold codec，执行一次“压缩 → 解压”，再继续原始 Attention 和生成流程。

三种算法共用本目录的同一套原生源码与 bridge：

| `--codec` | 含义 |
|---|---|
| `none` | Baseline，不处理 K/V |
| `tunstall_bf16_r160` | 原 Tunstall BF16 固定 1.60x，空间不足时量化回退 |
| `tunstall_bf16_r200` | Tunstall BF16 固定 2.00x，空间不足时 FP8 回退 |
| `r160_base_bf16` | 新 Base15 R160，利用 4 KiB 对齐余量保存 M1/M0 |

> 该入口只用于比较精度，不能用它测 TTFT 或吞吐。Transformers 的
> `device_map=balanced` 是按层放到四张 NPU 上，不是 vLLM TP4。为保持 codec
> 口径一致，脚本会在每层将 8 个 KV heads 显式拆成 4 组 × 2 heads，模拟
> Qwen3-32B TP4 的压缩单元。

## 前置条件

- 4 张可用 Ascend NPU，建议每张 64 GiB；
- 本地 Qwen3-32B BF16 模型与 tokenizer；
- 已匹配的 PyTorch、torch-npu、CANN 环境；
- Transformers 4.51+ 或 5.x、Accelerate 1.10+；
- Bash、C++17 编译器、`sha256sum` 和 `awk`；
- LongBench V2 固定 revision 的 503 条 `data.json`。

先进入这个可独立复制的目录：

```bash
export TEST_DIR=/mnt/longbench_v2_kvfold_transformers
cd "${TEST_DIR}"
```

不要安装 LongBench 官方 requirements 中的旧 vLLM。只在现有 torch-npu 环境
补充离线入口依赖：

```bash
python3 -m pip install -r requirements.txt
```

## 1. 构建并检查三种 codec

换机器后必须在目标机重新构建：

```bash
bash build_bridge.sh
python3 smoke_test_codec.py --codec all
```

## 2. 准备数据

```bash
python3 download_dataset.py --output /mnt/longbench-v2/data.json
```

下载慢时，直接在网络较好的机器下载后复制 `data.json`；运行期不需要联网，
也不需要外部 judge。

## 3. 先跑最小 Baseline smoke

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

python3 run_longbench.py \
  --model /mnt/models/Qwen3-32B \
  --dataset-json /mnt/longbench-v2/data.json \
  --codec none \
  --run-id baseline-smoke-001 \
  --subset smoke \
  --max-samples 1 \
  --output results/transformers-baseline-smoke.jsonl
```

默认配置已经固定为：BF16、SDPA、`device_map=balanced`、non-thinking、greedy、
YaRN factor 4、最大上下文 131072、最大输入 120000、最大输出 128。加载后会
严格检查模型是 Qwen3-32B（64 层、8 KV heads、head dim 128）、四张 NPU 都被
使用，且没有权重落到 CPU 或磁盘。

如果 Accelerate 需要显式预留显存，可以重复传入：

```bash
  --max-memory 0=56GiB \
  --max-memory 1=56GiB \
  --max-memory 2=56GiB \
  --max-memory 3=56GiB
```

具体上限应按目标机器空闲 HBM 调整，脚本不替机器硬编码。

## 4. 跑候选算法 smoke

以原 Tunstall R160 为例：

```bash
python3 run_longbench.py \
  --model /mnt/models/Qwen3-32B \
  --dataset-json /mnt/longbench-v2/data.json \
  --codec tunstall_bf16_r160 \
  --run-id tunstall-r160-smoke-001 \
  --subset smoke \
  --max-samples 1 \
  --output results/transformers-tunstall-r160-smoke.jsonl
```

测试另外两种算法只需将 `--codec` 和输出文件分别改为：

```text
tunstall_bf16_r200
r160_base_bf16
```

每条题目都会记录实际 input token SHA256 和该题经过 codec 的 block/mode
统计；候选运行结束后还会生成：

```text
<output>.codec-stats.json
```

脚本要求每题的 64 层全部命中 hook，否则立即报错。中断后可以用完全相同的
命令加 `--resume` 接着跑；metadata 不一致时会拒绝混合结果。

## 5. 配对比较 smoke

```bash
python3 compare_results.py \
  --baseline results/transformers-baseline-smoke.jsonl \
  --candidate results/transformers-tunstall-r160-smoke.jsonl \
  --codec tunstall_bf16_r160 \
  --allow-partial \
  --json-output results/transformers-tunstall-r160-smoke-comparison.json
```

比较脚本会检查两次运行的模型配置、数据、题目、Prompt、实际输入 token IDs、
Transformers/torch-npu 版本、YaRN、device map 和生成参数完全一致，并验证
候选的 codec 源码 ID、64 层覆盖、字节统计、模式计数和错误数。

## 6. 完整 503 题

Smoke 全部通过后，分别换新 run ID 和输出文件，同时去掉 `--subset smoke` 和
`--max-samples 1`（默认就是 `full`），依次运行 Baseline 与三个候选，再分别
比较。完整比较默认要求 503/503 条结果。

## 工作原理与边界

```text
Qwen3 Attention
  -> Q/K Norm -> RoPE
  -> DynamicCache.update()                 <-- 临时 monkeypatch
       -> 只选择全局对齐的完整 128-token block
       -> 8 KV heads 拆成 4 个 2-head TP4 codec block
       -> D2H -> compress -> decompress -> H2D
       -> 调用原 DynamicCache.update()
  -> 原 Attention -> greedy generation
```

- 不足 128 token 的尾部和逐 token Decode K/V 保持原值；
- 传入 Attention 的原始 K/V tensor 不会被原地修改；
- CPU 往返按小批 block 进行，防止一次复制整层长上下文 K/V；
- 当前 prefill 的 Attention 也使用重建后的 K/V，因此这是比真实
  `save -> external load` 更激进的精度压力测试；
- 120K prefill 必须实际走 Ascend 融合 SDPA。若当前 torch-npu/Transformers
  组合不支持，不要退回 eager（显式注意力矩阵会 OOM），应修复软件栈或使用
  单独的 vLLM 测试方案；
- `device_map=balanced` 适合功能和精度筛查，但单请求通常按层串行经过四张卡，
  运行完整 503 题可能很慢。

后续增加 codec 时仍扩展统一的 native registry、`codec_catalog.py` 和 codec
smoke；Transformers hook 与 LongBench runner 无需按算法复制。
