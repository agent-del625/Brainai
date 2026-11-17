# 评测基线

> 本文件是 `eval_all.py` 产出的指标的唯一对照口径。基线数字来自 Issue #1《项目背景与下一步计划》中「已验证的数字」，是后续一切改进「进步 / 退步 / 持平」的判据。

## 怎么跑

```bash
python3 eval_all.py                          # 全部探针，输出 logs/baseline.json（最新）+ 归档
python3 eval_all.py --selftest               # 无资产自检（校验指标函数与落盘）
python3 eval_all.py --only unified,caption   # 只跑指定探针
```

- 根目录默认 `/srv/brainai`，可用 `BRAINAI_ROOT` 环境变量或 `--root` 覆盖。
- 资产（数据 / 权重）缺失的探针记 `skip` 并写明原因，不中断其它探针。
- 每次跑完 `logs/baseline.json` 永远指向最新，历史存档为 `logs/baseline_<时间>.json`。

## 事实基线（Issue #1「已验证的数字」）

| 能力 | 指标 | 数值 | 说明 |
|---|---|---|---|
| 图像描述（独立 VLM，beam） | chrF | 59.7% | 束搜索 + 判断重排 |
| 统一模型 55.8M | 图像 chrF | 49.5% | 与上行为不同模型/口径 |
| 统一模型 55.8M | 声音识别 | 84% | |
| 声音分类（CNN + mel） | 准确率 | 93.9% | |
| 本地 ASR | 实时率 / 英文 WER | 0.22 / 约 2% | |
| 本地 TTS | VITS 中文 / espeak | 906ms / 15ms | |
| 端到端语音 | ASR+模型+TTS | 357+350+1040 ms | |
| 识物（150 类主体） | 准确率 | 77% | 显著性排序 + 候选约束 |

## 口径约定（两处历史数字不一致，本文件统一）

- **问答内容准确率**：Issue 记 **27%**，`report_overview.py` 记 **54.6%**。二者是不同时间点 / 不同评测方式，无法直接对账。
  本文件采用 `eval_all.py` 的 `qa` 探针口径：**固定测试集上「期望答案子串命中率」**。当前 `/api/ask` 仍是规则表（未接真模型），
  该口径可复现；接真模型后用同一测试集同口径复跑，即可量化对比。规则表口径下的当前值为 **75%（3/4）**。

## 基线最小格式（`logs/baseline.json`）

```json
{
  "schema_version": 1,
  "generated_at": "2026-11-17T18:44:57+08:00",
  "root": "/srv/brainai",
  "ckpt": "ckpt/model.pt",
  "seed": 0,
  "n": 32,
  "probes": {
    "unified": {"status": "ok", "metrics": {"重建MSE": [0.046], "视频→文本_R@1": 65.6}, "seconds": 1.2},
    "qa": {"status": "ok", "metrics": {"问答内容准确率": 75.0}, "seconds": 0.0}
  },
  "metrics": {
    "unified": {"重建MSE": [0.046], "视频→文本_R@1": 65.6},
    "qa": {"问答内容准确率": 75.0}
  }
}
```

## 探针与指标对照

| 探针 | 指标 | 复用来源 |
|---|---|---|
| `unified` | 重建 MSE/PSNR、视频→文本/音频→文本/视频→音频 R@1/R@5、生成字符准确率、参数量 | `eval.py` |
| `caption` | 图像描述 chrF | `report_final.py` / `cap_eval_beam.py` |
| `sound_vq` | VQ 码本使用率、重建 MSE、往返一致性、分频段误差 | `amae2.py` |
| `sound_clf` | 声音分类准确率（连续 vs 量化 token） | `asem.py` |
| `qa` | 问答内容准确率（规则表口径） | `api_server.py` |
| `latency` | 端到端首音延迟 / 总耗时 | `stream_pipeline.py` |
| `selftest` | 指标函数自检（无资产） | — |

## 复现约定

- 固定随机种子（`--seed 0`），固定测试集切分（沿用各脚本已有的 `perm`/`tr`/`te` 切分方式）。
- 数据 / 权重不在仓库里（`.gitignore` 排除 `data/`、`ckpt/`、`logs/`），复跑依赖树莓派或 Windows 训练机上的资产。
- 每阶段结束跑一次 `eval_all.py`，与上一版基线对比后再进下一阶段。
