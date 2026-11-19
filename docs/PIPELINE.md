# Brainai 全流程手册

> 目标：在树莓派 5（16GB）+ 一台带 GPU 的 Windows 机器上，从零训练一个**非 Transformer 主干的、类脑的、能生成新答案的多模态模型**，并把图像/声音/文字打通成可交互系统。
>
> 本文档记录**每一步怎么做、产物在哪、判据是什么、出过什么坑**。

---

## 0. 总览

### 0.1 当前进度（截至 2026-11-19）

| 模块 | 状态 | 关键数字 |
|---|---|---|
| 分词器（SetBPE） | ✅ 完成 | 词表 18,875，压缩率 1.45 字/token |
| 图像 DCT 编解码 | ✅ 验证 | 6 系数/块，PSNR 21.1 dB |
| 声音 DCT 编解码 | ✅ 验证 | 64 系数，PSNR 28.2 dB |
| 纯文本 LM | ✅ 可训练 | 100M 参数，困惑度 6-8（语料受限） |
| 统一 token 模型 | ✅ 跑通 | 8.1M，图像+声音+文字共享嵌入表 |
| 图像→字幕生成 | ❌ 塌缩 | 输出 "a man in a a a a..." |
| RL（DPO） | 📝 脚本就绪 | 未跑 |

### 0.2 硬件

| 机器 | 用途 | 配置 |
|---|---|---|
| 树莓派 5 | 推理、小模型训练、数据准备、Web 服务 | 16GB，Debian 13，4 核 |
| Windows 训练机 | 大模型训练 | i9-13900HK，RTX 3070 Laptop 8GB，64GB 内存 |

---

## 1. 环境

### 1.1 树莓派

```bash
# 系统包
sudo apt install -y tmux zstd unzip git

# Python 依赖（用 --break-system-packages 因为 Debian 13 有 PEP668 限制）
pip install --break-system-packages torch soundfile scipy numpy pyarrow pillow \
  librosa onnxruntime sherpa-onnx flask requests zstandard
```

### 1.2 Windows（重装后照做）

```powershell
# 1) 装 Python 3.11（勾选 Add to PATH）
# 2) 装 GPU 版 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 3) 其它
pip install numpy scipy pyarrow pillow
# 4) 验证
python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
```

### 1.3 远程访问约定

```bash
# 从树莓派访问 Windows（局域网优先）
sshpass -p '<密码>' ssh admin@192.168.1.3 'powershell -NoProfile -Command "..."'

# 传文件
sshpass -p '<密码>' scp 文件 admin@192.168.1.3:C:/work/文件
```

**规则**：
- PowerShell 脚本**只用 ASCII**，中文会破坏解析
- 列目录用 `cmd /c dir`，别用 `Get-ChildItem`
- 长任务放 `setsid nohup ssh ... powershell -File 脚本.ps1 &`，别用 `Start-Process`（父进程退出会被杀）

---

## 2. 数据管线

### 2.1 语料来源与清洗

| 语料 | 来源 | 清洗规则 | 产物 |
|---|---|---|---|
| 中文 | 中文维基（254k 条）+ 中文 C4（96 片中的 27 片） | 长度 30-400 字；中文字符占比 >50% | `data/big/zh_all.txt`（906MB） |
| 英文 | wikitext-103 + openwebtext | 长度 200-6000 字符；字母占比 >55% | `data/big/en.txt`（383MB） |
| 代码 | CodeSearchNet Python（3 片，32 万函数） | 长度 150-5000；≥5 行有效代码；最多 160 行 | `data/big/code.txt`（104MB） |

### 2.2 配比：**必须按我们的 token 数算，不能按文档数或词数**

这是踩过的最大坑：按文档数配 4:4:2，实际 token 占比是 **63% 中文 / 19% 英文 / 18% 代码**（因为英文句子平均只有 22.6 token，中文维基平均 264 token）。

正确做法：
1. 用分词器把每份语料编码，统计**实际 token 数**
2. 以英文为基准，中文截到同等 token 量，代码取一半
3. 得到 4:4:2

脚本：`lm2.py` 里的 `build_pool()`（按 token 配额重复采样文档）。

### 2.3 flickr30k（图文对）

```bash
# 下载（4.39GB，用循环续传，别用单次 curl）
for i in $(seq 1 200); do
  curl -sL -C - --max-time 300 -o flickr30k.zip \
    "https://hf-mirror.com/datasets/nlphuji/flickr30k/resolve/main/flickr30k-images.zip"
  sleep 5
done
unzip -q flickr30k.zip -d flickr30k
```

- 图：28,235 张 jpg（`data/downloads/flickr30k/flickr30k-images/`）
- 字幕：`data/downloads/f30k_caps.csv`（5 句/图）
- 编码成 DCT token：`build_f30k_tokens.py` → `data/f30k_dct_tokens.npz`（12,000 图 / 60,000 对）

### 2.4 声音

| 数据 | 数量 | 处理 |
|---|---|---|
| ESC-50 | 2,000 条 / 50 类 | 数据增强到 10,000 条（平移+噪声+增益） |
| UrbanSound8K | 3,276 条 / 10 类 | 从 parquet 里的音频字节重新算 mel |

---

## 3. 分词器：SetBPE

### 3.1 为什么不用现成的

- BPE：只看相邻对频次，学不到"词族结构"
- 我们的早期版本：频次阈值截断，是**静态词表**，切分不唯一

### 3.2 SetBPE 的做法

```
gain = 频次 × (1 + λ·log(1 + 词族数))      λ = 0.35
```

**词族数**：一个片段被多少个不同词当作前缀/后缀共享过（集合切分统计）。

效果：`field` 这种"频次一般但属于 field/fielder/fields 词族"的片段会被优先合并，而不是只被 `in`/`the` 这种高频词垄断。

### 3.3 训练

```bash
python3 setbpe3.py 8000        # 8000 次合并，约 55 分钟
# 产物：data/setbpe.pkl（词表 18,875 + 合并表）
```

### 3.4 判据

| 指标 | 旧（统计词表） | SetBPE |
|---|---|---|
| 词表 | 59,603 | **18,875** |
| 压缩率 | 0.99 字/token | **1.45 字/token** |

同一段文本 token 数少 32%，嵌入层参数少 68%。

### 3.5 编码流程（必须和训练一致）

1. 按空白切成"段"
2. 每段做 BPE 合并（按 merge 排名的**最低**优先）
3. 中文按字进入，英文转小写
4. 段之间插 SPACE token（id=4）

---

## 4. DCT 编解码

### 4.1 图像

```
8×8 块 DCT → zigzag 前 6 个系数（i+j≤2，低频带）→ 各自量化 → 整数码
```

| 参数 | 值 | 结果 |
|---|---|---|
| 图像尺寸 | 32×32 | 16 个块 |
| token 数 | 96（16 块 × 6 系数） | 词表 288 |
| PSNR | **21.66 dB** | 可解码回图像 |

**关键对比**：块均值（最朴素的 patch token）无论给多少比特都卡在 26.63 dB 以下（音频）或 19.77 dB（图像），因为**块均值丢了块内结构**。

### 4.2 声音

```
mel 80×128 → 2D DCT → 取幅值最大的 64 个系数 → 各 64 档量化
```

| 参数 | 值 |
|---|---|
| token 数 | 64 |
| PSNR | **28.18 dB** |
| 对比 | 块均值 810 比特才 26.63 dB，DCT 用 99 比特就到 27.01 dB |

### 4.3 生成实验的结论（失败但有价值）

试过「AR 生成低频 + 扩散补高频」和「扩散在低频带」两种，都不成功：

| 尝试 | 结果 | 原因 |
|---|---|---|
| AR 低频 + 扩散高频 | 5.16 dB（比基线 6.46 差） | **方向错了**：人眼忽略高频，扩散该在低频 |
| 扩散低频带（修正版） | 15.44 / 低通 18.74（基线 17.16 / 21.31） | 管线对了但**模型太小、数据太少** |

**结论**：树莓派 CPU 上做不了有意义的生成实验，需要 GPU。

---

## 5. 模型训练

### 5.1 纯文本 LM（推荐先做这一步）

```bash
# 生成词表 token 流（Windows 上跑）
python3 prep_all.py        # → data/setbpe_stream_all.npy（456.7M token）

# 训练
python3 lm4.py             # 100M 参数，seq 512，bf16
```

关键配置（照 MiniMind 的做法）：

| 项 | 值 | 为什么 |
|---|---|---|
| 上下文 | **512** | 小模型也要长上下文 |
| 语料组织 | **token 流切窗**（不是按文档 padding） | 零浪费 |
| 学习率 | 1.5e-4 + warmup 3000 + 余弦退火 | 小模型标准配方 |
| 精度 | **bf16**（不是 fp16） | fp16 会 NaN |
| 梯度累积 | 2（等效 batch 16） | 省显存 |

### 5.2 统一 token 多模态模型

```
[M_IMG] 96 个图像 DCT token → 字幕文字 → [EOS]
[M_AUD] 64 个声音 DCT token → 类别文字 → [EOS]
[T_QA]  文字问题 → [SEP] → 答案 → [EOS]
```

**三种模态共用一张 `nn.Embedding`**，词表 = 文字 18,875 + 图像 288 + 声音 64 = 19,227。

```bash
python3 uni_dct.py         # 8.1M（d=256, 4 层），2000 步
```

### 5.3 RL（脚本已就绪，未跑）

| 脚本 | 作用 |
|---|---|
| `rl_reward.py` | chrF 0.5 + 关键词覆盖 0.3 + 长度 0.1 + 去重 0.1 |
| `rl_sample.py` | 同题采样 5 条 → 打分 → 存偏好对 |
| `rl_train.py` | DPO（beta=0.1，lr 1.5e-5） |

---

## 6. 评测

### 6.1 指标定义

| 指标 | 用途 | 注意 |
|---|---|---|
| chrF(3-gram) | 生成质量 | 要和参考答案算 |
| 困惑度 | LM 训练进度 | 换语料后不可跨版本比较 |
| **低通 PSNR** | 图像重建 | **人眼只看低频，原始 PSNR 会误导** |
| 命中率 | 声音分类 | 类别名是否出现在生成文本 |

### 6.2 必须避免的评测错误

1. **用贪心解码评估生成** → 会看到"所有图输出同一句"的假象（束搜索才是真实水平）
2. **测试集和训练集混了** → 数字虚高（早期视频字幕 chrF 79% 是泄漏，真实 35.8%）
3. **换语料后比较困惑度** → 没有可比性

---

## 7. 踩过的坑（速查）

| 坑 | 现象 | 解法 |
|---|---|---|
| fp16 溢出 | loss 变 nan | 换 bf16，加 nan 检测 |
| 词表按文档数配比 | 中文占 63%，英文崩 | 按 token 数配 |
| 贪心解码 | 所有图同一句 | 用束搜索评估 |
| 数据集泄漏 | chrF 虚高 | 按图/视频划分 |
| VQ 码本塌缩 | 只用 1 个码字 | 改用 DCT 或加 EMA/重启 |
| 多进程抢显存 | 训练无输出 | 只留一个进程 |
| 日志缓冲 | 看不到进度 | `python -u` |
| PowerShell 中文 | 脚本解析失败 | 全部 ASCII |
| `Start-Process` 启任务 | SSH 断开后被终止 | 用 `setsid nohup ssh ... -File` |
| 下载大文件断流 | 卡在固定大小 | 循环 `curl -C -` 续传 |
| 硬杀进程 | 系统崩溃风险 | 用 `Stop-Process` 前先存盘 |
| 同时跑多个大内存任务 | 内存打满 → 崩溃 | 启动前查可用内存 |

---

## 8. Windows 重装后的恢复清单

```powershell
# 1) Python 3.11 + PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy pyarrow pillow

# 2) 拉代码
git clone https://github.com/agent-del625/Brainai.git C:\work

# 3) 从树莓派传数据（在树莓派执行）
scp data/big/{zh_all,en,code}.txt admin@192.168.1.3:C:/work/data/big/
scp data/setbpe.pkl admin@192.168.1.3:C:/work/data/
scp data/f30k_dct_tokens.npz admin@192.168.1.3:C:/work/data/

# 4) 编码 token 流
python -u prep_all.py

# 5) 训练
python -u lm4.py
```

---

## 9. 下一步

| 优先级 | 任务 | 判据 |
|---|---|---|
| 高 | 修 8.1M 模型的"a a a"塌缩 | 生成不同句子 |
| 高 | 词表降到 8k（省嵌入层参数） | 同样的困惑度下生成质量提升 |
| 中 | 补 Chinese C4 剩余分片 | 中文语料 350MB → 1GB |
| 中 | RL 阶段跑起来 | DPO 偏好准确率 >60% |
| 低 | 生成实验（需 GPU） | 低通 PSNR 超过基线 |

---

## 10. 目录速查

```
/srv/brainai/
├── data/
│   ├── big/{zh_all,en,code}.txt      语料
│   ├── corpus/                        旧语料（mix_*）
│   ├── setbpe.pkl                     SetBPE 词表
│   ├── f30k_dct_tokens.npz            图像 DCT token + 字幕
│   ├── dct_aud_tokens.npz             声音 DCT token
│   ├── downloads/flickr30k/           原始图片 4.3GB
│   └── tok_zhcode.pkl                 旧词表（含中文，已弃用）
├── setbpe3.py                         分词器训练
├── dct_tokenizer.py                   DCT 编解码器
├── build_f30k_tokens.py               图像 token 化
├── uni_dct.py                         统一 token 模型
├── lm4.py                             纯文本 LM
├── hybrid_gen4.py                     生成实验
├── test_uni_dct.py                    生成质量测试
└── ckpt/uni_dct.pt                    统一模型权重
```
