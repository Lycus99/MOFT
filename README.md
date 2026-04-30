# MOFT：面向医疗视觉语言模型的多目标微调（Multi-Objective Fine-Tuning）

<p align="center">
  <strong>MOFT</strong> 将「语言流畅度」与「医学语义准确性」解耦为两条独立训练轨迹，并在参数空间中依据线性模式连通性（Linear Mode Connectivity, LMC）进行融合，缓解单一损失下的梯度冲突。<br/>
  配套高质量多模态数据集 <strong>PSV2026</strong>（约 52K 样本）用于公平评测与可迁移监督。
</p>

<p align="center">
  <a href="https://github.com/Lycus99/MOFT">代码与数据仓库</a>
  ·
  <a href="#开源发布计划">开源计划</a>
  ·
  <a href="#引用">引用</a>
</p>

---

## 摘要

医疗视觉语言模型（VLM）在监督微调（SFT）中常因 token 分布倾斜：高频语法 token 主导梯度，稀释了对关键临床实体的学习信号，并在「流畅度」与「医学事实性」之间产生梯度冲突。**MOFT** 将训练拆分为两条轨迹——分别强调通用语言建模与医学实体加权——再在参数空间中线性融合权重，从机制上规避联合优化中的干扰。

我们在 **PSV2026** 上进行了系统评估（数据源自 PathVQA、SLAKE、VQA-RAD 等，并经商用 VLM 清洗与 critque 式增强）。主要数值结果（论文报告）：在 PSV2026 上，相对标准 SFT，MOFT 在 Lingshu-7B 上准确率 **63.0% → 65.5%（+2.5%）**，MedGemma-4B **58.4% → 60.8%（+2.4%）**，Qwen3VL-4B **61.2% → 63.1%（+1.9%）**；在外部 OmniAbnormalCT 上，与 PSV2026 联合训练可显著提升基线，且 MOFT 持续优于 SFT。

---

## 核心图示

> **使用前请将论文工程目录 `alternate_tj_latex_template_ap/figures/` 中的同名插图复制到本仓库 `docs/assets/`**，以便在 GitHub 上正常显示下图（若路径不同，请批量替换 URL）。

### 微调范式对比与 MOFT 动机

医疗 VLM 微调需要在语言连贯性与医学准确性之间取得平衡；MOFT 通过双轨迹训练与权重融合显式解耦两类目标。

<p align="center">
  <img src="docs/assets/fig_intro.png" alt="Fine-tuning paradigms comparison" width="92%"/>
</p>

### MOFT 框架总览

混合指令数据（标准 QA + 批判式增强 CR）、双目标轨迹训练，以及参数空间融合。

<p align="center">
  <img src="docs/assets/fig_algorithm.png" alt="MOFT framework overview" width="98%"/>
</p>

### PSV2026 数据构建流程

训练侧：VLM 精炼 QA、误差诱导与批判数据 **PSV2026-CR**；测试侧：查询扩展、五维评分细则与临床校验。

<p align="center">
  <img src="docs/assets/fig_dataset4.png" alt="PSV2026 curation pipeline" width="92%"/>
</p>

### LMC 与融合比例的经验验证

插值路径上的 NLL 接近线性期望（左）；测试准确率在插值系数约 0.5 处达到峰值（右），支持在平坦盆地内进行参数融合。

<p align="center">
  <img src="docs/assets/fig_lmc2.png" alt="Linear Mode Connectivity validation" width="92%"/>
</p>

### PSV2026 测试集统计

图像模态、问题类型与临床领域分布。

<p align="center">
  <img src="docs/assets/psv2026_statistics2.png" alt="PSV2026 test set statistics" width="98%"/>
</p>

### 按临床领域与成像模态的性能对比（相对 SFT）

哑铃图展示 DFT 与 MOFT 相对标准 SFT 的准确率变化。

<p align="center">
  <img src="docs/assets/dumbbell_unified_fonts.png" alt="Domain and modality performance" width="98%"/>
</p>

### Token 级 NLL 与不同医学实体权重

医学 / 非医学 token 的 NLL 行为与测试准确率随实体加权系数 \(c\) 的变化。

<p align="center">
  <img src="docs/assets/fig_analysis.png" alt="Per-token NLL and accuracy vs entity weighting" width="92%"/>
</p>

### 定性样例（SFT vs MOFT）

GitHub Markdown 对 PDF 预览支持不稳定，建议将 `fig_quali.pdf` 导出为 **`docs/assets/fig_quali.png`** 后取消下面注释：

<!--
<p align="center">
  <img src="docs/assets/fig_quali.png" alt="Qualitative SFT vs MOFT" width="98%"/>
</p>
-->

若暂时仅保留 PDF，可将文件置于 `docs/assets/fig_quali.pdf`，在 Release 或论文附录中提供链接。

---

## 仓库结构（规划）

下列为建议目录，便于复现论文实验与二次开发；实际以仓库内最新说明为准。

```
MOFT/
├── README.md
├── LICENSE
├── requirements.txt / environment.yml
├── configs/                 # 各骨干网络与训练配置（学习率、LoRA、实体权重 c、融合系数 λ 等）
├── data/
│   ├── psv2026/             # PSV2026 元数据与划分说明
│   └── scripts/             # 数据下载、校验与预处理
├── src/
│   ├── train_sft.py         # 标准 SFT 基线
│   ├── train_moft_branch.py # 语言学 / 医学两支轨迹（或统一入口 + objective 开关）
│   ├── fuse_weights.py      # 参数空间线性融合与可选校准
│   ├── inference.py       # 批量推理
│   └── evaluate/            # Rubric 评判、准确率与其它指标
├── checkpoints/             # （可选）放置说明与 .gitignore，大文件用 HF / 网盘
└── docs/
    └── assets/              # README 所用插图
```

---

## 开源发布计划

以下为分阶段交付清单，便于审稿、合规与社区复现；可根据伦理审查与机构政策调整时间节点。

| 阶段 | 内容 | 说明 |
|------|------|------|
| **P0** | **许可证与声明** | 选用代码许可证（如 Apache-2.0 / MIT）；数据若含临床风格内容需附使用范围、禁止用途与免责条款。 |
| **P1** | **数据集 PSV2026** | 发布 **PSV2026-QA**、**PSV2026-CR** 的训练划分；测试集查询与图像（若许可）；图像清单与来源映射（PathVQA / SLAKE / VQA-RAD）；批判增强与 GPT 生成部分的 **可复现脚本说明**（商用 API 不完全开源属正常，需记录版本与日期）。 |
| **P1** | **评测协议** | Rubric 生成流程说明、五维二元细则示例、LLM-as-a-judge 调用配置；可选发布 **脱敏后的 rubric JSON** 与人类评审摘要统计。 |
| **P2** | **模型权重** | 各骨干 **SFT 单端点**（\(c=1.0\) / 医学加权端）与 **MOFT 融合后** 权重；优先 Hugging Face Hub / Zenodo；附训练配置哈希与评测分数。 |
| **P2** | **核心训练脚本** | 医学实体 token 识别（如 GPT 标注管线或可替换的词典/NER 方案）、两支独立训练、checkpoint 保存与 **线性插值融合**；支持 LoRA / 全参的配置示例。 |
| **P3** | **推理与评估** | 批量推理脚本；主表与消融表的评测入口（与论文指标对齐）；OmniAbnormalCT 等外部数据集的 **数据准备说明**（遵循各数据集原始协议）。 |
| **P3** | **文档与复现** | 一键环境安装、`CONFIG` + `SEED` + 版本记录；已知问题与 FAQ；GPU 显存与耗时参考。 |

### 交付物核对清单（维护者可打勾）

- [ ] **PSV2026**：训练/验证/测试索引；数据来源与预处理文档；critique 数据字段说明  
- [ ] **评测**：Rubric 模板；judge 提示词版本；主指标计算脚本  
- [ ] **权重**：Lingshu-7B / MedGemma-4B / Qwen3VL-4B 等 MOFT 与基线  
- [ ] **训练**：`train_moft` 双轨迹 + `fuse`；实体加权 \(c\) 与融合 \(\lambda\) 扫描示例  
- [ ] **推理**：单图/批量；与论文定性图一致的可视化脚本（可选）  
- [ ] **伦理与隐私**：无真实患者标识确认；第三方数据许可证合集  

---

## 环境与快速开始（占位）

正式脚本发布后，此处将补充例如：

```bash
conda env create -f environment.yml
conda activate moft
python src/train_moft_branch.py --config configs/qwen3vl_moft.yaml
python src/fuse_weights.py --ckpt_ling ... --ckpt_med ...
python src/evaluate/run_psv2026.py --checkpoint ...
```

---

## 引用

若使用 MOFT、PSV2026 或本仓库成果，请引用论文（正式发表后补全卷期页码与 DOI）：

```bibtex
@article{moft2026,
  title   = {MOFT: Multi-Objective Fine-Tuning for Medical Vision-Language Models},
  author  = {Li, Yuchong and Zeng, Xiaojun and You, Caizhen and Wu, Pengbo and Guo, Zixian and Yang, Jian and Jia, Fucang and Zhang, Lei},
  journal = {IEEE Transactions on ...},
  year    = {2026},
  note    = {To appear}
}
```

---

## 致谢

数据集构建与评测使用了 PathVQA、SLAKE、VQA-RAD 等公开资源；商用模型辅助标注请在论文与网页中保留相应引用与使用条款说明。
