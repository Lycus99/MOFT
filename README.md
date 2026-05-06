# MOFT — Code: Medical loss scaling for MS-SWIFT

**Project page:** [https://lycus99.github.io/MOFT/](https://lycus99.github.io/MOFT/)  
**Paper:** *MOFT: Multi-Objective Fine-Tuning for Medical Vision-Language Models*

This repository ships a **minimal MS-SWIFT patch** that implements **`MedLossScale`**—token/segment-level loss reweighting for medical terms during supervised fine-tuning. It is released as **companion code** for MOFT; the project page links the full paper, **PSV2026** dataset, Hugging Face **model** and **dataset** artifacts, and aggregate results.

---

## MOFT in brief

Vision-language models (VLMs) for healthcare suffer when standard SFT uses a **uniform next-token loss**: frequent syntactic tokens dominate gradients, while **sparse medical entities** receive weak signals, and objectives for **linguistic fluency** vs **medical semantics** can **interfere** in the optimization landscape.

**Multi-Objective Fine-Tuning (MOFT)** addresses this by:

1. **Decoupling** training into two trajectories—one emphasizing linguistic fluency and one emphasizing medical semantics.
2. **Fusing** the resulting checkpoints in parameter space via **Linear Mode Connectivity (LMC)** to reduce gradient interference between objectives.

MOFT is evaluated together with **PSV2026**, a **52K** high-quality multimodal dataset built with **critique-based augmentation** to stress diagnostic rationale over flat memorization; PSV2026 improves upon simplified queries and noisy alignment common in resources such as PathVQA, SLAKE, and VQA-RAD, with stronger emphasis on spatial structure, anatomical landmarks, and pathological patterns.

### Representative results (PSV2026)

| Model | SFT | MOFT | Δ |
|-------|-----|------|---|
| Lingshu-7B | 63.0% | **65.5%** | +2.5% |
| MedGemma-4B | 58.4% | **60.8%** | +2.4% |
| Qwen3VL-4B | 61.2% | **63.1%** | +1.9% |

On the external **OmniAbnormalCT** benchmark, co-training with PSV2026 improves baselines; MOFT **consistently surpasses** standard SFT across settings reported on the [project page](https://lycus99.github.io/MOFT/).

---

## What this repository contains

This repo is **not** the full MOFT training stack; it provides the **MS-SWIFT patch** used to implement **medical-term–aware loss scaling** (`--loss_scale med`):

| Component | Role |
|-----------|------|
| **`MedLossScale`** (`med.py`) | Loads a JSON config (`loss_scale_config`) and assigns higher loss weights to **matched medical-term spans** in the assistant response. |
| **`med_calculate_loss_scale_v2`** (`utils.py`) | Lightweight weighting: **no blacklist** or external term-frequency files; background spans use **`coef1`**, term spans use **`coef2`**. |
| **`mapping.py`** | Registers the strategy as **`med`** in `loss_scale_map`. |

You only need **three Python files** to apply the patch. The loss-scale JSON referenced in `med.py` as **`loss_scale_config = "repsv_train_lingshu7b_correction.json"`** is **not** vendored in this repo (it can be large). Obtain **`repsv_train_lingshu7b_correction.json`** from the Hugging Face dataset **[ASD9987/PSV2026](https://huggingface.co/datasets/ASD9987/PSV2026)** and copy it to **`swift/loss_scale/config/`** so it matches `loss_scale_config`.

For **datasets, checkpoints, full training recipes, and figures**, follow the **[MOFT project page](https://lycus99.github.io/MOFT/)** (Paper, GitHub, Hugging Face model & dataset, BibTeX).

---

## Tested environment

Versions below come from the authors’ conda env **`swift28`** (`pip list`). Use them as a reference for reproducibility; compatible ranges may also work.

| Component | Version |
|-----------|---------|
| Python | 3.11 |
| ms_swift | 4.0.0.dev0 |
| torch | 2.8.0 |
| torchvision | 0.23.0 |
| torchaudio | 2.8.0 |
| transformers | 4.57.6 |
| accelerate | 1.12.0 |
| peft | 0.18.1 |
| trl | 0.24.0 |
| modelscope | 1.34.0 |
| deepspeed | 0.18.5 |
| flash_attn | 2.8.3 |
| datasets | 3.6.0 |
| numpy | 2.2.6 |

For a full environment pin, add a `requirements-freeze.txt` from `pip freeze` or follow the official ms-swift install docs.

---

## Prerequisites

Install **ms-swift** following the [official instructions](https://github.com/modelscope/ms-swift). Then locate the installed package:

```bash
python -c "import swift; print(swift.__file__)"
```

You should see a path ending with `.../site-packages/swift/__init__.py`. The patch files go under `.../site-packages/swift/loss_scale/`.

---

## Applying the patch (3 files)

Replace these files in your **installed** `ms-swift` tree (back up the originals first):

| File in this repo | Target path |
|-------------------|-------------|
| `med.py` | `site-packages/swift/loss_scale/med.py` |
| `utils.py` | `site-packages/swift/loss_scale/utils.py` |
| `mapping.py` | `site-packages/swift/loss_scale/mapping.py` |

Steps:

1. **Back up** the three upstream files above.
2. **Copy** this repository’s `med.py`, `utils.py`, and `mapping.py` into `swift/loss_scale/`, overwriting the existing files.
3. Download **`repsv_train_lingshu7b_correction.json`** from the Hugging Face dataset [**ASD9987/PSV2026**](https://huggingface.co/datasets/ASD9987/PSV2026) (this is the file set by `loss_scale_config` in `med.py`). Place it under **`swift/loss_scale/config/`**, or change `loss_scale_config` in `med.py` if you use another filename.

**Note:** Replacing `utils.py` and `mapping.py` wholesale may conflict with **future ms-swift upgrades**—reconcile changes manually after upgrading.

### Large JSON configs

The primary source for `repsv_train_lingshu7b_correction.json` is the Hugging Face dataset [**ASD9987/PSV2026**](https://huggingface.co/datasets/ASD9987/PSV2026). If you mirror it elsewhere, avoid committing very large files as plain Git; use **Git LFS** or a release attachment, and document the URL and **SHA256** in this README or a `DATA.md`.

### Optional: per-epoch random reset of `coef2`

The stock patch keeps **`coef2_reset_ratio = 0.0`** and has **`apply_epoch_random_reset`** commented out in `med.py`. If you enable random reset (`coef2_reset_ratio > 0`) and uncomment the method, you must also wire **`apply_epoch_random_reset`** in `swift/trainers/seq2seq_trainer.py` (multiply token loss by the scaled tensor after reset), matching the behavior of a SWIFT build that already exposes this hook.

---

## Training

Add to your existing SWIFT training command:

```bash
--loss_scale med
```

Hyperparameters **`coef1`** (background segments) and **`coef2`** (medical-term segments) are defined in `med.py`; adjust there or subclass `MedLossScale` if you prefer not to edit the file in `site-packages`.

---

## Repository layout (minimal)

```text
.
├── README.md
├── README_zh.md               # optional Chinese README
├── LICENSE                    # recommended: align with upstream ModelScope / ms-swift terms
├── med.py
├── utils.py
└── mapping.py
```

---

## Data and reproducibility

- **`repsv_train_lingshu7b_correction.json`** (`med.py`: `loss_scale_config = "repsv_train_lingshu7b_correction.json"`): available from the Hugging Face dataset **[ASD9987/PSV2026](https://huggingface.co/datasets/ASD9987/PSV2026)**. Download and place it in **`swift/loss_scale/config/`** next to your patched `ms-swift` install.
- Publish **SHA256** checksums for any mirrored copies of large JSON files.
- Do not rely on machine-specific absolute paths inside Python code when sharing publicly.
- **PSV2026** and evaluation protocols are described on the [project page](https://lycus99.github.io/MOFT/).

---

## License

This patch derives from **MS-SWIFT / ModelScope**-licensed code. Comply with the **upstream license**. MOFT-related assets (data, models) may have separate terms—see the project page and Hugging Face dataset/model cards.

---

## Citation

If you use **MOFT** or this code in research, please cite the paper (update venue when available) and **[ms-swift](https://github.com/modelscope/ms-swift)**. Replace the journal field when the official publication is announced.

**MOFT (paper — placeholder from project page; update when finalized):**

```bibtex
@article{moft2026,
  title   = {Multi-Objective Fine-Tuning for Medical Vision-Language Models},
  author  = {Li, Yuchong and Zeng, Xiaojun and You, Caizhen and Wu, Pengbo and Guo, Zixian and Yang, Jian and Jia, Fucang and Zhang, Lei},
  journal = {TO BE FILLED},
  year    = {2026},
  note    = {Project page: https://lycus99.github.io/MOFT/}
}
```

**This software repository (replace URL with your GitHub repo):**

```bibtex
@software{moft_swift_med_loss,
  title        = {MOFT: MS-SWIFT patch for medical loss scaling (MedLossScale)},
  author       = {Li, Yuchong and collaborators},
  year         = {2026},
  url          = {https://github.com/<your-org>/<your-repo>},
  note         = {Companion code for MOFT; see https://lycus99.github.io/MOFT/}
}
```

---

**Disclaimer:** This README describes research-oriented companion code; it is not an official ModelScope release. Use at your own risk in production.
