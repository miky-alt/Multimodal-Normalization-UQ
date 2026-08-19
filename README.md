# 🤖 Multimodal Uncertainty Quantification and Score Normalization for Vision-Language Models: A Comprehensive Tutorial

A hands-on tutorial notebook covering multimodal uncertainty quantification (UQ), score normalization, and alibration for Vision-Language Models (VLMs), using **UQLM** and **LM-Polygraph** unified through a lightweight internal utility called `uq_toolbox`.

## 📋 Overview

Vision-Language Models can generate fluent, confident-sounding answers even when those answers are wrong — and offer no built-in signal that anything is off. **Multimodal Uncertainty Quantification (UQ)** closes that gap by producing a reliability score alongside every generated answer, computed without changing the model itself. 

Furthermore, because raw uncertainty scores are often unbounded or uninterpretable, this repository guides you through a complete **score normalization and calibration pipeline** to translate those metrics into reliable probabilities for effective hallucination detection.



---

## 📁 Repository Structure

```
Multimodal-Normalization-UQ/
│
├── uq_mult_norm.ipynb             # Main tutorial notebook (Multimodal UQ & Normalization)
│
└── uq_toolbox/                    # Internal infrastructure utility
    ├── __init__.py
    ├── registry.py                # UQ technique registry
    ├── core/
    │   ├── pipeline.py            # Dataset-level and batch scoring pipelines
    │   ├── uq_engine.py           # Central routing engine (UQLM ↔ LM-Polygraph)
    │   └── response_evaluator.py  # Substring match and correctness evaluators
    ├── managers/
    │   └── model_manager.py       # Model loading, aliasing, white/black-box modes
    └── data/
        └── data_loader.py         # Dataset ingestion and preprocessing.
```
---

## 🗂️ Tutorial Structure (`uq_mult_norm.ipynb`)

| Section | Content |
| :--- | :--- |
| **1️⃣ Introduction** | The hallucination challenge in clinical VLM domains, framework architectures (`lm-polygraph` & `UQLM`), and learning objectives. |
| **2️⃣ Global Setup** | Dependencies, environment configurations, and random seed locking for scientific reproducibility. |
| **3️⃣ Multimodal UQ** | Decoder-side signals ($U_{\text{res}}$), visual/textual grounding, and evaluation across multiple paradigms. |
| **4️⃣ Normalization & Calibration** | Taxonomy, implementation, and evaluation of normalization vs. performance calibration techniques. |

---

## 🎨 UQ & Calibration Methods Covered

### 1. Multimodal Uncertainty Quantification Paradigms
* **Sample Diversity (Black-Box & White-Box):** `NumSemSets`, `DegMat`, `Eccentricity`, `SemanticEntropy`, `EigenScore`, `NonContradiction`, `SemanticSetsConfidence`, `BlackSemanticNegentropy`, `MonteCarloProbability`, `ConsistencyAndConfidence (CoCoA)`.
* **Information-Based (Probability & Attention-based):** `LabelProb`, `MaximumSequenceProbability`, `MeanTokenEntropy`, `TokenEntropy`, `MaximumTokenProbability`, `Attention Score`, `SequenceProbability`, `ProbabilityMargin`, `MeanTokenNegentropy`.
* **Ensemble:** `GeneralizedEnsemble` 
* **Reflexive:** `Linguistic1S`, `PTrue`, `Verbalized1S`.

### 2. Score Normalization & Calibration Pipeline
* **Pure Normalization:** Linear Scaling (`MinMaxNormalizer`), Percentile Mapping (`QuantileNormalizer`).
* **Performance Calibration:** Platt Scaling (`PlattPCCNormalizer`), Binned PCC (`BinnedPCCNormalizer`), Isotonic PCC (`IsotonicPCCNormalizer`).

---

## 🧰 uq_toolbox

`uq_toolbox` is a lightweight internal utility built for this tutorial to remove boilerplate infrastructure. It handles:
* loading and registering UQLM and LM-Polygraph models under simple aliases;
* routing uncertainty calls to the correct framework backend;
* standardizing  multimodal dataset ingestion.

---

## ⚙️ Requirements

### API Keys & Tokens
* **Hugging Face token** — required to download vision-language models (e.g., LLaVA, SmolVLM) and auxiliary datasets[cite: 2].

### Hardware & Software
* **GPU:** A CUDA-enabled GPU (NVIDIA T4 or better) is strongly recommended[cite: 2].
* **Python:** Python 3.9+[cite: 2].
* **Platform:** Google Colab.

---

## 🚀 Getting Started

1. Open `uq_mult_norm.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run the installation and setup cells step-by-step following the in-notebook execution guidelines.


