# 🤖 Multimodal Uncertainty Quantification and Score Normalization for Vision-Language Models: A Comprehensive Tutorial

A hands-on tutorial notebook covering multimodal uncertainty quantification (UQ), score normalization, and alibration for Vision-Language Models (VLMs), using **UQLM** and **LM-Polygraph** unified through a lightweight internal utility called `uq_toolbox`.

## 📋 Overview

Vision-Language Models can generate fluent, confident-sounding answers even when those answers are wrong — and offer no built-in signal that anything is off. **Multimodal Uncertainty Quantification (UQ)** closes that gap by producing a reliability score alongside every generated answer, computed without changing the model itself. 

Furthermore, because raw uncertainty scores are often unbounded or uninterpretable, this repository guides you through a complete **score normalization and calibration pipeline** to translate those metrics into reliable probabilities for effective hallucination detection.



---

## 📁 Repository Structure

UQ-Text-Mining/
│
├── notebooks/
│   └── uq_1008.ipynb             # Main tutorial notebook (Multimodal UQ & Normalization)
│
└── notebooks/uq_toolbox/         # Internal infrastructure utility
    ├── __init__.py
    ├── registry.py               # UQ technique registry
    ├── core/
    │   ├── pipeline.py           # Dataset-level and batch scoring pipelines
    │   ├── uq_engine.py          # Central routing engine (UQLM ↔ LM-Polygraph)
    │   └── response_evaluator.py # Substring match and correctness evaluators
    ├── managers/
    │   └── model_manager.py      # Model loading, aliasing, white/black-box modes
    └── data/
        └── data_loader.py        # Dataset ingestion and preprocessing (e.g., VQA-RAD)

---

## 🗂️ Tutorial Structure (`uq_1008.ipynb`)

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
* **Information-Based (Probability & Attention-based):** `LabelProb`, `MaximumSequenceProbability`, `MeanTokenEntropy`, `TokenEntropy`, `MaximumTokenProbability`, `Attention Score`, `SequenceProbability`, `ProbabilityMargin`, `MeanTokenNegentropy`[cite: 2].
* **Ensemble:** `GeneralizedEnsemble` (Optuna-optimized joint weight tuning via `uqlm`)[cite: 2].
* **Reflexive:** `Linguistic1S`, `PTrue`, `Verbalized1S`[cite: 2].

### 2. Score Normalization & Calibration Pipeline
* **Pure Normalization:** Linear Scaling (`MinMaxNormalizer`), Percentile Mapping (`QuantileNormalizer`)[cite: 2].
* **Performance Calibration:** Platt Scaling (`PlattPCCNormalizer`), Binned PCC (`BinnedPCCNormalizer`), Isotonic PCC (`IsotonicPCCNormalizer`)[cite: 2].

---

## 🧰 uq_toolbox

`uq_toolbox` is a lightweight internal utility built for this tutorial to remove boilerplate infrastructure[cite: 2]. It handles:
* loading and registering UQLM and LM-Polygraph models under simple aliases[cite: 2];
* routing uncertainty calls to the correct framework backend[cite: 2];
* standardizing VQA and multimodal dataset ingestion (e.g., VQA-RAD)[cite: 2];
* wrapping Optuna-based joint ensemble weight tuning and threshold calibration[cite: 2].

---

## ⚙️ Requirements

### API Keys & Tokens
* **OpenAI API key** — required for UQLM white-box/black-box scorers[cite: 2].
* **Hugging Face token** — required to download vision-language models (e.g., LLaVA, SmolVLM) and auxiliary datasets[cite: 2].

### Hardware & Software
* **GPU:** A CUDA-enabled GPU (NVIDIA T4 or better) is strongly recommended[cite: 2].
* **Python:** Python 3.9+[cite: 2].
* **Platform:** Google Colab (recommended) or a local Jupyter environment with GPU support[cite: 2].

---

## 🚀 Getting Started

### Option 1 — Google Colab (Recommended)
1. Open `notebooks/uq_1008.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**[cite: 2].
3. Run the installation and setup cells step-by-step following the in-notebook execution guidelines[cite: 2].

### Option 2 — Local Environment
```bash
git clone [https://github.com/miky-alt/UQ-Text-Mining.git](https://github.com/miky-alt/UQ-Text-Mining.git)
cd UQ-Text-Mining

pip install "pip<24.1"
pip install uqlm langchain-openai pandas plotly matplotlib \
  transformers accelerate sentence-transformers datasets \
  scikit-learn nltk ipywidgets
pip install "git+[https://github.com/IINemo/lm-polygraph.git@dev](https://github.com/IINemo/lm-polygraph.git@dev)"
pip install "git+[https://github.com/IINemo/llm-uncertainty-head.git](https://github.com/IINemo/llm-uncertainty-head.git)"
pip install --force-reinstall "protobuf==5.28.3"
pip install langchain-anthropic langchain-google-genai langchain-ollama
