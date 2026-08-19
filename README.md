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

---

## 📖 References

* <a name="ref-shelmanov-2026"></a>**[Shelmanov et al., 2026]** Artem Shelmanov, Maxim Panov, Roman Vashurin, Artem Vazhentsev, Ekaterina Fadeeva, Lyudmila Rvanova, and Timothy Baldwin. *"Uncertainty Quantification for Large Language Models"*. Tutorial at the 40th Annual AAAI Conference on Artificial Intelligence (AAAI-2026). January 20, 2026. MBZUAI & ETH Zürich.

* <a name="ref-yang-2025b"></a>**[Yang et al., 2025b]** Ziran Yang, Shibo Hao, Hao Sun, Lai Jiang, Qiyue Gao, Yian Ma, and Zhiting Hu. *"Understanding the Sources of Uncertainty for Large Language and Multimodal Models"*. ICLR Workshop: Quantify Uncertainty and Hallucination in Foundation Models: The Next Frontier in Reliable AI, 2025. [OpenReview](https://openreview.net/forum?id=5By0rus8z7).

* <a name="ref-lin-2024"></a>**[Lin et al., 2024]** Lin, Z., Trivedi, S., & Sun, J. (2024). *Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models*. Transactions on Machine Learning Research (TMLR). [arXiv:2305.19187](https://arxiv.org/abs/2305.19187).
* <a name="ref-kun-2023"></a> **[Kuhn et al., 2023]** Kuhn, L., Gal, Y., & Farquhar, S. (2023). *Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation*. International Conference on Learning Representations (ICLR). [arXiv:2302.09664](https://arxiv.org/abs/2302.09664).
* <a name="ref-vas-2025"></a>**[Vashurin et al., 2025]** Vashurin, R., Fadeeva, E., Vazhentsev, A., Rvanova, L., Vasilev, D., Tsvigun, A., Petrakov, S., Xing, R., Sadallah, A., Grishchenkov, K., Panchenko, A., Baldwin, T., Nakov, P., Panov, M., & Shelmanov, A. (2025). *Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph*. Transactions of the Association for Computational Linguistics, 13, 220–248. [doi:10.1162/tacl_a_00737](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00737/128713/Benchmarking-Uncertainty-Quantification-Methods).

* <a name="ref-chen-2024"></a> **[Chen et al., 2024]** Chao Chen, Kai Liu, Ze Chen, Yi Gu, Yue Wu, Mingyuan Tao, Zhihang Fu, and Jieping Ye. *"INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection"*. The Twelfth International Conference on Learning Representations (ICLR 2024). [OpenReview](https://openreview.net/forum?id=Zj12nzlQbz).

* <a name="ref-chen-mueller-2023"></a>**[Chen and Mueller, 2023]** Jiuhai Chen and Jonas Mueller. *"Quantifying Uncertainty in Answers from any Language Model and Enhancing their Trustworthiness"*. arXiv preprint arXiv:2308.16175. [arXiv:2308.16175](https://arxiv.org/abs/2308.16175).

* <a name="ref-farquhar-2024"></a>**[Farquhar et al., 2024]** Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. *"Detecting hallucinations in large language models using semantic entropy"*. Nature, 2024. [doi:10.1038/s41586-024-07421-0](https://doi.org/10.1038/s41586-024-07421-0).

* <a name="ref-vashurin-2025-mbr"></a>**[Vashurin et al., 2025a]** Roman Vashurin, Maiya Goloburda, Albina Ilina, Aleksandr Rubashevskii, Preslav Nakov, Artem Shelmanov, and Maxim Panov. *"Uncertainty Quantification for LLMs through Minimum Bayes Risk: Bridging Confidence and Consistency"*. arXiv preprint arXiv:2502.04964. [arXiv:2502.04964](https://arxiv.org/abs/2502.04964).
* <a name="ref-sriramanan-2024"></a>**[Sriramanan et al., 2024]** Gaurang Sriramanan, Siddhant Bharti, Vinu Sankar Sadasivan, Shoumik Saha, Priyatham Kattakinda, and Soheil Feizi. *"LLM-Check: Investigating Detection of Hallucinations in Large Language Models"*. Advances in Neural Information Processing Systems (NeurIPS 2024). [OpenReview](https://openreview.net/forum?id=LYx4w3CAgy).

* <a name="ref-vazhentsev-2026"></a>**[Vazhentsev et al., 2026]** Artem Vazhentsev, Lyudmila Rvanova, Gleb Kuzmin, Ekaterina Fadeeva, Ivan Lazichny, Alexander Panchenko, Maxim Panov, Mrinmaya Sachan, Preslav Nakov, Timothy Baldwin, and Artem Shelmanov. *"Efficient Hallucination Detection for LLMs Using Uncertainty-Aware Attention Heads"*. Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. [arXiv:2505.20045](https://arxiv.org/abs/2505.20045).

* <a name="ref-farr-2024"></a>**[Farr et al., 2024]** David Farr, Nico Manzonelli, Iain Cruickshank, and Jevin West. *"RED-CT: A Systems Design Methodology for Using LLM-labeled Data to Train and Deploy Edge Classifiers for Computational Social Science"*. arXiv preprint arXiv:2408.08217. [arXiv:2408.08217](https://arxiv.org/abs/2408.08217).

* <a name="ref-scalena-2025"></a>**[Scalena et al., 2025]** Daniel Scalena, Leonidas Zotos, Elisabetta Fersini, Malvina Nissim, and Ahmet Üstün. *"EAGer: Entropy-Aware GEneRation for Adaptive Inference-Time Scaling"*. arXiv preprint arXiv:2510.11170. [arXiv:2510.11170](https://arxiv.org/abs/2510.11170).

* <a name="ref-tian-2023"></a>  **[Tian et al., 2023]** Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn, and Christopher D. Manning. *"Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback"*. Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 5433–5442, 2023. [arXiv:2305.14975](https://arxiv.org/abs/2207.05221)

* <a name="ref-kadavath-2022"></a>  **[Kadavath et al., 2022]** Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, Scott Johnston, Sheer El-Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Sam Bowman, Stanislav Fort, Deep Ganguli, Danny Hernandez, Josh Jacobson, Jackson Kernion, Shauna Kravec, Liane Lovitt, Kamal Ndousse, Catherine Olsson, Sam Ringer, Dario Amodei, Tom Brown, Jack Clark, Nicholas Joseph, Ben Mann, Sam McCandlish, Chris Olah, and Jared Kaplan. *"Language Models (Mostly) Know What They Know"*. arXiv:2207.05221, 2022. [arXiv:2305.14975 ](https://arxiv.org/abs/2305.14975)



* <a name="ref-cole-2023"></a> **[Cole et al., 2023]** Jeremy R. Cole, Michael J. Q. Zhang, Daniel Gillick, Julian Martin Eisenschlos, Bhuwan Dhingra, and Jacob Eisenstein. *"Selectively Answering Ambiguous Questions"*. EMNLP 2023. [arXiv:2305.14613](https://arxiv.org/abs/2305.14613).

* <a name="ref-lamb-2026"></a>**[Lamb et al., 2026]** Tom A. Lamb, Desi R. Ivanova, Philip H. S. Torr, and Tim G. J. Rudner. "Improving Semantic Uncertainty Quantification in Language Model Question-Answering via Token-Level Temperature Scaling". Proceedings of the 29th International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR, Volume 300, Tangier, Morocco, 2026. https://arxiv.org/pdf/2604.07172.
