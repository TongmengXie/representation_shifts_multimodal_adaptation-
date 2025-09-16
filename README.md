# Representational Shifts in Multimodal Adaptation

## TL;DR
This repository investigates how internal representations of a large language model (LLM) change when adapted into a vision–language model (VLM).  
We compare **Gemma-2 2B** (LLM) with **PaliGemma-2 3B** (VLM), focusing on:
- Representation shifts using **Sparse Auto-Encoders (SAEs)**
- **Linear probes** for semantic transfer across modalities
- **Attribute-specific probes** (brightness, modality, redness, etc.)
- Suggestions for **extensions and repository improvements**

---

## Overview
This repository explores how the internal representations of an LLM change when adapted into a VLM.  
We analyse **Google’s Gemma-2 2B** LLM and its multimodal variant **PaliGemma-2 3B**, examining:
- Representation-level shifts
- Information that can be linearly extracted
- Alignment across modalities (text ↔ vision)

The code is organised around **three main experiments** plus a few supporting notebooks.  
A summary of each experiment and possible extensions is provided below.

---

## Experiments

### 1. SAE-based Representation Shift Analysis
**Goal:** Quantify how Gemma’s hidden activations change after multimodal training.  

- Notebook: `scripts/minimal.ipynb`  
- Uses a **Sparse Auto-Encoder (SAE)** from the `sae-lens` library, trained on Gemma.  
- Projects activations from **Gemma-2 2B** and **PaliGemma-2 3B** into a **shared sparse basis**.  
- Metrics: cosine similarity, L² distance, top-k activated basis vectors.  

**Key Findings:**
- Overall structure preserved (cosine ≈ 0.95).  
- **Visually grounded tokens** (objects, colours, spatial terms) shift most.  
- **Syntactic features** shift less.  
- Adaptation reallocates capacity towards **visual semantics**.

**Potential Extensions:**
- Layer-wise analysis across all decoder layers  
- Train SAEs on PaliGemma activations directly  
- Correlate shifts with downstream multimodal performance  
- Cross-language prompts  
- Combine with **linear probes**

---

### 2. Linear Probes: Semantic Transfer Between Modalities
**Goal:** Test whether semantics learned in text transfer to vision (and vice versa).  

- Notebook: `scripts/linear_probes.ipynb`  
- Dataset: COCO 2017 images + captions, balanced across categories (animals, vehicles, food, indoor scenes).  
- Probes: **logistic regression** (via `ProbeResult` in `scripts/probe_utils.py`).  

**Observations:**
- High in-modality accuracy (>90%).  
- Cross-modal transfer ≈ chance level → representations not linearly aligned.  
- Text emphasises **compositional semantics**; vision emphasises **colour/shape/spatial context**.  

**Potential Extensions:**
- Non-linear probes (MLPs, kernels)  
- Contrastive fine-tuning on image–text pairs  
- More categories (fine-grained classes, actions, colours)  
- Probe different hidden layers  

---

### 3. Linear Probes for Modality, Brightness, and Redness
**Goal:** Probe simple binary properties for interpretability.  

- **Modality classification** (`scripts/linear_probes_img_txt_cls.ipynb`):  
  - Dataset: `coco_imgs_text_balanced_val.csv`  
  - Probe can perfectly distinguish image vs text activations.  

- **Brightness probe** (`scripts/LinearProbes_brightness.ipynb`):  
  - Constructs brightened/darkened COCO images.  
  - Vision tower activations linearly encode brightness.  

- **Redness probe (planned):**  
  - Adjust red channel intensity, train probe on less-red vs more-red variants.  

**Potential Extensions:**
- Probe for other low-level attributes (saturation, contrast, blur, rotation).  
- Joint modality–attribute probes (modality + brightness/colour).  
- Extend to temporal/video signals.  

---

## Repository Improvements

- **Unified dataset scripts:** reduce duplication with a shared loader & augmentation pipeline.  
- **Configuration files:** move hyper-parameters into YAML/JSON for reproducibility.  
- **Save & load probes:** store weights, reports, and extracted features.  
- **Results logging & plots:** integrate visualisations and save figures.  
- **Documentation:** expand README with environment setup, GPU guidance, and citations (e.g. [Gemma Scope], PaliGemma papers).  

---

## Getting Started

1. **Clone & install dependencies:**
   ```bash
   git clone <repo_url>
   cd <repo_name>
   pip install torch transformers sae-lens scikit-learn pandas
