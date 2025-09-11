Representation Shifts in Multimodal Adaptation

This repository contains experiments exploring how multimodal models
represent different kinds of information. Two concrete probing tasks
are provided: a modality probe that distinguishes image tokens
from text tokens, and a brightness probe that asks whether the
model linearly encodes changes in image brightness. Both tasks
operate on COCO val2017 and use the PaliGemma
language–vision model
.

Project Structure

scripts/LinearProbes_ImageTextClassification.ipynb – Notebook that
performs a pairwise‑controlled modality probing experiment. Each
COCO image is paired with one randomly selected caption to control
semantics. A linear classifier is trained at every layer to
distinguish image versus text representations. The notebook
leverages shared utilities for data loading, activation
extraction and probe training.

scripts/LinearProbes_Brightness.ipynb – Notebook that builds a
brightness dataset by artificially darkening and brightening each
selected image. A single caption per image is used for semantic
control, but only the image representations are probed. Data are
split by base_id to avoid identity leakage: all variants of the
same image remain in either the training or test split. Linear
probes are trained across layers to see whether brightness is
linearly encoded.

utils/data_utils.py – Functions for constructing balanced
image–text datasets and brightness variants, and for performing
groupwise train/test splits. Use these helpers instead of
duplicating data handling logic across notebooks.

utils/activation_utils.py – Functions for preloading image bytes
and extracting hidden activations from PaliGemma in both “raw”
(vision tower) and “lm” (projected into the language model) modes.
A simple automatic mixed precision context is provided to reduce
GPU memory usage during extraction.

utils/probe_utils.py – A dataclass for holding probe results and
two functions for training logistic regression classifiers. Use
train_linear_probe_on_splits when you have explicit train/test
partitions (e.g., in pairwise‑controlled experiments). Use
train_linear_probe for random train/test splits when no prior
partition exists.

answer.js – Starter for generating presentation slides (not
modified in this project).

Running the Experiments

Install dependencies. A Python 3.11 environment with PyTorch,
Transformers, scikit‑learn, pycocotools, pandas and matplotlib is
required. The notebooks will attempt to install missing packages
at runtime if you run them as provided.

Download the COCO 2017 train/val images and annotations into
../data/train2017, ../data/val2017 and
../data/annotations_trainval2017/annotations. Adjust the
ANNO_DIR and IMG_DIR paths in the notebooks if your data
lives elsewhere.

Run scripts/LinearProbes_ImageTextClassification.ipynb. It
constructs a balanced dataset, splits by file_name to avoid
semantic leakage, extracts layer‑wise activations from
PaliGemma, trains probes using train_linear_probe_on_splits,
and plots accuracy/F1 across layers.

Run scripts/LinearProbes_Brightness.ipynb to generate the
brightness variants, perform the pairwise train/test split by
base_id, extract activations, train probes per layer and plot
results. The CSV and perturbed images are saved into
../data/brightness_pairs and ../data/brightness_dataset.csv.

Design Notes

The original notebooks contained duplicate code for loading data,
extracting hidden states and training probes. To improve
maintainability this refactor introduces modular utilities in
utils/. Each experiment now focuses on the high‑level logic
(construct dataset, split, extract activations, probe) while reusing
shared components. Two probe functions are provided: one that
performs its own random split (train_linear_probe) and one that
operates on pre‑defined splits (train_linear_probe_on_splits).

For the modality experiment we extract both image and text
representations using the PaliGemma language model (the vision tower
is only used internally via the processor). For the brightness
experiment we extract only image representations. In both cases
activations are mean‑pooled over the sequence dimension before
probing.