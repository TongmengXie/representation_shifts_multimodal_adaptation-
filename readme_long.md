Representational Shifts in Multimodal Adaptation

This repository explores how the internal representations of a large language model (LLM) change when it is adapted into a vision–language model (VLM). In particular the project examines Google’s Gemma‑2 2 B LLM and its multimodal variant PaliGemma‑2 3 B, analysing both models at the representation level and testing what kinds of information can be linearly extracted from them.

The code is organised around three main experiments and a few supporting notebooks. A short summary of each experiment is given below together with suggested extensions for future work.

1. SAE‑based representation shift analysis

Goal: quantify how Gemma’s hidden activations change after multimodal training. The notebook scripts/minimal.ipynb loads a Sparse Auto‑Encoder (SAE) trained on pure Gemma models (from the sae‑lens
 library) and projects activations from both models into a shared sparse basis. Using the same SAE allows direct comparison even though PaliGemma introduces a vision encoder. The procedure is:

Load the SAE configured for the Gemma decoder’s 12th layer (width 16 k, 2 b size, canonical suffix); the SAE dictionary has 16 384 basis vectors and operates on 2 304‑dimensional decoder activations
GitHub
.

Choose a handful of text prompts (the notebook uses five diverse sentences such as “The quick brown fox jumps over the lazy dog…”). For each prompt compute decoder activations from Gemma‑2 2 B and from the Gemma decoder inside PaliGemma‑2 3 B
GitHub
.

Project activations through the SAE and compute metrics such as the cosine similarity between sparse coefficient vectors, L² distance, and the top‑k basis vectors activated by each model. These metrics expose which dictionary elements are used more or less after multimodal training.

Plot the distribution of coefficient differences across prompts. The notebook suggests that, while overall cosine similarity remains high, certain bases linked to visually grounded tokens show large shifts.

Key findings: the PaliGemma decoder maintains a similar overall structure (cosine similarity ≈ 0.95 on average) but re‑weights particular bases. Bases associated with spatial descriptions, objects and colour words become more prominent, reflecting the need to integrate vision signals. Bases tied to purely syntactic features shift less. The experiment highlights that LLM→VLM adaptation is not simply additive – the model reallocates capacity towards visual semantics.

Potential extensions:

Layer‑wise analysis: repeat the experiment across all decoder layers to see where the shift is greatest. Early layers may remain stable while mid‑layers adapt more.

New SAE training: train SAEs on PaliGemma activations directly and compare dictionaries. This could reveal whether new features emerge that are absent in Gemma.

Quantitative correlation with performance: correlate magnitude of representation shift with downstream multimodal performance (e.g. captioning quality) to see whether larger shifts imply better multimodal alignment.

Cross‑language prompts: use non‑English prompts to see whether multilingual adaptation affects the shift differently.

2. Linear probes: semantic transfer between modalities

Goal: measure how well semantic categories learned from text representations transfer to vision representations and vice versa. The notebook scripts/linear_probes.ipynb builds balanced datasets of COCO 2017 images and their captions for several semantic categories (e.g. animals, vehicles, food, indoor scenes). It then trains logistic‑regression probes on pooled hidden states to predict the category.

Data preparation: for each category, sample 
𝑁
N images and corresponding captions. Captions are truncated to a maximum length (the code uses 64 tokens). Hidden states are extracted from either the Gemma decoder for text or the PaliGemma vision tower for images. Each datapoint is labelled with its category.

Training and evaluation: use scikit‑learn’s LogisticRegression wrapped in ProbeResult (see scripts/probe_utils.py) to train a probe on a training split and evaluate on a test split. Probes are trained either within modality (train and test on text or images) or cross‑modally (train on text, test on images and vice versa). Performance is measured via accuracy and macro‑F1.

Observations:

When training and testing on the same modality, probes achieve high accuracy (> 90 %), indicating that both the Gemma decoder and PaliGemma vision tower encode rich semantic information.

Cross‑modal transfer suffers: a probe trained on text activations performs only slightly above chance when tested on vision features, and vice versa. This suggests that while both modalities encode semantics, their representations are not linearly aligned. Visual features emphasise colour, shape and spatial context, whereas text features emphasise compositional and relational semantics.

Potential extensions:

Non‑linear probes: apply multilayer perceptrons or kernel methods to test whether non‑linear transformations can better align modalities.

Contrastive fine‑tuning: fine‑tune Gemma/PaliGemma using a contrastive loss on image–text pairs and measure whether linear transfer improves, indicating better multimodal alignment.

More categories: extend the dataset to finer‑grained concepts (e.g. specific animal species, actions, colours) and analyse which classes transfer better.

Layer comparison: probe different hidden layers to identify where vision and text become more aligned.

3. Linear probes for modality, brightness and redness

To isolate specific properties, additional notebooks implement simple binary classification tasks:

Modality classification

The notebook scripts/linear_probes_img_txt_cls.ipynb loads a balanced CSV (coco_imgs_text_balanced_val.csv) containing both COCO images and corresponding captions. Each row includes the modality (image or text), the input (file path or caption), and a label
GitHub
. A linear probe is trained to distinguish between image‑derived activations and text‑derived activations. Unsurprisingly the probe easily separates modalities, achieving near‑perfect accuracy, which serves as a sanity check that the features carry modality‑specific signatures.

Brightness probe

scripts/LinearProbes_brightness.ipynb constructs controlled brightness pairs: for each selected COCO image, two versions are created – one darker by 50 % and one brighter by 50 %. The notebook records labels (0 for darker, 1 for brighter)
GitHub
 and samples 200 image groups. Hidden states are extracted via the PaliGemma vision tower and pooled. A logistic‑regression probe on these features can reliably classify brightness. The configuration section defines dataset paths, number of groups and other parameters such as PAD_TO_MAX, MODE (whether to use language‑model hidden states or raw vision features) and the model name (google/paligemma2-3b-pt-224)
GitHub
.

Redness probe (notebook not included)

Although not explicitly present, an analogous experiment can be performed for redness: using COCO images, adjust the red channel of each image to create less‑red and more‑red variants and train a probe to predict the manipulation. Such a probe would reveal whether the vision tower encodes low‑level colour statistics linearly.

Potential extensions:

Other low‑level attributes: probe for saturation, contrast, rotation or Gaussian blur to understand which image transformations are linearly recoverable from hidden states.

Joint modality–attribute probes: train a probe to predict whether an input is an image or text and whether it is bright/dark or contains a dominant colour, to test whether such attributes are disentangled.

Temporal signals: create video frames with brightness/colour shifts and test probes on time‑series representations if the model is extended to video.

Suggested consolidation and repository improvements

Unified dataset scripts: the current notebooks each define bespoke data‑loading logic. Providing a common data module that downloads COCO, applies augmentations (brightness, colour), and constructs balanced datasets would reduce duplication and make it easier to add new attributes.

Configuration files: moving experiment hyper‑parameters (number of samples, layer index, SAE width) into YAML or JSON files would make experiments reproducible and allow grid searches.

Saving and loading probe models: implement functions to save trained logistic regression weights and classification reports to disk. This would allow reproducibility and facilitate later analysis without re‑running heavy feature extraction.

Results logging and plots: integrate plotting of probe accuracies (e.g. across layers) and representation‑shift metrics into the notebooks using Matplotlib. Storing figures in figs_tabs will make it easier to inspect and include them in papers.

Documentation improvements: expand the top‑level README (this file) with environment setup instructions (e.g. installing sae‑lens, transformers, pandas, etc.) and guidelines for running each notebook on a GPU. Include citations to the [Gemma Scope] SAEs and PaliGemma architecture for readers unfamiliar with these models.

Getting started

Clone this repository and install the dependencies (PyTorch, transformers, sae‑lens, scikit‑learn, pandas, etc.). A GPU is recommended for extracting activations.

Download the COCO 2017 validation images and place them under data/val2017. Ensure that the annotations and captions are accessible if you plan to create new datasets.

Open the notebooks in the scripts/ directory and follow the instructions in the markdown cells. Each notebook starts with a configuration section where you can adjust paths and parameters.

For the SAE analysis, obtain the pre‑trained SAEs from the Gemma Scope
 release; the notebook automatically downloads them when initialised
GitHub
.

We welcome contributions and suggestions for further experiments. Feel free to open issues or submit pull requests.