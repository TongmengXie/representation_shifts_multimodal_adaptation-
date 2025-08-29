Challenges for using the same Gemmascope for gemma-2 vs paligemma comparison: 

Problem 1: Wrong Model Class
PaliGemma uses PaliGemmaForConditionalGeneration, not AutoModelForCausalLM.

Problem 2: Different Architecture
PaliGemma combines a vision encoder with a Gemma language model, so:

It has different layer structures
The Gemma Scope SAEs were trained on pure Gemma models, not PaliGemma
The dimensions and layer mappings won't match

Problem 3: Patching tokenizers
An error pops up when patching Gemma-2 which says assertion error from decvice-side. It is a problem with token ID exceeding available vocab size
