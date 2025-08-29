Challenges for using the same Gemmascope for gemma-2 vs paligemma comparison: 

Problem 1: Wrong Model Class
PaliGemma uses PaliGemmaForConditionalGeneration, not AutoModelForCausalLM.

Problem 2: Different Architecture
PaliGemma combines a vision encoder with a Gemma language model, so:

It has different layer structures
The Gemma Scope SAEs were trained on pure Gemma models, not PaliGemma
The dimensions and layer mappings won't match

Problem 3: Tensor Size Mismatch
The error shows tensor size mismatch (11 vs 10), indicating different sequence lengths due to tokenization differences.