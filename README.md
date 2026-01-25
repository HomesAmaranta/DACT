# DACT

This is the PyTorch implementation of our paper: 
**Drift-Aware Continual Tokenization for Generative Recommendation**

## Overview

We propose DACT, a Drift-Aware Continual Tokenization framework with two stages: 
1.  **Tokenizer Fine-tuning**: Augmented with a jointly trained Collaborative Drift Identification Module (CDIM) that outputs item-level drift confidence and enables differentiated optimization for drifting and stationary items.
2.  **Hierarchical Code Reassignment**: Using a relaxed-to-strict strategy to update token sequences while limiting unnecessary changes.

![Framework](image/image.png)

## DACT Tokenizer

### Train
```bash
cd rqvae
bash finetune_book_dact.sh
```

### Tokenize
```bash
cd rqvae
bash generate_code_dact.sh
```

## Instantiation
### For TIGER
```bash
cd TIGER-backbone
bash finetune.sh
```

### For LC-Rec
```bash
cd LC-Rec-backbone/scripts_qwen
bash train.sh
```
