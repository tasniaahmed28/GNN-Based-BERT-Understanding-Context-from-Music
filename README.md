# GNN-Based BERT for Understanding Context from Music

CSE425 / EEE474 / CSE715 — Neural Networks Course Project
## Overview
A hybrid BERT + Graph Neural Network (GNN) system for understanding musical
context, implemented across four tasks of increasing complexity:

1. Task 1 — BERT baseline: text/metadata classifier over FMA track metadata (genre proxy for tags)
2. Task 2 — GNN on audio structure: GraphSAGE over chroma segment graphs, compared against a CNN mel-spectrogram baseline
3. Task 3 — GNN-BERT fusion: cross-attention and early-concat fusion variants, full 4-way ablation
4. Task 4 — Contrastive retrieval: dual-encoder InfoNCE training on MusicCaps, with zero-shot genre transfer evaluation

## Results Summary

Task1: Model: BERT-only; Test Macro-F1:	0.532; Test AUC-PR:0.575 	
Task2: Model: GraphSAGE (GNN-only); Test Macro-F1: 0.238; Test AUC-PR:0.256  
Task2: Model: CNN (mel-spectrogram baseline); Test Macro-F1: 0.3817; Test AUC-PR:0.444 
Task3: Model: Cross-attention fusion; Test Macro-F1: 0.485; Test AUC-PR: 0.579 
Task3: Model: Early-concat fusion (Best); Test Macro-F1: 0.576; Test AUC-PR: 0.622 
Task4: Model: Zero-shot genre transfer; Test Macro-F1: 0.107

Retrieval (Task 4, N=60 test pairs): Audio→Caption R@10 = 26.7%, Caption→Audio R@10 = 31.7% (chance = 16.7%).

## Datasets

- FMA-small (8,000 tracks, 8 genres): https://github.com/mdeff/fma — a balanced 2,000-track subsample was used for Tasks 2–3.
- MusicCaps (5,521 caption pairs): https://huggingface.co/datasets/google/MusicCaps — audio retrieved via a pre-extracted community mirror ('nicolaus625/cmi'); 400 successfully paired examples used for Task 4.
