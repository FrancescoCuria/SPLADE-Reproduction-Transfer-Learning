# From Distillation to Hard Negative Sampling: Reproduction & Experiments on SPLADE++

## Overview
This repository contains a reproduction study on additive techniques for SPLADE models and an original experiment on transfer learning. The project was based on the work by Formal et al. (SIGIR '22).  
**[Read the full Project Presentation (PDF)](ProjectPresentation.pdf)**

## Research Objectives
The project aims to answer several research questions:
* Validate the additive improvements of SPLADE++ training strategies (distillation, hard negative mining, pre-training) in a zero-shot retrieval context.
* Compare the zero-shot performance of sparse retrieval models against dense models on unseen domains.
* **Extension Experiment:** Investigate how fine-tuning affects zero-shot generalization by analyzing domain adaptation in sparse retrieval models.

## Experimental Setup
The evaluation framework is based on 7 datasets extracted from the BEIR benchmark:
* **Datasets:** SciFact, SCIDOCS, FIQA, NFCorpus, ArguAna, TREC-COVID, Webis-Touché-2020.
* **Evaluated Models:** BM25 (baseline), SPLADE v2, SPLADE Distil, SPLADE++ CoCondenser-SelfDistil, SPLADE++ CoCondenser-Ensemble Distil and Hybrid Models (SPLADE++ + BM25).
* **Metrics:** NDCG@10, MRR@10, P@10, R@100, R@1K, Average FLOPS, Average Sparsity.

## Main Results

### 1. Reproduction and SOTA
* Baseline Matching: Successfully validated the SPLADE++ training pipeline, matching official State-Of-The-Art (SOTA) figures (e.g., reaching an NDCG@10 of 72.5 on TREC-COVID).
* Zero-Shot Superiority: Confirmed that sparse neural retrievers outscale dense baselines in zero-shot settings, providing a better trade-off between retrieval efficiency and lexical interpretability.
* Hybrid Retrieval: Demonstrated that combining sparse and lexical representations (SPLADE++ + BM25) consistently maximizes overall performance, proving highly effective in complex, "debate-oriented" tasks like Touché-2020.

### 2. Fine-Tuning Experiment (Transfer Learning)
An experiment was conducted to adapt the model from the SciFact domain (scientific fact-checking) to the SCIDOCS domain (citation recommendation). The results revealed the "Stability-Plasticity" dilemma:
* Domain adaptation produced neutral results on the target task (SCIDOCS).
* A slight "negative transfer" occurred on the source dataset (SciFact) and out-of-domain benchmarks (FIQA), indicating that the strong pre-training resisted adaptation and the model overfitted to the specifics of the source data.
* In low-resource scenarios, the zero-shot configuration proved to be superior and more reliable than few-shot fine-tuning strategies.

## Tech Stack
* Deep Learning: PyTorch, Hugging Face Transformers
* Information Retrieval: SPLADE Models, BM25, BEIR Benchmark
