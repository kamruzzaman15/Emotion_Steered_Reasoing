# Emotion-Steered Mitigation of Biased Reasoning in Large Language Models

This repository contains code and research materials for studying the connection between biased reasoning in large language models (LLMs) and emotion-like internal representations.

The project asks two main questions:

1. Are biased model behaviors associated with specific emotion-like directions in hidden states?
2. Can these directions be used to reduce biased behavior without damaging general emotion reasoning or task performance?

The current version focuses on internal representation analysis, bias probing, and activation steering for mitigation.

## Overview

LLMs can produce biased outputs in many settings. For example, they may prefer stereotype-consistent continuations, make stereotyped guesses in ambiguous contexts, or change their responses when demographic information is added to the prompt.

Most prior work measures these problems only from the final output. This project studies the internal side of the problem. It analyzes hidden states to test whether biased behavior is linked to emotion-like representations inside the model.

The main idea is to:

1. construct model-specific emotion vectors from hidden states,
2. project benchmark representations onto these emotion directions,
3. compare biased and non-biased behaviors in emotion space,
4. learn a bias-related direction from emotion projection features, and
5. use protected activation steering to reduce harmful behavior while preserving useful emotion reasoning.

## Main Contributions

- Builds model-specific emotion vectors from hidden states.
- Probes LLM internal states using interpretable emotion-like directions.
- Analyzes biased behavior across multiple benchmarks, including:
  - StereoSet
  - GenAssocBias
  - BBQ
  - BOLD
- Studies whether emotion-space patterns can predict biased model behavior.
- Introduces a protected activation-steering method for mitigation.
- Evaluates whether steering reduces biased behavior while preserving:
  - general emotion reasoning,
  - general task performance,
  - output coherence.

## Method Summary

### 1. Emotion Vector Construction

We construct emotion vectors by prompting the model with short emotion-inducing contexts. For each emotion, hidden states are collected and averaged. These vectors are then centered against a global mean and normalized.

The resulting vectors represent model-specific emotion-like directions in hidden-state space.

### 2. Bias Probing

We run bias benchmarks through the model and collect hidden states from relevant layers. These hidden states are projected onto the emotion vectors.

The analysis compares projection patterns across conditions such as:

- stereotype vs. anti-stereotype completions,
- biased answers vs. uncertainty-preserving answers,
- ambiguous vs. unambiguous examples,
- pre-generation vs. post-generation representations.

This helps test whether certain emotion-like directions are more active when the model produces biased behavior.

### 3. Bias Direction Learning

We use emotion projection features to train a lightweight classifier that separates biased and non-biased behavior.

The classifier weights are then mapped back into hidden-state space to form a bias-related direction. This direction represents the part of the emotion space that is most associated with biased behavior.

### 4. Protected Steering

A direct removal of the full bias direction may damage useful model behavior. To avoid this, we define a protected emotion subspace that should be preserved for general emotion reasoning.

The final steering direction removes only the residual harmful component:

- preserve the part needed for general emotion reasoning,
- steer away from the remaining bias-related direction.

This allows mitigation while reducing the risk of hurting general emotional competence or task performance.

## Repository Structure

```text
.
├── data/
│   ├── stereoset/
│   ├── genassocbias/
│   ├── bbq/
│   └── bold/
├── src/
│   ├── emotion_vectors/
│   ├── probing/
│   ├── steering/
│   ├── evaluation/
│   └── utils/
├── configs/
├── notebooks/
├── scripts/
├── results/
├── figures/
├── requirements.txt
└── README.md
