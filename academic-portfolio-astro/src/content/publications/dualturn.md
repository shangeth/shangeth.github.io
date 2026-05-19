---
title: "DualTurn: Learning Turn-Taking from Dual-Channel Generative Speech Pretraining"
author: "Shangeth Rajaa"
date: "2026-03-09"
journal: "Interspeech 2026 (Accepted)"
external_url: "https://arxiv.org/abs/2603.08216"
description: "Dual-channel generative pretraining for learning natural turn-taking in spoken dialogue without labeled data. A 0.5B model that outperforms models 6x its size on turn prediction."
tags:
  - "Voice AI"
  - "Turn-Taking"
  - "Spoken Dialogue"
  - "Speech LLM"
---

## Abstract

One of the hard problems in voice AI is knowing when to speak and when to listen. DualTurn tackles this by pretraining on dual-channel conversational audio, generating both speakers' future audio autoregressively so the model learns natural conversational dynamics without any labeled data.

After fine-tuning, it predicts turn-taking signals that map to agent actions. Despite being 0.5B parameters, it beats much larger alternatives: wF1 0.633 vs. 0.389 compared to VAP on agent action prediction, and AUC 0.930 vs. 0.880 versus a 3.1B model on word-level turn prediction.
