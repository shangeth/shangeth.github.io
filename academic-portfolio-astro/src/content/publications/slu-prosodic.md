---
title: "Improving End-to-End SLU Performance with Prosodic Attention and Distillation"
author: "Shangeth Rajaa"
date: "2023-08-20"
journal: "Interspeech 2023, pp. 1114–1118"
external_url: "https://doi.org/10.21437/Interspeech.2023-1760"
description: "Two techniques for incorporating prosody into end-to-end SLU: prosody-attention and prosody-distillation. Up to 8% intent classification accuracy improvement on SLURP."
tags:
  - "Voice AI"
  - "Spoken Language Understanding"
  - "Prosody"
  - "Speech"
---

## Abstract

Most end-to-end SLU systems use pretrained ASR or LM features but completely ignore prosody, even though how something is said often matters as much as what is said. We propose two ways to fix this: prosody-attention, which builds attention maps from prosodic features across time, and prosody-distillation, which directly teaches the acoustic encoder to understand prosodic patterns rather than just concatenating them.

Prosody-distillation gives 8% and 2% intent accuracy gains on SLURP and STOP over the prosody baseline.
