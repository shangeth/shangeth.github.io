---
title: "SpeechLLM: Multi-Modal LLM for Speech Understanding"
date: "2024-06-26"
description: "A small multimodal LLM that reads paralinguistic signal — emotion, prosody, speaker traits — directly from speech audio instead of through an ASR transcript, built alongside the release of SpeechLLM at Skit.ai."
tags:
  - "Speech LLM"
  - "Voice AI"
  - "Speech Representation"
  - "Prosody"
---

[![GitHub](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/skit-ai/SpeechLLM.git)
[![HuggingFace speechllm-2B](https://img.shields.io/badge/🤗-speechllm--2B-blue.svg)](https://huggingface.co/skit-ai/speechllm-2B)
[![HuggingFace speechllm-1.5B](https://img.shields.io/badge/🤗-speechllm--1.5B-blue.svg)](https://huggingface.co/skit-ai/speechllm-1.5B)
[![Open in Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?logo=googlecolab&color=blue)](https://colab.research.google.com/drive/1uqhRl36LJKA4IxnrhplLMv0wQ_f3OuBM?usp=sharing)
[![Cite this work](https://img.shields.io/badge/Cite-BibTeX-yellow.svg)](#citation)

## Motivation

A conversational voice stack runs ASR, then an LLM, then TTS. The ASR emits text, so the LLM's input is text, so every decision the system makes is a function of the words alone.

The speech signal carries considerably more: speaker age, gender and accent; recording environment and channel quality; emotion and arousal, including whether frustration is rising or flat; prosody, which distinguishes a question from a statement with identical wording, and separates "I *said* Tuesday" from "I said *Tuesday*"; language choice, including mid-utterance code-switching; and turn-taking cues such as whether a pause is a hesitation or a completion.

None of this survives transcription. All of it was present in the encoder that produced the transcript. Two callers saying "yeah, that's fine", one confirming and one about to abandon the call, produce byte-identical LLM input and require opposite responses.

The question this experiment addresses is narrow: can an LLM read the non-lexical channel directly, at a parameter count small enough to eventually sit inside a live call? SpeechLLM is a research artifact, not a production system.

## Why the cascade resists incremental repair

Information loss is the obvious problem and the less interesting one. Three structural issues matter more.

**No gradient path between components.** The ASR receives no signal about what the LLM did with its output, so it cannot learn what the LLM needed. Each component is optimized against its own metric; the assembled system is optimized against nothing.

**Untrainable interface decisions.** Endpointing, barge-in handling and speaker continuity fall between components. With no differentiable path, they become hand-tuned thresholds.

**A symmetric bottleneck on output.** The TTS receives text and nothing else: no dialogue history, no caller state, not even its own preceding turn. Its prosody is therefore context-free by construction. In our demo system this is audible as pitch discontinuities across turns, since the vendor TTS re-derives delivery from scratch at each call.

OpenAI's [GPT-4o announcement](https://openai.com/index/hello-gpt-4o/) five weeks ago describes the same failure: a three-model pipeline cannot observe tone, multiple speakers or background noise, and cannot output laughter or emotion.

## Design space: getting audio into an LLM

Two viable routes as of early 2024.

**Discrete tokens.** Quantize speech so the LLM receives something close to its native input. Semantic units discard the paralinguistics of interest. Acoustic codec tokens preserve them but multiply sequence length and bias the model toward acoustic detail over content. [SpeechTokenizer](https://arxiv.org/abs/2308.16692) opens by establishing that neither token type is suitable for speech language modeling.

**Continuous embeddings.** Run a speech encoder, project its output into the LLM embedding dimension, prepend the resulting vectors. Information is preserved and sequences stay short. The cost is that continuous projections cannot be generated autoregressively, which constrains the output path (see below).

I used continuous embeddings.

**Encoder selection matters more than connector design,** because the encoder's pretraining objective determines what survives it. ASR-supervised encoders are trained to discard non-lexical variation. Self-supervised encoders retain more, and different objectives retain different things: [WavLM](https://arxiv.org/abs/2110.13900) adds denoising and utterance mixing to the [HuBERT](https://arxiv.org/abs/2106.07447) recipe specifically for speaker discrimination. Within any encoder, [information stratifies by depth](https://arxiv.org/abs/2107.04734), with acoustic detail in lower layers and lexical content in upper layers.

Encoders evaluated: HuBERT-XLarge, WavLM-Large, Whisper-small, SpeechTokenizer, and a CLIP-style contrastive audio encoder. **HuBERT-XLarge was best across all six tasks.** Several alternatives were competitive on transcription specifically; none was better across the board. Plain masked-prediction pretraining, with no quantization, supervision or contrastive objective, outperformed every alternative. I do not have a complete explanation for this.

LLM backbones considered: Gemma-2B, Phi-2, and [TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0). TinyLlama was selected on compute and cost grounds. The resulting system is roughly 2B parameters total.

## Embedding-space behaviour

An LLM's input embedding space is the image of a fixed discrete vocabulary. Injecting continuous vectors into it raises a question that the literature mostly leaves implicit: should the projector map audio onto the text manifold?

I do not think it should, and I do not think it does. Mapping each audio segment near the token it would transcribe to amounts to reimplementing ASR inside the projector, discarding exactly the information the architecture exists to preserve. Audio appears instead to occupy a separate region that the model accesses through attention rather than through the vocabulary.

**The alignment is unsupervised.** No frame-to-token correspondence appears anywhere in the loss. Next-token prediction over the output alone is sufficient to make the connector arrange audio frames such that existing attention heads can use them. That this succeeds with the language model [entirely frozen](https://arxiv.org/abs/2307.11795) suggests the mechanism transfers without retraining.

**Sequence layout is load-bearing.** The input is ordered: instruction, audio, closing marker, target. The instruction, including the list of requested attributes, is fully present in context before any audio frame is attended to. In a causal decoder this means the query conditions the reading of the audio, so identical speech embeddings are interrogated differently depending on the text. The continuous span is delimited by `<speech>` and `</speech>` tokens drawn from the model's own vocabulary.

This cross-channel conditioning is the argument for an LLM over a set of task-specific classifiers. Flat prosody, the lexical content "yeah, that's fine", and the preceding turn jointly resolve an ambiguity that none resolves alone. A discriminative emotion classifier has access to neither the words nor the dialogue history.

## Multi-task training over partially labelled corpora

Six target attributes: speech activity, transcript, gender, age, accent, emotion. No corpus contains all six.

| Corpus | Speech activity | Transcript | Gender | Emotion | Age | Accent |
|---|---|---|---|---|---|---|
| ARCA23K | ✓ (negative) | | | | | |
| CREMA-D | ✓ | ✓ | ✓ | ✓ | ✓ | |
| CommonVoice | ✓ | ✓ | ✓ | | ✓ | ✓ |
| IEMOCAP | ✓ | ✓ | ✓ | ✓ | | |
| LibriSpeech | ✓ | ✓ | ✓ | | | |
| ML Spoken Words | ✓ | ✓ (single word) | ✓ | | | |
| RAVDESS | ✓ | ✓ | ✓ | ✓ | | |

A conventional multi-head model requires one head and loss per attribute, a masking scheme for every absent label, and a fixed output schema.

Formulating the labels as text removes all of this. Each example's instruction enumerates only the attributes that example carries; the target is a JSON object with exactly those keys. Loss is masked across the instruction and the audio span and applied only to the target. Absent attributes are never requested and never contribute gradient. Seven corpora with disjoint label sets become a single training set with no masking logic.

Two augmentations: the requested field set is randomly subsampled on roughly 20% of examples, so field selection is learned rather than memorized as a fixed output shape; and instructions are sampled from approximately sixty paraphrases to avoid keying on a single string.

Consequences: arbitrary attribute subsets can be requested at inference with unchanged weights, and extending the schema (noise type, channel quality) requires data and a line of text rather than architectural change.

[Qwen-Audio](https://arxiv.org/abs/2311.07919) addresses the same problem at larger scale under the name one-to-many interference, using a hierarchical tag schema across thirty-plus tasks. Natural-language instructions achieve the same effect without a schema to maintain.

This does not eliminate task interference. Across checkpoints, accent accuracy improved while age accuracy and word error rate degraded. The formulation removes the interference from the architecture, not from the optimization.

## Model and limitations

![SpeechLLM architecture](https://github.com/skit-ai/SpeechLLM/raw/main/assets/speechllm.png)

Encoder, connector, LLM. 16kHz input; JSON output containing any requested subset of the six attributes. Roughly 2B parameters, single forward pass. [Colab](https://colab.research.google.com/drive/1uqhRl36LJKA4IxnrhplLMv0wQ_f3OuBM).

Metadata prediction was chosen over conversational response generation because it isolates the perception question and yields output directly consumable by a dialogue policy.

Known limitations:

1. **Duration dependence.** On utterances near one second, paralinguistic prediction degrades sharply relative to multi-second speech. Speaker attributes appear to require duration that short segments do not provide.
2. **Read versus spontaneous speech.** Performance on spontaneous emotional speech is substantially worse than on read speech, by a margin that makes read-speech evaluation largely uninformative for conversational deployment.
3. **Encoder tap point.** I used the final encoder layer, which given the depth-stratification result is the most lexical and least speaker-bearing layer available. An intermediate tap or learned layer weighting is an obvious untested improvement.
4. **Not streaming.** The encoder is non-causal, which precludes real-time use.

**Deployment caution.** A model inferring gender, age and accent from voice will be least accurate for speakers already underserved by speech systems, and the cost of those errors is not uniformly distributed. Appropriate use is as a soft prior on dialogue policy: verification, pacing, escalation to a human operator. Not as an identity assertion and not as an access control mechanism.

## Direction: generation, and full duplex

Perception is one stage. The system above still emits a text string to a TTS, so the output-side bottleneck described earlier applies to it unchanged.

**Generation.** Controlling delivery rather than only content reopens the tokenizer question under a tighter constraint: output units must support high-quality synthesis and carry sufficient semantic structure for the LLM to reason over. This is where the continuous-embedding choice becomes costly, since the output path must return to discrete representations.

**Joint perception and generation.** If a single model both perceives and generates over shared units, the two directions provide supervision to each other. Predicting future audio supervises prosody, speaker characteristics and environment in a way no annotated corpus does. The seven corpora above collectively required substantial annotation effort to yield six coarse attributes; unlabelled conversational audio contains far more information about the same properties.

**Full duplex.** Human turn transitions cluster near 200ms after turn end across languages ([Stivers et al., 2009](https://www.pnas.org/doi/10.1073/pnas.0903616106)); cascaded systems require seconds. The gap follows from an architecture that cannot begin computing until a voice-activity threshold fires. End-to-end, turn-taking becomes a learned function of recent prosody, pause structure and acoustic context rather than a threshold. Two prior results indicate tractability: [Voice Activity Projection](https://aclanthology.org/2022.sigdial-1.51/) learns turn-taking self-supervised from raw waveforms without turn-taking annotation, and [dGSLM](https://arxiv.org/abs/2203.16502) generates speech, laughter and naturalistic turn-taking across two channels simultaneously from 2000 hours of untranscribed two-channel conversation.

The duration-dependence limitation above bears directly on this. A full-duplex system reasons over a few hundred milliseconds at a time, precisely the regime where paralinguistic prediction failed here. Either these attributes require tracking across a conversation rather than per-segment inference, or the representation needs to change.

## Evaluation gap

Available metrics are word error rate and accuracy on coarse attribute labels. Nothing measures whether a system responded appropriately to rising frustration, which is the capability this line of work exists to enable. Every quantity I reported measures the part of the problem that was already measurable. Absent a benchmark for conversational appropriateness, the field will continue optimizing transcription.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{Rajaa_SpeechLLM_Multi-Modal_LLM,
author = {Rajaa, Shangeth and Tushar, Abhinav},
title = {{SpeechLLM: Multi-Modal LLM for Speech Understanding}},
url = {https://github.com/skit-ai/SpeechLLM}
}
```

## References

<ol class="references">
<li>Borsos, Z., et al. (2022). <em>AudioLM: a Language Modeling Approach to Audio Generation.</em> <a href="https://arxiv.org/abs/2209.03143" target="_blank" rel="noopener noreferrer">arXiv:2209.03143</a>.</li>
<li>Chen, S., et al. (2022). <em>WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing.</em> IEEE JSTSP. <a href="https://arxiv.org/abs/2110.13900" target="_blank" rel="noopener noreferrer">arXiv:2110.13900</a>.</li>
<li>Chu, Y., et al. (2023). <em>Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models.</em> <a href="https://arxiv.org/abs/2311.07919" target="_blank" rel="noopener noreferrer">arXiv:2311.07919</a>.</li>
<li>Ekstedt, E., &amp; Skantze, G. (2022). <em>Voice Activity Projection: Self-supervised Learning of Turn-taking Events.</em> Interspeech 2022, 5190-5194.</li>
<li>Ekstedt, E., &amp; Skantze, G. (2022). <em>How Much Does Prosody Help Turn-taking? Investigations using Voice Activity Projection Models.</em> SIGDIAL 2022, 541-551.</li>
<li>Fathullah, Y., et al. (2023). <em>Prompting Large Language Models with Speech Recognition Abilities.</em> <a href="https://arxiv.org/abs/2307.11795" target="_blank" rel="noopener noreferrer">arXiv:2307.11795</a>.</li>
<li>Hsu, W.-N., et al. (2021). <em>HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units.</em> IEEE/ACM TASLP, 29, 3451-3460.</li>
<li>Ma, Z., et al. (2024). <em>An Embarrassingly Simple Approach for LLM with Strong ASR Capacity.</em> <a href="https://arxiv.org/abs/2402.08846" target="_blank" rel="noopener noreferrer">arXiv:2402.08846</a>.</li>
<li>Nguyen, T. A., et al. (2023). <em>Generative Spoken Dialogue Language Modeling.</em> TACL. <a href="https://arxiv.org/abs/2203.16502" target="_blank" rel="noopener noreferrer">arXiv:2203.16502</a>.</li>
<li>OpenAI (2024). <em>Hello GPT-4o.</em> <a href="https://openai.com/index/hello-gpt-4o/" target="_blank" rel="noopener noreferrer">openai.com/index/hello-gpt-4o</a>.</li>
<li>Pasad, A., Chou, J.-C., &amp; Livescu, K. (2021). <em>Layer-Wise Analysis of a Self-Supervised Speech Representation Model.</em> ASRU 2021, 914-921.</li>
<li>Rajaa, S. (2023). <em>Improving End-to-End SLU performance with Prosodic Attention and Distillation.</em> Interspeech 2023, 1114-1118.</li>
<li>Rajaa, S., Dalmia, S., &amp; Nethil, K. (2022). <em>Skit-S2I: An Indian Accented Speech to Intent Dataset.</em> <a href="https://arxiv.org/abs/2212.13015" target="_blank" rel="noopener noreferrer">arXiv:2212.13015</a>.</li>
<li>Rubenstein, P. K., et al. (2023). <em>AudioPaLM: A Large Language Model That Can Speak and Listen.</em> <a href="https://arxiv.org/abs/2306.12925" target="_blank" rel="noopener noreferrer">arXiv:2306.12925</a>.</li>
<li>Stivers, T., et al. (2009). <em>Universals and cultural variation in turn-taking in conversation.</em> PNAS, 106(26), 10587-10592.</li>
<li>Tang, C., et al. (2023). <em>SALMONN: Towards Generic Hearing Abilities for Large Language Models.</em> ICLR 2024. <a href="https://arxiv.org/abs/2310.13289" target="_blank" rel="noopener noreferrer">arXiv:2310.13289</a>.</li>
<li>Tushar, A. (2023). <em>Speech-First Conversational AI Revisited.</em> Skit Tech.</li>
<li>Zhang, X., et al. (2023). <em>SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models.</em> ICLR 2024. <a href="https://arxiv.org/abs/2308.16692" target="_blank" rel="noopener noreferrer">arXiv:2308.16692</a>.</li>
</ol>
