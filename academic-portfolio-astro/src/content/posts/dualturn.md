---
title: "DualTurn: Learning Turn-Taking from Dual-Channel Generative Speech Pretraining"
date: "2026-04-03"
description: "Speech-to-speech models know when to speak but can't reason. Cascaded LLM pipelines can reason but only react to silence. DualTurn pretrains on dual-channel human conversation to bring S2S-level turn-taking into a standard ASR-LLM-TTS stack."
tags:
  - "Voice AI"
  - "Turn-Taking"
  - "Full-Duplex"
  - "Spoken Dialogue"
  - "Speech LLM"
---

[![GitHub](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/anyreachai/dualturn)
[![HuggingFace](https://img.shields.io/badge/🤗-DualTurn-blue.svg)](https://huggingface.co/collections/anyreach-ai/dualturn)
[![arXiv](https://img.shields.io/badge/arXiv-2603.08216-b31b1b.svg)](https://arxiv.org/abs/2603.08216)
[![Cite this work](https://img.shields.io/badge/Cite-BibTeX-yellow.svg)](#citation)

<div class="status-highlight">
🎉 <strong>Accepted at Interspeech 2026!</strong> Excited to present DualTurn in person in Sydney, September 28 – October 1, 2026.
</div>

<div class="status-highlight">
Update, July 2026: see how this went from paper to production in <a href="/posts/dualturn-production">DualTurn in Production</a>.
</div>

## The turn-taking problem hasn't actually been solved

Every voice agent has to answer the same question tens of times per call: has the other person finished talking? Get it wrong in one direction and the agent talks over the user; get it wrong in the other and the user sits through a dead-air pause wondering if the call dropped. Neither failure is rare in production voice AI today, and the reason is structural, not incidental.

The default mechanism is still a silence timer sitting on top of [voice activity detection](https://www.cekura.ai/blogs/voice-activity-detection) (VAD): the moment acoustic energy drops for long enough, the system assumes the turn is over and fires. The industry has spent real effort making this less crude. [Semantic endpointing](https://gradium.ai/content/semantic-vad-voice-agents-turn-detection-2026) reads the transcript for syntactic and lexical completeness instead of just measuring silence duration; [LiveKit's Turn Detector](https://livekit.com/blog/solving-end-of-turn-detection) and [Deepgram's Flux](https://deepgram.com/learn/introducing-flux-conversational-speech-recognition) fuse acoustic and linguistic cues into a learned classifier; recent work adds neural-codec features ([Udupa et al., 2025](https://arxiv.org/abs/2506.07081)) and multimodal fusion ([Li et al., 2025](https://arxiv.org/abs/2509.23938)) to the same basic recipe. All of it is a real improvement over a fixed-duration timeout.

But look at where every one of these systems sits in the pipeline: downstream of VAD. VAD has to fire a candidate pause before the smarter model ever gets to weigh in. The classifier can be arbitrarily sophisticated about *whether* a detected pause is a real end of turn, but it has no opinion on turns that haven't paused yet. It is a better reflex, not anticipation. Structurally, every production system in this category is reactive. It can only classify silence that has already happened.

This is also, if you look closely, a **half-duplex** problem dressed up as a classification problem. A half-duplex system is a walkie-talkie: one side transmits, the other is deaf to anything but a fixed interrupt signal. That is what a standard ASR-LLM-TTS pipeline is functionally doing. It generates and plays a full agent turn, and "listening" during that turn usually means nothing more than watching for a VAD spike to cut playback for barge-in. It is not actually processing the user's channel the way a person would. A **full-duplex** system keeps both channels open and modeled continuously and simultaneously, the way two people on a call behave: you can hear the other person while you're still forming your own sentence, overlap for a moment, back off, backchannel without stopping them. Continuous, simultaneous processing of both channels is what makes anticipation possible in the first place. A half-duplex system cannot anticipate a turn it isn't listening for yet, no matter how good its classifier is once the pause finally shows up.

Full-duplex speech-to-speech (S2S) models are built around exactly that continuous, simultaneous processing, and it's worth being precise about why they end up good at turn-taking as a side effect. Models like [Moshi](https://arxiv.org/abs/2410.00037), [dGSLM](https://arxiv.org/abs/2203.16502), [PersonaPlex](https://arxiv.org/abs/2602.06053), and [NTPP](https://arxiv.org/abs/2506.00975) are trained to generate *both* speakers' audio streams jointly, frame by frame, continuously. There's no separate turn-detection module, because turn-taking was never factored out as a separate problem in the first place. To predict the next audio frame on either channel well, the model has to implicitly track who's about to speak, whether a pause is a hesitation or a completion, whether an overlap is a backchannel or an interruption. Generation *is* the anticipation mechanism here. Moshi reports human raters preferring it specifically on prosody, turn-taking, and empathy over cascaded baselines, and that's not a coincidence. It's what joint dual-channel generative training optimizes for.

The catch is everything these models are *not* trained to do. They're end-to-end audio-in, audio-out language models, but generic ones: no tool calling, no retrieval, no reliable structured output, none of the reasoning scaffolding that a production voice agent (booking a flight, pulling account data, checking a policy) actually needs. And their turn-taking competence lives inside the joint generative process itself, so there's no clean way to lift it out and bolt it onto a conventional ASR-LLM-TTS stack, which is what most deployed systems still are, for good reason: reasoning and tool-calling are considerably easier to get right in a modular pipeline with a text LLM at the center.

So the field has two model classes, each with the thing the other lacks. S2S models have the turn-taking, cascaded pipelines have the reasoning. **[DualTurn](https://arxiv.org/abs/2603.08216)** is my attempt to move the turn-taking half across that boundary. It takes exactly the ingredient that makes S2S models good at turn-taking (dual-channel generative pretraining), but uses it only to build a *perception* component: a small model that listens to both channels of a live call full-duplex, continuously, and emits interpretable signals a standard half-duplex pipeline can act on, rather than a model that has to also generate the reply.

The closest existing attempt at this is [Voice Activity Projection](https://doi.org/10.21437/Interspeech.2022-10955) (VAP), a self-supervised, dual-channel model that continuously predicts future voice activity from raw audio rather than waiting on VAD. VAP is the right idea, but its 5.8M-parameter CPC encoder collapses everything into a single binary voice-activity probability. It cannot tell a backchannel ("mhm", "right") from an actual turn-taking bid, and needs additional fine-tuning just to beat chance at backchannel prediction at all ([Inoue et al., 2025](https://aclanthology.org/2025.naacl-long.367/)). Text-based models like [TurnGPT](https://doi.org/10.18653/v1/2020.findings-emnlp.268) ignore prosody entirely. Single-channel audio classifiers ([Roddy et al., 2018](https://doi.org/10.21437/Interspeech.2018-2124)) can flag that speech ended but, without the other speaker's channel, can't tell you what kind of ending it was. The strongest prior model, [Wang et al. (2024)](https://doi.org/10.1109/ICASSP48485.2024.10447196), fuses a 3.1B-parameter LLM with audio and gets good word-level predictions, at a parameter count well past what a low-latency, always-on turn-taking component can afford.

## Architecture

<img src="/images/posts/dualturn-architecture.png" alt="DualTurn architecture: Stage-1 generative pretraining of the backbone and depth predictor, and Stage-2 fine-tuning of twelve classification heads" />

Both channels, the user's and the agent's, go through a frozen [Mimi](https://arxiv.org/abs/2410.00037) neural codec, which turns each 24kHz waveform into a sequence of 8 RVQ codebooks at 12.5 frames/second. Rather than the discrete codebook indices, DualTurn uses Mimi's *continuous* 512-dim encoder embeddings per channel, an ablation choice that turns out to matter a lot, as I'll get to below. Each channel gets its own MLP projection, the two representations are concatenated, and the result feeds a Qwen2.5-0.5B backbone.

Twelve lightweight classification heads (six per channel) read off the backbone's final hidden state: two-layer MLPs with GELU and dropout for the sparse, event-like signals (EOT, HOLD, BOT, BC), and plain linear projections for the dense per-frame signals (VAD, FVAD). At inference the model streams both channels continuously with a 240ms stride (3 audio frames per step) and KV-caching, with no waiting for VAD to nominate a pause. That's the full-duplex listening part in practice: both channels are processed every 240ms regardless of who's talking, even though the agent's own speech still comes out of a standard TTS turn at the end of the pipeline. Latency is about 78ms on CPU, 27ms on an A100, well inside budget for a component that has to run continuously alongside the rest of the stack rather than firing occasionally.

### Stage 1: generative pretraining, no labels

The backbone is first trained to autoregressively predict *both* speakers' next audio frame simultaneously, with a small (~10.6M-parameter) depth predictor generating the next-frame RVQ tokens per channel from the backbone's output. This is the same objective family that gives S2S models their turn-taking sense: the model can't get good at predicting what either speaker says next without implicitly modeling semantics, prosody, and the interaction pattern between the two channels (who's about to jump in, whether a pause is filled or terminal, whether an overlap is a backchannel). Once pretraining finishes, the depth predictor is thrown away (DualTurn never needs to actually generate audio) and only the backbone carries into Stage 2.

### Stage 2: turn-taking signals, still no manual labels

Six per-channel signals are fine-tuned on top of the pretrained backbone, and critically, every label is derived automatically from voice-activity alignment, with no human annotation anywhere in the pipeline:

| Signal | Definition |
|---|---|
| **EOT** (end-of-turn) | Speech offset where the other speaker takes the floor within 4s |
| **HOLD** | Any other speech offset (i.e., not an EOT) |
| **BOT** (beginning-of-turn) | Speech onset (≥1s) following the other speaker |
| **BC** (backchannel) | Isolated utterance ≤1s with ≥1s silence on both sides |
| **VAD** | Binary voice activity, per frame |
| **FVAD** (future VAD) | Mean voice activity over four look-ahead windows: 0–240ms, 240–480ms, 480–960ms, 960ms–2s |

EOT and HOLD are complementary by construction (exactly one fires at every speech offset), which is what lets the model learn to separate a genuine turn end from a mid-turn thinking pause without anyone labeling the difference. The 4-second EOT window (versus VAP's 1 second) matters more than it looks: about 12% of Switchboard turn transfers involve pauses longer than a second, and a 1s cutoff just throws that data away. Labels are also smoothed with an asymmetric Gaussian (σ=3 frames before an event, σ=1 after), which explicitly trains the model to anticipate each signal up to 240ms early rather than only recognize it after the fact.

These six raw signals aren't directly useful to a dialogue policy; what a pipeline actually needs is an action. So Stage 2 maps them to five:

| Action | Definition |
|---|---|
| **ST** (start-talking) | User offset; agent begins speaking within 4s |
| **CL** (continue-listening) | User offset; user resumes within 2s |
| **SL** (start-listening) | Overlap onset; incoming speech is >1s |
| **CT** (continue-talking) | Overlap onset; incoming speech is <1s |
| **BC** (backchannel) | Agent vocalization <1s during user speech |

ST/CL are essentially VAP's shift/hold decision; SL, CT, and BC are the overlap and backchannel behavior that no prior turn-taking model addresses directly. The mapping from six signals to five actions can be a zero-parameter heuristic (e.g. `EOT_user > 0.5 AND BOT_agent > 0.5 → ST`) or a multinomial logistic regression probe fit on held-out data. The LR coefficients are legible: the start-talking action weighs `EOT_user` and `VAD_agent` positively while weighing `VAD_user` and `BOT_user` negatively, which is exactly the "user has stopped and isn't restarting" pattern you'd hand-design if you were writing the heuristic yourself.

## Results

Training used about 453 hours of dual-channel conversational audio: 289 hours of [otoSpeech](https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-280h) (English, full-duplex, 24kHz) and 220 hours of [Switchboard](https://doi.org/10.1109/ICASSP.1992.225858) (telephone speech, 8kHz). A 138-session Switchboard split and a disjoint 113-session otoSpeech split were held out for evaluation, the same test protocol used by VAP and Wang et al., so the numbers are directly comparable.

**Word-level turn prediction**, against Wang et al.'s 3.1B audio+text model and a text-only GPT-2 baseline:

| Model | AUC(Continue) | AUC(Backchannel) | AUC(Turn) | Avg AUC | EER |
|---|---|---|---|---|---|
| GPT-2 (text-only) | 0.851 | 0.774 | 0.862 | 0.829 | 24.5 |
| RP+HuBERT+hist (3.1B) | 0.903 | 0.818 | 0.920 | 0.880 | 19.3 |
| DualTurn (EOT signal alone) | 0.918 | 0.904 | 0.919 | 0.914 | 15.2 |
| DualTurn (heuristic) | 0.940 | 0.925 | 0.924 | 0.930 | 13.2 |
| DualTurn (LR probe) | **0.961** | **0.979** | **0.950** | **0.963** | **9.7** |

A single raw signal from the 0.5B model, with zero combination logic, already beats a 3.1B fusion model that has both audio and transcript access. The equal-error-rate column tells the same story in a more intuitive unit: it goes from 24.5% for a text-only model, to 19.3% once you add audio and a 3.1B backbone, down to 9.7% for DualTurn's LR probe.

**Agent action prediction**, against VAP:

| Model | wF1 (Switchboard) | BC F1 (Switchboard) | wF1 (otoSpeech) | BC F1 (otoSpeech) |
|---|---|---|---|---|
| VAP (native) | 0.276 | — | — | — |
| VAP (LR probe) | 0.389 | 0.000 | 0.461 | 0.000 |
| **DualTurn** | **0.633** | **0.349** | **0.707** | **0.512** |

VAP's BC F1 is not a rounding artifact. It's a model with no dedicated backchannel signal, and no amount of post-hoc logistic regression over its outputs can invent one. DualTurn's 0.349 (chance ≈ 0.080) is the first result I know of where a self-supervised, dual-channel turn-taking model, trained the same way VAP is on the same kind of unlabeled data, gets meaningfully above chance on this.

**Anticipation.** Plotting shift/hold AUC against time relative to speech offset makes the gap visual: DualTurn's curve sits above VAP's at every point, and the two are already clearly separated well before the turn actually ends, not just after:

<img src="/images/posts/dualturn-anticipation.png" alt="Shift-vs-hold AUC across time relative to speech offset: DualTurn leads VAP by roughly 220ms" />

Median reaction time is -360ms relative to turn end for DualTurn versus -140ms for VAP, about 220ms of extra anticipation. That also shows up as fewer mistakes: start-talking-for-continue-listening confusions drop from 27.4% to 22.4%, ST F1 improves from 0.808 to 0.829, and interruptions fall by 5 percentage points.

One obvious objection to all of this: DualTurn is trained on a broader 4-second turn definition than VAP's original 1-second one, so maybe the gains are just an artifact of an easier label. Re-running VAP's own 1-second frame-level protocol, VAP's home turf, rules that out:

| Protocol (VAP's original 1s definitions) | Shift/Hold | Short/Long | Shift-Prediction | BC-Prediction |
|---|---|---|---|---|
| VAP | 0.843 | 0.916 | 0.720 | 0.838 |
| DualTurn | **0.985** | **0.979** | **0.764** | **0.864** |

DualTurn wins on all four, including the exact task VAP was designed and tuned for. The gain is in the representation, not the label definition.

## Why does pretraining matter this much?

The most useful result in the paper, honestly, isn't a leaderboard number. It's the ablation that explains *where* the backchannel gain actually comes from, because the answer is not "a bigger model."

Compare an 8M-parameter LSTM against the 0.5B-parameter LLM backbone, both *without* Stage-1 pretraining: wF1 0.602 vs. 0.604, BC F1 0.077 vs. 0.079. Statistically identical. Sixty-two times the parameters buys essentially nothing on backchannel detection if the model never went through dual-channel generative pretraining. Now compare the pretrained LLM (A) against the same architecture without pretraining (C): BC F1 jumps from 0.079 to 0.349, a 340% relative improvement. Over 99% of the total backchannel gain traces to pretraining, not architecture. The LLM's capacity advantage over the LSTM contributes roughly +0.002 BC F1 on its own.

That's the core claim of the paper stated as a measurement rather than a slogan: **the backbone is a vessel, not the source of turn-taking knowledge. Generative pretraining is the teacher.** A model only benefits from more capacity once that capacity has something to hold.

Two more analyses back this up:

**Signal difficulty splits cleanly along the same line.** VAD and FVAD, dense per-frame signals, are learned fine by every variant, including the plain LSTM. BOT, BC, and EOT are sparse interactional events that require modeling two-speaker dynamics, and pretraining is what closes the gap on exactly those: +188% on BOT, +41% on BC, near-zero change on VAD/FVAD. Pretraining isn't making the model uniformly better. It's specifically teaching it the interactional structure that per-frame classifiers never needed.

**Multi-scale attention emerges without being asked for.** Probing attended temporal distance per head/layer in the pretrained backbone shows six layers attending short-range (<1s, presumably frame-level acoustic detail) and three attending long-range (L7 at 14.9s, L9 at 12.6s, L11 at 14.8s), tracking conversational context at a horizon an 8M LSTM structurally can't reach. Strip out pretraining and only three layers stay short-range, with long-range attention scattered rather than structured; this is the mechanistic reason C and D perform the same. The pretrained model also develops action-specific attention with no supervision telling it to: the user channel gets attended 3.77× more for continue-listening decisions and 2.21× more for backchannel decisions, and the two action classes reach back over different windows (3.41s for BC, 4.91s for CT).

A codebook ablation adds a consistent, independent line of evidence: zeroing individual Mimi codebooks shows CB0 (semantic content) contributing 56% of shift/hold discrimination, CB1 another 26%, and the remaining six fine-acoustic codebooks together only 18%. Turn-taking is mostly a semantics-and-prosody problem, not a fine-acoustic-detail problem, which lines up with [Gravano & Hirschberg (2011)](https://doi.org/10.1016/j.csl.2010.10.003).

Two negative results are worth keeping too, because they cut against intuitions I had going in. Keeping the Stage-1 generative loss active as an auxiliary objective during Stage-2 hurts: BC F1 drops from 0.349 to 0.077, because the generative and classification gradients compete and the sparse classification signal loses. And adding a text-output (ASR) objective during Stage-1, specifically to align the representation with the LLM's pretrained text capability, also hurts Stage-2 performance (BC F1 0.349 → 0.085). Audio-only pretraining produces a better turn-taking representation than a text-aligned one. On a more mundane note, LoRA adapters slightly *beat* full fine-tuning of all 500M parameters (BC F1 0.349 vs. 0.337) using 55× fewer trainable parameters, and continuous Mimi embeddings clearly beat discrete codebook indices as input (BC F1 0.349 vs. 0.072), since quantization throws away exactly the prosodic nuance the sparse signals depend on.

## Limitations

None of this is deployment-ready as-is, and it's worth being specific about where it falls short rather than leaving that implicit.

**Backchannel precision is still modest.** 0.349 F1 sounds strong against chance (≈0.080), but it decomposes into recall 0.458 and precision 0.282, meaning roughly 72% of predicted backchannels are still false positives. Real, not solved. Backchannels are under 8% of evaluation events, so the signal is real but sparse enough that it shouldn't be a standalone trigger in production. It belongs as a soft input to a decision policy, at a raised threshold, not as the sole gate on whether the agent says "mhm."

**453 hours, English, two-party only.** That's a small pretraining corpus by generative-model standards, and everything reported here is single-language, two-speaker telephone/online conversation. Multi-party dynamics (who's addressed, who's expected to yield) and cross-lingual turn-taking norms (which do differ, prosodically and structurally) are untested.

**The signal-to-action mapping is still linear.** Both the heuristic and the LR probe can only combine what the twelve heads already output; neither can recover information the pretrained representation didn't surface. A more expressive probe, or joint end-to-end training of the mapping, is an obvious next step that wasn't tried here.

**Evaluation is still Switchboard-shaped.** Both benchmarks are telephone-style, sequential, two-party dialogue. Real deployed voice agents increasingly involve tool-use latency (the agent goes silent mid-turn while a function call resolves, which looks like a pause but isn't one) and multi-turn task state that this evaluation doesn't exercise.

## Where this leaves the field

Scaling to larger, multilingual, multi-party corpora and improving the generative pretraining recipe itself are the natural next steps. The 453-hour, English-only, two-party setup here is a floor, not a ceiling. But the result I'd want to hold onto even as the numbers move is the ablation, not the leaderboard: architecture bought almost nothing without pretraining, and pretraining bought almost everything without architecture. If that generalizes, the lesson for anyone building the next generation of turn-taking components isn't "use a bigger model." It's "give the model something to predict that forces it to model both speakers at once." The backbone is the vessel. Generative pretraining is the teacher.

Code, the trained checkpoints, and the datasets are public: [GitHub](https://github.com/anyreachai/dualturn) has the training and inference code, and the [Hugging Face collection](https://huggingface.co/collections/anyreach-ai/dualturn) has the open model weights and datasets, if you want to run it yourself or pick the ablations apart.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{rajaa2026dualturn,
  author        = {Rajaa, Shangeth},
  title         = {{DualTurn: Learning Turn-Taking from Dual-Channel Generative Speech Pretraining}},
  year          = {2026},
  eprint        = {2603.08216},
  archivePrefix = {arXiv},
  primaryClass  = {eess.AS},
  url           = {https://arxiv.org/abs/2603.08216}
}
```

## References

<ol class="references">
<li>Défossez, A., et al. (2024). <em>Moshi: A Speech-Text Foundation Model for Real-Time Dialogue.</em> <a href="https://arxiv.org/abs/2410.00037" target="_blank" rel="noopener noreferrer">arXiv:2410.00037</a>.</li>
<li>Deepgram (2026). <em>Introducing Flux: Conversational Speech Recognition to Solve the Biggest Problem in Voice Agents — Interruptions.</em> <a href="https://deepgram.com/learn/introducing-flux-conversational-speech-recognition" target="_blank" rel="noopener noreferrer">deepgram.com</a>.</li>
<li>Ekstedt, E., &amp; Skantze, G. (2020). <em>TurnGPT: A Transformer-based Language Model for Predicting Turn-taking in Spoken Dialog.</em> Findings of ACL: EMNLP 2020, 2981-2990.</li>
<li>Ekstedt, E., &amp; Skantze, G. (2022). <em>Voice Activity Projection: Self-supervised Learning of Turn-taking Events.</em> Interspeech 2022, 5190-5194.</li>
<li>Godfrey, J. J., Holliman, E. C., &amp; McDaniel, J. (1992). <em>Switchboard: Telephone Speech Corpus for Research and Development.</em> ICASSP 1992, 517-520.</li>
<li>Gravano, A., &amp; Hirschberg, J. (2011). <em>Turn-taking cues in task-oriented dialogue.</em> Computer Speech &amp; Language, 25(3), 601-634.</li>
<li>Hu, E. J., et al. (2022). <em>LoRA: Low-Rank Adaptation of Large Language Models.</em> ICLR 2022. <a href="https://arxiv.org/abs/2106.09685" target="_blank" rel="noopener noreferrer">arXiv:2106.09685</a>.</li>
<li>Inoue, K., Lala, D., Skantze, G., &amp; Kawahara, T. (2025). <em>Yeah, Un, Oh: Continuous and Real-time Backchannel Prediction with Fine-tuning of Voice Activity Projection.</em> NAACL 2025, 7171-7181. <a href="https://aclanthology.org/2025.naacl-long.367/" target="_blank" rel="noopener noreferrer">aclanthology.org</a>.</li>
<li>Li, G., et al. (2025). <em>Easy Turn: Integrating Acoustic and Linguistic Modalities for Robust Turn-Taking in Full-Duplex Spoken Dialogue Systems.</em> <a href="https://arxiv.org/abs/2509.23938" target="_blank" rel="noopener noreferrer">arXiv:2509.23938</a>.</li>
<li>LiveKit (2026). <em>Solving End-of-Turn Detection: LiveKit Turn Detector v1.0.</em> <a href="https://livekit.com/blog/solving-end-of-turn-detection" target="_blank" rel="noopener noreferrer">livekit.com</a>.</li>
<li>Nguyen, T. A., et al. (2023). <em>Generative Spoken Dialogue Language Modeling.</em> TACL, 11, 250-266. <a href="https://arxiv.org/abs/2203.16502" target="_blank" rel="noopener noreferrer">arXiv:2203.16502</a>.</li>
<li>otoearth (2025). <em>otoSpeech-full-duplex-280h: Full-Duplex Conversational Speech Dataset.</em> <a href="https://huggingface.co/datasets/otoearth/otoSpeech-full-duplex-280h" target="_blank" rel="noopener noreferrer">huggingface.co</a>.</li>
<li>Qwen Team (2025). <em>Qwen2.5 Technical Report.</em> <a href="https://arxiv.org/abs/2412.15115" target="_blank" rel="noopener noreferrer">arXiv:2412.15115</a>.</li>
<li>Rajaa, S. (2026). <em>DualTurn: Learning Turn-Taking from Dual-Channel Generative Speech Pretraining.</em> Interspeech 2026. <a href="https://arxiv.org/abs/2603.08216" target="_blank" rel="noopener noreferrer">arXiv:2603.08216</a>.</li>
<li>Roddy, M., Skantze, G., &amp; Harte, N. (2018). <em>Investigating Speech Features for Continuous Turn-Taking Prediction Using LSTMs.</em> Interspeech 2018, 586-590.</li>
<li>Roy, R., et al. (2026). <em>PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models.</em> <a href="https://arxiv.org/abs/2602.06053" target="_blank" rel="noopener noreferrer">arXiv:2602.06053</a>.</li>
<li>Udupa, S., Kumar, K., Majumdar, S., Balam, J., &amp; Ginsburg, B. (2025). <em>Streaming Endpointer for Spoken Dialogue using Neural Audio Codecs and Label-Delayed Training.</em> <a href="https://arxiv.org/abs/2506.07081" target="_blank" rel="noopener noreferrer">arXiv:2506.07081</a>.</li>
<li>Wang, J., et al. (2024). <em>Turn-taking and Backchannel Prediction with Acoustic and Large Language Model Fusion.</em> ICASSP 2024.</li>
<li>Wang, Q., et al. (2025). <em>NTPP: Generative Speech Language Modeling for Dual-Channel Spoken Dialogue via Next-Token-Pair Prediction.</em> <a href="https://arxiv.org/abs/2506.00975" target="_blank" rel="noopener noreferrer">arXiv:2506.00975</a>.</li>
</ol>
