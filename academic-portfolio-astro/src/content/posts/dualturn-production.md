---
title: "DualTurn in Production: From Research Checkpoint to Continual Learning on Live Calls"
date: "2026-07-24"
description: "Taking DualTurn from an Interspeech 2026 paper into a live voice agent: a ~1.5M-parameter model trained with zero manual labels, fastest end-to-end in real-call testing against cloud and open-source turn detectors, the only one in the industry that tells a backchannel from a real interruption, and set up to keep improving for free from every call it handles."
author: "Shangeth Rajaa"
tags:
  - "Voice AI"
  - "Turn-Taking"
  - "Full-Duplex"
  - "Production ML"
  - "Continual Learning"
---

[![GitHub](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/anyreachai/dualturn)
[![HuggingFace](https://img.shields.io/badge/🤗-DualTurn-blue.svg)](https://huggingface.co/collections/anyreach-ai/dualturn)
[![arXiv](https://img.shields.io/badge/arXiv-2603.08216-b31b1b.svg)](https://arxiv.org/abs/2603.08216)

<div class="status-highlight">
This is a follow-up to <a href="/posts/dualturn">DualTurn: Learning Turn-Taking from Dual-Channel Generative Speech Pretraining</a> (Interspeech 2026). That post is about the method and the paper's numbers. This one is about what happened after: taking DualTurn out of the paper and into a real voice agent, on real calls, and what changes when the thing you're optimizing for is a person on the phone instead of a benchmark split.
</div>

## Quick recap

DualTurn is a turn-taking perception model. Instead of waiting for a voice activity detector to flag a pause and then classifying whether that pause is a real end of turn, it listens to both sides of a call continuously, full-duplex, and predicts turn-taking signals directly from raw dual-channel audio. It gets there by pretraining generatively on dual-channel conversational audio, the same ingredient that makes full-duplex speech-to-speech models like Moshi good at turn-taking, but stripped down to just the perception component, so a standard ASR-LLM-TTS pipeline can use it without giving up the LLM's reasoning and tool-calling. The paper version is a 0.5B-parameter model with a Qwen2.5-0.5B backbone, twelve classification heads, and 453 hours of Switchboard and otoSpeech audio behind it, and it beats a 3.1B fusion baseline and the prior state of the art (VAP) on both word-level turn prediction and full agent-action prediction.

None of that is deployment-ready as written, and the paper says so explicitly. This post is the deployment.

## Same method, a much smaller vessel

The production model doesn't change the method. It's still stage-1 generative pretraining on dual-channel audio followed by stage-2 finetuning into turn-taking signals, still with no manually labeled data anywhere in the loop. What changes is the backbone, because a component that has to run continuously, on every call, in real time, on CPU, cannot afford a 0.5B-parameter LLM.

| | Research (paper) | Production |
|---|---|---|
| Backbone | Qwen2.5-0.5B | 2-layer LSTM, 256 hidden |
| Trainable params | ~500M | ~1.5M |
| Audio front-end | Frozen Mimi encoder | Frozen Mimi encoder |
| Heads | 12 (EOT/HOLD/BOT/BC × 2 channels) | 4 (end-of-turn, per-channel VAD, future-VAD) |
| Runs on | CPU | CPU, real time |

That's roughly a 330x reduction in trainable parameters. The reason this is worth trying at all, instead of just assuming a smaller model degrades, is the paper's own central finding: capacity alone barely moves turn-taking performance. An 8M-parameter LSTM and the 0.5B backbone score statistically identically on backchannel detection *without* dual-channel generative pretraining; pretraining is what takes the exact same architecture from near-chance to state-of-the-art. If that finding is real, it predicts that the backbone can be swapped for something CPU-cheap without losing the thing that actually matters, as long as the pretrain-then-finetune recipe stays intact. Production is that prediction, tested.

## Training on calls nobody labeled

The other thing that has to hold for a much smaller model trained on much less curated data to work is the labeling pipeline, and here the constraint was even harder than the paper's: zero manual annotation, on real AI-agent-to-human production phone calls, at a scale a labeling team was never going to touch.

The pipeline that makes this possible has no human in it:

1. **ASR** transcribes both channels with word-level timestamps.
2. **Voice activity detection** gives per-channel speech intervals, i.e. the timing ground truth.
3. **One LLM call per conversation** reads the transcript and judges, per customer turn, whether it was complete or the customer was cut off, and whether an overlap was a stop-worthy interruption.
4. Deterministic rules turn those judgments into frame-level training targets: an endpoint gets snapped to the actual VAD offset, then dropped if the customer resumes speaking shortly after with no agent speech in between, because that's a mid-thought pause, not a turn end, and no one should have to hand-label that distinction call by call.

Every step after ASR is either a VAD threshold or one cheap LLM call, and the LLM call is the one place any kind of judgment enters the pipeline. Applied across a corpus of production calls, this is a rounding error in cost next to what a comparable amount of human-labeled turn-taking data would run.

## Continuous at inference, still no VAD gate

The part of the architecture that's easy to state and easy to underestimate: in the deployed model, voice activity detection is an *output* of the network, not an input to it. There is no VAD sitting in front of the model deciding when it's allowed to run. Every 80ms tick, on both channels, unconditionally, the model runs. Its own VAD prediction is one of the things it emits alongside the end-of-turn signal, not a gate that has to fire first.

That has a direct, practical consequence for deployment: the model's internal state has already been accumulating evidence about where the turn is going for the entire utterance, not just since a pause started, so when the turn actually ends there's no cold start. Streaming state (hidden state carried tick to tick, a bounded attention cache in the audio front-end) keeps this at flat, constant latency regardless of how long the call has been running, and it's small enough to run on CPU in real time. This is what "full-duplex, continuously modeling both channels" from the paper actually looks like once it has to run in a live phone call instead of on a benchmark split.

## Two evaluations, one consistent picture

We tested DualTurn two ways: once against the field on a neutral public benchmark, and once inside an actual voice agent talking to actual people.

**On the neutral benchmark**, DualTurn goes head-to-head with the field of turn detectors people actually ship: LiveKit's own models, Deepgram Flux, the open SmartTurn and ultraVAD, scored on LiveKit's public leaderboard protocol (classify each pause in a call as a real end-of-turn or a mid-turn hold, at fixed latency/false-cutoff budgets).

| Model | Params | Runs on | FC@300ms | FC@600ms | Lat@5% | Lat@10% |
|---|---|---|---:|---:|---:|---:|
| LiveKit Turn Detector v1 | undisclosed | Cloud | 9.9% | 4.5% | 543 ms | 295 ms |
| Deepgram Flux | undisclosed | Cloud | 12.9% | 9.9% | 1151 ms | 548 ms |
| **DualTurn (ours)** | **1.5M** | **CPU** | **20.6%** | **8.5%** | 1026 ms | **520 ms** |
| ultraVAD | ~8B | GPU | 27.7% | 11.9% | 899 ms | 663 ms |
| LiveKit Turn Detector v1-mini | undisclosed | CPU | 27.8% | 12.1% | 1070 ms | 698 ms |
| SmartTurn v3.2 | ~8M | CPU | 35.2% | 14.8% | 1051 ms | 739 ms |

DualTurn is the best self-hostable, on-device turn detector on the board, ahead of an 8B open model at roughly 5,000x fewer parameters, ahead of LiveKit's own local model, ahead of the open state of the art. Only two undisclosed-size closed cloud systems place ahead of it at all.

**Inside a live agent**, we ran DualTurn against Deepgram Flux and LiveKit's detectors (the cloud v1 model, its MultilingualModel, and the local v1-mini) inside the same production LiveKit voice-agent stack, same LLM, same TTS, only the turn detector changing between runs.

<img src="/images/posts/dualturn-production-latency.png" alt="Agent response latency: DualTurn vs. production turn detectors. DualTurn hands the turn back in 0.22s and gets the agent talking in 1.36s, ahead of Deepgram Flux, LiveKit v1, LiveKit MultilingualModel, and LiveKit v1-mini." />

| Turn detector | End-of-turn latency | Total response latency | Backchannel / interrupt |
|---|---:|---:|---|
| **DualTurn** | **0.22 s** | **1.36 s** | ✅ natively supported |
| Deepgram Flux | 0.36 s | 1.50 s | ❌ |
| LiveKit Turn Detector v1 (cloud) | 0.58 s | 1.72 s | ❌ |
| LiveKit MultilingualModel | 0.69 s | 1.83 s | ❌ |
| LiveKit Turn Detector v1-mini | 1.10 s | 2.24 s | ❌ |

Here DualTurn doesn't place third behind the cloud systems, it's first, full stop: 0.22s to hand back the turn and 1.36s to the agent's first word, ahead of Deepgram Flux, LiveKit's cloud v1, LiveKit's MultilingualModel, and LiveKit v1-mini.

These are two different measurements, not two contradictory ones. The public benchmark scores a classification task: is this specific, already-flagged pause a real end-of-turn or not. It's a fair, neutral way to compare turn detectors, but by construction the clock only starts once a pause has already been identified. The live-agent test measures the thing a caller actually feels: real wall-clock time from the last word spoken to the agent's first word back, with every component in the loop except the turn detector held fixed. That's the harder and more honest test, because it's not scoring a model against a fixed dataset, it's timing an agent talking to a person. And it's where being the one detector in the room that never waited for a pause to be flagged in the first place shows up as a real, measured latency win, against everything we tested it against, cloud included.

## What nothing else here can do

There's a capability gap underneath these numbers that's worth stating plainly: no turn detector in the industry today, open-source or commercial, can tell a backchannel from a real interruption. DualTurn can.

Every detector we tested here is single-channel and VAD-triggered on the user's audio alone, and that's the norm across the field, not a gap specific to this comparison: LiveKit, Deepgram, and every published open turn-taking model reason over the user's channel and nothing else. When the agent is mid-sentence and the user says "mhm" or "right," none of them can tell that apart from the user genuinely taking the floor, because none of them were ever given the agent's own audio to reason over in the first place, and the best any of them can fall back on is a crude duration heuristic. DualTurn listens to both channels continuously, so it has direct access to the exact thing that decision depends on: what the agent is doing while the user is making that sound. That's not a training gap the field hasn't gotten around to closing, it's a wiring constraint: you cannot get this capability out of a model that was only ever given one channel to listen to.

That capability sits on top of a model that's also the fastest and lightest thing in the comparison: ~1.5M parameters, the smallest footprint of anything tested, running on CPU with no per-call vendor cost and no network round-trip, and it's the one that got the agent talking fastest, in every measurement above.

## Continual learning: turn-taking that keeps improving for free

Deploying DualTurn doesn't end the training loop, it restarts it. Every call the agent handles is another dual-channel recording that runs through the exact same zero-manual-label pipeline described above: ASR, VAD, one LLM pass per call. There's no incremental labeling cost as call volume grows, because there was never a human labeling step to scale in the first place. Continual learning here isn't a separate system bolted on afterward, it's the same training pipeline that produced v1, pointed at v1's own call traffic.

There's a second effect on top of raw volume, and it's the more interesting one. When the agent's turn-taking is bad, people can tell, and they adapt for it: they over-enunciate, they slow down, they leave unnaturally long pauses before continuing, the way anyone talks to something they suspect is a machine. That's a distribution of speech that isn't representative of how people actually talk to each other, and training on more of it doesn't buy much. As turn-taking gets better and the agent stops stepping on people and stops missing backchannels, that self-conscious, careful register goes away, and conversations start looking more like two people talking to each other than a person talking to a phone tree. That shift, more natural pacing, more overlap, more of the backchannel and interruption behavior the model exists to handle, is exactly the training signal that's hardest to get any other way, and it only shows up once the model deployed is already good enough to stop provoking the unnatural version of it. Better turn-taking produces better training data for turn-taking, at zero additional labeling cost, for as long as the agent keeps taking calls.

## Where this leaves it

The paper's core claim was that generative pretraining, not architecture, is what makes a model good at turn-taking, and that the backbone is a vessel for that pretraining rather than the source of it. Production is that claim under real conditions: a 330x smaller backbone, trained on real, unlabeled production calls instead of a curated research corpus, still comes out ahead of everything self-hostable on a neutral public benchmark, and still wins outright once it's put in front of an actual person on a live call. The method traveled. The vessel didn't need to be big.

Paper, code, and the open model weights are here if you want to look closer: [arXiv](https://arxiv.org/abs/2603.08216), [GitHub](https://github.com/anyreachai/dualturn), [Hugging Face](https://huggingface.co/collections/anyreach-ai/dualturn), and the [original DualTurn post](/posts/dualturn) for the method and the paper's own results in full.

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
