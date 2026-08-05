---
title: "Human-like is genre-specific: a turn-taking analysis of voice AI vs. people"
date: "2026-06-25"
description: "Most of what gets called \"robotic\" in voice AI turns out to be genre-specific style, not a defect. The one real, directional problem, measured against human telephone chat, task dialogue, and casual conversation, is response speed (~4.5x slower), and fixing it is entangled with the one trait that already looks human."
tags:
  - "Voice AI"
  - "Turn-Taking"
  - "Conversational AI"
  - "Speech Perception"
image: "/images/posts/turn-taking-response-gap.png"
---

[![Cite this work](https://img.shields.io/badge/Cite-BibTeX-yellow.svg)](#citation)

Talk to a voice agent and you usually know within a sentence or two that it isn't a person. We reach for
vague words like "robotic" or "stilted" and move on. But conversational rhythm isn't vague. It's
measurable: how fast people answer, how often the floor changes hands, how long each turn runs, how much
speakers overlap, how they hand off without colliding. These are the mechanics of **turn-taking**, and
they are a large part of what makes a conversation feel alive.

I measured turn-taking across several kinds of conversation: human telephone chat, human task dialogue,
casual human conversation, and a **task voice-agent**'s calls. The useful finding is more specific than
"agents are slow." Exactly **one** thing is clearly, directionally worse: **response speed.** Almost
everything else people call "robotic" turns out to be either a *style* difference with no universal
"better," or a side-effect of that one timing problem. And the human "target" itself isn't a single
number; it depends on the medium and the genre. So "make it sound human" is the wrong instruction.
"Answer faster, and match the right human *style* for the job" is closer.

A note on scope up front: this is a **descriptive cross-section**, not a benchmark of the industry. The
agent figures come from one task domain (appointment-style calls); the human references are public and
proprietary corpora measured the same way. Treat it as a case study of a pattern, not a leaderboard.

## Turn-taking is a handful of measurable behaviors

A conversation's rhythm decomposes into a small set of quantities:

| metric | what it captures | does "more/less" mean "better"? |
|---|---|---|
| **response gap** | silence before the next person starts | **yes, faster is generally better** (to a point) |
| **floor-changes / min** | how often the turn passes between speakers | no, genre-dependent |
| **turn length** | how long turns run (monologuing) | no, genre/role-dependent |
| **talk balance** | how evenly the two share the floor | no, role-dependent |
| **overlap / talk-over** | how much both talk at once | no; *cooperative* overlap is positive, only floor-grabbing is disruptive |
| **backchannels** | "uh-huh / yeah" listening cues | no, "more ≠ better"; context-appropriate |

That third column matters, and it's where most "human-likeness" talk goes wrong. Only response speed has
a clean direction (faster responses reliably increase felt connection in human conversation
[Templeton et al., PNAS 2022](https://www.pnas.org/doi/10.1073/pnas.2116915119); modal human gap ≈ 200 ms,
[Stivers et al., 2009](https://www.pnas.org/doi/10.1073/pnas.0903616106)). The rest are *style* dimensions
whose "right" value depends on the kind of conversation. Overlap is the clearest case: sociolinguistics
distinguishes **cooperative overlap**, a marker of engagement and rapport in Deborah Tannen's
"high-involvement" style, from **competitive** floor-grabbing; overlap per se is not a defect.

The conversations: **Switchboard** (recorded telephone chit-chat) and **[SpokenWOZ](https://arxiv.org/abs/2305.13040)**
(a spoken task-oriented corpus: booking and information calls) as public human references, a set of
**casual online** human conversations, and a body of **task voice-agent** calls. I computed the metrics
above on each; this post is about what they say, not how they're computed.

## There is no single "human" baseline, and it's mostly about *medium*, not casual-vs-task

The first finding is about people. Two things vary the human "rhythm," and they're different things.

**Speed varies with the medium, not with casual-vs-task.** Telephone chit-chat (Switchboard) answers in
**~0.33 s** and task dialogue (SpokenWOZ) in **~0.40 s**, essentially the same, both near the ~0.2–0.4 s
human norm. The slow human number is **casual *online* conversation (~1.5 s)**, and that's a **mediation
effect**: network latency and missing cues stretch gaps. Face-to-face transitions average ~135 ms, but the
same speakers over video run ~440 ms, and remote responses approach a second
([Boland et al., "Zoom disrupts the rhythm of conversation"](https://www.researchgate.net/publication/356018506_Zoom_disrupts_the_rhythm_of_conversation)).
So "casual" is **not** slow; *mediated* is slow. (As a check on the measurement itself: scored as pooled
response latency, a stricter cut of the same gap, the Switchboard calls come out to ~0.22 s, close to the
canonical Switchboard floor-transfer-offset of ~187 ms, per
[Roberts, Torreira & Levinson, 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4429583/). That's evidence
the underlying measurement behaves.)

<img src="/images/posts/turn-taking-genre-grid.png" alt="Turn-taking by conversation type: response gap, floor-changes per minute, talk balance, and turn length across Switchboard, SpokenWOZ, casual online conversation, and the AI voice agent" />

**Style varies with genre.** What casual and task dialogue genuinely differ on is *how the floor is
shared*, not speed. Task dialogue is brisk and alternating: the floor changes ~13.7 times per minute of
speech, turns are short, participation is near-even (talk-balance 0.84), think "what date?", "the 15th,"
"morning or afternoon?" Casual conversation has longer turns, more overlap, and far more backchannels. On
backchannels specifically the literature gives a cited contrast, roughly **7/min in casual vs ~5.5/min in
task** dialogue (Danish corpus), and notes that *more isn't automatically better*; what matters is
context-appropriateness ("[the more does not necessarily mean the better](https://www.researchgate.net/publication/43968975)").
(I don't quote my own backchannel numbers: they aren't comparably measurable. Hand-transcribed corpora
capture "uh-huh"s; ASR-based ones drop most.)

The takeaway: there is no one "human" rhythm to imitate. Speed has a clear target (fast, ~0.3–0.5 s,
regardless of genre); *style* depends on whether the conversation is casual or transactional.

## The agent vs. human task dialogue

For a task agent, the right human comparison is human *task* dialogue (SpokenWOZ), not casual chat.

<img src="/images/posts/turn-taking-response-gap.png" alt="Median response gap before the other party answers: 0.33s for Switchboard telephone chat, 0.40s for SpokenWOZ task dialogue, 1.50s for casual online conversation, and 1.80s for the AI voice agent, about 4.5x slower than human task dialogue" />

**On speed, the gap is real and large: ~0.4 s for human task dialogue vs ~1.8 s for the agent, about 4.5×
slower.** And this is *not* a genre artifact: the agent is slower than human *casual telephone* (~0.33 s)
too. This is the one finding that survives every caveat below. It's genuine dead air while a cascaded
ASR→LLM→TTS stack thinks.

The other differences need more care, because (a) most of these metrics have no universal "better," and
(b) the comparison isn't as clean as "same genre." SpokenWOZ is *collaborative booking* (two people
trading questions, naturally symmetric); the agent's calls are *information delivery* (the agent conveys
details, the human confirms, naturally asymmetric). So part of any reciprocity / balance / turn-length gap
is **task type**, not human-vs-AI. With that caveat:

| metric (task setting) | human task (SpokenWOZ) | task voice agent | how to read it |
|---|---|---|---|
| response gap | 0.40 s | **1.80 s** | **~4.5× slower, a real, directional gap** |
| floor-changes / speech-min | 13.7 | 5.97 | less back-and-forth (but partly task-type) |
| talk balance | 0.84 | 0.31 | more one-sided (partly task-type, info delivery) |
| turn length (p90) | 8.3 s | 14.0 s | longer than *task* humans, though casual humans run longer still (telephone p90 ≈ 17 s) |
| talk-over (competitive overlap) | low | low | the agent does **not** talk over people |

Two things worth stating plainly. First, **"the agent monologues" is genre-relative.** Its turns are
longer than human *task* turns (14 s vs 8 s), but *shorter* than casual telephone turns (~17 s). People
monologue too, in the right setting. So this is "less like brisk task dialogue," not "uniquely verbose."

Second, the agent's low talk-over is **not necessarily a virtue to bank.** An agent that waits ~1.8 s
before speaking structurally *can't* overlap, so "it doesn't interrupt" may be a **by-product of being
slow**, not independent politeness. That matters for what comes next: making it faster could *erode* the
low-overlap behavior, so the two have to be tuned together.

## What's actually off, in three tiers

Decomposing turn-taking turns "it feels off" into something honest about direction:

| tier | axes | reading |
|---|---|---|
| **Clearly worse (directional)** | response speed | ~4.5× slower than humans; the one unambiguous problem. (Crispness of hand-offs is the same timing dimension, not a separate one.) |
| **Differs from human task dialogue, but genre/role-dependent, not "worse"** | reciprocity, turn length, talk balance | the agent is less brisk and more one-sided than human *task* dialogue, but these have no universal "better," and part of the gap is the agent's task being info-delivery |
| **At human level** | talk-over, false-starts | the agent doesn't grab the floor or cut into pauses, though some of this may follow from its slowness (above) |

So the genuinely solid, directional conclusion is **narrow and strong: the agent is much too slow.** The
"monologue/one-sided" differences are real but soft (genre- and task-confounded), and the good behaviors
are partly entangled with the slowness. This is a more useful map than "the agent is bad at conversation":
**fix speed first; treat the floor-sharing differences as genre-specific tuning, not universal defects.**

That's worth pausing on, because it cuts against the usual complaint about spoken dialogue systems, which
is the *opposite*: most are accused of interrupting too aggressively and rarely backchanneling
([Apple, ICLR 2025](https://arxiv.org/abs/2503.01174)). This one fails the other way, too slow rather than
too aggressive, which matters for what you'd actually fix first. Politeness turns out to be the easy,
well-tooled problem in this literature; speed is the hard part.

## Why speed is the one to fix first, but keep it in proportion

Response speed is more than a latency number; in human conversation it's a **social signal**. People
answer each other in ~200 ms, faster than deliberate thought, and faster responses make conversations feel
more connected; because replies under ~250 ms are too fast to consciously stage, timing acts as an
*honest* signal of engagement ([Templeton et al., PNAS 2022](https://www.pnas.org/doi/10.1073/pnas.2116915119)).
That's why a 4.5× gap matters even when nothing is "wrong" with the words.

But keep the magnitude honest:

<img src="/images/posts/turn-taking-perception-thresholds.png" alt="Response latency against human-perception thresholds: human task dialogue at 0.4s in the natural band, the AI voice agent at 1.8s in the noticeably-slower-but-still-tolerable band, well before the 2s unnatural line, the 3s satisfaction-drop line, and the 4s breakdown line" />

Classic interface thresholds put ~0.1 s as "instant," ~1 s as the limit for staying in flow
([Miller, 1968](https://dl.acm.org/doi/pdf/10.1145/1476589.1476628); Nielsen, 1994). Conversationally,
controlled studies find delays become clearly unnatural only past ~2 s, and user satisfaction in a
[human–robot study](https://ir.library.oregonstate.edu/downloads/h415pk244) held up until about **3 s**
(and depended on task complexity: simple questions soured sooner). The agent's ~1.8 s is therefore **~4.5×
the human rate but still short of the ~2–3 s tolerance cliffs**: the cost is a conversation that feels
*noticeably less crisp and less connected*, not one that's unusable. That's the right framing, meaningful
rather than catastrophic, and it's why responsiveness is the highest-leverage fix rather than an
emergency.

## The mechanism: anticipation, not waiting

Why are cascaded agents stuck around 1.5–2 s? Partly compute, but mostly **how they decide the caller is
done.** Most systems wait for a fixed silence, typically 0.5–1 s, then begin. Humans don't wait; they
**anticipate**, reading prosody, syntax, and meaning to project where a turn will end, often planning
their reply before the speaker finishes ([Gravano & Hirschberg, turn-taking cues in task-oriented dialogue](https://www.sciencedirect.com/science/article/abs/pii/S0885230810000690)).

And silence is a coarse signal. In real conversation most speaker transitions are separated by very short
intervals, the majority under half a second
([Heldner & Edlund, 2010](https://staff.fnwi.uva.nl/r.fernandezrovira/teaching/cosp/cosp2016/docs/HeldnerEdlund2010.pdf)).
A detector that waits ~500–1000 ms of silence to declare "your turn" is, by construction, slower than the
human rhythm and blind to the moments humans actually use to hand off. The lever for the speed gap isn't a
faster LLM; it's **predicting turn-ends instead of waiting for silence to prove them**. Full-duplex models
like [Moshi](https://arxiv.org/abs/2410.00037) (≈200 ms) show how far the other end of that design space
reaches.

## So how do you make a task voice agent more human?

Match the human profile *for the job*, and be honest about which targets are firm and which are
genre-specific:

| axis | human task reference | target for a task agent | firmness |
|---|---|---|---|
| response gap | ~0.4 s | **~0.8 s** (cascaded can't hit human ~0.4 s, but well-tuned cascaded systems land in the 0.7–1.0 s band, per [Patamia et al., 2025](https://www.mdpi.com/2227-7080/13/12/591)) | **firm, directional** |
| turn length | ~8 s | shorter, broken-up reads | genre-specific (toward task style) |
| reciprocity | ~13.7/min | more hand-backs | genre-specific (toward task style) |
| talk balance | ~0.84 | more even | genre/role-specific; adjust for your task's asymmetry |
| talk-over / false-starts | low | keep low, but watch it as you speed up | guardrail (entangled with speed) |

Three notes. **(1) The firm target is speed**: anticipatory endpointing so the agent can begin near the
half-second humans expect; everything else is secondary. **(2) The floor-sharing targets are
genre-specific**, not universal maxima: aim toward the *task* style (shorter turns, more hand-backs),
adjusted for how asymmetric your task genuinely is. A status-delivery call is *meant* to be more one-sided
than a collaborative booking. **(3) Speed and restraint are coupled**: the agent's good low-overlap
behavior may partly come from waiting, so as you cut latency, watch that it doesn't start talking over
people. A casual *companion* agent would target the opposite on overlap and backchannels, which is the
whole point: **"human-like" is not one setting.**

## Caveats

This is a **descriptive cross-section**, not a causal or industry-wide claim: the agent data is one task
domain (one system), the casual-online set is a single corpus (its ~1.5 s is at the high end of even the
online-conversation literature), and I lean on published perception research, not my own A/B outcomes, to
argue that speed matters. The "genre-matched" comparison still mixes two task *types* (collaborative
booking vs information delivery), so the floor-sharing gaps are partly task structure. And most metrics
have **no universal "better" direction**; they're genre/role-specific, which is the post's whole point. I
only claim a direction for response speed. Backchannel rates aren't comparably measurable across these
corpora (transcription captures them unevenly), so I cite published rates rather than my own.

## Takeaway

Most of what gets called "robotic" turns out not to be a defect at all. **Monologuing, one-sidedness, low
back-and-forth: these are genre and role differences with no universal "better,"** and part of the gap
here is simply that information delivery is naturally more one-sided than the collaborative booking calls
it's being measured against. The one real, directional problem is response speed: **about 4.5× slower
than people, in both casual and task conversation**, and that gap alone explains most of the "robotic"
feeling. It doesn't come free to fix, though. The agent's one clearly human-like trait, not talking over
people, may just be a side effect of waiting so long to respond, so closing the latency gap without
watching that trade-off risks curing one problem by creating another. Human-likeness isn't a single
naturalness dial to turn up. It's answering in roughly the half-second humans expect, matching the
conversational *style* the job actually calls for, and keeping the restraint that speed was quietly buying
it.

## Citation

If you use this work, please cite:

```bibtex
@misc{rajaa2026turntaking,
  author       = {Rajaa, Shangeth},
  title        = {{Human-like is genre-specific: a turn-taking analysis of voice AI vs. people}},
  year         = {2026},
  howpublished = {\url{https://shangeth.com/posts/turn-taking-human-vs-ai/}},
  note         = {Blog post}
}
```

## References

<ol class="references">
<li>Arora, S., Lu, Z., Chiu, C.-C., Pang, R., &amp; Watanabe, S. (2025). <em>Talking Turns: Benchmarking Audio Foundation Models on Turn-Taking Dynamics.</em> ICLR 2025. <a href="https://arxiv.org/abs/2503.01174" target="_blank" rel="noopener noreferrer">arXiv:2503.01174</a>.</li>
<li>Boland, J. E., Fonseca, P., Mermelstein, I., &amp; Williamson, M. (2022). <em>Zoom disrupts the rhythm of conversation.</em> Journal of Experimental Psychology: General, 151(6), 1272-1282. <a href="https://pubmed.ncbi.nlm.nih.gov/34748361/" target="_blank" rel="noopener noreferrer">pubmed.ncbi.nlm.nih.gov</a>.</li>
<li>Défossez, A., et al. (2024). <em>Moshi: A Speech-Text Foundation Model for Real-Time Dialogue.</em> <a href="https://arxiv.org/abs/2410.00037" target="_blank" rel="noopener noreferrer">arXiv:2410.00037</a>.</li>
<li>Godfrey, J. J., Holliman, E. C., &amp; McDaniel, J. (1992). <em>Switchboard: Telephone Speech Corpus for Research and Development.</em> ICASSP 1992, 517-520.</li>
<li>Gravano, A., &amp; Hirschberg, J. (2011). <em>Turn-taking cues in task-oriented dialogue.</em> Computer Speech &amp; Language, 25(3), 601-634. <a href="https://www.sciencedirect.com/science/article/abs/pii/S0885230810000690" target="_blank" rel="noopener noreferrer">sciencedirect.com</a>.</li>
<li>Heldner, M., &amp; Edlund, J. (2010). <em>Pauses, gaps and overlaps in conversations.</em> Journal of Phonetics, 38(4), 555-568. <a href="https://staff.fnwi.uva.nl/r.fernandezrovira/teaching/cosp/cosp2016/docs/HeldnerEdlund2010.pdf" target="_blank" rel="noopener noreferrer">PDF</a>.</li>
<li>Li, H. Z., Cui, Y., &amp; Wang, Z. (2010). <em>Backchannel Responses and Enjoyment of the Conversation: The More Does Not Necessarily Mean the Better.</em> International Journal of Psychological Studies, 2(1), 25-34. <a href="https://www.researchgate.net/publication/43968975" target="_blank" rel="noopener noreferrer">researchgate.net</a>.</li>
<li>Maslych, M., Katebi, M., Lee, C., Hmaiti, Y., Ghasemaghaei, A., Pumarada, C., Palmer, J., Segarra Martinez, E., Emporio, M., Snipes, W., McMahan, R. P., &amp; LaViola Jr., J. J. (2025). <em>Mitigating Response Delays in Free-Form Conversations with LLM-powered Intelligent Virtual Agents.</em> CUI '25. <a href="https://arxiv.org/abs/2507.22352" target="_blank" rel="noopener noreferrer">arXiv:2507.22352</a>.</li>
<li>Miller, M. R. (2025). <em>Timing Matters: Effects of Response Delay on Perceived Naturalness in Robot Conversations</em> (Master's project, Oregon State University). <a href="https://ir.library.oregonstate.edu/downloads/h415pk244" target="_blank" rel="noopener noreferrer">ir.library.oregonstate.edu</a>.</li>
<li>Miller, R. B. (1968). <em>Response time in man-computer conversational transactions.</em> AFIPS '68 (Fall, part I), 267-277. <a href="https://dl.acm.org/doi/pdf/10.1145/1476589.1476628" target="_blank" rel="noopener noreferrer">dl.acm.org</a>.</li>
<li>Nielsen, J. (1994). <em>Usability Engineering.</em> Morgan Kaufmann Publishers.</li>
<li>Patamia, R. A., Dinh, H. P. T., Liu, M., &amp; Cosgun, A. (2025). <em>Turn-Taking Modelling in Conversational Systems: A Review of Recent Advances.</em> Technologies, 13(12), 591. <a href="https://www.mdpi.com/2227-7080/13/12/591" target="_blank" rel="noopener noreferrer">mdpi.com</a>.</li>
<li>Roberts, S. G., Torreira, F., &amp; Levinson, S. C. (2015). <em>The effects of processing and sequence organization on the timing of turn-taking: a corpus study.</em> Frontiers in Psychology, 6, 509. <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC4429583/" target="_blank" rel="noopener noreferrer">pmc.ncbi.nlm.nih.gov</a>.</li>
<li>Si, S., Ma, W., Gao, H., Wu, Y., Lin, T.-E., Dai, Y., Li, H., Yan, R., Huang, F., &amp; Li, Y. (2023). <em>SpokenWOZ: A Large-Scale Speech-Text Benchmark for Spoken Task-Oriented Dialogue Agents.</em> NeurIPS 2023. <a href="https://arxiv.org/abs/2305.13040" target="_blank" rel="noopener noreferrer">arXiv:2305.13040</a>.</li>
<li>Stivers, T., et al. (2009). <em>Universals and cultural variation in turn-taking in conversation.</em> PNAS, 106(26), 10587-10592. <a href="https://www.pnas.org/doi/10.1073/pnas.0903616106" target="_blank" rel="noopener noreferrer">pnas.org</a>.</li>
<li>Tan, F. F.-Y., Messerschmidt, M. A., Yin, W., &amp; Nov, O. (2026). <em>The Impact of Response Latency and Task Type on Human-LLM Interaction and Perception.</em> <a href="https://arxiv.org/abs/2604.06183" target="_blank" rel="noopener noreferrer">arXiv:2604.06183</a>.</li>
<li>Tannen, D. (1984). <em>Conversational Style: Analyzing Talk Among Friends.</em> Ablex Publishing.</li>
<li>Templeton, E. M., et al. (2022). <em>Fast response times signal social connection in conversation.</em> PNAS, 119(4), e2116915119. <a href="https://www.pnas.org/doi/10.1073/pnas.2116915119" target="_blank" rel="noopener noreferrer">pnas.org</a>.</li>
</ol>
