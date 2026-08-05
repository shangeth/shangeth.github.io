---
title: "Comparing Voice Agents Against the Right Human: A Genre-Matched Turn-Taking Study"
date: "2026-06-25"
description: "The checklist for making a voice agent \"sound human\": less interrupting, more backchanneling, an even exchange, mostly targets things that aren't broken. Measured against the right human (task dialogue, not casual chat), almost everything holds up except response speed, and fixing that carelessly breaks the one thing the agent already gets right."
tags:
  - "Voice AI"
  - "Turn-Taking"
  - "Conversational AI"
  - "Speech Perception"
image: "/images/posts/turn-taking-response-gap.png"
---

[![Cite this work](https://img.shields.io/badge/Cite-BibTeX-yellow.svg)](#citation)

## You already talk two different ways

Call your best friend and ask what they're doing this weekend. Then call a bank to schedule an
appointment. You won't run either conversation the same way, and you already know that without thinking
about it.

With your friend, you trade long stories, jump in mid-sentence to react, say "yeah, totally" a dozen
times just to keep things moving, and let whoever's talking hold the floor for a while. With the bank,
you get straight to it: they ask, you answer, you ask, they answer, quick and alternating, nobody talking
over anybody, and the call ends the moment the task is done. Neither one sounds robotic. They're just
different jobs.

That's not just an impression, it shows up in measurement. Switchboard, a large corpus of recorded
**casual** telephone conversation, and SpokenWOZ, a corpus of real human-to-human **task-oriented** calls
(booking, scheduling, information requests), let you compare the two registers directly: same kind of
measurement, both real people, only the genre of the conversation changes.

| axis | casual conversation (Switchboard) | task dialogue (SpokenWOZ) |
|---|---|---|
| response gap | 0.33 s | 0.40 s |
| floor-changes / min | 7.0 | 13.7 |
| turn length (p90) | 17.3 s | 8.3 s |
| overlap | 6.0% | 1.7% |
| interruptions / min | 0.96 | 0.38 |
| talk balance | 0.73 | 0.84 |

Response speed barely moves. People answer in about a third of a second in casual conversation and about
four-tenths of a second on a task call, both the same kind of fast, well within the normal human range
(response times around 200 milliseconds are typical across languages generally
[Stivers et al., 2009](https://www.pnas.org/doi/10.1073/pnas.0903616106)). Speed isn't what tells these
two conversations apart.

What tells them apart is the shape of the turn-taking. Task calls change speakers about twice as often
per minute, run much shorter turns, and involve far less overlap and far fewer interruptions. Even the
most "obviously humanizing" behavior doesn't work the way you'd expect: task calls are actually the more
evenly balanced ones, not the casual conversations, probably because a task call is naturally a
back-and-forth exchange, while casual conversation gives more room for one person to hold the floor
telling a story.

None of this makes the bank call less human. It's a different, and equally legitimate, way of talking.
Which is the whole point: "natural" was never one thing, even for people. It's a fit between the
conversation's job and its shape. So before asking why a voice agent feels robotic, it's worth asking a
more basic question first: robotic compared to which kind of human, the one on a casual call, or the one
doing the task?

## The wrong bar for a voice agent

So which register should a voice agent be measured against? If it's booking an appointment or walking
through an account update, its job matches the task call, not the catch-up-with-a-friend call. That
sounds obvious once you say it. But it's not how "make it sound human" usually gets applied. The instinct
when an agent feels off is to import the traits of casual conversation: never interrupt, backchannel
constantly, keep the exchange perfectly even, don't let either side run long. Those are good instincts
for a friend. They're the wrong bar for an agent doing a bank's job, and grading against the wrong bar is
a mistake made before anything has even been measured.

The comparison that actually matters is a task voice agent against human task dialogue, SpokenWOZ, not
Switchboard: real people doing the same kind of job the agent is doing. That's the bar the rest of this
piece uses, measured the same way across four sets of calls, two public human corpora, a set of casual
online human conversation, and a body of real task voice-agent calls (appointment-style: scheduling,
information delivery), so every number below is comparable to every other. Worth being precise about
scope, too: this argument is about task agents specifically. A casual, companion-style voice agent should
probably be graded closer to Switchboard than SpokenWOZ. The lesson doesn't transfer directly to that
case.

## The one real gap: speed

Here's where the genre-matched comparison earns its keep. Human task dialogue answers in about 0.40
seconds. The task voice agent answers in about 1.80 seconds. That's roughly 4.5 times slower, and it's
the one place in this whole comparison where the direction is unambiguous: faster is better, no genre
caveat required.

<img src="/images/posts/turn-taking-response-gap.png" alt="Median response gap before the other party answers: 0.33s for casual conversation (Switchboard), 0.40s for task dialogue (SpokenWOZ), 1.50s for casual online conversation, and 1.80s for the AI voice agent, about 4.5x slower than human task dialogue" />

It's tempting to wonder whether this is a byproduct of task calls being quick, brisk exchanges rather
than an actual defect. It isn't: the agent is slower than casual conversation too (0.33 seconds), which
rules out "the bar itself is just fast" as an excuse. However the human side is measured, the agent is
answering several times slower than a person would. (There's a separate human dataset of casual
conversation happening online, over video or a similar remote medium, that clocks in slower still, around
1.5 seconds, but that's a mediation effect, network lag and missing visual cues stretch out response
gaps, not evidence that casual talk is naturally slow
[Boland et al., "Zoom disrupts the rhythm of conversation"](https://pubmed.ncbi.nlm.nih.gov/34748361/).
Take the call away from a screen and the numbers above are what you get.)

This is genuine dead air, the gap while a speech-recognition-to-language-model-to-speech-synthesis
pipeline processes and generates a reply before the agent can start talking.

Response speed isn't just a technical latency number, either. In human conversation it acts as a social
signal: people answer each other in around 200 milliseconds, too fast to be consciously planned, and
controlled experiments show that shortening or lengthening that gap directly changes how connected two
people feel, even with nothing else about the conversation different
[Templeton et al., PNAS 2022](https://www.pnas.org/doi/10.1073/pnas.2116915119). So a 4.5x gap isn't a
rounding error. It's also not catastrophic, though: the tolerance research puts a genuinely broken
conversation somewhere past two to three seconds of delay
[Maslych et al., 2025](https://arxiv.org/abs/2507.22352), and in a controlled
[human-robot study](https://ir.library.oregonstate.edu/downloads/h415pk244), satisfaction held up until
about three seconds. At 1.8 seconds, the agent sits just under that line. The honest way to size it is
"noticeably less crisp and connected," not "unusable."

## Checked against the right bar, the rest holds up

With the bar set correctly, what about everything else on the usual checklist: turn length, how evenly
the floor is shared, how often it changes hands, whether the agent talks over people?

<img src="/images/posts/turn-taking-genre-grid.png" alt="Turn-taking by conversation type: response gap, floor-changes per minute, talk balance, and turn length across casual conversation, task dialogue, casual online conversation, and the AI voice agent" />

| metric (task setting) | human task dialogue | task voice agent |
|---|---|---|
| floor-changes / min | 13.7 | 5.97 |
| talk balance | 0.84 | 0.31 |
| turn length (p90) | 8.3 s | 14.0 s |
| talk-over (competitive overlap) | low | low |

On paper this looks like a list of problems: the agent changes speakers less than half as often, is far
more one-sided, and runs longer turns. Look closer, though, and most of it is explained by the job, not
by the agent being bad at conversation. The human task calls in SpokenWOZ are collaborative booking, two
people trading questions back and forth, naturally symmetric. The agent's calls are information delivery,
the agent conveys details and the caller mostly confirms, naturally asymmetric. Some of that one-sidedness
would show up in a human doing the exact same job. The turn-length gap tells a similar story from another
angle: the agent's turns are longer than task-dialogue turns (14 seconds vs. 8), but still shorter than
the turns people run in casual conversation (past 17 seconds). It isn't "the agent monologues," it's "the
agent is less brisk than a booking call," a real claim, but a much smaller and less alarming one.

And on the axis that sounds most like textbook politeness, not talking over people, the agent is already
at the human level. That's worth pausing on, because it cuts against the usual complaint about spoken
dialogue systems. The published research on voice AI turn-taking generally reports systems that interrupt
too aggressively and rarely backchannel
([Apple, ICLR 2025](https://arxiv.org/abs/2503.01174)). This one is too passive, not too aggressive.
Whatever's actually wrong here, it isn't the thing most turn-taking research is built to catch.

## Why the fix isn't free

That last point comes with a catch, and it's the reason this isn't just a "reduce latency" problem you
can hand off and forget about.

An agent that takes 1.8 seconds to respond cannot, by definition, talk over the person it's listening to;
it hasn't started generating a reply yet. So the fact that this agent doesn't interrupt people may not be
independent good behavior. It may simply be a side effect of being slow. That matters for what happens
next: shrink the latency without addressing anything else, and the by-product goes away with it. The
safety margin that "not being ready to speak yet" was quietly providing disappears, and there's a real
risk of trading a well-understood, well-diagnosed problem, the agent is too slow, for a less obvious one,
the agent now talks over people, which is arguably worse, because it's the harder failure to catch and
the exact failure most of the existing turn-taking literature is already warning about.

Speed and restraint, in other words, aren't two separate dials here. They're coupled, at least in a
system that currently gets its politeness for free by being slow. Fixing one without a plan for the other
doesn't fix the conversation. It just moves the defect somewhere less obvious.

## What actually closes the gap

Why are cascaded agents stuck around 1.5 to 2 seconds in the first place? Partly compute, but mostly how
they decide the caller is finished talking. Most systems wait for a fixed length of silence, typically
half a second to a full second, and only then start generating a reply. Humans don't do this. They
anticipate, reading prosody, syntax, and meaning to project where a turn is headed, often planning a
reply before the other person has finished speaking
([Gravano & Hirschberg, 2011](https://www.sciencedirect.com/science/article/abs/pii/S0885230810000690)).
Silence is also a coarse signal to wait on: most real speaker transitions are separated by very short
gaps, the majority under half a second
([Heldner & Edlund, 2010](https://staff.fnwi.uva.nl/r.fernandezrovira/teaching/cosp/cosp2016/docs/HeldnerEdlund2010.pdf)),
so a detector waiting on 500 to 1000 milliseconds of silence is, by construction, slower than the rhythm
it's trying to match, and blind to the exact moments humans use to hand off. The lever here isn't a
faster language model. It's predicting the end of a turn instead of waiting for silence to prove it
happened: full-duplex models like [Moshi](https://arxiv.org/abs/2410.00037), built around exactly that,
get down to around 200 milliseconds.

Put together, the targets for a task voice agent look genre-specific rather than universal:

| axis | human task reference | target | why |
|---|---|---|---|
| response gap | ~0.4 s | ~0.8 s | the firm target; ~0.4 s isn't realistic yet for a cascaded pipeline, but well-tuned cascaded systems land in the 0.7-1.0 s band [Patamia et al., 2025](https://www.mdpi.com/2227-7080/13/12/591) |
| turn length, reciprocity, balance | matches SpokenWOZ | move toward task style, adjusted for the task's asymmetry | genre-specific, not a universal maximum |
| talk-over | low | keep low while speed improves | the guardrail from the section above |

Only the first row is a hard target with a real direction. The rest are "move toward the human task
profile, adjusted for how asymmetric the job actually is," not a checklist to max out.

## Back to the two phone calls

A few honest limits first: this is a descriptive comparison, not a controlled experiment; the agent data
comes from one task domain; and most of the perception research above is published literature, not a
test run for this piece specifically.

Now go back to the two calls this piece opened with. Nobody expects the bank call to sound like the one
with your friend, and grading it that way would be a mistake before the data even comes out. The same
mistake is easy to make with a voice agent: reach for the traits of casual conversation, more
backchanneling, less monologuing, a perfectly even exchange, as the definition of "sounds human," when
the right comparison is how a person would run the exact same task-oriented call. Measured against that
bar, almost everything on the usual checklist is fine. The one real gap is response speed, and closing it
only helps if the fix also watches the thing it's currently buying for free: an agent that doesn't talk
over people. Fix the actual problem, and don't lose the one thing already working while doing it.

## Citation

If you use this work, please cite:

```bibtex
@misc{rajaa2026turntaking,
  author       = {Rajaa, Shangeth},
  title        = {{Comparing Voice Agents Against the Right Human: A Genre-Matched Turn-Taking Study}},
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
<li>Maslych, M., Katebi, M., Lee, C., Hmaiti, Y., Ghasemaghaei, A., Pumarada, C., Palmer, J., Segarra Martinez, E., Emporio, M., Snipes, W., McMahan, R. P., &amp; LaViola Jr., J. J. (2025). <em>Mitigating Response Delays in Free-Form Conversations with LLM-powered Intelligent Virtual Agents.</em> CUI '25. <a href="https://arxiv.org/abs/2507.22352" target="_blank" rel="noopener noreferrer">arXiv:2507.22352</a>.</li>
<li>Miller, M. R. (2025). <em>Timing Matters: Effects of Response Delay on Perceived Naturalness in Robot Conversations</em> (Master's project, Oregon State University). <a href="https://ir.library.oregonstate.edu/downloads/h415pk244" target="_blank" rel="noopener noreferrer">ir.library.oregonstate.edu</a>.</li>
<li>Patamia, R. A., Dinh, H. P. T., Liu, M., &amp; Cosgun, A. (2025). <em>Turn-Taking Modelling in Conversational Systems: A Review of Recent Advances.</em> Technologies, 13(12), 591. <a href="https://www.mdpi.com/2227-7080/13/12/591" target="_blank" rel="noopener noreferrer">mdpi.com</a>.</li>
<li>Si, S., Ma, W., Gao, H., Wu, Y., Lin, T.-E., Dai, Y., Li, H., Yan, R., Huang, F., &amp; Li, Y. (2023). <em>SpokenWOZ: A Large-Scale Speech-Text Benchmark for Spoken Task-Oriented Dialogue Agents.</em> NeurIPS 2023. <a href="https://arxiv.org/abs/2305.13040" target="_blank" rel="noopener noreferrer">arXiv:2305.13040</a>.</li>
<li>Stivers, T., et al. (2009). <em>Universals and cultural variation in turn-taking in conversation.</em> PNAS, 106(26), 10587-10592. <a href="https://www.pnas.org/doi/10.1073/pnas.0903616106" target="_blank" rel="noopener noreferrer">pnas.org</a>.</li>
<li>Templeton, E. M., et al. (2022). <em>Fast response times signal social connection in conversation.</em> PNAS, 119(4), e2116915119. <a href="https://www.pnas.org/doi/10.1073/pnas.2116915119" target="_blank" rel="noopener noreferrer">pnas.org</a>.</li>
</ol>
