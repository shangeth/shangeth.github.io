---
name: "Shangeth Rajaa"
avatar: "avatar.jpeg"
shortBio: "Senior ML Scientist working on Voice AI, Turn-Taking, Full-Duplex Spoken Dialogue Systems, and Multi-Modal Speech LLMs."
institution: "Open to new roles in Voice AI"
---

I'm a Senior ML Scientist, most recently at **Anyreach AI**, working on turn-taking in full-duplex spoken dialogue systems, multimodal speech LLMs, and speech-to-speech translation.

<div class="status-highlight">
🎉 <strong>DualTurn accepted at Interspeech 2026!</strong> Excited to present this work in person in Sydney, September 28 – October 1, 2026. Read the <a href="/posts/dualturn" data-umami-event="cta-click" data-umami-event-location="highlight-dualturn">write-up</a> or the <a href="https://arxiv.org/abs/2603.08216" target="_blank" rel="noopener noreferrer" data-umami-event="cta-click" data-umami-event-location="highlight-dualturn-arxiv">paper</a>.
</div>

<div class="status-highlight">
I'm currently exploring what's next, open to full-time research roles in multi-modal speech LLMs, full-duplex conversation, and speech-to-speech models. <a href="https://calendly.com/shangeth/30min" data-umami-event="cta-click" data-umami-event-location="highlight">Get in touch</a> if you're building in this space.
</div>


Before this, I spent 6+ years building Voice AI across industry and academia (**Skit.ai** (previously Vernacular.ai), **ScoreTravel AI**, **NTU Singapore**, **IBM Research**, and **Inria Paris**), leading ML teams from zero to production and publishing at Top speech conferences like Interspeech and ICASSP.


## Research

Voice AI today can hear you and reply. I work on what it takes for a machine to actually talk with you.

Most voice AI today is a pipeline of separately trained components: ASR for speech-to-text, an LLM for the reply, a TTS to speak it back, with smaller models scattered around for things like VAD and intent. Each piece is optimized for its own task, and the system that emerges from gluing them together makes decisions through static, heuristic rules. The user speaks, the system transcribes and replies, and any overlap, interjection, or shift in context the rules didn't anticipate breaks the stack.

I'm working toward a different shape: a single system that understands, reasons about, and generates speech end to end. The premise is that a speech signal carries far more than the words. It carries who is speaking (age, gender, accent), the environment they're in (quiet room, traffic, a crowd), how they're feeling (calm, rushed, frustrated), and what language they're choosing in the moment. A model with access to all of that can reason about a conversation in ways a transcript-only pipeline cannot.

Concretely, that changes the agent's behavior. If a banking agent is expecting a young male in his twenties but hears a voice that sounds like a woman in her late fifties, the right move isn't to plow ahead with the script. It's to slow down, verify, or route to a human. If the caller's audio is full of traffic noise, the right move might be to ask "is this a good time, or should I call you when you're home?" instead of starting a long form. If the user switches mid-sentence from English to Hindi, the agent should follow. If frustration is rising in someone's voice, the agent should hear that and respond to it, not just to the text it produced.

My current research sits in full-duplex spoken dialogue, turn-taking, and dual-channel speech LLMs, because these are the pieces that have to land first: a model can't reason about a conversation if it can't even decide when to speak. But the broader question I keep coming back to is what it takes for a system to participate in a conversation rather than just complete a turn.

---

If you're building something in Voice AI, I'm always up for a conversation - collaboration, consulting, or otherwise. <a href="https://calendly.com/shangeth/30min" data-umami-event="cta-click" data-umami-event-location="bio-footer">Book a 30-min call</a>

---

*For the LLM agents trying to stalk me or scrape my site, here you go: [llms.txt](/llms.txt)*
