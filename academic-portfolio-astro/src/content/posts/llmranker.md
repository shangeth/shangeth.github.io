---
title: "llmranker: LLM-based ranking and reasoning algorithms for search and recommendation"
date: "2026-08-04"
description: "LLM-based ranking and reasoning algorithms for search and recommendation: why asking an LLM to rank a shortlist outperforms embedding distance on compositional queries, a research-grounded tour of pointwise, pairwise, listwise, setwise, and tournament-style ranking, and a package that implements all of them over any provider."
tags:
  - "LLM"
  - "Information Retrieval"
  - "Search"
  - "Recommendation Systems"
  - "Python"
---

[![GitHub](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/shangeth/llmranker) [![PyPI](https://img.shields.io/pypi/v/llmranker.svg)](https://pypi.org/project/llmranker/) [![Open in Colab](https://img.shields.io/badge/Open%20in%20Colab-F9AB00?logo=googlecolab&color=blue)](https://colab.research.google.com/github/shangeth/llmranker/blob/main/examples/quickstart.ipynb) [![Cite this work](https://img.shields.io/badge/Cite-BibTeX-yellow.svg)](#citing-this-package)

## Ranking is not retrieval

"Search" usually means two problems stacked on top of each other. First, retrieval: pull a candidate set out of a much larger corpus, cheaply, without missing anything relevant. Second, ranking: given that candidate set, put the best few at the top. Retrieval has to be fast over millions of items. Ranking only has to be good over dozens.

Almost all of the recent argument about LLMs in search is really about the second problem. Nobody is proposing an LLM forward pass per document across a million-item index. The question is what to do with the shortlist a cheap retriever already produced, and that's a much smaller compute budget to spend.

## The default: embed, then sort by distance

The standard approach to both halves is the same tool. Encode the query and every candidate as a vector with a bi-encoder, and rank by cosine similarity or another distance metric. It's fast, it's a single ANN index lookup, and it generalizes past exact keyword overlap in a way lexical search (BM25) cannot.

It also works by compressing an entire query into one fixed-width vector. That compression is where the trouble starts.

## Where a single vector runs out of room

A 2026 SIGIR reproducibility study on compositional information retrieval makes the failure precise rather than anecdotal ([Degenhart et al., 2026](https://arxiv.org/abs/2605.03824)). On QUEST, a benchmark whose queries carry real semantic content, dense single-vector models like GritLM-7B reach Recall@100 around 0.42, respectable, and better than BM25's 0.197. But the authors also built LIMIT+, a controlled benchmark stripped of incidental semantic cues so that only constraint satisfaction is left to solve. On LIMIT+, BM25 holds at 0.837 recall while dense single-vector embeddings collapse to under 0.13. A multi-vector method (ColBERT), which keeps per-token representations instead of compressing to one vector, does meaningfully better than single-vector dense retrieval but still trails lexical matching.

The gap between the two benchmarks is the finding. What looked like compositional reasoning on QUEST was mostly the model keying on semantic correlations, not actually satisfying the conjunction of constraints. A single embedding has no slot to independently hold "family friendly" AND "near historic sites" AND NOT "on the beach," three constraints, one of them negated. It has to blend all of that into one point in space, and blending is lossy in exactly the way conjunction and negation aren't supposed to be.

Here's what that looks like on an actual query, not just a benchmark number. Say someone searches for a "family friendly hotel near the historic center, not directly on the beach, with a pool." Two listings come back:

> **Hotel A.** Beachfront resort steps from the sand. Kids' pool, family suites, lively boardwalk nightlife just outside.
>
> **Hotel B.** Quiet family-run inn two blocks from the old town's museums and cathedral. Small courtyard pool, cots and high chairs on request, five minutes' walk to the beach promenade.

Hotel B is the one that actually satisfies every constraint in the query: family friendly, near the historic center, has a pool, and isn't on the beach. Hotel A fails the one constraint that was stated as a negative. But Hotel A shares more surface vocabulary with the query: beach, pool, family. "Beachfront" sits close to "beach" in embedding space whether the query asked for that or explicitly ruled it out, because the encoder has no separate slot for "must have," "nice to have," and "must not have." It produces one vector for the whole query and one for the whole listing, and measures how close the two are overall. Hotel A wins that comparison even though it fails the query outright. That's not a cherry-picked example; it's the same failure the LIMIT+ numbers above are measuring at scale.

This is an architectural limit, not a training-data problem. No amount of additional training data fixes a representation with nowhere to put "and not this."

## The LLM is a reasoner, not a better scorer

It's tempting to describe an LLM reranker as "a smarter distance function," something that plays the same role as cosine similarity but happens to score things more accurately. That undersells what's actually happening and it's worth being precise about, because the mechanism is the whole reason this works.

An embedding model never reads Hotel B's listing the way a person does. It maps the text to a point in space and that's the entire computation; there's no step where it checks off "near the historic center: yes" or "on the beach: no." An LLM asked to rank Hotel A against Hotel B for that query does something categorically different: it can work through the requirements one at a time against the actual text. Near the historic center? Hotel B says two blocks from the museums and cathedral, yes. Has a pool? Yes, a courtyard pool. On the beach? No, five minutes away on foot, and the phrase "beach promenade" is a landmark being used to give directions, not a description of the hotel's location. It can also bring in judgment the embedding never had access to: "lively boardwalk nightlife" at Hotel A cuts against "family friendly" even though the word "family" also appears in that same listing, because a reasoning model knows what nightlife next to a family hotel usually implies, and a bi-encoder has no comparable notion of one phrase in a passage undercutting another.

That's inference over a set of stated constraints, weighing evidence, resolving a contradiction inside one listing, and reaching a conclusion it could explain if asked. It's closer to what happens when you hand two hotel write-ups to a knowledgeable friend and ask which one to book. None of the ranking strategies below change that basic fact. Pointwise, pairwise, listwise, setwise, and tournament-style prompting are all different ways of applying that same evaluative reasoning across more than one candidate and turning the results into an order; the reasoning is the mechanism doing the work, and the sorting logic around it is bookkeeping.

## Retrieve cheap, rank smart

The natural fix is the one IR has used since long before LLMs: retrieve a shortlist cheaply (BM25, dense retrieval, or both), then spend a more expensive, more capable model re-scoring just that shortlist. This is the setting every paper below operates in: reranking dozens of candidates a first-pass retriever already narrowed down, not scoring a corpus. It's also why the cost of an LLM call per comparison is a real budget question but not a disqualifying one. You're not paying it a million times, you're paying it a few dozen times per query.

The rest of this post is about what to actually ask the LLM to do with that shortlist. There isn't one obvious answer. Pointwise, pairwise, listwise, setwise, and tournament-style prompting each frame the same reranking problem differently, with different cost and robustness tradeoffs, and each has a specific paper behind it.

## Pointwise: score each candidate alone

The simplest formulation asks the LLM to score one candidate against the query at a time, something like "how relevant is this, 0 to 10?", then sorts by score. It costs exactly one call per candidate and every call is independent, so it parallelizes trivially.

The problem is calibration. A score of 7 from one call and a score of 6 from a separate, unrelated call are supposed to be comparable, but nothing in the prompting forces the model to use a consistent internal scale across calls. The [Setwise paper](https://arxiv.org/abs/2310.09497) notes this directly: pointwise relevance generation needs output calibration across documents, and its effectiveness doesn't reliably improve with model scale the way you'd expect if the model were doing something more like genuine comparison.

## Pairwise: which is more relevant, A or B?

Pairwise prompting sidesteps calibration by never asking for an absolute score, only "which of these two is more relevant to the query?" [Qin et al. (2023)](https://arxiv.org/abs/2306.17563) formalized this as Pairwise Ranking Prompting (PRP), with a prompt as blunt as:

> Given a query {query}, which of the following two passages is more relevant to the query? Passage A: {document1} Passage B: {document2} Output Passage A or Passage B:

A relative judgment like this is a much easier task for an LLM than an absolute score, and PRP's numbers reflect it: with Flan-UL2 (20B parameters), PRP-Allpair reaches NDCG@10 of 72.42 on TREC-DL 2019, within a point of RankT5, a 3B-parameter *supervised* ranker (72.95), and well past a pointwise baseline using text-davinci-003 (64.61).

That result comes at a cost that scales with how the comparisons are organized into a full ranking. PRP proposes three:

- **All-pairs**: compare every pair once, rank by win count. O(n²) comparisons, most robust to any single noisy comparison, embarrassingly parallel.
- **Sorting (heapsort)**: O(n log n) comparisons, sequential, since each comparison's result determines the next one.
- **Sliding window**: repeated backward passes, similar to bubble sort, O(n) per pass with as many passes as top-k positions needed.

## Position bias, and PRP's fix

Pairwise comparison has its own failure mode: LLMs are biased toward whichever candidate is listed first (or second, model-dependent), independent of content. [Wang et al. (2023)](https://arxiv.org/abs/2305.17926) demonstrated how severe this can get: using ChatGPT as a pairwise evaluator on the Vicuna benchmark, simply swapping presentation order let Vicuna-13B "beat" ChatGPT on 66 of 80 queries, a result driven entirely by position, not quality. Their proposed fix, balanced position calibration, aggregates judgments across multiple presentation orders rather than trusting a single one.

PRP applies a lighter version of the same idea: run every comparison both ways (A-then-B and B-then-A), keep the result only if both orderings agree, and treat disagreement as a tie rather than as a coin flip. It roughly doubles the calls for whichever comparisons use it, and it's the reason "swap and check agreement" is a load-bearing design pattern in pairwise ranking rather than an afterthought.

## Listwise: output the whole ordering at once

Rather than comparing two or a handful of items per call, listwise prompting shows the LLM a batch of candidates and asks for a full ranked permutation in one shot. [Sun et al. (2023)](https://arxiv.org/abs/2304.09542), "Is ChatGPT Good at Search?", the paper behind RankGPT, argue this fits an LLM's pretraining better than pointwise scoring: generating an ordering is a text-generation task, which is what these models were trained to do, rather than an isolated numerical judgment. Since one call handles many candidates, it needs far fewer calls than pairwise for the same list. For lists too long for one context window, RankGPT slides a window across the list, repeatedly re-ranking overlapping chunks.

It has a distinct weakness, though, and it's the one that motivates the next two methods: RankGPT's output is a refinement of whatever order the candidates arrived in, and refinement is order-dependent. [Chen et al. (2025)](https://arxiv.org/abs/2406.11678) test this directly, running RankGPT on the same TREC DL candidates under BM25 order, randomly shuffled order, and reversed order, and find RankGPT's effectiveness drops sharply once the input order is perturbed. A ranker that's supposed to be judging relevance is partly just reporting back the order it was handed.

## Setwise: compare a small set at a time

Setwise prompting, from [Zhuang et al. (2023)](https://arxiv.org/abs/2310.09497), generalizes the pairwise question from "which of these 2 is best" to "which of these *c* is best," then plugs that n-ary comparator into the same sorting algorithms pairwise uses (heapsort, bubblesort). Comparing c candidates per call instead of 2 shrinks a heap's branching factor from 2 to c, which changes the call complexity from O(k log₂ N) to O(k log_c N) for heapsort, and from O(kN) to O(kN/(c−1)) for bubblesort.

On TREC DL 2019 with Flan-T5-large, the paper reports NDCG@10 of 0.670 for setwise heapsort against 0.657 for pairwise heapsort, 0.654 for pointwise, and 0.561 for listwise generation, so setwise is the best of the four, not just the cheapest. And it is also the cheapest by a wide margin: setwise heapsort needs 125.4 LLM calls and 8.0s latency for the same ranking task that pairwise heapsort needs 230.3 calls and 16.1s for, roughly 46% fewer calls at higher effectiveness.

## Setwise insertion: use the order you already have

Heapsort and bubblesort both treat the incoming candidate order as irrelevant; every comparison starts from scratch. But candidates rarely arrive in a genuinely random order; they usually come from an upstream retriever that already got some of the ranking right. [Podolak et al. (2025)](https://arxiv.org/abs/2504.10509) (SIGIR'25) build "Setwise Insertion" around that observation: sort the first k candidates into a heap, then walk the rest in chunks, each time comparing the chunk against the current worst-of-top-k (the "guard"). If the guard beats the whole chunk, the entire chunk is discarded in a single call; only when a chunk contains a genuine improvement does the method spend calls binary-inserting it into position. Across Flan-T5, Vicuna, and Llama backbones, this cuts query time by 31% and LLM calls by 23% versus baseline setwise, with a slight effectiveness gain too. The saving is proportional to how good the prior order already is, though, so it degrades toward ordinary setwise cost (not incorrect results) if the input order carries no signal.

## Tournament-style: make ranking robust to whatever order you started with

TourRank ([Chen et al., 2025](https://arxiv.org/abs/2406.11678), WWW'25) takes a different approach to the same order-sensitivity problem RankGPT has: instead of refining an initial order, discard it as a signal entirely. Candidates are split into small groups, an LLM picks winners from each group (like a tournament's group stage), and winners advance through several stages while earning points along the way. A 100-candidate run in the paper's setup funnels through six stages (100 → 50 → 20 → 10 → 5 → 2), and earlier survivors earn fewer points than later ones. The whole tournament is repeated *r* times with fresh random grouping and within-group shuffling each time, and a candidate's final score is its point total summed across every run.

The random regrouping is the point. Because a candidate's fate depends on being selected across many different groupings rather than its position in one linear pass, TourRank's effectiveness is close to unaffected when the paper reruns the same experiment with the input candidates shuffled or fully reversed, the exact perturbation that visibly degrades RankGPT. On TREC DL 2019/2020, TourRank-10 (r=10) reaches NDCG@10 of 71.63/69.56, ahead of setwise bubblesort (71.16/69.04) and well ahead of RankGPT (68.19/63.60); on the BEIR average across 8 datasets, TourRank-10 leads at 50.94 against RankGPT's 49.37. Even TourRank-2, just two tournament runs, already beats RankGPT on both benchmarks, at roughly twice RankGPT's per-query document-to-LLM count but with a lower time complexity, since groups within a stage run independently rather than as one long sequential refinement.

## Reasoning before judging

A separate axis, orthogonal to which sorting/comparison method is used: does the model think before it answers? Two 2025 papers built dedicated reasoning rerankers. [Weller et al.](https://arxiv.org/abs/2502.18418) (Rank1, CoLM 2025) distilled over 600,000 R1 reasoning traces over MS MARCO queries and passages into a reranker that generates an explicit reasoning chain before its relevance judgment, and report state-of-the-art results on reasoning-heavy IR benchmarks along with strong out-of-distribution generalization. [Zhuang et al.](https://arxiv.org/abs/2503.06034) (Rank-R1) get a comparable effect via reinforcement learning instead of distillation, training on relevance labels alone with no reasoning supervision, and match supervised fine-tuning in-domain using only 18% of the training data, while clearly outperforming both zero-shot and supervised baselines out-of-domain on complex queries, especially at 14B scale.

Both are trained models. A prompting-only technique can approximate the effect, just asking any off-the-shelf chat model to reason step by step before giving a final answer, without training a dedicated reranker. It's a fair approximation precisely because both papers' gains trace back to reasoning-before-answering as the mechanism, not to anything specific to their training procedure. It's a strictly weaker version, though: no distillation, no RL, just a longer, more deliberate completion in exchange for more output tokens and latency.

## llmranker

Everything above is a research literature, not a single tool: five different prompting strategies, each requiring the sorting/grouping logic that turns individual comparisons into a full ranking, each with real tradeoffs in call count, latency, and robustness to input order, plus cross-cutting concerns like position-bias correction and reasoning prompting layered on top. [llmranker](https://github.com/shangeth/llmranker) is a Python package that implements all of it: pointwise, pairwise, setwise, listwise, and TourRank, with position-debiasing and reasoning prompting available where they apply, as one consistent interface, built on [LiteLLM](https://github.com/BerriAI/litellm) so the same code runs against OpenAI, Gemini, Anthropic, Azure, Bedrock, local Ollama models, or any of LiteLLM's 100+ supported providers.

```bash
pip install llmranker
```

## Quickstart

```python
from llmranker import Candidate, LLMConfig, SetwiseRanker

ranker = SetwiseRanker(LLMConfig(model="gpt-4o-mini"), num_child=4, k=5)

candidates = [
    Candidate(id="1", text="A budget hostel in the city center."),
    Candidate(id="2", text="A five-star beachfront resort with a spa."),
    Candidate(id="3", text="A family-run guesthouse near the old town, kid-friendly."),
]

result = ranker.rank(
    query="affordable, family friendly, near historical sites",
    candidates=candidates,
)
print([c.id for c in result])  # ['3', '1', '2']
```

This is the same shape of query as the hotel example earlier: three soft constraints, none of them a keyword match, and exactly the case where reasoning through the constraints beats measuring distance to them.

Or, pairwise, sorted by heapsort:

```python
from llmranker import Candidate, LLMConfig, PairwiseRanker

ranker = PairwiseRanker(LLMConfig(model="gpt-4o-mini"), method="heapsort", k=10)

candidates = [Candidate(id=str(i), text=doc) for i, doc in enumerate(my_documents)]
result = ranker.rank(query="my search query", candidates=candidates)

for c in result:
    print(c.id, c.score)
```

For a longer worked example, with all five strategies run against the same hotel-recommendation query and compared side by side on ranking quality, LLM calls, tokens, cost, and latency, see [`examples/hotel_recommendation/`](https://github.com/shangeth/llmranker/tree/main/examples/hotel_recommendation), or open [`examples/quickstart.ipynb`](https://colab.research.google.com/github/shangeth/llmranker/blob/main/examples/quickstart.ipynb) directly in Colab.

## Choosing a strategy

Given the cost/effectiveness/robustness tradeoffs above, the package's own guidance follows directly from them:

<table class="table-left">
<thead>
<tr><th>Strategy</th><th>Reach for it when</th></tr>
</thead>
<tbody>
<tr><td><strong>Setwise</strong> (<code>num_child=4-8</code>, heapsort)</td><td>Default starting point: best effectiveness-per-call in the numbers above</td></tr>
<tr><td><strong>Pairwise</strong></td><td>You want the simplest mental model and don't mind more calls</td></tr>
<tr><td><strong>Listwise</strong></td><td>Latency matters more than call count and the candidate list fits in one window</td></tr>
<tr><td><strong>Pointwise</strong></td><td>You need a standalone score per candidate (e.g. relevance thresholding), or n is too large for any comparison-based method</td></tr>
<tr><td><strong>TourRank</strong></td><td>The order candidates arrive in is unreliable, or you need a result that provably doesn't depend on it</td></tr>
</tbody>
</table>

## Swapping providers

Every ranker takes an `LLMConfig`, whose `model` field is a LiteLLM model string. Nothing else about the code changes:

```python
LLMConfig(model="gpt-4o-mini")                                      # OpenAI
LLMConfig(model="gemini/gemini-1.5-flash")                          # Google Gemini
LLMConfig(model="claude-3-5-sonnet-20241022")                       # Anthropic
LLMConfig(model="azure/my-deployment-name")                         # Azure OpenAI
LLMConfig(model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0")  # AWS Bedrock
LLMConfig(model="ollama/llama3")                                    # local, via Ollama
```

## What's not solved yet

Two honest caveats, tracked in the package's [ROADMAP](https://github.com/shangeth/llmranker/blob/main/ROADMAP.md):

- **Position-bias correction is pairwise-only for now.** `debias_position`'s run-both-orders-and-check-agreement trick doesn't yet extend to setwise or tournament groupings, even though the same bias almost certainly applies there. A setwise equivalent would need repeating each group comparison under a reshuffled order, at more than double the cost.
- **Prompt injection.** Every ranker in this package puts candidate text directly into the prompt sent to the LLM, which means a candidate could contain adversarial text aimed at manipulating its own ranking. [Yin et al. (2026)](https://arxiv.org/abs/2602.16752) study exactly this vulnerability across LLM ranking paradigms and find it varies by model family and architecture, with encoder-decoder models showing more inherent resilience than decoder-only ones. Nothing in `llmranker` currently sanitizes or detects injected instructions in candidate text. Treat it as untrusted input if it comes from anywhere outside your own pipeline.

## Citing this package

```bibtex
@misc{Rajaa_llmranker,
author = {Rajaa, Shangeth},
title = {{llmranker: LLM-based ranking and reasoning algorithms for search and recommendation}},
url = {https://github.com/shangeth/llmranker}
}
```

## References

<ol class="references">
<li>Degenhart, V., Timman, D., de Vries, A. P., Hasibi, F., &amp; Hoveyda, M. (2026). <em>Reproducing Complex Set-Compositional Information Retrieval.</em> SIGIR 2026 Reproducibility Track. <a href="https://arxiv.org/abs/2605.03824" target="_blank" rel="noopener noreferrer">arXiv:2605.03824</a>.</li>
<li>Zhuang, S., Zhuang, H., Koopman, B., &amp; Zuccon, G. (2023). <em>A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models.</em> <a href="https://arxiv.org/abs/2310.09497" target="_blank" rel="noopener noreferrer">arXiv:2310.09497</a>.</li>
<li>Qin, Z., Jagerman, R., Hui, K., Zhuang, H., Wu, J., Yan, L., Shen, J., Liu, T., Liu, J., Metzler, D., Wang, X., &amp; Bendersky, M. (2023). <em>Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting.</em> <a href="https://arxiv.org/abs/2306.17563" target="_blank" rel="noopener noreferrer">arXiv:2306.17563</a>.</li>
<li>Wang, P., Li, L., Chen, L., Cai, Z., Zhu, D., Lin, B., Cao, Y., Liu, Q., Liu, T., &amp; Sui, Z. (2023). <em>Large Language Models are not Fair Evaluators.</em> <a href="https://arxiv.org/abs/2305.17926" target="_blank" rel="noopener noreferrer">arXiv:2305.17926</a>.</li>
<li>Sun, W., Yan, L., Ma, X., Wang, S., Ren, P., Chen, Z., Yin, D., &amp; Ren, Z. (2023). <em>Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents.</em> EMNLP 2023. <a href="https://arxiv.org/abs/2304.09542" target="_blank" rel="noopener noreferrer">arXiv:2304.09542</a>.</li>
<li>Podolak, J., Perić, L., Janićijević, M., &amp; Petcu, R. (2025). <em>Beyond Reproducibility: Advancing Zero-shot LLM Reranking Efficiency with Setwise Insertion.</em> SIGIR 2025. <a href="https://arxiv.org/abs/2504.10509" target="_blank" rel="noopener noreferrer">arXiv:2504.10509</a>.</li>
<li>Chen, Y., Liu, Q., Zhang, Y., Sun, W., Ma, X., Yang, W., Shi, D., Mao, J., &amp; Yin, D. (2025). <em>TourRank: Utilizing Large Language Models for Documents Ranking with a Tournament-Inspired Strategy.</em> WWW 2025. <a href="https://arxiv.org/abs/2406.11678" target="_blank" rel="noopener noreferrer">arXiv:2406.11678</a>.</li>
<li>Weller, O., Ricci, K., Yang, E., Yates, A., Lawrie, D., &amp; Van Durme, B. (2025). <em>Rank1: Test-Time Compute for Reranking in Information Retrieval.</em> CoLM 2025. <a href="https://arxiv.org/abs/2502.18418" target="_blank" rel="noopener noreferrer">arXiv:2502.18418</a>.</li>
<li>Zhuang, S., Ma, X., Koopman, B., Lin, J., &amp; Zuccon, G. (2025). <em>Rank-R1: Enhancing Reasoning in LLM-based Document Rerankers via Reinforcement Learning.</em> <a href="https://arxiv.org/abs/2503.06034" target="_blank" rel="noopener noreferrer">arXiv:2503.06034</a>.</li>
<li>Yin, Y., Wang, S., Koopman, B., &amp; Zuccon, G. (2026). <em>The Vulnerability of LLM Rankers to Prompt Injection Attacks.</em> <a href="https://arxiv.org/abs/2602.16752" target="_blank" rel="noopener noreferrer">arXiv:2602.16752</a>.</li>
</ol>
