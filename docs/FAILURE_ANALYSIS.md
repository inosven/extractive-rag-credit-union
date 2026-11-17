# Where the remaining answers are lost

Measured on the dev set (345 generated questions, 301 of them answerable) with document expansion on, using
`tools/miss_analysis.py`. A candidate "contains the answer" only when all required strings of the gold answer are
in that one candidate, so an answer spread over two candidates counts as a miss. The script prints counts only.

## The answer is not among the 12 candidates: 45 of 301 questions (15%)

| Why | Questions | Share |
|---|---|---|
| The strings are not on the site at all (wrong gold, or metadata only) | 0 | 0% |
| The strings are on the site, but no single page has them all | 0 | 0% |
| One page has them all, but no single passage does | 19 | 42% |
| A passage has them all and is ranked 13 to 50 by the passage search | 16 | 36% |
| A passage has them all and is ranked below 50 | 10 | 22% |

By category (missed / questions): program information 14/121, company facts 7/33, how-to 7/43, entity attribution
6/19, tier questions 6/15, current values 4/17, as-of-date questions 1/9.

What this says:

- **Chunking, 42%.** The largest group is answers longer than one 120-word window, such as a list of three awards
  or a count and the places it refers to. The page is found; the answer is cut in two. Larger windows would fix
  some of these and make every prompt longer; letting the selector return two passages is the more direct fix.
  Neither is done.
- **Ranking, 58%.** The passage exists and is ranked too low. Document expansion already moved many of these up
  (recall@12 0.78 to 0.85). Two further retrievers were measured, below.

## Two things tried on the ranking part

| Added | Recall@12 | Cost per question | Final score (dev) | Kept |
|---|---|---|---|---|
| Local embedding search (bge-small) as a third list in the fusion | 0.784 to 0.821 (without expansion) | 6 ms | 0.715 to 0.709 | off by default |
| Local cross-encoder (ms-marco-MiniLM-L-6) re-ranking the first 50 fused results | 0.850 to 0.870 | 580 ms median, 840 ms p95 | not run | no |

The cross-encoder finds the answer for 2 more questions in 100, but adds about 0.6 s on a CPU to an answer that
already spends about 1 s in the selector call, against a 2 s limit. The answer is among the first 50 fused results
for 90% of questions, so a faster re-ranker has at most 5 points of recall to gain here.

## The answer is among the candidates and is still lost

From `tools/recall_stats.py` on the same set, without expansion (345 questions): 14 times the selector picked
another passage, 13 times the answer came with a value the gold answer forbids (usually a neighbouring tier or an
older figure in the same passage), 7 times the system refused although a candidate had the answer. Refusal
questions: 42 of 44 correct.
