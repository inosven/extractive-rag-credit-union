# Sample site

A made-up credit union ("Maple Ridge") in the same format as the real data: three snapshots of thirteen pages
(rates, loan rates, fee schedule, product and help pages, branches, two news releases and a blog post), 20
questions in `eval.jsonl` and one labelled conflict. Between the snapshots a certificate rate, an auto loan rate,
a card fee, Saturday hours and the member count change, and an old blog post still quotes a 2024 rate.

    python harness/run_eval.py --submission submission --snapshots sample/snapshots \
        --eval sample/eval.jsonl --conflicts-gold sample/conflicts_gold.json --out results_sample

The system was written for another website and was not tuned on this one. Rule-only it answers 17 of the 20
questions; it misses a balance-tier question, the Saturday hours and one scholarship amount. It reports the
certificate rate change as a conflict but not the stale rate in the blog post.
