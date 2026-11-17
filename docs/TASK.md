# The task

The system reads several dated crawls ("snapshots") of one credit union's public website, oldest first, and then
answers member questions the way a careful person would: with the value that is current, for the right tier or
term, with a source, and with "I don't know" when the site does not say.

What makes it hard is the website, not the questions:

- a rate changes, but an old page, PDF or news release still shows the old number;
- a promotion ends and its page stays up;
- one page gives the dividend rate and another the APY, and they look like a disagreement;
- a table has tiers by balance or term, and a careless reader sees every tier as a conflict.

## Interface

`submission/solution.py` has a `build()` function that returns an object with three methods (the exact types are
in `harness/contract.py`):

| Method | What it does |
|---|---|
| `ingest(snapshot)` | called once per snapshot, oldest first |
| `answer(question, as_of=None)` | answers from the newest snapshot, or as the site stood on the `as_of` date |
| `conflicts()` | lists where the website contradicts itself, which value is believed and why |

## Scoring

`harness/` is the scorer that came with the assignment, unchanged. Per question it checks that the required
values are in the answer (numbers compared by value), gives zero if a listed wrong or out-of-date value appears,
gives credit for a refusal only on questions the site cannot answer, and adds a small credit for citing the right
page. The final score combines accuracy (55%), the conflict list (15%), consistency across reworded and repeated
questions (15%) and speed (15%, full marks when 95% of answers take 2 s or less).

## Data

The website snapshots were provided for the assignment and may not be redistributed, so they are not in this
repository; neither are the assignment's own practice questions. `sample/` holds a small made-up site in the same
format, so that the whole pipeline can be run without them.
