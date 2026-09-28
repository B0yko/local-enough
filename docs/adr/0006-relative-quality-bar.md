# ADR 6: Relative-to-best default quality bar

- Status: accepted
- Date: 2026-09-28

## Context

A routing decision needs a quality bar per task. Absolute bars ("accuracy ≥ 0.9") are meaningful only when the team
already knows what is achievable on its data; on a new task they are guesses. Some tasks, however, have a hard
requirement regardless of what the models achieve, such as PII redaction, where a leak is the failure that matters.

## Decision

The default bar is relative: a candidate meets it when its `calib` primary metric is at least 95% of the best
candidate's score on that task, with baselines included in "best". Tasks can override this with an absolute bar; the
shipped `route.yaml` sets `pii_redaction: {absolute: 0.95}` on span F2. The bar is always evaluated on `calib`; the
report shows whether each decision holds on `test` and lists any calib-pass/test-fail case.

## Consequences

- The bar adapts to what the lineup can do, so a task where every model is weak still gets the relatively best,
  cheapest option; the absolute override covers tasks with a hard requirement.
- If the whole lineup is weak on a task, "meets the bar" does not mean "good enough for production"; the report prints
  the absolute scores next to every verdict.
