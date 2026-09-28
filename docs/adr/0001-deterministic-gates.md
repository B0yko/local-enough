# ADR 1: Deterministic gates instead of self-reported confidence

- Status: accepted
- Date: 2026-09-28

## Context

The router has to decide, per request, whether the cheap primary model's answer is good enough or whether to
escalate. Common options are asking the model for a confidence score, using token log-probabilities, or training a
classifier on past outcomes. Self-reported confidence from small models is poorly calibrated, log-probabilities are
not exposed by every OpenAI-compatible server (and not at all by some cloud providers), and a learned escalation
model would need labelled production traffic that a new deployment does not have.

## Decision

Escalation is driven only by deterministic evidence checks ("gates") computed from the input, the parsed output and
the non-LLM baselines:

- classification: the label is in the label set and agrees with the TF-IDF baseline;
- extraction: valid JSON against the task schema, required fields present, free-text fields grounded in the source,
  normalised fields re-derivable from the source (phone digits, country, date from a date expression and the
  reference date, product enum, amount digits);
- pii_redaction: every predicted span occurs in the text and the regex baseline found nothing the model missed;
- summarisation: within the word limit and every number and date in the summary occurs in the source;
- entity_matching: agreement with the rapidfuzz baseline outside the baseline's uncertainty band.

Thresholds (fuzzy grounding ratio, entity-matching band) are tuned on the `calib` split only. The same gate code runs
in `route --simulate` and in the live router.

## Consequences

- Escalation decisions are reproducible and explainable, and simulation predicts live routing closely.
- Gates catch format and grounding failures, not every semantic error: a well-formed, grounded but wrong answer
  passes. The measured test-split quality of the routed system is reported next to each bar for that reason.
- When the primary candidate is the baseline a gate compares against, that gate reduces to its format checks.
