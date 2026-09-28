# ADR 2: Template-generated synthetic data

- Status: accepted
- Date: 2026-09-28

## Context

Four of the five tasks (extraction, PII redaction, summarisation, entity matching) concern business records that
cannot be published when real: customer emails, HR notes, meeting transcripts, vendor masters. Public datasets for
these tasks are either not redistributable, not in this output format, or labelled in ways that do not match the
task definitions here. Generating data with an LLM would make the gold labels only as good as that LLM and would
bias the benchmark towards the generating model family.

## Decision

`scripts/build_datasets.py --seed 7` renders the four synthetic datasets and the judge sets from hand-written
templates and seeded value pools, with no LLM in the loop, so every gold label is exact by construction. The fifth
task uses the public BANKING77 dataset (CC-BY-4.0). Fictional values follow reserved ranges (RFC 2606 email domains,
Ofcom and NANP fictional phone ranges, IBAN-shaped strings with valid check digits) and names come from curated
pools checked against a brand blocklist. Tests assert byte-identical regeneration, the hygiene rules and a
sentence-repetition limit on transcripts.

## Consequences

- Labels are exact and the data is redistributable under Apache-2.0.
- Template data is more regular than real mail, so scores on real data may be lower. The README says so, and
  bring-your-own tasks exist so a team can rerun the same pipeline on its own labelled records.
