# Datasets

Everything under `src/local_enough/data/datasets/<task>/` is produced by
`scripts/build_datasets.py --seed 7`: hand-written sentence templates and seeded value pools, **no LLM in the
loop**, so gold labels are exact by construction. Regenerating with the same seed reproduces every synthetic
file byte-for-byte (see `tests/test_datasets_hygiene.py`).

Four of the five tasks are fully synthetic and Apache-2.0, same as the code. `classification` is the one
exception: it bundles a stratified subset of the public **BANKING77** dataset.

Regenerate everything (BANKING77 needs network access once):

```
uv run python scripts/build_datasets.py --seed 7 --banking77-dir /path/to/cache
```

Add `--skip-banking77` to regenerate only the four synthetic tasks offline.

## extraction

- **Source / licence:** synthetic, Apache-2.0.
- **Files:** `calib.jsonl` (100 rows), `test.jsonl` (200 rows), `task.yaml`.
- **Seed:** 7. **Distinct templates:** 24 (opening sentences, seats/budget/meeting-request phrasings, urgency
  hints, closings; phone/amount/date formats are generated separately, see below).
- **Metric:** field accuracy after type-aware normalisation (primary), plus exact match and invalid-JSON rate.
- **Generator:** `build_extraction()` in `scripts/build_datasets.py`.

Inbound B2B enquiry emails. Each item's `text` starts with a `Date: <weekday>, <d> <Mon> <yyyy>` header (also
stored as `reference_date`), then `From:`/`Subject:` lines and a body. Gold has all 11 CRM fields
(`company_name`, `contact_name`, `contact_email`, `phone`, `country`, `product`, `seats`, `budget_amount`,
`budget_currency`, `requested_meeting_date`, `urgency`). Phone numbers are drawn only from the fictional UK
mobile range (07700 900xxx) and US/CA range (555-01xx) in varied formats (spaced, dashed, bracketed, `+44`/`0`
prefixes, `+1`/`00` prefixes). Requested meeting dates use relative expressions ("next Tuesday", "in two
weeks", "on the 14th") resolved by the generator against the reference date, so the gold ISO date is exact.
Countries (GB, US, CA) are given by name, by code, or left implicit (inferred only from the phone/address).

## classification

- **Source / licence:** [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets) (PolyAI), **CC BY
  4.0**. Casanueva, I., Temcinas, T., Gerz, D., Henderson, M., Vulic, I. (2020). *Efficient Intent Detection
  with Dual Sentence Encoders*. Proceedings of the 2nd Workshop on NLP for ConvAI, ACL 2020.
- **Files:** `calib.jsonl` (154 rows, 2/intent), `test.jsonl` (308 rows, 4/intent, disjoint from calib),
  `train.jsonl` (10,003 rows, the full train split, bundled only for the TF-IDF baseline), `task.yaml`,
  `LICENSE-BANKING77.txt` (full CC BY 4.0 legal text, copied verbatim from the PolyAI repository).
- **Seed:** 7 (used only to sample calib/test from the official test split).
- **Metric:** accuracy (primary), plus macro-F1 and invalid-label rate.
- **Generator:** `build_classification()` in `scripts/build_datasets.py`.
- **Source files and sha256** (downloaded once from
  `https://github.com/PolyAI-LDN/task-specific-datasets`, `banking_data/`):
  - `train.csv`: `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b`
  - `test.csv`: `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d`

  The generator verifies both hashes before sampling and refuses to proceed on a mismatch.

Query -> one of 77 intent labels (the official category strings, e.g. `card_arrival`). `calib` and `test` are
sampled disjoint, stratified per intent, from the official 3,080-row test split, with the seed. `train.jsonl`
is the unmodified 10,003-row train split, converted from CSV to JSONL only.

**Attribution and licence statement:** this product includes a stratified subset of the BANKING77 dataset
by PolyAI, licensed under CC BY 4.0. The bundled files are unchanged apart from CSV-to-JSONL conversion and
sampling; see `NOTICE` and `src/local_enough/data/datasets/classification/LICENSE-BANKING77.txt`.

## pii_redaction

- **Source / licence:** synthetic, Apache-2.0.
- **Files:** `calib.jsonl` (100 rows), `test.jsonl` (200 rows), `task.yaml`.
- **Seed:** 7. **Distinct templates:** 71 (6 PII-type sentence templates x4, 5 hard-negative kinds x3, 4
  document-category fillers x4, plus support/HR/call-note topic pools).
- **Metric:** span F2 (primary; recall weighted twice), plus recall, precision, leak rate and the zero-leak
  document rate.
- **Generator:** `build_pii()` in `scripts/build_datasets.py`.

Support tickets, HR notes, invoice disputes and call notes with 0-6 gold spans each of `PERSON`, `EMAIL`,
`PHONE`, `IBAN`, `ADDRESS`, `DATE_OF_BIRTH`; about 15% of documents carry no PII at all. Every span's character
offsets are asserted against the document text at generation time. Hard negatives that appear in the text but
are never gold spans: company names, product names, order/invoice reference codes, non-birth dates, and
labelled general company switchboard numbers (still drawn from the same fictional phone ranges). IBANs are
fictional account numbers with **valid mod-97 check digits**, verified in the generator against a published
test IBAN before use.

## summarisation

- **Source / licence:** synthetic, Apache-2.0.
- **Files:** `calib.jsonl` (40 transcripts), `test.jsonl` (80 transcripts), `judge_calib.jsonl` (200
  construction-labelled summaries over the 40 calib transcripts), `judge_holdout_transcripts.jsonl` (40 extra
  transcripts, disjoint from calib/test), `judge_holdout.jsonl` (200 construction-labelled summaries over the
  40 holdout transcripts), `task.yaml`.
- **Seed:** 7. **Distinct templates:** 150 (meeting topics, workstreams, due-date/amount phrasings,
  decision/task actions, opening/closing/small-talk/superseded/rejected/elaboration turns, and the 3 judge
  summary phrasing styles per fact kind).
- **Metric:** rubric pass rate, bias-corrected (primary; report shows "not judged" until `judge score` has
  run), plus mean fact recall.
- **Generator:** `build_summarisation()` in `scripts/build_datasets.py`.

600-1000 word meeting transcripts with speaker turns, built from a large combinatorial template pool so that
no single sentence (split on sentence punctuation, whitespace/case-normalised) appears in more than 20% of
transcripts (verified for the whole 160-transcript pool in `tests/test_datasets_summarisation.py`; the worst
offender in the current build is under 5%). Each transcript has 4-6 required facts (decisions, action items
with an owner and due date, amounts) and 5-10 distractors (small talk, superseded decisions, rejected
options).

The **judge-calib** and **judge-holdout** sets are **construction-labelled, never "human-labelled"**: for
each transcript, 5 template-rendered summaries (faithful; one required fact dropped; wrong number or date;
wrong owner; one invented commitment), each in one of 3 phrasing styles rotated by seed. Every summary record
carries the per-fact construction labels (`present`, `correct`), the unsupported-claim count, the variant,
the style, and `label_pass` computed by the same code rule the benchmark uses (at least `ceil(0.8n)` facts
present and correct, zero unsupported claims, within the 120-word limit).

## entity_matching

- **Source / licence:** synthetic, Apache-2.0.
- **Files:** `calib.jsonl` (100 pairs), `test.jsonl` (200 pairs), `task.yaml`.
- **Seed:** 7. **Distinct templates:** 10 (6 positive transformations: legal-form variant, abbreviation,
  typo, transliteration, missing fields, moved office; 3 hard-negative transformations: other country, parent
  vs subsidiary, similar name on the same street; plus the unrelated-pair negative).
- **Metric:** F1 on the match class (primary), plus precision and recall.
- **Generator:** `build_entity_matching()` in `scripts/build_datasets.py`.

Two company records (`name`, `street`, `postcode`, `city`, `country`, `domain`, `vat_id`, `phone`) -> a
`match` boolean, with about 40% positive pairs in both splits (exact ratio verified in
`tests/test_datasets_entity_matching.py`). VAT ids are fictional VAT-style strings, not validated against any
real country's algorithm.

## Bring-your-own example

`examples/custom-task/` ships a tiny made-up 3-label support-ticket classifier (`billing`, `bug_report`,
`how_to`; 30 calib / 60 test / 150 train rows, 30 distinct templates), generated by the same seeded
`build_custom_example()` function so it regenerates alongside the bundled tasks. See
`docs/add-your-task.md` for the bring-your-own-task schema.

## Fictional values (applies to every synthetic task)

- Email addresses use only the RFC 2606 domains `example.com`, `example.org`, `example.net` and their
  subdomains.
- Phone numbers use only the fictional UK mobile range `07700 900xxx` (`+44 7700 900xxx`) and the North
  American fictional range `555-01xx` (`+1 <area code> 555 01xx`).
- IBANs are structurally valid (correct BBAN length per country, correct mod-97 check digits) with random
  account numbers.
- Person and company names come from curated fictional pools, checked in the generator against a ~100-entry
  blocklist of well-known real brands (word-boundary match, so it never flags an unrelated substring).
- Fictional companies stay in four generic sectors: software, office supplies, logistics, facilities.
- No output ever contains the string "Andrii" or "Boiko" (checked by the generator and by
  `tests/test_datasets_hygiene.py`).

## Tests

`tests/test_datasets_hygiene.py` and the per-task `tests/test_datasets_*.py` files check: byte-identical
regeneration of the four synthetic tasks into a tmp directory (classification's BANKING77 part runs only when
`LOCAL_ENOUGH_BANKING77_DIR` is set); RFC 2606 domains and allowed phone ranges only; no blocklisted brand
names; no "Andrii"/"Boiko"; split sizes; the ~40% entity-matching match rate; the ~15% PII-free document rate;
PII span offset exactness; the summarisation sentence-repetition cap; and the judge-set label logic.
