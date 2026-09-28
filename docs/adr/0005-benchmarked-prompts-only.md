# ADR 5: A router that serves only benchmarked task prompts

- Status: accepted
- Date: 2026-09-28

## Context

A general OpenAI-compatible gateway accepts any prompt. The routing decisions here rest on measured accuracy of each
model on a specific prompt and output format; a different prompt would invalidate them.

## Decision

The router exposes one model alias per task (`local-enough/<task>`). The client sends only the raw input as the user
message; the router applies the exact prompt template used in the benchmark, parses the output with the benchmark's
parser and returns the canonical parsed output. Arbitrary passthrough prompts and streaming are out of scope for
v0.1 and are on the roadmap.

## Consequences

- The quality and cost the router delivers are the ones measured, and simulation can replay routing exactly.
- Clients must adapt to one alias per task instead of sending free-form prompts.
