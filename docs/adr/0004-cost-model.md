# ADR 4: Cost model, dedicated vs shared machine, and the break-even feasibility rule

- Status: accepted
- Date: 2026-09-28

## Context

Cloud prices are per token; the local-versus-cloud decision is per task and per month. A local pilot often looks
free because the machine is already paid for, and a break-even volume is meaningless if one machine cannot serve
it. The router also serves several tasks from the same machine, which a per-task view does not capture.

## Decision

The formulas are in [docs/cost-model.md](../cost-model.md) and implemented as pure functions in
`local_enough/costmodel.py`:

- fixed cost per machine per month = amortised purchase price × allocation + idle energy × allocation + ops;
- energy per task from incremental watts and sustained throughput (Pass B throughput × soak throttle factor);
- capacity per month = sustained tasks per hour × busy hours per day × 30;
- break-even volume = fixed / (cheapest bar-meeting cloud USD per task − local energy per task), reported as
  "never" when the denominator is ≤ 0, "infeasible on this hardware" when it exceeds one machine's capacity,
  "n/a — local below bar" or "n/a — no cloud model meets the bar" when a side has no candidate.

Two scenarios, always labelled:

- **Dedicated** (per-task tables, frontier charts, break-even and verdicts): the machine serves only task t at its
  monthly volume V_t.
- **Shared** (router mixed-workload table): one machine's fixed cost is shared across all locally served volume;
  utilisation is Σ V_t / capacity_t over locally served tasks, and above 1 the fleet is ceil of that sum.

Identical extra machines scale fixed cost and capacity together, so if one machine cannot reach break-even within
its capacity, no number of identical machines can; the verdict is "infeasible on this hardware" rather than a
machine count. A test covers this property.

## Consequences

- Local cost depends on the monthly volume: the same model is expensive at low volume and cheap at high volume, and
  the report shows both the fully loaded and the energy-only cost.
- The hardware price is an input that must be entered (0 is rejected), with its source and date in the report.
