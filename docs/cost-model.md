# Cost model

This document defines the cost model implemented as pure functions in
[`src/local_enough/costmodel.py`](../src/local_enough/costmodel.py). All quantities marked **(t)**
are per task kind (classification, extraction, pii_redaction, summarisation, entity_matching).
Every function is deterministic, unit-tested, and takes no configuration object directly -- callers
pass plain numbers, so the model has no hidden state.

## Two scenarios

The report and router always label which scenario a number uses:

- **Dedicated.** A machine dedicated to serving one task at its own monthly volume `V_t`. This is
  what the per-task tables, the frontier charts and the break-even table use.
- **Shared.** One machine (or a small fleet of identical machines) serves every task the router
  sends locally, and its fixed cost is spread across all of that locally served volume. This is
  what the router's mixed-workload table uses.

The two scenarios never mix in one number: a dedicated-scenario `$/1k tasks` figure is never
compared directly against a shared-scenario one without saying so.

## Volume

`reference_monthly_volume` (`route.yaml`) is the total mixed monthly volume across every task.
Per-task volume is

```
V_t = reference_monthly_volume × workload_mix[t]
```

unless `expected_monthly_volume.<t>` is set in `route.yaml`, in which case that value overrides the
computed `V_t` for task `t`.

## Fixed cost (dedicated scenario, per machine)

```
fixed_usd_per_month = purchase_price_usd × allocation / (lifetime_years × 12)
                    + idle_watts × 24 × 30 / 1000 × electricity_usd_per_kwh × allocation
                    + ops_usd_per_month
```

The first term amortises the purchase price linearly over its lifetime. The second is the cost of
running the machine idle around the clock (24 h × 30 days), converted from watts to kWh. Both are
scaled by `allocation`, the share of the machine charged to this workload (`1.0` for a machine
dedicated to `local-enough`). `ops_usd_per_month` covers anything else (for example, none in the
reference setup).

`fixed_usd_per_month` is implemented in [`costmodel.fixed_usd_per_month`](../src/local_enough/costmodel.py).

## Throughput and energy (t)

```
sustained_tasks_per_hour(t) = pass_b_tasks_per_hour(t) × throttle_factor
```

Pass B measures throughput at concurrency 4 after a short warm-up; the throttle factor (Pass C,
last-5-minute throughput ÷ first-5-minute throughput from a 20-minute soak) corrects that short
measurement for thermal throttling over a sustained run.

```
energy_usd_per_task(t) = incremental_watts / (sustained_tasks_per_hour(t) / 3600) / 3.6e6
                        × electricity_usd_per_kwh
```

`incremental_watts` is the mean load wattage minus the mean idle wattage while the model server is
running (see `power-probe`). Dividing by tasks-per-second gives joules per task; dividing by
3.6 × 10⁶ (joules per kWh) gives kWh per task; multiplying by the electricity price gives USD per
task. When `sustained_tasks_per_hour(t)` is 0 the function returns `inf` (undefined throughput
makes energy cost undefined too) rather than raising, so callers can propagate it into a "never"
verdict instead of crashing.

## Dedicated-scenario cost and capacity (t)

```
local_usd_per_task(t) = fixed_usd_per_month / V_t + energy_usd_per_task(t)
capacity_per_month(t) = sustained_tasks_per_hour(t) × busy_hours_per_day × 30
machines_needed(t)    = ceil(V_t / capacity_per_month(t))
```

`local_usd_per_task` spreads the whole machine's fixed cost over the task's own volume, which is
optimistic when the same machine also serves other tasks -- that sharing is exactly what the shared
scenario below corrects for.

## Break-even (t)

```
break_even_V(t) = fixed_usd_per_month / (cloud_usd_per_task(t) − energy_usd_per_task(t))
```

where `cloud_usd_per_task(t)` is the cheapest cloud model that meets the quality bar for task `t`.
[`costmodel.break_even`](../src/local_enough/costmodel.py) returns a `BreakEven(volume, verdict,
machines_needed)` with one of these verdicts:

| Verdict | Condition |
|---|---|
| `n/a — no cloud model meets the bar` | no cloud candidate meets the bar for this task |
| `n/a — local below bar` | no local candidate meets the bar for this task |
| `never` | the denominator is ≤ 0 (local's energy cost alone is not cheaper than cloud) |
| `infeasible on this hardware` | `break_even_V > capacity_per_month(t)`: the crossover volume exceeds what one machine can sustain |
| `break-even` | otherwise -- there is a volume within this machine's capacity at which local becomes cheaper than cloud |

`machines_needed` is reported alongside every verdict: it is `ceil(V_t / capacity_per_month(t))` for
the task's *actual* expected volume, independent of the break-even verdict, so the break-even table
can show it even for a "never" or "infeasible" row.

### Identical extra machines never rescue an infeasible break-even

Adding `N` identical dedicated machines multiplies both `fixed_usd_per_month` and
`capacity_per_month(t)` by `N`, while `break_even_V` scales by the same `N`
(`fixed_usd_per_month × N / denominator`). The ratio `break_even_V / capacity_per_month(t)` --
which decides `break-even` vs. `infeasible on this hardware` -- is therefore unchanged by `N`. If
one machine cannot break even within its own capacity, no number of identical extra machines can
either; only a cheaper, more efficient or higher-capacity machine changes the verdict. `never` is
unaffected by `N` for the same reason: its denominator does not depend on fixed cost or capacity at
all. This property is checked by a seeded grid of ~200 random parameter combinations in
`tests/test_costmodel.py::test_identical_extra_machines_never_create_a_break_even_seeded_grid`
(the `hypothesis` library is not a project dependency, so the grid is a fixed-seed `random.Random`
sweep instead of property-based generation).

## Shared-machine scenario (router, mixed workload)

The router sends several task kinds to the same local machine. Utilisation across the tasks it
serves locally is

```
utilisation = Σ V_t / capacity_per_month(t)   (over locally served tasks)
machines    = max(1, ceil(utilisation))
```

Above a utilisation of 1, the router's mixed workload needs `ceil(utilisation)` identical machines,
each contributing its own `fixed_usd_per_month`. That total fixed cost is then spread over all
locally served volume, and each task's own energy cost is added on top:

```
shared_fixed_usd_per_task = (fixed_usd_per_month × machines) / Σ V_t   (over locally served tasks)
local_usd_per_task(t)     = shared_fixed_usd_per_task + energy_usd_per_task(t)
```

[`costmodel.shared_machine`](../src/local_enough/costmodel.py) implements this and returns the
utilisation, machine count, the shared fixed rate, and a per-task `local_usd_per_task` dict.

## Sensitivity grid

For the headline task, [`costmodel.sensitivity_grid`](../src/local_enough/costmodel.py) recomputes
`fixed_usd_per_month` and the break-even verdict across `lifetime_years ∈ {2, 3, 4}` ×
`purchase_price multiplier ∈ {0.75, 1.0, 1.25}` (9 cells), holding throughput, energy and cloud
cost fixed. This shows how sensitive the verdict is to two assumptions Andrii chose by hand: how
long the machine stays in service, and how much it actually cost (list price vs. a discount or a
price rise).

## Worked example (illustrative numbers)

These numbers are for illustration only -- they are not the measured reference-run figures, which
come from `docs/cost-model.md`'s companion, the README's break-even table, once Pass A/B/C and
`power-probe` have actually run.

Inputs: `purchase_price_usd=1200`, `allocation=1.0`, `lifetime_years=3`, `idle_watts=5`,
`electricity_usd_per_kwh=0.30`, `ops_usd_per_month=0`; a local model sustaining 450 tasks/hour
(Pass B throughput 500 tasks/hour × a throttle factor of 0.9) at `incremental_watts=20`;
`busy_hours_per_day=8`; a classification volume of `V_t = 50,000 × 0.30 = 15,000` tasks/month; the
cheapest cloud model meeting the bar costs `$0.003`/task.

```
fixed_usd_per_month   = 1200×1/(3×12) + 5×24×30/1000×0.30×1        = 33.333 + 1.080  = $34.41/month
sustained_tasks/hour   = 500 × 0.9                                                    = 450
energy_usd_per_task    = 20 / (450/3600) / 3.6e6 × 0.30                               = $0.0000133/task
capacity_per_month     = 450 × 8 × 30                                                 = 108,000 tasks/month
local_usd_per_task     = 34.41/15,000 + 0.0000133                                     = $0.0023076/task
                                                                                        ≈ $2.31 per 1,000 tasks
break_even_V            = 34.41 / (0.003 − 0.0000133)                                 ≈ 11,522 tasks/month
verdict                 = "break-even" (11,522 ≤ 108,000 capacity)
machines_needed(V_t)    = ceil(15,000 / 108,000)                                      = 1
```

Local becomes cheaper than the cheapest qualifying cloud model above about 11,500 tasks/month for
this task, well inside the ~108,000-task/month capacity of one machine at 8 busy hours/day -- so the
dedicated-scenario verdict for this illustrative task is `break-even`.
