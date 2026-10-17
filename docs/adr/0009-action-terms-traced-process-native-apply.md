# Action terms: a traced process, a native apply

> Status: **Accepted**. Amends [ADR 0005](0005-onnx-traced-terms-superseding-the-declarative-dsl.md)'s
> Action row ("fully native, no ONNX at all") and §7's Action bullet: a task's own
> action class has its `process_actions` and `reset` traced, in the stateful shape of
> ADR 0005 §3, while every term's `apply_actions` stays native and synchronous. Bumps
> the document `format` from 5 to 6 (ADR 0006 §7): an action entry gains fields an
> older engine would ignore and so run wrong.

## Context

ADR 0005 kept Action a closed, native set mirroring mjlab's action classes, for one
reason: `apply_actions` runs every physics substep, the hottest loop, where an awaited
ONNX call does not belong. The adapter matched a term by class name and skipped
anything else with a warning.

Tasks define their own action classes, and the skipped ones are not cosmetic:

- **In-hand rotation** (`JointPositionDeltaAction`) integrates clipped deltas against
  the previous target, clamps the sum to the soft joint limits, restarts it at the
  measured pose on reset, and with `interpolate_decimation` ramps the PD target from
  the previous step's target to the new one across the substeps.
- **luwu** (`HeldJointPositionAction`) zeroes the action and holds each joint at its
  measured position for `hold_duration_s` after a reset.
- Smoothing terms (an EMA over mjlab's target) inherit mjlab's apply and only change
  the target.

Each of these keeps state across control steps, which a policy wrapper cannot hold
without changing the policy's interface. mjlab already splits a term in two:
`process_actions`, once per control step, turns the raw action into
`_processed_actions` and updates whatever the term keeps; `apply_actions`, once per
substep, turns that into the actuator target. The browser ran both halves in one
function every substep.

mjlab's actuators also delay their command by `ActuatorCfg.delay_*` physics steps,
which the browser did not run (it warned).

## Decision

### 1. Two halves, run where mjlab runs them

`process` runs once per control step, after the policy and before the substeps;
`apply(k)` runs at each substep `k`, synchronously and with no ONNX, as ADR 0005 §7
requires of the hot loop. A native term's process is the formula mjlab's class
computes; a traced term's is a graph, awaited once per control step.

### 2. A task's own class: `process_actions` and `reset` traced

The adapter turns a task's own buildable action cfg mjswan has no class for into a
`TracedActionCfg`; one of mjlab's own classes without one (differential IK) is skipped
with a warning, as before. The build constructs the term on the scene's trace env and
traces both bodies, the term's state promoted to explicit I/O as a stateful command's is:

```
process(raw, slots..., prev_<f>..., [rand]) -> (next_<f>..., processed, [processed_from])
reset(slots..., prev_<f>..., [rand])        -> (next_<f>...)
```

- **State** is every tensor attribute either body writes, found by identity and
  in-place version, with each tensor held so a freed one's id cannot pass for a new
  one's; its initial value is what `cfg.build(env)` leaves. A reset that writes nothing
  has no graph.
- **Env reads** become slots as for any traced term: the joint positions a reset
  restarts from, the episode clock a hold reads.
- **Refused**, with the fix in the message: a body that changes Python state (a
  counter, an attribute it creates, a deque of past actions), writes the sim (through
  any entity, or into an entity's buffers), or reads a live mjlab object other than its
  entity cached at init (its data, the scene, the sim, the env, a sensor); two state
  fields that share one tensor, which the browser would carry apart; a slot the
  browser's reader does not serve; a reset that reads the episode clock, which the
  browser zeroes before its reset terms and mjlab after them; a term whose action width
  differs from the joints it drives, or that names no joints.
- **Verified before it ships**: both graphs run against the term's own bodies over
  random actions, resets and episode clocks, and a disagreement fails the build. A
  boolean-mask row write (`x[mask] = ...`) traces as a `where`, so it keeps working
  when the mask changes.

### 3. Every term's `apply_actions`: native, named rather than traced

Tracing apply would put an awaited graph on every substep. Instead the entry names
what it does, from a small vocabulary the browser implements:

| Field | What apply does each substep | From |
|---|---|---|
| `type` | `joint_position` (target minus encoder bias), `joint_position_relative` (measured position plus the action, mjlab's `RelativeJointPositionAction`), `joint_velocity`, `torque`, and the existing reference, command and muscle kinds | the class |
| `interpolate` | ramps from `processed_from` (the processed action before this step) to `processed`, `alpha = (k + 1) / decimation` | in-hand rotation |
| `hold_s` | while `float32(episode_length * step_dt) < hold_s`, the target is the joint's measured position | luwu |
| `delay` | per mjlab delay buffer (one actuator class and delay setting): its `DelayBuffer`, lag in `[min_lag, max_lag]` physics steps with `hold_prob`, `update_period`, `per_env_phase` | the actuator cfgs |

A class that inherits `apply_actions` from mjlab's position, relative, velocity or
effort action gets that kind with no binding. One that overrides it needs
`mjswan.register_action(cfg_name, ActionBinding(apply=..., hold_s=..., interpolate=...))`,
and the build refuses it without one. The binding's fields may be callables of the
task's cfg, so `hold_s=lambda cfg: cfg.hold_duration_s` follows the task. `interpolate`
and `hold_s` act on a position target, so the build refuses them on a velocity or
torque apply. The build looks the binding up itself, so it may be registered before or
after `add_policy`.

### 4. Command delays for every term

The build reads `delay_*` off the actuator cfgs as it reads the PD gains, for every
kind of term. mjlab zeroes the targets at a reset and writes them once before the
next step, so a delayed actuator acts on zero for its first `lag` substeps; the
browser starts each delay's history from that zero frame.

### 5. Observations that read a term's internals

An observation may read `get_term(name).<field>`: a traced term's state field, or any
term's `_processed_actions`. It is served as a new slot, `{action_term, field}`, fed as
float32 whatever the field's dtype. Any other internal raises, as does a property that
reads the term's entity: the browser holds no copy of either, so its value would bake.
The build serializes the actions before the observations, so it knows which fields a
traced term carries.

### 6. Document format 6

An action entry may carry `interpolate`, `hold_s`, `delay`, `process` and `reset`, and
`type` may be `joint_position_relative`; an input slot may be
`{action_term, field}`. A format-5 engine would run such a term as a plain position
term, so `DOCUMENT_FORMAT` is 6.

## Considered options

- **Trace `apply_actions` too.** Rejected: an awaited ONNX run per substep, against
  ADR 0005 §7, for behavior that in every task found reduces to a handful of shapes.
- **A wrapper around the policy.** It can integrate a delta, but not ramp across the
  substeps, hold per substep, or delay a command, and it changes the policy's
  interface.
- **Native classes for each task's term.** Every task would need an mjswan release,
  and the term's arithmetic would drift from the task's own code. The trace runs the
  task's code.

## Consequences

- In-hand rotation and luwu-style tasks build. `tests/dump_action_fixture.py` steps
  such terms in mjlab and `actionParity.test.ts` replays them through the real
  process graph and native apply, substep by substep; the real in-hand rotation term,
  with its trained policy and a 10-step delay, matched mjlab over 300 steps and seven
  resets.
- An apply outside the vocabulary (one reading a sensor per substep, writing effort
  and position together) is refused at build time. Widening the vocabulary is an
  mjswan change.
- A traced term costs one awaited graph run per control step.
- Randomized delay lags are drawn from the browser's seeded RNG: mjlab's distribution,
  not its stream. The browser keeps one delay per action term, so joints of two terms
  behind one mjlab delay buffer draw their lags apart; the build warns when that lag is
  random. Lags an event sets through `Actuator.set_lags` are not run.
- ADR 0005 §3's stateful *event* terms remain unimplemented: an event's trace carries
  no state.
