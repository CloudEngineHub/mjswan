# ONNX-traced MDP term bodies, mjlab-native runtime internals

> Status: **Accepted (design)** — §1 ("no new `manifest.json`") is superseded, and §2's PRNG
> rule and §4's graph-reference base amended, by
> [ADR 0006](0006-swn-simulation-document.md). Supersedes the *term-body representation* of
> [ADR 0003](0003-declarative-mdp-terms-alongside-custom-js.md) and preserves the
> *external engine API* of [ADR 0004](0004-headless-engine-core.md). This is a
> pre-1.0 rewrite of the runtime's MDP-execution layer with no
> backward-compatibility constraint. It replaces the declarative JSON
> composition-graph DSL with term bodies **traced to ONNX at build time and run
> by ONNX Runtime Web** (already shipped for policy inference), and realigns the
> in-browser internals to mirror mjlab's manager/`envs.mdp` layout. Scope,
> security posture, and the five-manager restriction are inherited from ADRs
> 0003/0004 and are **not** relitigated here — only the representation changes.
>
> **Superseded in part by [ADR 0006](0006-swn-simulation-document.md):** §1's
> *"No new `manifest.json`"* no longer holds — the build writes one
> `manifest.json` at the document root. ADR 0006 also amends §2 (PRNG behaviour
> at an MDP switch) and §4 (graph references resolve against the scene
> directory). §3 and §5–§9 stand as written.
>
> **Amended by [ADR 0009](0009-action-terms-traced-process-native-apply.md):** Action
> is no longer fully native. A task's own action class has its `process_actions` and
> `reset` traced in §3's stateful shape; every term's `apply_actions` stays native and
> synchronous, as §7 and §8 require. §7's `relative_joint_position` is implemented as
> `joint_position_relative`; its tendon and site kinds are still not run. §3's stateful
> Event terms were never implemented: an event's trace carries no state.

## Context

ADR 0003 chose to represent MDP term bodies as a **declarative composition graph
(JSON DAG)** traced from mjlab-style Python functions, interpreted in the browser
by a tiny hand-written interpreter (`core/dsl/interpreter.ts`, ~80 lines) over a
closed primitive registry (`core/dsl/primitives.ts`, ~635 lines). ADR 0004 lifted
the engine out of React into a pure-TS `createEngine`, with a stable
bytes-in/snapshot-out API and an app-owned catalog parsed by `mjswan/manifest`.

The DSL delivered on its security thesis (term bodies are effect-free data, no
`eval`, no per-simulation code, set-membership validation for Cloud). But it
carries a structural maintenance tax that grows with coverage:

- **Every primitive must be reimplemented in TypeScript and kept numerically in
  lockstep with mjlab/torch.** `primitives.ts` is a second, hand-written copy of
  the math mjlab already defines. Divergence is a live risk, which is exactly why
  `scripts/verify_dsl_migration.py` (op-parity/envelope structural check) and a
  vitest numeric harness exist. Adding a new mjlab data field or op is an engine
  PR plus a parity re-check.
- **The primitive set is closed.** A term that needs an op the registry lacks
  cannot be expressed until the op lands in the engine — the "PR a primitive"
  on-ramp of ADR 0003 §Consequences.

`torch.onnx.export` removes both: the traced graph **is** the computation, run by
a standard, already-audited runtime (ONNX Runtime Web, already loaded for the
policy — `core/policy/OnnxModule.ts`, `executionProviders: ['wasm']`). There is no
second implementation of the math to maintain and no closed op registry to grow —
whatever torch can express and export, the browser can run. The effect-free /
Cloud-safe property is preserved (an ONNX graph is pure data interpreted by a
fixed runtime; ADR 0004 §10's *effectful-vs-effect-free* line still holds — ONNX
term bodies fall on the effect-free side, same as the DSL did).

mjswan remains an **inference/playback** system (ADR 0003 scope): Reward,
Curriculum, Metrics, Recorder are out; the five managers are Observation,
Termination, Action, Event, Command; "reset" is `EventTermCfg(mode="reset")`, not
a sixth manager.

## Decision

Replace the declarative DSL representation of term bodies with ONNX, restricted
to the five managers, with a strict execution split: each manager's **term body**
(author-written logic) is traced to an ONNX graph at build time and run by ONNX
Runtime Web; each manager's **orchestration** (triggers, scheduling, aggregation,
noise/scale/clip/history pipelines) is native TypeScript. **Action is fully
native — no ONNX at all** (unchanged from ADR 0003: closed built-in set, hottest
loop, once per physics substep).

| Manager | Orchestration (native TS) | Term body |
|---|---|---|
| Observation | concatenate, noise, clip, scale, history — port of `ObservationManager.compute_group` minus `func()` | **ONNX** (fused, §Fusion) |
| Termination | OR-reduce across terms; `time_out` split out | **ONNX** (fused, §Fusion) |
| Action | full — `ActionKind` enum + `process`/`apply`, synchronous | **none** |
| Event | `EventManager.apply()` timer/mode logic (interval countdown, reset gating, startup-once, global vs per-instance) | **ONNX**, called only when the native trigger fires |
| Command | `CommandTerm.compute()` resample timer | **ONNX**, `(prev_state, resample_mask, rand) → (next_state, command)` |

### 1. Representation, not artifacts — extend `config.json` + `policy.json`

**No new `manifest.json`.** The build reuses the existing artifact contract
(`config.json` + per-policy `policy.json`, parsed by `mjswan/manifest`); only the
**DSL portion is rewritten for ONNX**. A term that was serialized as a DSL
envelope

```json
{ "kind": "observation", "nodes": [...], "output": "n7" }
```

becomes an ONNX reference with the native pipeline metadata alongside it:

```json
{
  "name": "base_ang_vel",
  "onnx": "obs/base_ang_vel.onnx",
  "source_url": "git://.../velocity_env_cfg.py#L82",
  "noise": { "type": "uniform", "min": -0.2, "max": 0.2 },
  "scale": 1.0, "clip": null, "history_length": 0
}
```

Actions keep their existing declarative descriptor (`kind`/`target_ids`/`scale`/
`offset_mode`/`clip`) — no ONNX, no change. `mjswan/manifest` continues to parse
the same top-level shape; `PolicyInput.config` stays an opaque parsed object the
engine interprets. The `uses_custom_js` flag and the declarative-only Cloud policy
(ADR 0003) carry over unchanged: an ONNX term is effect-free and Cloud-safe.

### 2. Determinism — orchestrator-owned seeded PRNG, `rand` as explicit input

All randomness inside Event/Command term bodies is generated by a **seeded PRNG
in the TypeScript orchestrator** and passed to the graph as an explicit `rand`
input tensor. This is **for bit-for-bit session replay** (the "share this exact
run" viewer feature): the seed is owned and persisted by the orchestrator, so a
recorded rollout replays identically later.

This is *not* framed as "ONNX random ops are non-deterministic and must be
avoided" (the earlier rationale). Matching mjlab's own random draws is neither
required nor well-defined — torch's RNG is not bit-portable across CPU/CUDA, so
mjlab guarantees only seeded reproducibility *within a fixed setup*, not portable
determinism. mjswan offers its **own** replayable-session guarantee, which is
strictly cleaner, and threading `rand` explicitly is the right design for it
regardless (it also happens to sidestep ONNX `RandomUniform`/`RandomNormal`
cross-EP/version divergence, since the ONNX spec does not mandate a PRNG
algorithm). Do **not** use ONNX's random ops.

### 3. Stateful terms — hidden state promoted to explicit `(prev_state) → (next_state, …)`

Statefulness does not preclude tracing. A stateful term is exported exactly like
an RNN cell or a KV-cache: its hidden state is promoted to an explicit input and
output, and the **native orchestrator holds it across frames** (the same way it
already holds `time_left` for the resample timer).

- **Command:** `forward(prev_state, resample_mask, rand) → (next_state, command)`.
  `_resample_command` runs under a `torch.where` gated by `resample_mask`;
  `_update_command` always runs. For `UniformVelocityCommand`, `prev_state` is the
  handful of small tensors it carries (`vel_command_b`, `heading_target`,
  `is_heading_env`, `is_standing_env`, …) — all trace once threaded explicitly.
  **Reset unifies to `resample_mask = true`** (no separate reset code path).
- **Stateful Event terms** use the same shape where needed.
- The genuinely non-DAG-able parts are handled natively for unrelated reasons, not
  because a term is "a class": the **joystick GUI override** is viewer
  interactivity (native, §7), and the **metrics dict** is training-only logging
  (dropped, out of scope). mjlab modeling commands as classes is not evidence
  against tracing — `self.foo` is just Python's idiom for recurrent state.

**Schema consequence:** `policy.json` must declare each stateful term's state
tensors — **names, shapes, and initial values** — so the orchestrator can
allocate, persist, and reset them (reset = `resample_mask = true`). Initial state
(e.g. `vel_command_b` = zeros; masks sampled at first resample) is captured at
export time.

### 4. Fusion reduces graph count, not call frequency

Per-manager fusion (ADR 0003/proposal §10) is a **build-time** optimization that
fuses a manager's terms into one graph, cutting ORT-Web `run()` calls from
O(terms) to O(1) per manager. The governing principle:

> **Fusion changes how many graphs exist, never how often they are called.**
> The native trigger logic still gates execution; a manager that fires on some
> frames still runs `ort.run()` only on those frames, fused or not.

- **Observation and Termination: fusion mandatory (v1).** Both run every control
  step and are on the policy's critical path (observation feeds the policy), so
  they must be one graph each. Per-call overhead (JS↔WASM crossing, tensor
  setup) recurs every frame and dominates for small graphs; fusion is what keeps
  the per-frame ORT cost at ~2 `run()` calls (obs + term) on top of the policy's.
- **Command: may fuse (low priority).** It is called every frame regardless, so
  fusing its (typically one) term changes little; reuse the same mechanism when
  convenient.
- **Event: fuse per `mode`, but calls stay gated.** Fuse the terms within a mode
  into one graph, yet **on a frame where the native trigger does not fire,
  `ort.run()` is not called at all.**

Do **not** fuse the observation graph into the policy graph — it would collapse
obs+policy to one `run()` but breaks the "`policy.onnx` is the trained artifact"
separation and complicates command/joystick injection.

Session hygiene: create sessions once at load and cache them (`onnx/` session
manager); prefer the WASM EP for these small graphs (WebGPU's upload/download
round-trip loses); pre-allocate input tensors and write in place per frame rather
than re-allocating (the current policy path re-allocates every frame —
`runtime.ts` — which the new path should avoid).

### 5. Port trigger *semantics*, not the tensor implementation — N=1

mjlab's `(num_envs,)` tensors, `.nonzero()`, and `env_ids` masks exist to make
thousands of environments efficient in one GPU kernel launch during training.
**v1 targets strictly a single environment (N=1)**, where none of that payoff
exists — importing it verbatim is complexity tax, not faithfulness.

Port the *semantics*: interval countdown, resample-time sampling,
`min_step_count_between_reset` gating, global-vs-per-instance timers. In the
orchestrator, `time_left` is a plain number, `fired`/`resample` are plain
booleans, and reset is `if (done) reset()` — **not** `doneMask.nonzero() →
resetIds`.

ORT-Web's I/O is tensor-shaped even at batch 1, so each `ort.run()` has a
one-line wrap/unwrap. **That tensor shape must not leak past the `onnx/`
boundary** — the orchestrator deals only in scalars and short fixed-size arrays.
(If a small demo grid is ever wanted, the answer is N independent engine
instances per ADR 0004 §2, not a batched loop — each instance keeps scalar
state.)

### 6. dtype at the WASM↔ONNX boundary

MuJoCo's `mjtNum` is float64; ONNX graphs are float32. The state read from WASM
MuJoCo (`collectRawState()`) converts the float64 buffer to a `Float32Array`
**once**, at that read site, before feeding any graph. float32 is sufficient for
playback.

### 7. Internals mirror mjlab; external API preserved

- **Preserved (ADR 0004, the product API):** `createEngine` / `MjswanEngine`,
  the `subscribe` snapshot, `camera` / `commands` verbs, `mjswan/manifest`,
  bytes-in delivery, multi-instance, `dispose`. This surface may be improved but
  is not discarded.
- **Rewritten (internal MDP-execution layer):** the DSL interpreter and
  `Dsl{Observation,Termination,Event}` are replaced by ONNX-graph-backed
  managers. The TypeScript internals are **realigned to mirror mjlab's layout**
  (`envs/mdp/` + `managers/`), matching the Python side. The proposal's
  `runtime/{physics,onnx,managers,engine.ts}` is the *implementation behind*
  `createEngine`, not a replacement for it.
- **Action** stays a native `ActionKind` enum mirroring
  `mjlab.envs.mdp.actions` (`joint_position`, `relative_joint_position`,
  `joint_velocity`, `joint_effort`, `tendon_*`, `site_effort`). `process(raw) =
  clamp(raw*scale + offset, clip)` is shared; only the sim-write differs per kind;
  `joint_position` subtracts `encoder_bias` before writing. `applyAction()`
  contains **no `await`/async ONNX call** — synchronous, once per substep.

### 8. Runtime step loop (authoritative)

```ts
// one frame = one policy step
const obs = await ortObs.run(collectRawState());   // fused obs graph (§4)
const currentObs = nativePostprocess(obs);         // noise/clip/scale/history/concat
const policyOut = await ort.run("policy.onnx", { obs: currentObs });
actionNative.processAction(policyOut);

for (let i = 0; i < decimation; i++) {
  actionNative.applyAction(i);   // sync, no ONNX
  wasm.step();
}

const done = (await ortTerm.run(collectRawState())).done;   // fused term graph
if (done) nativeReset();                                    // fires mode="reset" events

wasm.forward();   // exactly once, mirroring mjlab's ManagerBasedRlEnv.step() ordering

if (eventTrigger.fired(dt)) {                    // native gate; no run() otherwise
  const { delta, recomputeLevel } = await ortEvent.run({ state: collectRawState() });
  nativeApplyEventDelta(delta);
  if (recomputeLevel !== "none") wasm.recomputeConstants(recomputeLevel);
}

const resampleMask = commandTrigger.tick(dt);    // scalar/boolean
const { nextState, command } = await ortCommand.run({
  state: collectRawState(), prevState: cmdState, resampleMask, rand: prng.next(),
});
cmdState = nextState;                            // orchestrator holds state (§3)
// joystick override, if present, replaces `command` here — native (§3)

render(wasm.data);
```

`wasm.forward()` is called **exactly once** (not after the decimation loop *and*
after reset); this mirrors mjlab's own `step()` ordering and must not be
"optimized" into two calls. Tensor wrap/unwrap lives inside each `ortX.run`
wrapper (§5), not here.

### 9. Traceability and non-traceable terms

- Every `.onnx` carries a `source_url` (or content hash) in `policy.json`,
  pointing back to the Python term function it was traced from.
- If `torch.onnx.export` cannot trace a term (data-dependent control flow, ragged
  structures, string manipulation), the build **fails loudly and names the term**
  — never force-traces, never ships source to the browser. Such a term is
  reimplemented as reviewed native TypeScript merged into the engine (the same
  treatment Action gets, and the trusted-only escape hatch of ADR 0004 §10).

## Considered options

- **Keep the declarative JSON DSL (status quo, ADR 0003).** Rejected: forces a
  hand-written TS reimplementation of every op kept numerically in sync with
  mjlab (the `primitives.ts` + `verify_dsl_migration.py` + vitest-parity tax), and
  a closed primitive set that gates new mjlab fields behind engine PRs. ONNX makes
  the traced graph the computation, removing both.
- **ONNX random ops (`RandomUniform`/`RandomNormal`).** Rejected: no replayable
  seed the orchestrator can persist, and cross-EP/version divergence (spec does
  not mandate a PRNG). Explicit `rand` input (§2) is strictly more controllable.
- **A new `manifest.json`.** Rejected: duplicates the existing `config.json` +
  `policy.json` + `mjswan/manifest` contract. Rewrite only the term representation
  inside it (§1).
- **Port mjlab's batched multi-env tensors line-for-line.** Rejected: GPU-scale
  machinery with no payoff at N=1; port the semantics with scalar state (§5).
- **Command as a native/class-only term (no tracing).** Rejected: stateful is not
  un-DAG-able — promote state to explicit I/O and hold it natively (§3).
- **Fuse the observation graph into the policy graph.** Rejected: breaks the
  trained-artifact boundary and complicates command/joystick injection (§4).

## Consequences

- **Removed:** `src/mjswan/dsl/` (`trace.py`, `env.py`, `ops.py`, `node.py`,
  `event.py`), `template/src/core/dsl/` (`interpreter.ts`, `primitives.ts`,
  `types.ts`, tests), `core/observation/DslObservation.ts`,
  `core/termination/DslTermination.ts`, `core/event/DslEvent.ts`, and
  `scripts/verify_dsl_migration.py` (superseded by the numeric parity harness
  below).
- **Build path:** the term-serialization step in the Builder/`policy.py` path is
  rewritten to wrap each term body in a small `torch.nn.Module` and trace it with
  `torch.onnx.export` (dynamic batch axis, static everything else), resolving every
  `SceneEntityCfg` regex to static indices at trace time (as mjlab's managers do at
  `_prepare_terms()`), and baking those indices as graph constants. The policy
  network reuses mjlab's existing `export_policy_to_onnx` path — no second
  policy-export mechanism. `onnxruntime` becomes a build-time parity dependency;
  `onnxruntime-web` (already present) now runs term bodies too, not just the policy.
- **Mandatory validation:** a numeric-parity harness runs N steps through the real
  mjlab Python env and, in parallel, feeds the same states through the exported
  `.onnx` graphs via `onnxruntime` (not `torch`), asserting `allclose` within
  tolerance for every term's output every step. A term that fails to trace fails
  the build.
- **Config validation** is unchanged from ADR 0003: reject any task config
  referencing `reward`, `curriculum`, `metrics`, or `recorders`.
- Cloud stays declarative-only and effect-free-safe; an ONNX term is effect-free
  data, so the ADR 0003/0004 security posture is preserved, not reopened.

## Phased execution plan

1. **Build path + parity harness**, targeting mjlab's Cartpole end-to-end in
   Python only. Exit: exported ONNX graphs match the live mjlab env within
   tolerance for every term, every step.
2. **WASM physics + native Action module + step-loop skeleton.** Cartpole headless
   in Node against the exported bundle; compare trajectories to the mjlab
   reference.
3. **Observation/Termination/Event/Command ONNX wiring in-browser** (fused per §4),
   behind the existing `createEngine` API. Cartpole interactive in a browser tab.
4. **Port a locomotion task** (Unitree G1 velocity-flat) to stress Event/Command
   trigger timing and the noise/history observation pipeline at a larger term
   count.
5. **Viewer UX:** command override via on-screen joystick (native, replaces the
   command tensor for that frame — §3/§7), debug-visualization parity where
   feasible in WebGL.

## Acceptance criteria

- [ ] Cartpole: an N-step browser rollout matches the Python mjlab reference within
      tolerance on observations and termination flags, given identical initial
      state and action sequence.
- [ ] A recorded session replays bit-for-bit from its persisted PRNG seed (§2).
- [ ] G1 velocity-flat: interactive playback sustains target frame rate on
      reference hardware/browser.
- [ ] No term's Python source ships to the browser as executable text (audit: grep
      the built bundle — it must not be found).
- [ ] `config.json` / `policy.json` validate against their schema; the build
      rejects any task config referencing `reward`, `curriculum`, `metrics`, or
      `recorders`.
