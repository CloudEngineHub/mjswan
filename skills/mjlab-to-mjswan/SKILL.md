---
name: mjlab-to-mjswan
description: Port an mjlab task to mjswan: turn a local or remote GitHub repo that registers mjlab tasks into a browser app. Use when the user wants to run, visualize, or deploy an mjlab task (theirs or a third party's) in the browser with mjswan.
---

Port one mjlab task from a target repo into a browser app built by [mjswan](https://github.com/ttktjmt/mjswan). The target is any repo that registers mjlab tasks, given as a local path or a GitHub URL.

**Scope.** Inside the target repo everything you write stays on the Python side: no TypeScript (`ts_src`), no `ViewerConfig`. mjswan itself is edited in exactly one case: a missing capability that is generic rather than task-specific, which step 8 turns into a pull request.

## Ground rules

- mjswan reimplements none of mjlab's term functions: a task's real observation / termination / event / command functions are traced to ONNX at build time and run in the browser beside the policy. A fix therefore means *making mjlab's own function traceable*, never rewriting its math.
- When something fails, read the exception in full, then the module it came from in the installed package: `python -c "import mjswan, pathlib; print(pathlib.Path(mjswan.__file__).parent)"`. That source is the only current truth.
- Two fix attempts per term, then record it as skipped and move on. A partial port that builds and passes parity is the deliverable; all-or-nothing is not.

## 1. Acquire the target

A GitHub URL → `git clone <url> ./<repo-name>`, then work inside it. A local path → use it as is.

Then make mjswan importable from the **same interpreter** that can import the target's task registrations. What this pipeline needs is the `mjlab` extra (mjlab + torch: `add_scene_mjlab`, term tracing, `.pt` conversion) plus `onnxruntime`, which step 7 runs the exported graphs under:

- Repo already has an environment → add to it: `uv pip install "mjswan[mjlab]" onnxruntime`.
- Fresh clone with nothing → `uv venv && uv pip install -e . && uv pip install "mjswan[mjlab]" onnxruntime`.
- No `uv` available → install into the interpreter that already runs the target, against its own `sys.prefix`.

Every source is its own extra and `import mjswan` touches none of them, so install only the one the checkpoints come from: `wandb` for a W&B run path, `hf` for anything fetched from the Hub, e.g. `"mjswan[mjlab,wandb]"`.

mjswan pins `mujoco` **exactly**, its `mjlab` extra pins `mjlab` **exactly**, and the `mjlab` extra also brings mjlab's own bounded `requires-python` with it, so the interpreter mjswan alone accepts is not always the one this pipeline accepts. A target that pins either package differently will not co-resolve. If the install fails on one, **stop and report the resolver's output verbatim**: resolving it is the user's call, not yours.

## 2. Find the task ids

Registration happens as an import side effect, and every repo does it differently: `mjlab.tasks`, a `<pkg>/tasks/__init__.py`, a `src/tasks/` on `sys.path`, or a `bootstrap_*()` function that must be *called*, sometimes only after an env var or a data file is in place.

Look at the repo's own README run instructions first, then grep for modules touching `mjlab.tasks.registry` or calling `register`. Confirm each candidate by diffing the registry around the import:

```sh
python - <<'PY'
from mjlab.tasks.registry import list_tasks
before = set(list_tasks())
import your_repo.tasks             # or: from ... import bootstrap_fn; bootstrap_fn()
after = set(list_tasks())
print(f"{len(before)} already registered, {len(after - before)} new")
print(sorted(after - before))
PY
```

An empty diff means the candidate registers nothing of its own: wrong candidate, or its registration needs a call and arguments you have not supplied yet.

Show the discovered task ids to the user and let them pick one. Ask **here**, not later, for anything the registration itself needs: env vars, clip or dataset paths, gated-dataset access, credentials.

## 3. Pre-flight, seconds before any build

`load_env_cfg` only reads config, so use it to find in seconds what a full build would take minutes to reach:

```sh
python - <<'PY'
from mjlab.tasks.registry import load_env_cfg
from mjswan.mjlab import adapt_actions
import your_repo.tasks
cfg = load_env_cfg("Mjlab-Velocity-Flat-Unitree-G1", play=True)
for name, term in (adapt_actions(cfg.actions) or {}).items():
    print("action", name, type(term).__name__, term.unsupported_reason)
PY
```

A task's own action class, one mjswan has no class of its own for, is traced (ADR 0009): `adapt_actions` returns a `TracedActionCfg` for it, and the build traces its `process_actions` and `reset` against the trace env and checks both graphs against the term's own code. Its `apply_actions` runs natively in the browser. A class that inherits it from mjlab's position, relative, velocity or effort action needs nothing more; one that overrides it is refused until the port names what it does, reading the options off the task's cfg rather than restating them:

```python
mjswan.register_action(
    "JointPositionDeltaActionCfg",
    mjswan.ActionBinding(
        apply="joint_position", interpolate=lambda cfg: cfg.interpolate_decimation
    ),
)
```

`ActionBinding` covers a position, relative, velocity or torque target, a ramp across the substeps (`interpolate`) and a hold of the measured pose after a reset (`hold_s`). An apply outside that, a trace the build refuses (its message names the cause, such as a Python counter or a sim write), or a term with a non-`None` `unsupported_reason` makes the task unportable as configured. A refused body can sometimes port through `ActionBinding(trace_override=...)`, which swaps a trace-friendly body onto the built term; otherwise report the reason verbatim, then go to step 8, which decides whether mjswan should learn it. Never hardcode which action classes those are; the build's messages and the `unsupported_reason` field are the source of truth.

A warning that a term **sets** fields mjswan's class does not have, or that an actuator sets what the browser does not run (anything past PD within the effort limit and the command delay: a torque-speed curve, friction), names settings the browser drops while the term itself ports. Each is a behaviour gap: report it verbatim beside the skipped terms. A generic one goes to step 8 like a skipped term; one the model can carry instead, such as an actuator's friction as the joint's `frictionloss`, moves there. A lag an event sets through `Actuator.set_lags` (older mjlab's `sync_actuator_delays`) does not trace: drop the event, and give the actuator cfg the lag range itself when the delay matters.

One shape still ports by hand: a term that takes no column of the policy's output (`action_dim == 0`) and writes joint targets of its own, such as an arm-motion generator beside a locomotion policy. Drop it, and the terms that read it, from the env config. Then pass a `CommandJointPositionActionCfg` for its joints, and the command it reads, to `add_policy(actions=..., commands=...)`, never to the env config, whose terms mjlab builds itself. The command can be a `ui_command` of sliders, or a traced command that reproduces the dropped term's motion.

### Take the configuration from mjlab

A feature that makes the author restate what mjlab already declares is the wrong design. mjlab's config is the source; mjswan reads it.

- A new mjlab action class: name the mjswan cfg class exactly as mjlab names its own, or add the mapping to `_ACTION_CLASS_ALIASES` in `mjlab/action.py`. `_adapt_action_cfg` then copies every matching dataclass field by itself and the port needs no extra argument. A task's own class needs neither: it is traced.
- Anything the task's env config or runner config already carries (`env_cfg`, `resolve_runner_defaults`) is read from there, never restated at the call site.
- A keyword argument the author has to pass by hand is the last resort, not the first.

### Where the change goes

Find the neighbouring case in the mjswan tree first: each already carries a worked example of the thing being added. An action apply `ActionBinding` cannot name is the one exception to the no-TypeScript rule at the top of this file, because apply runs natively every physics substep (ADR 0009) and needs an engine-side branch in `applyAction.ts` as well as a binding field or cfg class; that prohibition covers per-port `ts_src`, not mjswan's own engine.

### Change discipline

- Follow mjswan's `AGENTS.md`.
- One generic unit test at `tests/test_<feature>.py`; `tests/test_muscle_action.py` is the precedent. Test the feature directly. Do not add a demo scenario, a golden file, or the target's task to `examples/`.
- `make format && make type && uv run pytest -m "not slow"` before pushing.

### Open it

```sh
gh repo fork ttktjmt/mjswan --clone /tmp/mjswan-pr
cd /tmp/mjswan-pr && git checkout -b <branch>
# implement, test
gh pr create --repo ttktjmt/mjswan --title "<title>" --body "<body>"
```

The body names the task and term that motivated the change, states which package defines that term (the generality evidence), and says what the change does not cover.

The port stays blocked until the PR lands. Finish every other part of it, and report this one as waiting on the PR, with its URL.

## 9. Report

Write `mjswan_app/README.md`: the target's source URL, the task id, prerequisites (credentials, env vars, data files), the run command, and how the build is handed on — `app.save_document()` writes `mjswan_app/dist.swn`, which `mjswan serve`, `mjswan info` and `mjswan publish` each accept in place of the directory.

Then tell the user, in this order:

1. the `mjswan info` tree, then terms traced and parity result (`report.summary()` verbatim if anything failed);
2. terms skipped and settings dropped, each with what unblocks it: an author-side `terms.py` replacement, a policy wrapper, a `ts_src` TypeScript class (out of scope here), or the mjswan PR you opened in step 8, with its URL;
3. anything you stopped on: dependency conflicts, missing credentials, task-specific gaps that mjswan must not absorb;
4. that browser behaviour is unverified by this pipeline: parity proves the graph matches mjlab, but mjlab integrates with `mujoco_warp` while the browser runs MuJoCo's WASM build, and a policy can behave differently under the two.
