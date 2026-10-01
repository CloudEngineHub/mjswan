---
icon: octicons/workflow-16
---

# mjlab-to-mjswan

Give it a repository that defines [mjlab](https://github.com/mujocolab/mjlab){:target="_blank"}
tasks, local or on GitHub. A coding agent ports one task, with its trained checkpoints, into
a browser app.

!!! tip
    For mjlab's own tasks with checkpoints on W&B or the Hugging Face Hub,
    [`Builder.from_mjlab`](../guides/mjlab.md#1-one-liner-builderfrom_mjlab) does this in one
    line of Python.

## Usage

1. [Install](index.md#install) the `mjswan` plugin.
2. Run the skill on a GitHub URL or a local path:

    ```
    /mjswan:mjlab-to-mjswan https://github.com/mujocolab/g1_spinkick_example
    ```

3. Pick a task, and answer the agent's questions.
4. Open the app, from the repository root:

    ```sh
    python -m mjswan_app.main
    ```

## Requirements

- `git` and Python 3.10 to 3.13. The agent installs mjswan itself. No GPU needed.
- Checkpoints: a W&B run, a Hugging Face repository, or `model_*.pt` files.

!!! warning
    If the repository pins other `mujoco` or `mjlab` versions than mjswan, the agent stops
    and leaves the choice to you.

## What the agent does

1. Finds the repository's tasks, and asks you to pick one.
2. Checks, in seconds, that mjswan supports the task's actions.
3. Converts the checkpoints to ONNX.
4. Writes `mjswan_app/` and builds it, fixing what fails.
5. Checks that the converted code computes the same numbers as mjlab (parity check).
6. Opens a pull request to mjswan, only if mjswan lacks something other tasks need too.
7. Reports what was ported and what was skipped.

!!! warning "Watch it before you share it"
    The parity check compares numbers, not behavior. mjlab trains with `mujoco_warp`, the
    browser runs MuJoCo's WebAssembly build, and a policy can behave differently.

## Output

```
mjswan_app/
├── main.py      the build script
├── terms.py     rewritten terms, only if some failed to convert
├── README.md    how to run it
└── dist/        the built app
```

To share `dist/`, see [Deployment](../guides/deployment.md) or
[Publishing to Cloud](../guides/publishing.md).
