---
icon: octicons/sparkle-16
---

# Skills

Skills let a coding agent, such as Claude Code, do mjswan work for you.

| Skill | What it does |
|---|---|
| [mjlab-to-mjswan](mjlab-to-mjswan.md) | Turns an mjlab task from any repository into a browser app |

## Install

All skills come in one Claude Code plugin, `mjswan`, from the `ttktjmt` marketplace (the
mjswan repository).

=== "Terminal"

    ```
    /plugin marketplace add ttktjmt/mjswan
    ```

    ```
    /plugin install mjswan@ttktjmt
    ```

=== "VS Code"

    Type `/plugins`, add `ttktjmt/mjswan` in the **Marketplaces** tab, then install
    `mjswan`.

=== "Other agents"

    Copy a skill's whole folder from
    [`skills/`](https://github.com/ttktjmt/mjswan/tree/main/skills){:target="_blank"} to
    where your agent loads skills.

Start a skill with `/mjswan:<skill-name>`, or just describe the job. Update or remove the
plugin from the plugin manager (`/plugin` or `/plugins`).
