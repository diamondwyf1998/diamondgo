# Agent Handoff

This file records coordination notes for Codex agents working on DiamondGo.

## GitHub

- Repository: https://github.com/diamondwyf1998/diamondgo
- Default branch: `main`
- Local branch `master` tracks `origin/main`.
- Local `origin` uses SSH over port 443:
  `ssh://git@ssh.github.com:443/diamondwyf1998/diamondgo.git`
- A GitHub SSH key titled `diamondgo-codex-20260605` has been added for Codex push access.
- Do not store SSH private keys, passphrases, or GitHub credentials in this repository.

## Current Commit Order

- `0e47ab5` Initial DiamondGo source snapshot
- `e49cbac` Add evaluation replay dashboard
- `936f629` Add multi-worker training loop

## Work Ownership

- Evaluation, reporting, dashboards, result browsing, and migration packaging are owned by the evaluation/display agent.
- Training-loop changes, MCTS performance work, and multi-worker training changes are owned by the training agent unless the user explicitly transfers that work.
- Before editing, run `git status --short` and avoid overwriting uncommitted work from another agent.

## Evaluation Dashboard

`src/diamondgo/eval_checkpoints.py` writes these files during evaluation:

- `results.json`
- `report.md`
- `dashboard.html`

The dashboard supports checkpoint match selection, game selection, move replay, root value display, and top search candidate display.

Current server dashboard:

`/root/diamondgo/artifacts/eval-every-50-vs-initial/dashboard.html`

## Server Migration Artifacts

The latest source migration artifacts on the server are:

- `/root/autodl-tmp/diamondgo-migration/diamondgo-source.bundle`
- `/root/autodl-tmp/diamondgo-migration/diamondgo-source.tar.gz`

Regenerate these after important source commits if the user may switch servers.
