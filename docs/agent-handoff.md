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

## Current Training/Evaluation Defaults

- Default 9x9 model-input komi is `0.5`; the observation plane still uses `komi / 10.0`.
- Default 9x9 scoring komi is `6.5`; terminal winner, value targets, score margins, and SGF `KM` use `score_komi`.
- Old checkpoints that do not include `score_komi` are evaluated with their serialized `config.komi` unless a new run explicitly sets `--score-komi`.
- Default 9x9 self-play cap is `120` moves.
- `src/diamondgo/eval_suite.py` runs standard checkpoint matches at step tiers such as `50,200,500` against `initial` and `previous`.
- `src/diamondgo/tactical_eval.py` runs fixed capture/atari probes and reports target top-1/top-3/rank.
- `multiworker_train` behavior metrics include black/white win rates, color-bias alerts, and signed score-margin summaries.

## Server Migration Artifacts

The latest source migration artifacts on the server are:

- `/root/autodl-tmp/diamondgo-migration/diamondgo-source.bundle`
- `/root/autodl-tmp/diamondgo-migration/diamondgo-source.tar.gz`

Regenerate these after important source commits if the user may switch servers.
