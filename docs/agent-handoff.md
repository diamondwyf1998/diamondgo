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

## Human-Readable Records

- `docs/training-notes.md` is for people first. Keep it readable: important
  phenomena, operations, interpretations, next actions, and pointers to evidence.
- Put raw tables, metric snapshots, artifact inventories, and checkpoint lists in
  `docs/training-data-log.md`, then reference them from the notes.
- Visual/qualitative claims should point to human-inspectable artifacts. For
  games and tactical tests, prefer SGF, HTML viewers, rendered boards/casebooks,
  screenshots, or dashboards in addition to JSON.
- When adding a tactical probe or board-reading claim, include the checkpoint or
  cycle and a rendered artifact path so a human can inspect whether the case is
  meaningful.

## Frontend Reuse Rules

- Do not generate a new replay/play frontend from scratch for each result
  bundle. Reuse the stable viewers in `artifacts/viewers/` and pass data through
  query params, JSON files, or server APIs.
- `artifacts/viewers/play-ai.html` is the reusable human-vs-AI UI. The server
  must expose `/api/info` so the page can display the loaded checkpoint and
  default simulations without hand-editing HTML.
- The play server must use the same rules backend as training/evaluation via
  `demo_cpu.make_rules` and checkpoint `rules_backend`. Do not add a separate
  UI-only Go rules implementation; it will drift on ko, suicide, scoring, or
  pass semantics.
- `artifacts/viewers/selfplay-viewer.html` is the reusable self-play replay UI.
  New self-play artifacts should provide `summary.json` and
  `cycle-xxxxx-moves.json` in the expected shape instead of copying the viewer.
- Generated casebooks should keep their data separate from presentation where
  practical. If a temporary standalone HTML casebook is still used, verify
  rendered board count, controls, and encoding in the browser before handing it
  to the user.
- Keep generated HTML copy ASCII-only unless the page has already been tested
  in the local static server. This avoids recurring mojibake in artifact pages.

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
- Checkpoint snapshots are dense at the start: save every `5` cycles through
  cycle `50`, then every `10` cycles by default. Override with
  `--early-checkpoint-cycles`, `--early-checkpoint-every`, and
  `--checkpoint-every` only when the run intentionally needs a different
  archive cadence.
- `src/diamondgo/eval_suite.py` runs standard checkpoint matches at step tiers such as `50,200,500` against `initial` and `previous`.
- `src/diamondgo/tactical_eval.py` runs fixed capture/atari probes and reports target top-1/top-3/rank.
- `multiworker_train` behavior metrics include black/white win rates, color-bias alerts, and signed score-margin summaries.

## Server Migration Artifacts

The latest source migration artifacts on the server are:

- `/root/autodl-tmp/diamondgo-migration/diamondgo-source.bundle`
- `/root/autodl-tmp/diamondgo-migration/diamondgo-source.tar.gz`

Regenerate these after important source commits if the user may switch servers.

## Server Expiration Reminder

- The current rented server is expected to expire tomorrow relative to the
  2026-06-06 overnight run.
- Before the server expires, upload all source, scripts, notes, and compact
  result summaries to GitHub.
- Do not assume large training artifacts are safely stored just because the
  source is pushed. Checkpoints, generated dashboards, SGFs, JSON traces, and
  monitor logs should be inventoried and either:
  - compressed into a migration/download bundle, or
  - uploaded through an explicit large-artifact path such as Git LFS or GitHub
    Releases.
- At minimum, preserve the final `latest.pt`, selected checkpoint snapshots,
  `train.log`, `metrics.jsonl`, `gpu_monitor.csv`, eval/tactical reports, and
  the human-readable notes that point to them.
- To upload a run's saved checkpoints to GitHub LFS from a server checkout, run:
  `python scripts/archive_run_checkpoints.py artifacts/<run-dir> --commit --push`.
  This writes `artifacts/checkpoint-archive/<run-dir>/manifest.json`, stages the
  `.pt` files through Git LFS, commits them, and pushes to `origin/main`.
