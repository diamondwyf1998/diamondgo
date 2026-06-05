# Training Notes

This file records the human-facing story of training observations and
operations. Keep only important facts, phenomena, and next actions here. Put
large tables and raw measurements in `docs/training-data-log.md`, then reference
them from this file.

Human-readable record rule:

- Notes are for people first. They should help a reader understand what
  happened, why it matters, what to inspect next, and where the supporting data
  lives.
- Do not turn this file into a raw dump. Put large tables, metric snapshots,
  and artifact inventories in `docs/training-data-log.md`.
- For qualitative claims about games, tests, shapes, tactical probes, or search
  behavior, prefer a human-inspectable artifact such as an HTML viewer, SGF,
  rendered board/casebook, screenshot, or dashboard. JSON alone is not enough
  when a person needs to judge the position.
- If an artifact is meant to support a visual claim, include the exact
  checkpoint/cycle and a link/path to the rendered view.

Source labels:

- `User observation`: qualitative reading or hypothesis first raised by the
  user while reviewing games.
- `Agent measurement`: aggregate statistics, scripted samples, or generated
  dashboard/table results.
- `Technical operation`: code/config/training operation performed by an agent.
- `Hypothesis`: interpretation that still needs targeted tests.

## 2026-06-05 Notes

### Configuration And Operation Changes

- `Technical operation`: Main training/evaluation hyperparameters are tracked
  separately from qualitative notes. This includes model size, workers,
  simulations, optimizer, replay size, evaluation games, and tactical probe
  settings.
  - Data reference: `docs/training-data-log.md#training-hyperparameters-and-runtime-config`
- `Technical operation`: Increased the self-play move cap during the experiment
  sequence:
  - early runs and first multiworker checkpoints used `max_moves=80`
  - the 630-690 stage used `max_moves=160`
  - the 700+ optimized run used `max_moves=120`
  - Data reference: `docs/training-data-log.md#run-and-config-timeline`
- `Technical operation`: Komi handling was changed by recent source work,
  partly by the other agent:
  - legacy artifacts generally serialize `komi=0.5`
  - current code separates model-input `komi=0.5` from scoring `score_komi=6.5`
  - important intent: do not change the model input plane when continuing old
    checkpoints; change only terminal scoring/value labels via `score_komi`
  - Data reference: `docs/training-data-log.md#run-and-config-timeline`
- `Technical operation`: A fresh `komi=6.5` run was briefly started, then
  stopped after realizing `komi` is part of the model input. The active
  follow-up run instead resumes the old 0.5-komi model with `komi=0.5` and
  `score_komi=6.5`.
  - Run reference:
    `artifacts/multiworker-9x9-resume0p5-score6p5-100sims-120moves-2h-20260605`
  - Data reference: `docs/training-data-log.md#score-komi-continuation-cycles-857`
- `Technical operation`: The score-komi continuation was extended by another
  two hours. The original training Python process is allowed to finish
  naturally; a watcher then resumes the run's latest checkpoint for another
  120 minutes and runs the final eval/tactical probes after the extension.
  - Extension script:
    `/root/diamondgo/extend_resume_scorekomi65_2h.sh`
  - Data reference: `docs/training-data-log.md#score-komi-continuation-cycles-857`
- `Technical operation`: After the extension had started, remaining training
  time was still over one hour, so a cutoff watcher was added at about
  `21:42 CST`. It should stop the extension training about one hour later and
  then run the same final eval/tactical probes.
  - Cutoff log:
    `/root/diamondgo/artifacts/multiworker-9x9-resume0p5-score6p5-100sims-120moves-2h-20260605/cutoff-after-1h.log`
  - Data reference: `docs/training-data-log.md#score-komi-continuation-cycles-857`
- `Technical operation`: The `score_komi=6.5` continuation was stopped early
  after the user flagged that White win rate looked too high. The follow-up
  run resumes from the stopped latest checkpoint, keeps model-input `komi=0.5`,
  and changes terminal scoring/value labels to `score_komi=2.5`.
  - Training script: `tools/server/run_resume_scorekomi25_1h.sh`
  - Finalizer script: `tools/server/finalize_scorekomi25_1h.sh`
  - Data reference: `docs/training-data-log.md#score-komi-25-continuation-cycles-1065`
- `Technical operation`: The `score_komi=2.5` follow-up was also stopped early
  at cycle `1080` after the White skew persisted. Checkpoints were preserved
  for later inspection rather than overwritten.
  - Preserved server path:
    `/root/diamondgo/artifacts/multiworker-9x9-resume0p5-score2p5-100sims-120moves-1h-20260605`
  - Data reference: `docs/training-data-log.md#score-komi-25-continuation-cycles-1065`
- `Technical operation`: Started a fresh-from-zero design for the next run
  after reviewing AlphaGo Zero / Leela Zero / KataGo-style implementation
  choices. The next run removes the constant komi input plane, increases the
  network to `64 channels x 4 residual blocks`, adds root Dirichlet exploration
  noise, uses a late-game temperature drop, and trains with random board
  symmetries.
  - Reason: fixed-komi input was acting as a constant feature in our 9x9 setup,
    while stronger public implementations either use color/history planes or a
    richer rules/score feature system.
  - Training script:
    `tools/server/run_fresh_nokomi_4x64_score2p5_noise_aug_2h.sh`
  - Data reference: `docs/training-data-log.md#fresh-no-komi-input-4x64-run`
- `Technical operation`: We are preparing a fresh retraining run rather than
  another continuation. The intended direction is to make the model a little
  larger and remove komi from the input features.
  - Planned model change: `residual_blocks=4`.
  - Planned input change: use `--no-input-komi`, so the network sees only own
    stones, opponent stones, and side-to-play.
  - Important: this cannot resume old checkpoints, because old checkpoints were
    trained with a 4-plane input stem and `--no-input-komi` changes the stem to
    3 planes. Start from scratch unless a conversion experiment is deliberately
    designed.
  - Current working-tree support: `input_komi` config plumbing exists across
    rules, self-play, training, eval, and tactical eval; root policy
    temperature/noise and dihedral augmentation knobs are also being added.
  - Data reference:
    `docs/training-data-log.md#planned-fresh-retraining-config`
- `Technical operation`: Built a dedicated single-game viewer for qualitative
  checkpoint inspection:
  - `artifacts/selfplay-showcase-swing-760-830-20260605/viewer.html`
  - Reason: older batched self-play dashboards mixed three parallel games into
    one position stream, which is misleading for reading a single game.
  - Data reference: `docs/training-data-log.md#single-game-showcase-cycles-760-830`

### Observed Phenomena

- `Agent measurement`: Win-rate asymmetry appeared strongly in the 700+ run.
  Black was favored around 700-759, White became strongly favored around
  770-819, and Black became favored again after about 820.
  - Main checkpoint range: `700-856`
  - Reversal checkpoints to inspect: `760, 770, 780, 790, 800, 810, 820, 830`
  - Data reference: `docs/training-data-log.md#black-win-rate-and-margin-cycles-700-856`
- `Agent measurement`: Black tends to prefer the center in several sampled
  checkpoints. The strongest examples are around `630, 650, 660, 690, 700,
  730, 750`, where Black's first 20 moves land in the center 5x5 much more
  often than White's.
  - Data reference: `docs/training-data-log.md#center-5x5-opening-distribution`
- `Agent measurement`: Around the color reversal, Black's opening distribution
  looks less stable. In small samples:
  - `760`: first move fixed at `F4` in all three sampled games
  - `770-800`: White wins most sampled games, and Black is less center-heavy
  - `810`: one sampled game starts with Black `pass`, which is abnormal
  - `830`: first move fixed at `G2` in all three sampled games, and Black wins
    all three sampled games
  - Data reference: `docs/training-data-log.md#single-game-showcase-cycles-760-830`
- `User observation`: During the White-favored phase, Black stones can look more
  dispersed and easier for White to surround in large regions. This is an
  observation from visual inspection, not yet a quantified metric.
  - Checkpoints to revisit visually: `770, 780, 790, 800, 810`
  - Viewer reference:
    `artifacts/selfplay-showcase-swing-760-830-20260605/viewer.html`
- `User observation`: When Black's win rate reversed back in the 820+ range,
  its style looked more stable and less scattered. A plausible qualitative
  reason is that Black no longer left a large, loose distribution of stones
  that White could surround.
  - Checkpoints to compare visually: `800, 810, 820, 830`
  - Data reference: `docs/training-data-log.md#single-game-showcase-cycles-760-830`
- `Agent measurement`: After switching only terminal scoring to `score_komi=6.5`
  while keeping the old model-input `komi=0.5`, the continuation run rapidly
  swung from a Black advantage at cycle `857` to a strong White advantage from
  about `859` onward. This should be interpreted as a rule-label shock /
  adaptation signal, not as a clean strength improvement.
  - Main checkpoint range so far: `857-871`
  - Data reference: `docs/training-data-log.md#score-komi-continuation-cycles-857`
- `User observation`: During the active `score_komi=6.5` continuation, White
  win rate looked too high. Late-cycle measurements confirmed this: cycles
  `1059-1064` all had White win rate above `90%`, with cycle `1064` at `31/32`
  White wins.
  - Technical response: stop the `score_komi=6.5` run and start a
    `score_komi=2.5` continuation.
  - Data reference: `docs/training-data-log.md#score-komi-continuation-cycles-857`
- `Agent measurement`: Lowering scoring komi to `2.5` did not quickly recover
  balance when resuming from the White-favored weights. Cycles `1065-1080`
  still had White win rates between `75.0%` and `96.9%`.
  - Interpretation: the inherited policy/value state matters; just changing
    terminal scoring labels midstream is not enough for a quick correction.
  - Data reference: `docs/training-data-log.md#score-komi-25-continuation-cycles-1065`
- `Hypothesis`: The previous weak behavior may be partly from representation
  and self-play noise, not only from scoring komi. A constant komi plane gives
  the network a feature that cannot explain position differences inside a fixed
  experiment, and `temperature=1.0` for the whole game may keep late-game play
  too random. The fresh 4x64 run is meant to test this directly.
- `Agent measurement`: The fresh no-komi-input 4x64 run did not immediately
  reproduce the extreme White skew seen in the continuation runs. Cycle `1`
  was White-favored (`21/32` White wins), while cycle `2` swung to Black
  (`19/32` Black wins). This is only an early health check, not a strength
  conclusion.
  - Data reference: `docs/training-data-log.md#fresh-no-komi-input-4x64-run`

### Tests Still Needed

- `User observation`: The previous `32 channels x 2 residual blocks` line is
  not exhausted. It may still improve with higher MCTS search or more advanced
  training tricks, so it should remain as a live comparison branch rather than
  being treated as failed.
  - Follow-up: run a stronger 2-layer baseline with more search, then compare
    it against the current fresh 4x64 model at matched training budgets.
  - Data reference: `docs/training-data-log.md#unfinished-2-layer-baseline-questions`
- Tactical learning checks were started for checkpoints `760-830`.
  - White one-stone capture is learned in this probe: `8/8` top1/top3.
  - Black capture is weak: black one-stone capture is `0/8` top1 and only
    `3/8` top3; black two-stone capture is `0/8` top3.
  - Simple escape-from-atari is not learned in these cases: black and white
    escape-atari probes are both `0/8` top3.
  - Capture-to-defend is partly learned: black `6/8` top1 and `7/8` top3,
    white `2/8` top1 and `3/8` top3.
  - Filling eyes is a major problem, especially for Black: the bad black
    fill-eye move is top1 in `6/8` checkpoints and top3 in `8/8`.
  - Self-atari/dead-shape probes are less catastrophic than fill-eye but still
    show bad black edge/corner moves in top3 for `3/8` checkpoints.
  - Rendered casebook:
    `artifacts/tactical-swing-760-830-20260605/casebook.html`
  - Data reference: `docs/training-data-log.md#tactical-probes-cycles-760-830`
- Tactical checks still need broader coverage:
  - test old `850/latest`, score-komi continuation `860/870/latest`, and later
    finished continuation checkpoints
  - test more capture/atari shapes, including ladders, snapbacks, and edge
    liberties
  - turn fill-eye into a larger distributional metric instead of two hand-built
    cases
  - audit whether these failures are policy-prior failures, MCTS/value failures,
    or both
- Opening policy checks:
  - for checkpoints `760, 770, 780, 790, 800, 810, 820, 830`, record empty-board
    top-10 priors and MCTS top candidates
  - specifically check how often `pass` appears before the board is near terminal
- Head-to-head checks:
  - compare nearby checkpoints around the reversal, such as `760 vs 770`,
    `770 vs 780`, `800 vs 810`, `810 vs 820`, instead of only self-play within
    one checkpoint
- Shape/distribution checks:
  - quantify whether Black's stones are more dispersed during the White-favored
    phase
  - possible metrics: connected-component count, largest group size, surrounded
    empty-region/contact metrics, center/edge split by color

### Working Interpretation

`Hypothesis`: The current evidence points to self-play distribution instability
rather than a smooth monotonic strength increase. The color reversal and fixed
first-move patterns suggest that the model may periodically collapse into
transient opening habits. The `810` first-move pass is a special warning sign
and should be tested directly before trusting later qualitative conclusions.

`Hypothesis`: The 820+ Black recovery may be related to a more stable style:
fewer scattered stones, less exposure to being surrounded, and possibly more
coherent group structure. This is currently based on user visual inspection and
needs a shape/distribution metric before treating it as a measured result.
