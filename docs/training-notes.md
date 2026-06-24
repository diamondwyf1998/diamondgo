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
- Result viewers should be reusable. Prefer stable frontends in
  `artifacts/viewers/` plus JSON/server data over newly copied standalone HTML,
  so labels, encoding, controls, and checkpoint identity do not drift between
  experiments.
- When the user asks for "大测评展示", prepare the complete display bundle:
  latest training-result summary, evaluation dashboards/reports, training
  curves, recorded self-play game viewer, and human-vs-AI play page loaded with
  the latest checkpoint. Prefer recorded training games when they exist; only
  generate new showcase games when recorded games are unavailable.
- When the user asks for "小测评", prepare a light display bundle without a
  full evaluation matrix: sample several recorded self-play checkpoints from
  the newest active run, open the reusable self-play viewer, and load the
  latest checkpoint into the human-vs-AI play page.
- When the user asks for "比较实验", pick representative checkpoints and choose
  reference opponents that best answer the current comparison question. Start
  with approximately `1x` and `2x` training-progress opponents when they are
  meaningful, but do not treat those ratios as fixed. If those pairings are
  unavailable, unfair, or uninformative, choose other sensible ratios/checkpoints
  and clearly label the rationale and approximation.

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
- `Technical operation`: The `4x64` fresh run changed both depth and width:
  residual blocks increased `2 -> 4`, and channels increased `32 -> 64`.
  This raised trainable parameters from `54,461` to `316,669`, about `5.82x`,
  not the roughly `2x` change expected from increasing depth alone. This should
  have been reported explicitly before launch.
  - Communication rule: before future training starts, report every important
    changed hyperparameter in the chat, especially model size, parameter count,
    search count, scoring komi, input planes, and exploration/training tricks.
  - Control follow-up: run a `32 channels x 4 residual blocks` no-komi-input
    variant with the same non-model tricks to isolate the effect of depth.
  - Data reference: `docs/training-data-log.md#model-size-comparison`
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
- `Technical operation`: The fresh no-komi-input 4x64 run was paused at about
  cycle `66` so the user could prepare an overnight deployment. The finalizer
  was stopped before the training process, so no automatic eval/tactical jobs
  were launched by this pause.
  - Preserved checkpoint reference:
    `artifacts/multiworker-9x9-fresh-nokomi-4x64-score2p5-100sims-noise-aug-2h-20260605/latest.pt`
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

## 2026-06-06 Notes

### Fresh No-Komi 4x64 Checkpoint 60

- `Technical operation`: The latest fresh no-komi-input `4x64` checkpoint was
  pulled for qualitative review and interactive play. The local self-play
  viewer was generated for `cycle-00060`, and the browser play server was
  switched to the same checkpoint.
  - Self-play viewer:
    `artifacts/selfplay-showcase-fresh-nokomi-4x64-cycle60-20260606/viewer.html`
  - Human play page:
    `artifacts/viewers/play-ai.html`
  - Play server note: the local browser play server was adapted so no-komi
    checkpoints use a 3-plane input state instead of the old 4-plane komi
    state.
  - Data reference:
    `docs/training-data-log.md#fresh-no-komi-input-4x64-checkpoint-60-showcase`
- `Agent measurement`: The `cycle-00060` latest self-play trace contains `32`
  games. In that sample, Black won `26` and White won `6`, so the current
  fresh run has a visible Black skew in self-play despite using `score_komi=2.5`.
  This is a sample from one latest cycle, not a final strength estimate.
  - Data reference:
    `docs/training-data-log.md#fresh-no-komi-input-4x64-checkpoint-60-showcase`

### Cross-Generation Strength Checks

- `Agent measurement`: The fresh no-komi-input `4x64` `cycle-00040`
  checkpoint beat early old-generation checkpoints very strongly in quick
  cross-generation matches. Each match used `10` games with candidate Black
  `5` times and White `5` times, `100` simulations per move, and the same
  opening sampling convention as the evaluation dashboard.
  - Against old checkpoints `40, 50, 80, 200`: results were `10/10`,
    `10/10`, `9/10`, and `8/10`.
  - Against old checkpoints `300, 400, 500`: results were `7/10`, `10/10`,
    and `10/10`.
  - Important caveat: the `500` opponent comes from
    `multiworker-9x9-100sims-90min-20260605`, because the earlier
    `overnight-9x9-20260605` line stops at `cycle-00410`.
  - Data reference:
    `docs/training-data-log.md#cross-generation-evaluation-fresh-cycle-40`

### Tactical Skills

- `Agent measurement`: The previous 12-case tactical casebook was rerun for
  fresh no-komi-input checkpoints `10,20,30,40,50,60`.
  - Capture and atari-defense probes improved only slightly: by cycle `60`,
    capture is `1/1 of 4` top1/top3 and atari defense is `0/1 of 4`
    top1/top3.
  - Fill-eye remains the clearest bad habit: for every tested checkpoint,
    `1/2` fill-eye bad moves are still top1/top3.
  - Self-atari/dead-shape bad moves did not enter top1/top3 in these six
    checkpoints, though one appears in top10.
  - Rendered casebook:
    `artifacts/tactical-fresh-nokomi-4x64-cycle10-60-20260606/casebook.html`
  - Data reference:
    `docs/training-data-log.md#tactical-probes-fresh-no-komi-4x64-cycles-10-60`

### Partial Eval Before 200-Sim Continuation

- `User request`: Before the overnight job, run an eval pass, then start a
  16-hour task with MCTS increased to `200`.
- `Agent measurement`: The eval pass completed the initial-opponent matches
  before being stopped to avoid delaying the overnight training launch.
  - Cycle `40` and `50` both scored `18/20` against the initial net.
  - Cycle `60` and latest/cycle `66` both scored `9/20`, a visible drop in this
    quick eval slice.
  - Important interpretation update: the `cycle 60/66` losses were caused by
    triggering early `pass` and premature game endings, not by the new model
    being weaker than the initial net in normal play. Treat these losses as an
    evaluation/termination pathology and a pass-policy problem.
  - This is not a full strength conclusion because the previous-opponent tier
    was not completed and the completed tier is contaminated by early-pass
    failures.
  - Data reference:
    `docs/training-data-log.md#partial-eval-before-200-sim-overnight-continuation`
- `Technical operation`: Training metrics and eval dashboards now include
  first-pass, second-pass, terminal double-pass, pass-by-color, and early-pass
  alert fields. This does not change rules or training targets; it makes the
  pass pathology visible during the overnight run.

### 200-Sim Overnight Continuation

- `Technical operation`: The next overnight run continues from the fresh
  no-komi-input `4x64` latest checkpoint instead of starting over.
  - Resume source: fresh run latest at cycle `66`.
  - New training directory:
    `artifacts/multiworker-9x9-fresh-nokomi-4x64-score2p5-200sims-noise-aug-16h-20260606`
  - Training self-play search increases from `100` to `200` simulations.
  - Architecture stays `64` channels x `4` residual blocks, with `316,669`
    trainable parameters.
  - Score komi stays `2.5`; input komi stays disabled.
  - Data reference:
    `docs/training-data-log.md#200-sim-overnight-continuation-configuration`
- `Hypothesis`: If the earlier weak tactical/shape behavior is mainly from
  search being too shallow, the 200-sim continuation should show cleaner
  self-play shapes and better tactical casebook results without requiring a
  new architecture. If the problem is mostly value/policy training quality, it
  may still plateau or oscillate.
- `Agent measurement`: The 200-sim continuation has now reached metrics/trace
  cycle `403`; the newest numbered checkpoint file pulled locally is
  `cycle-00400.pt`.
  - Latest self-play sample: `32` games, Black `26`, White `6`, Black win rate
    `81.25%`, mean Black score margin `+2.406`.
  - Loss has fallen to `1.3428`, with policy entropy `0.5819` and throughput
    around `28.68` positions/sec.
  - Early pass is improved versus the pathological eval losses but not gone:
    latest cycle has early first-pass `<40` rate `21.88%`.
  - Data reference:
    `docs/training-data-log.md#200-sim-continuation-cycle-403-snapshot`
- `Agent measurement`: The tactical casebook was rerun every `50`
  cycles for the 200-sim continuation, using checkpoints
  `100,150,200,250,300,350,400`. The two two-stone capture probes were removed
  afterward because they were not good basic capture tests, leaving `10`
  retained probes.
  - Atari-defense becomes the clearest improvement: from `0/1 of 4` top1/top3
    at cycle `100` to `4/4 of 4` top1/top3 from cycle `250` onward.
  - Fill-eye is mostly fixed in this probe set: bad fill-eye moves are
    `0/0 of 2` top1/top3 for all tested cycles except cycle `250`, which is
    `1/1 of 2`.
  - Active one-stone capture remains inconsistent: capture targets stay around
    `0-1` top1 hits and `1` top3 hit out of `2`.
  - Self-atari/dead-shape avoidance is mostly clean early, but the bad black
    edge/corner move returns by cycle `400` as `1/1 of 2` bad top1/top3.
  - Rendered casebook:
    `artifacts/tactical-fresh-nokomi-4x64-200sims-cycle100-400-20260606/casebook.html`
  - Data reference:
    `docs/training-data-log.md#tactical-probes-200-sim-continuation-cycles-100-400`
- `Agent measurement`: A second-level tactical casebook was added for the same
  200-sim line, using checkpoints `100,150,200,250,300,350,400,410`.
  - The tests cover hard-surrounded straight/bent three killing moves, double
    atari, and third-line atari of a second-line stone. The second-line probe
    was corrected to use only two attacking stones, so it is no longer a direct
    last-liberty capture test.
  - Double atari is the strongest result: `2/2` top3 in every tested
    checkpoint, with `1-2` top1 hits.
  - Life/death is not learned by this probe: cycle `410` is `0/0 of 4`
    top1/top3 for straight/bent three vital points, and the targets are not
    even top10.
  - Second-line third-line atari is learned in this probe: cycle `410` is
    `2/2 of 2` top1/top3, with both colors choosing `E3` as top1.
  - Rendered casebook:
    `artifacts/tactical-level2-fresh-nokomi-4x64-200sims-cycle100-410-20260606/casebook.html`
  - Data reference:
    `docs/training-data-log.md#tactical-level-2-probes-200-sim-continuation-cycles-100-410`

### Score-Komi 4.5 Continuation

- `User observation/request`: The 200-sim continuation still looks too
  Black-favored, so the next continuation should raise the scoring komi to
  `4.5`.
- `Technical operation`: The `score_komi=2.5` 200-sim run was stopped and
  preserved at cycle `410`; its old finalizer was stopped before it spent GPU
  time evaluating a superseded configuration.
  - Stop snapshot: Black `29`, White `3`, Black win rate `90.62%`.
  - First-pass median was `65.0`; early first-pass `<=40` was `6/32`, so this
    snapshot was Black-skewed even though the early-pass alert was not firing.
- `Technical operation`: A new continuation starts from the preserved
  `score_komi=2.5` latest checkpoint but changes terminal scoring/value labels
  to `score_komi=4.5`.
  - Model input remains no-komi: `input_komi=false`, `komi_metadata=0.5`.
  - Architecture and search are unchanged: `4x64`, `316,669` parameters,
    `200` MCTS simulations, `8` workers, `32` games/cycle.
  - Runtime is set to `6` hours so it can finish and run eval before the
    rented server expires.
  - Data reference:
    `docs/training-data-log.md#score-komi-45-continuation-configuration`

### Dual-GPU 2x96 Score-Komi 4.5 Fresh Run

- `Technical operation`: A new server run was started on the dual-GPU machine
  with a larger but shallower model: `2` residual blocks x `96` channels,
  `score_komi=4.5`, no komi input, `250` self-play simulations, `12` workers,
  `8` games per worker, and `96` games/cycle. This is a fresh run rather than a
  continuation from the previous `4x64` line.
  - Data reference:
    `docs/training-data-log.md#dual-gpu-2x96-score45-recorded-self-play-cycles-5-160`
- `User observation/request`: For new checkpoints, use already recorded
  self-play games when possible instead of regenerating display games. This
  keeps the displayed games tied to the actual training run and avoids mixing
  display artifacts with new ad hoc samples.
- `Agent measurement`: Recorded self-play for cycles
  `5,25,50,80,110,140,160` was pulled into a reusable catalog and opened in the
  shared self-play viewer.
  - Local catalog:
    `artifacts/selfplay-recorded-dualgpu-2x96-score4p5-20260607`
  - Viewer example:
    `http://127.0.0.1:8765/viewers/selfplay-catalog-viewer.html?dataset=selfplay-recorded-dualgpu-2x96-score4p5-20260607&cycle=160&game=20`
  - Black remains favored in these recorded cycles. Black win rate ranges from
    `54.17%` at cycle `5` to `75.00%` at cycle `160`, with cycle `50` already
    at `70.83%`.
  - Early pass before move `40` drops quickly from `31.25%` at cycle `5` to
    roughly `5-9%` in the middle/late sampled cycles, though it is not gone.
- `Agent measurement`: A reusable cross-run match script was added so future
  comparisons do not hardcode "same cycle" as the only pairing mode.
  - Script:
    `scripts/run_cross_run_matches.py`
  - It supports explicit checkpoint pairs, same-cycle pairing when requested,
    and close/approximate pair lists.
  - Data reference:
    `docs/training-data-log.md#cross-run-matches-dual-gpu-2x96-score45-vs-old-4x64-score25`
- `Agent measurement`: In 10-game cross-run matches against the earlier
  no-komi-input `4x64`, `score_komi=2.5` line, the new `2x96`,
  `score_komi=4.5` line is already competitive at similar early checkpoint
  counts, but not yet stronger than much later old checkpoints.
  - Exact available same-cycle result: new cycle `80` beat old cycle `80` by
    `8/10`.
  - Exact available double-cycle result: new cycle `50` versus old cycle `100`
    was `5/10`.
  - Close same-ish results were `7/10`, `6/10`, `5/10`, and `8/10` for
    `new50-vs-old60`, `new110-vs-old100`, `new140-vs-old150`, and
    `new160-vs-old150`.
  - Close double-ish results were weaker: `4/10`, `1/10`, `3/10`, and `3/10`
    for `new80-vs-old150`, `new110-vs-old200`, `new140-vs-old300`, and
    `new160-vs-old300`.
  - Interpretation: the fresh `2x96` run is learning useful play quickly, but
    the current evidence does not show it surpassing the older line at about
    twice the training age. The sample size is small, and color asymmetry is
    still visible, so these numbers should guide follow-up tests rather than
    settle strength.
- `Correction / preflight record`: Before the exact/close same-era comparison,
  an earlier preflight match was run against the much later old `4x64`,
  `score_komi=5.5` continuation. This was not the comparison the user wanted
  for same-cycle progress, but it is still useful as a sanity check against a
  substantially more trained previous line.
  - Pairs were `new50-vs-old530`, `new80-vs-old560`, `new110-vs-old590`,
    `new140-vs-old620`, and `new160-vs-old650`, all `10` games each.
  - Candidate wins were `0/10`, `3/10`, `1/10`, `3/10`, and `1/10`.
  - Interpretation: this should be read as "early new line is far behind a
    much later old continuation", not as evidence about same-cycle learning
    speed.
  - Data reference:
    `docs/training-data-log.md#preflight-cross-run-matches-dual-gpu-2x96-score45-vs-late-old-4x64-score55`

### Comparison Experiment: Dual-GPU 2x96 After18h vs Old 4x64

- `User request`: Run a "比较实验": pick several checkpoints and compare
  against approximately `1x` and `2x` training-progress old checkpoints. If
  exact pairings are unavailable, choose close reasonable substitutes.
- `Agent measurement`: Four candidate checkpoints from the after18h
  dual-GPU `2x96`, `score_komi=4.5` line were tested: cycles
  `240,250,260,270`. Each pair used `10` games, `5` as Black and `5` as White,
  with both sides at `100` simulations.
  - `1x` approximate opponents: old `4x64`, `score_komi=2.5` cycles
    `250,250,250,300`.
  - `2x` approximate opponents: old `4x64` cycles `500,500,520,530`; these
    come from later score-komi `6.5/5.5` lines, so they are not clean same-rule
    architecture comparisons.
  - Dashboard:
    `artifacts/compare-experiment-1x-2x-dualgpu2x96-20260607/dashboard.html`
  - Game replay dashboard:
    `artifacts/compare-experiment-1x-2x-dualgpu2x96-20260607/games_dashboard.html`
  - Data reference:
    `docs/training-data-log.md#comparison-experiment-dual-gpu-2x96-after18h-vs-old-4x64`
- `Agent measurement`: Results are noisy and not monotonic. Against `1x`
  approximate opponents the new line scores `4/10`, `2/10`, `6/10`, `4/10`.
  Against `2x` approximate opponents it scores `7/10`, `3/10`, `7/10`,
  `3/10`.
- `Interpretation`: This quick comparison does not support a simple "later new
  checkpoint is always stronger" story. The strongest pair results are cycles
  `240` and `260`, while cycles `250` and `270` are weak. Because the `2x`
  opponents mix later score-komi settings, treat the experiment as a practical
  progress probe rather than a clean scientific isolation of architecture.
- `Agent measurement`: The stronger-looking comparison pairs were rerun at
  `300` simulations for both sides.
  - Dashboard:
    `artifacts/compare-experiment-strong-300sims-dualgpu2x96-combined-20260607/dashboard.html`
  - Game replay dashboard:
    `artifacts/compare-experiment-strong-300sims-dualgpu2x96-combined-20260607/games_dashboard.html`
  - `new240` looks more stable at higher search: `7/10` against old `250` and
    `5/10` against old `500`.
  - `new260` does not preserve its 100-sim strength: it is `5/10` against old
    `250` and `0/10` against old `520`.
  - Interpretation: the 100-sim comparison overestimated at least some
    `new260` matchups. Higher-search comparisons should be used before calling
    a checkpoint clearly stronger.

### Small Eval: Score6.5 Cleanup 400-Sim Active Run

- `User request`: Do a "小测评" for the new training line: inspect a few
  recorded self-play checkpoints and load the latest model into human-vs-AI.
- `Technical operation`: The active server run is the cleanup continuation:
  `2x96`, `score_komi=6.5`, terminal dead-stone cleanup enabled, `400`
  self-play simulations, `12` workers, `96` games/cycle.
- `Agent measurement`: A lightweight local display was prepared from recorded
  self-play cycles `300,330,360,365`; no new showcase games were generated.
  - Small eval entry:
    `artifacts/small-eval-score6p5-cleanup-400sims-20260608/index.html`
  - Recorded self-play catalog:
    `artifacts/small-eval-selfplay-score6p5-cleanup-400sims-20260608/index.html`
  - Latest recorded viewer:
    `http://127.0.0.1:8765/viewers/selfplay-catalog-viewer.html?dataset=small-eval-selfplay-score6p5-cleanup-400sims-20260608&cycle=365&game=1`
  - Human-vs-AI:
    `http://127.0.0.1:8787/viewers/play-ai.html?checkpoint=dualgpu2x96-score6p5-cleanup-cycle-00366-latest&v=small-eval-score6p5-cleanup-366`
- `Agent measurement`: At the pulled latest checkpoint/metrics snapshot,
  `latest.pt` is cycle `366`. Latest metrics show Black win rate `65.6%`,
  early first-pass `<40` rate `5.2%`, mean moves `123.385`, loss `0.959`, and
  terminal cleanup counts `17` Black stones / `58` White stones.
- `Agent measurement`: A same-line `+50 cycle` PK check was run on the cleanup
  line, using `100` simulations and `10` games per pair.
  - Dashboard:
    `artifacts/pairwise-plus50-cleanup-score6p5-100sims-10games-20260608/dashboard.html`
  - Game replay dashboard:
    `artifacts/pairwise-plus50-cleanup-score6p5-100sims-10games-20260608/games_dashboard.html`
  - Later checkpoints score `8/10`, `5/10`, `6/10`, and `6/10` for
    `330-vs-280`, `340-vs-290`, `350-vs-300`, and `360-vs-310`.
  - Interpretation: the line shows some improvement over +50 cycles, but it is
    not cleanly monotonic. The `350-vs-300` and `360-vs-310` wins are still
    color-skewed toward candidate-as-Black, so use color split when judging
    progress.
- `User request / agent measurement`: On 2026-06-09, another small eval was
  prepared at latest cycle `546` for the same cleanup run.
  - Small eval entry:
    `artifacts/small-eval-score6p5-cleanup-cycle546-20260609/index.html`
  - Recorded self-play catalog:
    `artifacts/small-eval-selfplay-score6p5-cleanup-cycle490-546-20260609/index.html`
  - Latest recorded viewer:
    `http://127.0.0.1:8765/viewers/selfplay-catalog-viewer.html?dataset=small-eval-selfplay-score6p5-cleanup-cycle490-546-20260609&cycle=546&game=1`
  - Human-vs-AI:
    `http://127.0.0.1:8787/viewers/play-ai.html?checkpoint=dualgpu2x96-score6p5-cleanup-cycle-00546-latest&v=small-eval-score6p5-cleanup-546`
  - Latest metrics at cycle `546`: loss `0.975935`, positions/sec `20.288`,
    Black win rate `62.5%`, White win rate `37.5%`, mean moves `114.677`,
    first-pass median `59.5`, early first pass `<=40` was `11/96`, terminal
    cleanup removed `22` Black stones / `54` White stones.
  - Important observation: latest recorded cycle `546`, game `1` starts with
    Black `pass` even though `E5` had far more visits in the recorded top
    actions. This suggests pass sampling/temperature behavior still needs
    inspection, even if the overall early-pass rate is not catastrophic.
- `Agent measurement`: A new same-line `+50 cycle` PK check was run on
  2026-06-09 using cycle `540` versus cycle `490`, `100` simulations, and `10`
  games.
  - Dashboard:
    `artifacts/pairwise-plus50-cleanup-score6p5-cycle540-vs490-100sims-10games-20260609/dashboard.html`
  - Game replay dashboard:
    `artifacts/pairwise-plus50-cleanup-score6p5-cycle540-vs490-100sims-10games-20260609/games_dashboard.html`
  - Result: cycle `540` scored `6/10`; as Black `5/5`, as White `1/5`.
  - Pass behavior: first-pass median `66.0`; first pass `<=40` was `1/10`;
    terminal double-pass games `7/10`.
  - Interpretation: `540` is somewhat ahead of `490`, but the result is still
    highly color-skewed. Treat this as "slight improvement with color bias",
    not a clean strength jump.
- `User request / technical operation`: When the 18-hour dual-GPU run was close
  to finishing, the user asked to immediately continue for another `3` hours if
  the run had completed. A queue watcher was installed so this would happen
  automatically.
  - The original 18h finalizer was intentionally stopped; otherwise it would
    have started eval immediately and competed with the requested continuation
    training.
  - The continuation started from the 18h `latest.pt` at `2026-06-07 17:52
    +08`, finished at `20:57 +08`, and then ran eval/tactical probes.
  - All important artifacts were copied back locally under
    `artifacts/server-runs/20260607-dualgpu-2x96-18h-plus3h/`.
  - The 18h segment ended at cycle `236` with `2.45M` total positions; the +3h
    continuation ended at cycle `272` with `2.86M` total positions.
  - The final continuation self-play snapshot is still strongly Black-favored:
    cycle `272` has Black `76.04%`, White `23.96%`, and a `black` color-bias
    alert.
  - The valid continuation `n+50` eval slice is `cycle-272` vs `cycle-250`,
    where `cycle-272` won `17/20`. This match had first-pass `<=40` in only
    `1/20` games, which is much healthier than the vs-initial slices.
  - Caveat: the `step-200` and `step-500` "vs previous" continuation reports
    are effectively vs initial because the continuation checkpoint directory
    does not contain a selected previous checkpoint at those step intervals.
  - Tactical probes remain weak at `cycle-272`: only the white one-stone
    capture probe is top1/top3; black capture and both atari-escape probes
    fail.
  - Data reference:
    `docs/training-data-log.md#dual-gpu-2x96-score45-18h--3h-continuation`
- `User request / technical operation`: After the 18h+3h result, the user asked
  to raise the scoring komi to `6.5`, raise self-play search to `400` MCTS
  simulations, and start a `7` day continuation.
  - This continues from the `cycle-272` latest checkpoint of the
    `score_komi=4.5`, `250`-simulation dual-GPU line.
  - Kept constant for comparability: `2x96` model, `input_komi=false`,
    `komi_metadata=0.5`, `score_margin_reward_scale=0.2`, `max_moves=150`,
    `12` workers, `96` games/cycle, `record_every=5`.
  - Changed for this run: `score_komi=6.5`, `selfplay_simulations=400`,
    `time_limit_minutes=10080`.
  - Started on the server at `2026-06-07 22:23:53 +08`.
  - Server PIDs at launch: train `225750`, finalizer `225751`.
  - A server-side `status_3h.log` was added in the run directory because
    remote training continues even if the local Codex app or user computer is
    offline; app heartbeat checks may be delayed while the local environment is
    unavailable.
  - Initial GPU snapshot showed the process had entered self-play: GPU0 about
    `41%` / `4118 MiB`, GPU1 about `33%` / `1991 MiB`.
  - Data reference:
    `docs/training-data-log.md#dual-gpu-2x96-score65-400-sim-7-day-continuation`
- `User observation / technical operation`: The user observed that terminal
  dead stones were already causing problems and asked to add clear obvious-dead
  cleanup into the active training round, but only after testing it first.
  - The active no-cleanup run was left running during implementation/testing.
  - Tests passed for the conservative cleanup path: `5 passed, 2 skipped` in
    terminal reward/play-server focused tests, plus a tiny CPU `sgfmill`
    training smoke with `terminal_dead_stone_cleanup=true`.
  - A more aggressive edge-cleanup idea was tested and rejected because it can
    falsely kill edge groups; this is exactly the kind of hidden label bug we
    want the notes to preserve.
  - The cleanup being used is intentionally conservative: it only removes a
    whole group with fewer than two solid eyes when removing it creates a
    non-edge-touching empty region bordered only by opponent stones. It does
    not solve edge deaths, seki, ko, or complex life-and-death.
  - After tests passed, the no-cleanup score6.5/400 segment was stopped at
    cycle `276`, backed up to
    `/root/autodl-tmp/diamondgo-checkpoint-backups/score6p5_400sims_precleanup_cycle276_latest.pt`,
    and a cleanup-enabled 7-day continuation was launched from that checkpoint.
  - New server PIDs at launch: train `230774`, finalizer `230775`.
  - Data reference:
    `docs/training-data-log.md#dual-gpu-2x96-score65-400-sim-7-day-continuation`

### Comparison Experiment: Score6.5 Cleanup 300-Sim Cycle596

- `User request`: Run a new "比较实验" at `300` simulations.
- `Agent measurement`: The active cleanup run had reached latest cycle `596`.
  `latest.pt` was snapshotted before evaluation as
  `latest-cycle-00596.pt` so the comparison target would not move while
  training continued.
- `Experiment shape`: `10` games per pair, `5` as Black and `5` as White,
  both sides at `300` MCTS simulations, `opening_temperature_moves=0`.
  - Dashboard:
    `artifacts/compare-experiment-score6p5-cleanup-300sims-cycle596-20260609/dashboard.html`
  - Game replay dashboard:
    `artifacts/compare-experiment-score6p5-cleanup-300sims-cycle596-20260609/games_dashboard.html`
  - Data reference:
    `docs/training-data-log.md#score65-cleanup-300-sim-comparison-experiment-cycle596`
- `Result`: latest cycle `596` is very strong in this slice: it beat
  `cycle-540` by `10/10` and `cycle-500` by `10/10`, with both color splits
  `5/5`.
- `Caveat`: `latest596-vs-500` had early first-pass `<=40` in `5/10` games,
  so inspect games before treating the sweep as a clean tactical-strength
  result.
- `Important anomaly`: `cycle-590` lost `0/10` to `cycle-540` at the same
  search setting, despite `latest596` sweeping `cycle-540`. This suggests
  either a sharp local checkpoint trough around `590`, or a change between
  cycles `590` and `596` that greatly improves match play. Do not infer a
  monotonic curve from sparse checkpoints.
- `Bridge result`: `cycle-540` vs `cycle-490` became `5/10` at `300` sims,
  with candidate-as-Black `0/5` and candidate-as-White `5/5`; all games were
  White wins. This reframes the earlier `100`-sim `540-vs-490 = 6/10` result
  as strongly color-sensitive rather than a clean strength jump.

### Old 4x64 Comparison: Score6.5 Cleanup Cycle602

- `User request`: In addition to same-line comparisons, compare the current
  score6.5 cleanup model with older 4x64 checkpoints.
- `Agent measurement`: The active cleanup run had advanced to cycle `602`.
  `latest.pt` was snapshotted as `latest-cycle-00602.pt` before the comparison
  so training would not move the evaluated candidate.
- `Experiment shape`: `10` games per pair, `5` as Black and `5` as White,
  both sides at `300` MCTS simulations, `opening_temperature_moves=0`.
  - Dashboard:
    `artifacts/compare-experiment-score6p5-cleanup-vs-old4x64-300sims-cycle602-20260610/dashboard.html`
  - Game replay dashboard:
    `artifacts/compare-experiment-score6p5-cleanup-vs-old4x64-300sims-cycle602-20260610/games_dashboard.html`
  - Data reference:
    `docs/training-data-log.md#score65-cleanup-vs-old-4x64-300-sim-comparison-cycle602`
- `Result`: cycle `602` scored `5/10` against old 4x64 `score2.5 cycle300`
  and old 4x64 `score5.5 cycle530`, then `10/10` against old 4x64
  `score5.5 cycle590` and `cycle650`.
- `Important caveat`: the two `5/10` results were pure color splits:
  candidate-as-Black `5/5`, candidate-as-White `0/5`. This is not a clean
  "equal strength" signal; it says the matchup is heavily color/rule/komi
  sensitive under the current candidate scoring setup.
- `Useful signal`: no pair had early first-pass `<=40`, so this comparison is
  not obviously polluted by the earlier early-pass failure mode.
- `Follow-up`: The user noticed that many replayed games had identical
  positions. This was real: the greedy comparison used
  `opening_temperature_moves=0`, so each candidate/opponent/color combination
  mostly repeated one line.
  - A display/diagnostic rerun used `opening_temperature_moves=12` with the
    same `300` simulations per side.
  - New replay dashboard:
    `artifacts/compare-experiment-score6p5-cleanup-vs-old4x64-300sims-opening12-cycle602-20260610/games_dashboard.html`
  - Diversity improved from only `2/10` unique first-20-move sequences per pair
    in the greedy run to `10/10` unique first-20-move sequences per completed
    pair in the opening-temperature run.
  - This should be the default for human-facing showcase/evaluation games.
    Keep greedy `opening_temperature_moves=0` only when intentionally measuring
    deterministic best-line strength.

### Small Eval: Score6.5 Cleanup Cycle770

- `User request`: Run a "small eval" for the active score6.5 cleanup line.
- `Agent measurement`: The active server run had reached cycle `770`.
- `Display bundle`:
  `artifacts/small-eval-score6p5-cleanup-cycle770-20260611/index.html`
  - Recorded self-play viewer:
    `artifacts/small-eval-selfplay-score6p5-cleanup-cycle700-770-20260611/index.html`
  - Play AI:
    `http://127.0.0.1:8787/viewers/play-ai.html?checkpoint=dualgpu2x96-score6p5-cleanup-cycle-00770-latest&v=small-eval-score6p5-cleanup-770`
  - Pairwise replay:
    `artifacts/pairwise-plus50-cleanup-score6p5-cycle770-vs720-100sims-opening12-10games-20260611/games_dashboard.html`
  - Data reference:
    `docs/training-data-log.md#small-eval-score65-cleanup-cycle770`
- `Self-play snapshot`: Recorded cycles `700`, `730`, `760`, and `770`
  were loaded from training cycle records, not regenerated. At cycle `770`,
  Black won `64/96`, early first pass `<=40` occurred in `19/96`, and terminal
  cleanup removed `10` black stones / `46` white stones.
- `Pairwise plus-50`: cycle `770` vs cycle `720`, `100` sims each, `10` games,
  `opening_temperature_moves=12`. Result was `5/10`: candidate as Black `4/5`,
  candidate as White `1/5`, with `10/10` unique first-12-move lines.
- `Interpretation`: The latest checkpoint is playable and the recorded games
  are available for inspection, but the plus-50 slice does not show a clean
  strength jump. The color split is still prominent.

### Training Control: MCTS Doubled From 400 To 800

- `User request`: Double the active training MCTS count.
- `Technical operation`: The active score6.5 cleanup run was not overwritten.
  The old `400`-simulation training process and its finalizer were stopped,
  its `latest.pt` was backed up, and a new `800`-simulation continuation was
  started from the same line.
- `Transition point`: Before stopping, the old run had latest metric cycle
  `834`; the latest regular checkpoint file was `cycle-00830.pt`. The new
  continuation resumes from the old run's `latest.pt`, so later raw cycle IDs
  are not directly comparable to the earlier 400-sim segment without noting
  the search-budget change.
- `Backup`:
  `/root/autodl-tmp/diamondgo-checkpoint-backups/score6p5_cleanup_400sims_to800_20260611-113012_latest.pt`
- `New run`:
  `/root/diamondgo/artifacts/multiworker-9x9-cont-dualgpu-2x96-score6p5-cleanup-margin0p2-800sims-max150-after400-20260611`
- `New scripts`:
  `tools/server/run_cont_dualgpu_2x96_score6p5_cleanup_margin0p2_800sims_after400_20260611.sh`
  and
  `tools/server/finalize_cont_dualgpu_2x96_score6p5_cleanup_margin0p2_800sims_after400_20260611.sh`
- `Preserved config`: `score_komi=6.5`, `input_komi=false`,
  `terminal_dead_stone_cleanup=true`, `score_margin_reward_scale=0.2`,
  `2x96`, `12` workers, `8` games/worker, `max_moves=150`, same noise,
  temperature, replay, batch, and optimizer settings.
- `Changed config`: self-play simulations per move changed from `400` to
  `800`.
- `Time handling`: The continuation keeps the original 7-day experiment end
  time (`2026-06-14T23:05:43+08:00`) instead of silently starting a new full
  7-day clock. At launch this became `5015` remaining minutes.
- `Immediate resource check`: The new Python process started successfully with
  `simulations=800`; GPU memory rose to about `10.8 GiB` on GPU0 and `6.0 GiB`
  on GPU1, still within the 12 GiB cards.

### Training Control: Score Komi Raised To 7.5

- `User request`: Change the active training komi to `7.5`.
- `Technical operation`: The active `score_komi=6.5`, `800`-simulation
  continuation was stopped and preserved. Its `latest.pt` and `metrics.jsonl`
  were backed up, then a new continuation was started from that `latest.pt`
  with `score_komi=7.5`.
- `Transition point`: Before stopping, the `score6.5/800` segment had latest
  metric cycle `927`; the latest regular checkpoint file was `cycle-00920.pt`
  and the latest recorded SGF was `cycle-00925.sgf`.
- `Backup`:
  `/root/autodl-tmp/diamondgo-checkpoint-backups/score6p5_cleanup_800sims_to_score7p5_20260612-185546_latest.pt`
- `New run`:
  `/root/diamondgo/artifacts/multiworker-9x9-cont-dualgpu-2x96-score7p5-cleanup-margin0p2-800sims-max150-after6p5-20260612`
- `New scripts`:
  `tools/server/run_cont_dualgpu_2x96_score7p5_cleanup_margin0p2_800sims_after6p5_20260612.sh`
  and
  `tools/server/finalize_cont_dualgpu_2x96_score7p5_cleanup_margin0p2_800sims_after6p5_20260612.sh`
- `Preserved config`: `input_komi=false`, `komi_metadata=0.5`,
  `terminal_dead_stone_cleanup=true`, `score_margin_reward_scale=0.2`,
  `2x96`, `800` self-play simulations, `12` workers, `8` games/worker,
  `max_moves=150`, same noise, temperature, replay, batch, and optimizer
  settings.
- `Changed config`: `score_komi` changed from `6.5` to `7.5`.
- `Time handling`: The continuation keeps the original experiment end time
  (`2026-06-14T23:05:43+08:00`). At launch this became `3129` remaining
  minutes.
- `Immediate resource check`: The new process started successfully with
  `score_komi=7.5`; GPU memory was high at about `11.8 GiB / 12 GiB` on GPU0
  and `7.6 GiB / 12 GiB` on GPU1, so monitor for OOM or stalled metrics.

### Server Expiry Stop And Final Preservation

- `User request`: Stop the server work and make sure repository/artifacts are
  uploaded because the server expires today.
- `Technical operation`: All DiamondGo training/eval/finalizer wrapper
  processes and leftover GPU Python workers were stopped. A final GPU check
  showed no compute apps and about `1 MiB` used on each GPU.
- `Important result`: The `score7.5/800` continuation did start successfully,
  but it did not finish a full cycle before the stop request. It produced
  config/log metadata only, no metrics row, no checkpoint, and no SGF.
- `Last complete active-training data before the stop`: the preceding
  `score6.5/800` segment reached latest metric cycle `927`; latest regular
  checkpoint was `cycle-00920.pt`, latest recorded SGF was `cycle-00925.sgf`,
  and recent-12 black win rate was about `67.54%`.
- `Local final preserve`: downloaded and SHA-verified under
  `artifacts/server-final-preserve-20260614/`, total about `626.7 MB`.
  Contents include selected checkpoint archives, transition checkpoint
  backups, cycle-record archives for the main cleanup runs, and the
  `score7.5/800` metadata archive.
- `Server git caveat`: the server working copy still reported old HEAD
  `7199135`; operational scripts had been copied to the server directly.
  The authoritative source/docs/scripts are on GitHub via local commits through
  `7d2785e` and later preservation notes.

## 2026-06-16

### Planned Comparison: Previous-Two-Move Input

- `User request`: Prepare the next comparison experiment with only one new
  model trick: add the previous two moves as neural-network input planes.
- `Technical operation`: Added optional `history_moves` plumbing through rules,
  model construction, training configs, checkpoint configs, eval loaders,
  tactical eval, and the play server. The default is `0`, so old scripts and
  checkpoints keep their old input shape.
- `Input encoding`: with `--no-input-komi --history-moves 2`, the network input
  has `5` planes: own stones, opponent stones, to-play-black, previous move,
  and previous-previous move. A pass move is encoded as an all-zero history
  plane.
- `Training-script plan`: new scripts were added instead of overwriting old
  ones:
  - `tools/server/run_fresh_dualgpu_2x96_history2_score6p5_cleanup_nomargin_800sims.sh`
  - `tools/server/finalize_fresh_dualgpu_2x96_history2_score6p5_cleanup_nomargin_800sims.sh`
- `Planned config`: fresh start, `2x96`, `score_komi=6.5`, `komi=0.5` metadata,
  `input_komi=false`, `history_moves=2`, conservative terminal dead-stone
  cleanup enabled, score-margin reward disabled (`0.0`), `800` self-play
  simulations by default, `12` workers, `8` games per worker, `96` games/cycle,
  `max_moves=150`, same root noise/temperature/augmentation settings as the
  recent dual-GPU line.
- `Trace policy`: full root search-tree distributions are now optional in
  training records. The planned run keeps normal cycle SGF/trace records every
  `5` cycles, but only keeps full root distributions every `20` cycles and only
  for the first `5` games; other records keep the top `5` actions.
- `Comparison caution`: because the input shape changes, compare this run by
  total positions/games/wall-clock plus eval results. Raw cycle IDs alone are
  not a clean proxy for equal training exposure.

## 2026-06-22

### Training Control: Score Komi Back To 6.5

- `User request`: Change the active 4x64 history-2 training line from
  `score_komi=7.5` back to `score_komi=6.5`.
- `Technical operation`: Stopped the active score7.5 run, preserved its
  `latest.pt`, cleaned orphan `multiprocessing.spawn` self-play workers that
  continued holding GPU memory, then restarted from the preserved checkpoint
  with only `score_komi` changed.
- `Transition point`: the score7.5/temp0.7/600-sim segment had latest metric
  cycle `836` at the stop point.
- `Backup`:
  `/root/autodl-tmp/diamondgo-checkpoint-backups/score7p5_to_score6p5_20260622-003019_cycle836_latest.pt`
- `Previous run`:
  `/root/diamondgo/artifacts/multiworker-9x9-cont-dualgpu-4x64-history2-score7p5-cleanup-margin0p2-600sims-max150-temp0p7-late0p2-30w-notimelimit-after-temp0p75-20260621-145816`
- `New run`:
  `/root/diamondgo/artifacts/multiworker-9x9-cont-dualgpu-4x64-history2-score6p5-cleanup-margin0p2-600sims-max150-temp0p7-late0p2-30w-notimelimit-after-score7p5-cycle836-20260622-003019`
- `New PID file`:
  `/root/diamondgo/artifacts/dualgpu_4x64_history2_score6p5_margin0p2_600sims_max150_temp0p7_late0p2_30w_notimelimit.train.pid`
- `Preserved config`: `komi_metadata=0.5`, `input_komi=false`,
  `terminal_dead_stone_cleanup=true`, `score_margin_reward_scale=0.2`, `4x64`,
  `history_moves=2`, `600` self-play simulations, `30` workers, `8`
  games/worker, `240` games/cycle, `max_moves=150`,
  `train_steps_per_cycle=64`, `batch_size=256`, `replay_size=100000`,
  `temperature=0.7` for the first `16` moves, `late_temperature=0.2`,
  root Dirichlet noise `alpha=0.15` and fraction `0.25`,
  `root_policy_temperature=1.1`, `record_every=5`, and full root traces every
  `20` cycles for the first `5` games.
- `Changed config`: `score_komi` changed from `7.5` to `6.5`.
- `Immediate resource check`: the new process started successfully with
  wrapper PID `181740` and training Python PID `181744`; GPU memory after
  restart was about `12.3 GiB / 24 GiB` on GPU0 and `6.7 GiB / 24 GiB` on
  GPU1. No OOM or traceback was observed before handing back to heartbeat
  monitoring.

## 2026-06-24

### Preparing The First 13x13 Experiment

- `User request`: stop the current 9x9 line and prepare a 13x13 experiment
  while keeping the current architecture and core training tricks.
- `Technical operation`: stopped the active 9x9 run at latest metric cycle
  `1171`, preserved its `latest.pt`, and verified that no DiamondGo training
  or self-play worker process remained alive.
- `Backup`:
  `/root/autodl-tmp/diamondgo-checkpoint-backups/9x9_score6p5_margin0p2_600sims_stop_for_13x13_20260624-201836_cycle1171_latest.pt`
- `Code fix`: the lower-level configs already had `board_size`, but
  `batched_demo.py`, `multiworker_train.py`, and
  `multiworker_train_dualgpu.py` did not expose or pass through
  `--board-size`. This is now wired through, so 13x13 runs can be launched
  without editing source constants.
- `Planned config`: fresh 13x13 start, `4x64`, `history_moves=2`,
  `input_komi=false`, `komi=0.5` metadata, conservative terminal dead-stone
  cleanup enabled, score-margin reward scale `0.2`, `200` self-play
  simulations, `30` workers, `8` games per worker, `240` games/cycle,
  `max_moves=250`, `temperature=0.7` for the first `16` moves,
  `late_temperature=0.2`, same root noise and training batch settings as the
  current 9x9 history-2 line.
- `Dynamic score komi`: start from `score_komi=2.5` and choose among
  `2.5,4.5,6.5,7.5,8.5`. If the rolling black win rate exceeds `70%`, move
  one ladder step upward for the next cycle; if the rolling white win rate
  exceeds `70%`, move one ladder step downward. The default rolling window is
  the latest `5` cycles, using available completed cycles before cycle 5.
- `Fresh-start caveat`: 9x9 checkpoints cannot be directly resumed because the
  action count changes from `82` to `170`; trunk-transfer would need a separate
  migration script and is not part of this first 13x13 run.
- `Debug result`: local CPU smoke tests and server dual-GPU smoke tests passed.
  A 30-worker server stress test failed under the default open-file limit
  `1024` with `Too many open files`; the same stress test passed after
  setting `ulimit -n 65535`.
- `Run script`:
  `tools/server/run_fresh_dualgpu_13x13_4x64_history2_autokomi_margin0p2_200sims.sh`
- `Human-facing plan`:
  `docs/13x13-training-plan-20260624.md`

### 13x13 Early-Pass Intervention

- `User observation/request`: the initial 13x13 run had an early-pass failure
  mode; manually mask pass within the first `120` moves and consider a modest
  learning-rate increase.
- `Agent measurement`: before the intervention, the active 13x13 auto-komi run
  was stopped at latest metric cycle `26`. The latest cycle had
  `score_komi=8.5`, black win rate `1.0`, `pass_move_fraction=0.3255`,
  first-pass median about `6`, and all `240` games ended by pass.
- `Backup`:
  `/root/autodl-tmp/diamondgo-checkpoint-backups/13x13_autokomi_pre_passmask_lr_20260624-231552_cycleunknown_latest.pt`
- `Technical operation`: added optional `min_pass_move`. When positive, pass is
  removed from MCTS legal actions while the current game has fewer than that
  many played moves. The mask is applied before root and leaf expansion, so the
  search tree, policy target, and sampled action use the same action set.
- `Safety detail`: pass is not masked if no board point is legal, preventing an
  empty MCTS root in small-board or full-board edge cases.
- `Changed config for the continuation`: `min_pass_move=120`; AdamW learning
  rate raised from `0.001` to `0.0015`. Other 13x13 defaults remain unchanged
  unless explicitly overridden by the launch script.
