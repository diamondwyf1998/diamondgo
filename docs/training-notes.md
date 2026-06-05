# Training Notes

This file records the human-facing story of training observations and
operations. Keep only important facts, phenomena, and next actions here. Put
large tables and raw measurements in `docs/training-data-log.md`, then reference
them from this file.

## 2026-06-05 Notes

### Configuration And Operation Changes

- Increased the self-play move cap during the experiment sequence:
  - early runs and first multiworker checkpoints used `max_moves=80`
  - the 630-690 stage used `max_moves=160`
  - the 700+ optimized run used `max_moves=120`
  - Data reference: `docs/training-data-log.md#run-and-config-timeline`
- Komi handling was changed by recent source work, partly by the other agent:
  - legacy artifacts generally serialize `komi=0.5`
  - current code separates model-input `komi=0.5` from scoring `score_komi=6.5`
  - important intent: do not change the model input plane when continuing old
    checkpoints; change only terminal scoring/value labels via `score_komi`
  - Data reference: `docs/training-data-log.md#run-and-config-timeline`
- A fresh `komi=6.5` run was briefly started, then stopped after realizing
  `komi` is part of the model input. The active follow-up run instead resumes
  the old 0.5-komi model with `komi=0.5` and `score_komi=6.5`.
  - Run reference:
    `artifacts/multiworker-9x9-resume0p5-score6p5-100sims-120moves-2h-20260605`
  - Data reference: `docs/training-data-log.md#score-komi-continuation-cycles-857`
- Built a dedicated single-game viewer for qualitative checkpoint inspection:
  - `artifacts/selfplay-showcase-swing-760-830-20260605/viewer.html`
  - Reason: older batched self-play dashboards mixed three parallel games into
    one position stream, which is misleading for reading a single game.
  - Data reference: `docs/training-data-log.md#single-game-showcase-cycles-760-830`

### Observed Phenomena

- Win-rate asymmetry appeared strongly in the 700+ run. Black was favored around
  700-759, White became strongly favored around 770-819, and Black became
  favored again after about 820.
  - Main checkpoint range: `700-856`
  - Reversal checkpoints to inspect: `760, 770, 780, 790, 800, 810, 820, 830`
  - Data reference: `docs/training-data-log.md#black-win-rate-and-margin-cycles-700-856`
- Black tends to prefer the center in several sampled checkpoints. The strongest
  examples are around `630, 650, 660, 690, 700, 730, 750`, where Black's first
  20 moves land in the center 5x5 much more often than White's.
  - Data reference: `docs/training-data-log.md#center-5x5-opening-distribution`
- Around the color reversal, Black's opening distribution looks less stable.
  In small samples:
  - `760`: first move fixed at `F4` in all three sampled games
  - `770-800`: White wins most sampled games, and Black is less center-heavy
  - `810`: one sampled game starts with Black `pass`, which is abnormal
  - `830`: first move fixed at `G2` in all three sampled games, and Black wins
    all three sampled games
  - Data reference: `docs/training-data-log.md#single-game-showcase-cycles-760-830`
- Qualitative board reading: during the White-favored phase, Black stones can
  look more dispersed and easier for White to surround in large regions. This is
  an observation from visual inspection, not yet a quantified metric.
  - Checkpoints to revisit visually: `770, 780, 790, 800, 810`
  - Viewer reference:
    `artifacts/selfplay-showcase-swing-760-830-20260605/viewer.html`
- After switching only terminal scoring to `score_komi=6.5` while keeping the
  old model-input `komi=0.5`, the continuation run rapidly swung from a Black
  advantage at cycle `857` to a strong White advantage from about `859` onward.
  This should be interpreted as a rule-label shock / adaptation signal, not as a
  clean strength improvement.
  - Main checkpoint range so far: `857-871`
  - Data reference: `docs/training-data-log.md#score-komi-continuation-cycles-857`

### Tests Still Needed

- Tactical learning checks:
  - whether the model has learned to capture
  - whether the model has learned atari and atari defense
  - whether the model avoids obvious self-atari or group-death moves
  - Candidate tool: `src/diamondgo/tactical_eval.py`
  - Important checkpoints to test: old `850/latest`, score-komi continuation
    `860/870/latest`, and later finished continuation checkpoints.
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

The current evidence points to self-play distribution instability rather than a
smooth monotonic strength increase. The color reversal and fixed first-move
patterns suggest that the model may periodically collapse into transient
opening habits. The `810` first-move pass is a special warning sign and should
be tested directly before trusting later qualitative conclusions.
