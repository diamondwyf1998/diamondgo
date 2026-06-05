# Replay and Analysis UI Options

We need three things:

- human vs DiamondGo;
- DiamondGo self-play replay;
- visibility into search visits, priors, values, and principal variations.

## Recommendation

Start with Sabaki plus a DiamondGo GTP engine.

Sabaki is the best first UI because it is a mature local SGF editor, supports
GTP engines, supports board analysis for compatible engines, and works well as a
general Go board for non-KataGo engines. The integration cost is manageable:
implement standard GTP first, then add a small KataGo/Sabaki-compatible analysis
command for visits and candidate moves.

Windows portable download:

- https://github.com/SabakiHQ/Sabaki/releases/download/v0.52.2/sabaki-v0.52.2-win-x64-portable.exe

Suggested local path:

```powershell
C:\Users\diamo\Documents\diamondgo\tools\sabaki-v0.52.2-win-x64-portable.exe
```

After download, open the CPU smoke SGF:

```powershell
C:\Users\diamo\Documents\diamondgo\artifacts\cpu-demo-9x9.sgf
```

## Options

### Sabaki

Use for:

- playing against DiamondGo through GTP;
- opening SGF files from self-play;
- autoplaying games;
- seeing comments, variations, and board markup;
- eventually displaying live analysis once we implement compatible output.

Integration path:

1. `python -m diamondgo.gtp --board-size 9 --checkpoint ...`
2. Standard GTP commands: `protocol_version`, `name`, `version`, `boardsize`,
   `clear_board`, `komi`, `play`, `genmove`, `legal_moves`.
3. Analysis commands: `analyze` / `genmove_analyze` with top moves, visits,
   winrate/value, and principal variation.

### KaTrain

Use as reference UX, not the first custom-engine UI.

KaTrain has excellent play/review workflow, but it is strongly built around
KataGo and KataGo's analysis engine. It is useful for understanding what we want
from a teaching/review interface, and for comparing our weak baby model against
real KataGo behavior.

### LizGoban

Use as reference for search-process visualization.

LizGoban is focused on Leela Zero/KataGo-style real-time analysis and exposes
interesting search behavior. It is a good target if we decide to mimic the
Leela/KataGo analysis protocol more completely.

### Lizzie

Useful reference, but not the first pick.

Lizzie is historically important for Leela Zero and KataGo analysis. For our
workflow, Sabaki is simpler as the first bridge, and LizGoban/KaTrain are better
modern references for rich analysis UX.

## DiamondGo-Specific UI Contract

For early experiments, every searched move should be exportable in two forms:

- SGF comments: top-k moves, visits, policy prior, mean value, and PV.
- Analysis stream: one line per analysis update, compatible with Sabaki's
  `analyze` expectation where practical.

This keeps us from hiding learning behavior behind a black box. If the UI shows
a weird move, we should be able to inspect whether it came from raw policy,
search correction, low visits, value overconfidence, or legality masking.
