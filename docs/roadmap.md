# DiamondGo Roadmap

This roadmap is optimized for learning signal, not immediate playing strength.
Each stage should produce visible behavior we can inspect.

## Stage 0: Project Wiring

- Choose the first rules backend.
- Define config objects for board, MCTS, model, self-play, and training.
- Add a small residual network.
- Add command entrypoints that can print and validate configs.

Success criterion: the repository has a coherent shape and can run smoke checks.

## Stage 1: Minimal 9x9 AlphaZero

- Encode board state into network planes.
- Implement legal move masking through the rules adapter.
- Implement PUCT MCTS.
- Generate self-play games on 9x9.
- Train on policy visit counts and game outcome value.

Success criterion: a checkpoint beats a random/legal baseline on 9x9.

## Stage 2: Observability

- Save selected self-play positions.
- Track policy entropy, value loss, policy loss, and illegal-move mass before masking.
- Add checkpoint arena matches.
- Produce compact experiment reports.

Success criterion: we can explain what changed after each trick, not just whether
the loss moved.

## Stage 3: Engineering Tricks To Compare

Candidate ablations:

- Dirichlet noise on root priors.
- Temperature schedules.
- Replay buffer size and sampling strategy.
- Virtual loss and batched inference.
- Symmetry augmentation.
- Resignation thresholds.
- Checkpoint gating.

Each trick should get a small experiment note with:

- hypothesis;
- exact config diff;
- observed behavior;
- whether it is kept.

## Stage 4: KataGo-Inspired Extensions

- Ownership or territory-style auxiliary head.
- Score belief/value variants.
- Rule variation support.
- More careful komi handling.
- Playout cap randomization.

These come after the minimal loop is inspectable.
