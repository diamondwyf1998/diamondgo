from diamondgo.overnight_train import should_save_cycle_checkpoint


def test_default_checkpoint_schedule_is_dense_early_then_every_ten() -> None:
    saved = [
        cycle
        for cycle in range(1, 76)
        if should_save_cycle_checkpoint(cycle, checkpoint_every=10)
    ]

    assert saved == [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70]


def test_checkpoint_schedule_keeps_explicit_late_interval() -> None:
    saved = [
        cycle
        for cycle in range(48, 66)
        if should_save_cycle_checkpoint(
            cycle,
            checkpoint_every=20,
            early_checkpoint_cycles=50,
            early_checkpoint_every=5,
        )
    ]

    assert saved == [50, 60]
