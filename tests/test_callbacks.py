from hippo.training.callbacks import EarlyStopping


def test_early_stopping_triggers_after_patience():
    es = EarlyStopping(patience=3, mode="max")
    assert es.step(0.5) is True       # 첫 값 = 개선
    assert es.step(0.4) is False      # bad 1
    assert es.step(0.4) is False      # bad 2
    assert es.should_stop is False
    assert es.step(0.4) is False      # bad 3 -> 중단
    assert es.should_stop is True


def test_early_stopping_resets_on_improve():
    es = EarlyStopping(patience=2, mode="max")
    es.step(0.5)
    es.step(0.4)              # bad 1
    assert es.step(0.6) is True       # 개선 -> 리셋
    assert es.num_bad == 0
    assert es.should_stop is False


def test_early_stopping_min_mode():
    es = EarlyStopping(patience=2, mode="min")
    assert es.step(1.0) is True
    assert es.step(0.5) is True       # 작아짐 = 개선
    assert es.step(0.9) is False      # 나빠짐
