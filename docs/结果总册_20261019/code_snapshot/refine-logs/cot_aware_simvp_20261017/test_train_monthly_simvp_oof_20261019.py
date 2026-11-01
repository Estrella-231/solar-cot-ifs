from types import SimpleNamespace

from train_monthly_simvp_oof_20261019 import build_train_dataset


def main():
    calls = []

    class FakeDataset:
        def __init__(self, args, norm, split):
            calls.append(split)
            self.split = split

    base = SimpleNamespace(MonthlyDataset=FakeDataset)
    ds = build_train_dataset(base, object(), {})
    assert ds.split == "train"
    assert calls == ["train"]
    print("OOF_TRAINER_TRAIN_ONLY_WITNESS_PASS")


if __name__ == "__main__":
    main()
