from audit_future_ghi_oof_fold_feasibility import audit, bjt_day, sequence_frames
import csv


def main():
    assert len(sequence_frames("hunan_202404011645")) == 24
    assert sequence_frames("hunan_202404011645")[0] == "202404011645"
    assert bjt_day("202404011645") == "2024-04-02"
    with open("_oof_fold_witness_index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "seq_id"])
        w.writeheader()
        for i, start in enumerate(("202604010000", "202604020000", "202604030000", "202604040000")):
            w.writerow({"split": "train", "seq_id": "hunan_" + start})
        w.writerow({"split": "val", "seq_id": "hunan_202604050000"})
    try:
        r = audit(__import__("pathlib").Path("_oof_fold_witness_index.csv"), ks=(2,), salt="test")
        assert r["source_train_sequences"] == 4
        assert r["validation_payload_opened"] is False
        assert r["candidate_folds"]["2"]["oof_coverage_fraction"] <= 1.0
        assert all(f["train_heldout_frame_intersection"] == 0 for f in r["candidate_folds"]["2"]["folds"])
    finally:
        __import__("pathlib").Path("_oof_fold_witness_index.csv").unlink(missing_ok=True)
    print("OOF_FOLD_FEASIBILITY_WITNESS_PASS")


if __name__ == "__main__":
    main()
