# Retry 02 note

PBS job `215545.tc6000` and first isolated retry `215547.tc6000` both received six A800s and exited before data access/inference because the scheduler's human-readable `qstat -f` output folds the `exec_gpus` field in a way that defeated the initial parsers. The receipts and logs are preserved in their original directories; neither attempt created predictions or read validation targets, test, or buffer payloads.

This directory is a second isolated retry. Its allocation parser reads the XML-form `qstat -fx` `<exec_gpus>` element, extracts every GPU ID, and is regression-tested against completed job `215547.tc6000` before submission. Model weights, cache, sample contract, inference, score, and six-GPU request remain unchanged. The retry has a separate log, predictions, score, and receipt path.
