# TODOS

## Resume / idempotency for batch runs (from IMA-186 eng review)
- **What:** On re-run over an existing output dir, skip wells whose output already exists.
- **Why:** A 1536-well plate takes hours. A crash/Ctrl-C at well 800 currently forces a full re-run. Resume turns a crash into a cheap retry.
- **Pros:** Big-plate runs become interruptible/restartable; pairs with skip-and-report failure policy (re-run just the failed wells).
- **Cons:** Must define overwrite-vs-skip semantics and a reliable "well is complete" check (partial writes must not count as complete).
- **Context:** IMA-186 ships default overwrite/reprocess-all. Overwrite semantics deserve a small design pass. Natural home is IMA-192 (batch a folder of acquisitions).
- **Depends on:** IMA-186 (CLI + batch orchestration) landing first.
