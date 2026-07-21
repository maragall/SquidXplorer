# IMA-192 branch — what got committed, and why so little

Branch: `juliomaragall/ima-192-batch-a-folder-of-acquisitions-subfolders-optional`
Remote: pushed to `origin`, tracking set.
Base: `main` @ `b93b179` (repo scaffold only).

## TL;DR

This branch contains **exactly one commit** and a **28-line diff**. That is not
an accident or an oversight. This was a planning turn (`/plan-eng-review`), not
an implementation turn, and the feature it plans is blocked on code that does not
exist yet (IMA-186). Almost all the work product from the session lives in files
that are deliberately outside git. This document explains what is in the commit,
what is not, and why.

---

## The one commit

### `b69f51c` — IMA-192: capture deferred subprocess-per-acquisition isolation TODO

```
 TODOS.md | 28 ++++++++++++++++++++++++++++
 1 file changed, 28 insertions(+)
```

**What it adds:** a single `TODOS.md` at the repo root with one deferred item —
subprocess-per-acquisition isolation.

**Why it exists:** the plan's failure-isolation design (finding 4) wraps each
acquisition in `try/except` so one bad acquisition does not abort the batch.
That catches Python exceptions but not uncatchable failures: an OS OOM-kill, a
C-extension segfault, or a SIGKILL takes the whole process down and ends the
batch with no summary. True hard isolation means running each acquisition in its
own subprocess. The outside-voice reviewer flagged this. We judged it
disproportionate to a 2-point, Low-priority, optional feature, so instead of
building it we recorded it with full context and a revisit trigger (revisit if
OOMs actually occur, or when plate sizes grow).

**Why it is the only committed artifact:** it is the only review output that
belongs in the repository. It is real, forward-looking engineering context that
the next person needs regardless of whether IMA-192 is ever built. Everything
else the session produced is either a plan (local, gitignored) or tooling
metadata (outside the repo entirely).

---

## What is NOT in git, and where it actually lives

The review produced four other artifacts. None are in the commit. Here is each
one, its real absolute path, and why it is not committed.

| Artifact | Absolute path | Why not committed |
|----------|---------------|-------------------|
| The full plan | `/Users/julioamaragall/CEPHLA/worktrees/SquidMIP/ima-192/.spec/open/ima-192.md` | `.spec/` is in `.gitignore` — specs are local working docs by project convention |
| Test plan | `/Users/julioamaragall/.gstack/projects/maragall-SquidMIP/julioamaragall-juliomaragall-...-eng-review-test-plan-20260704-172938.md` | Lives under `~/.gstack/`, outside the repo — gstack tooling metadata |
| Implementation tasks (JSONL) | `/Users/julioamaragall/.gstack/projects/maragall-SquidMIP/tasks-eng-review-20260704-172938.jsonl` | Same — consumed by `/autoplan`, not source |
| This document | `/Users/julioamaragall/CEPHLA/worktrees/SquidMIP/ima-192/COMMITS-EXPLAINED.md` | Written for your review; currently untracked |

The `.gitignore` that keeps the spec out of git:

```
.spec/
.sprint/
__pycache__/
*.pyc
.DS_Store
```

So when you merge this branch to `main`, the diff you review is **only
`TODOS.md`**. The plan itself never enters version control on this branch.

---

## Why the branch is this light

Three reasons, in order of importance:

1. **It is a plan, not an implementation.** `/plan-eng-review` locks decisions
   before code is written. The deliverable is agreement on architecture, edge
   cases, and test coverage — not source files.

2. **The feature is blocked on IMA-186.** IMA-192 loops over the
   single-acquisition CLI (IMA-186), which is `Todo/unstarted`. Per your own
   review order, IMA-192 is #7 — last, or skipped for v1. There is nothing to
   build against yet, so there is nothing to commit beyond the deferred-work
   marker.

3. **Project convention keeps specs out of git.** `.spec/` is gitignored, so the
   plan document (the largest artifact) is local by design.

---

## What the branch actually encodes (the decisions, not the code)

Even though the diff is 28 lines, the branch represents six locked decisions plus
one cross-model reversal. These live in the (gitignored) plan and drive the eight
implementation tasks that will land when IMA-186 exists:

1. **Fold batch into IMA-186's CLI** — one code path, not a separate command.
2. **Auto-detect single-vs-batch, with ambiguity as a hard error** — this is the
   one that reversed my initial call. I first recommended "parent wins" on an
   ambiguous path; the outside voice showed that silently produces 1 plate
   instead of N, which is exactly the failure acceptance criterion 1 forbids.
   You steered the synthesis: keep auto-detect, but make ambiguity a loud error.
3. **Non-recursive** — direct children only.
4. **Junk children warned and skipped; zero recognized children is a hard error**
   (never a silent exit 0).
5. **Sequential outer loop, parallel inner** — reuse the memory-bounded
   per-well worker pool (built in IMA-188); no nested pools.
6. **Failure isolation = per-acquisition records + summary table + exit code**
   (0 all-ok, 1 any-failed).

Plus the output layout default (mirror subfolder names into
`--output-dir/<name>/`, overwrite only with `--force`).

---

## Build-order context that shaped the commit

From your review order (2026-07-05), IMA-192 is #7. Two forward-dependencies
follow, and both are recorded in the plan:

- The "fold batch in" decision must be carried into **IMA-186's design review at
  slot #6**, because by the time #7 is reached IMA-186 already exists. If it is
  not deliberate there, IMA-192 becomes a retrofit.
- **T8** (characterize `open_reader`'s error behavior on a non-acquisition)
  belongs at **slot #1 (IMA-189)**, not here. The detection rule assumes the
  reader fails cleanly and cheaply; that assumption has to be pinned when the
  reader is committed.

Neither of these is in the commit because neither is IMA-192's code to write.
They are notes to the upstream slots.

---

## What your feedback could change

Given how light the branch is, the highest-leverage feedback is likely about
**what should or should not be committed**, for example:

- Should the plan spec be committed after all (drop `.spec/` from `.gitignore`),
  so the design travels with the branch instead of living only on my machine?
- Should `TODOS.md` carry more than the one deferred item (e.g. capture T8's
  relocation to IMA-189 as a visible cross-ticket TODO)?
- Should this branch exist at all before IMA-186, or should the decisions be
  moved wholesale into IMA-186's ticket and this branch closed?

Tell me where you land and I will modify the branch accordingly.
