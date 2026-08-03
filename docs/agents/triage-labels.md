# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the
actual label strings used in this repo's issue tracker.

| Label in mattpocock/skills | Label in our tracker | Meaning                                  |
| -------------------------- | -------------------- | ---------------------------------------- |
| `needs-triage`             | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`               | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent`          | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human`          | `ready-for-human`    | Requires human implementation            |
| `wontfix`                  | `wontfix`            | Will not be actioned                     |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the corresponding
label string from this table.

## How a label is applied

The tracker is markdown files in `Cephla-Lab/AI-docs`, not an issue tracker with real labels
(see `issue-tracker.md`). A label is a `Status:` line near the top of the file:

    Status: ready-for-agent

The `to-do` / `in-progress` / `done` folder is the coarse state; the `Status:` line is the finer
triage role within it. A file in `done/` needs no `Status:` line.

Edit the right-hand column above if the vocabulary ever changes.
