# Issue tracker: AI-docs (Cephla-Lab/AI-docs)

Engineering work for this repo is tracked as **markdown documents** in a separate repo:
`~/CEPHLA/AI-docs`, under `SquidXplorer/` (github.com/Cephla-Lab/AI-docs). It is not GitHub
Issues — this repo's Issues tab is unused — and it is not Linear.

`support.cephla.com/tickets` is **manual, human-only** intake. Agents cannot reach it and must
not assume its contents. Items reach the agent loop only when a human copies them into AI-docs.

`IMA-###` ids are **legacy**. They appear in `.spec/STATE.md`, older branch names,
`docs/ima-*-eng-review.md`, and commit messages. No new ids are issued; a document's path and
title are its identity.

## The status semaphore

A work item is a file, and its **folder is its status**:

    SquidXplorer/to-do/        not started
    SquidXplorer/in-progress/  being worked
    SquidXplorer/done/         complete AND merged to main

A new project gets **its own directory** with the same three folders, as
`SquidXplorer/SimpleXplorer/` already does.

Only the user moves a doc to `done/` — ask, don't move it yourself.

## Naming

Per `SquidXplorer/README-naming.md`, titles must read from outside, because
dev-dashboard.cephla.com shows them to the whole team:

    YYYY-MM-DD-<feature>-design.md     what and why, the architecture
    YYYY-MM-DD-<feature>-plan.md       numbered tasks, each with verification
    YYYY-MM-DD-<feature>-decisions.md  calls made, with the reasoning
    action-items/YYYY-MM-DD-<topic>-sweep.md

The slug names the feature or subsystem, never an internal artefact. Don't prefix with
`squidxplorer-` or `viewer-`; the directory already says it.

## When a skill says "publish to the issue tracker"

Create a file under `~/CEPHLA/AI-docs/SquidXplorer/to-do/` following the naming convention
above. AI-docs is a **separate git repo** — commit there separately from this one.

## When a skill says "fetch the relevant ticket"

Read the file at the referenced path. Check all three semaphore folders; the user will normally
give you the path or the feature slug.

## Triage state

The folder is the coarse status. Finer triage roles are a `Status:` line near the top of the
file — see `triage-labels.md` for the strings.

## Human decisions: action items

A decision an agent cannot make is **not** a comment in the doc. It is an unchecked task with an
@mention in `SquidXplorer/action-items/YYYY-MM-DD-<topic>-sweep.md`, with indented `- Why:` and
`- From:` sub-bullets. The dashboard groups these by assignee. Prefix genuinely urgent items with
`!!`, sparingly.

Coordination intel (merge order, PR dependencies, conflict warnings) appends as a dated unchecked
item to `SquidXplorer/NOTES.md`. Merge priorities and target dates live in
`SquidXplorer/PR-PLAN.md` — consult it before planning merge order. Multi-PR programs are declared
in `SquidXplorer/INITIATIVES.md`; keep an initiative's keywords in branch and PR titles so it
aggregates.

## Wayfinding operations

Used by `/wayfinder`. An effort is a **directory**, and the semaphore is the ticket state — no
separate `Status: claimed` convention needed.

- **Map**: `SquidXplorer/<effort>/map.md` — Destination, Notes, Decisions-so-far, Not yet
  specified, Out of scope. Lives at the effort root, outside the semaphore folders.
- **Child ticket**: `SquidXplorer/<effort>/to-do/NN-<slug>.md`, numbered from `01`, the question
  in the body. A `Type:` line records `research`/`prototype`/`grilling`/`task`.
- **Blocking**: a `Blocked by: NN, NN` line near the top. Unblocked when every file it names sits
  in `done/`.
- **Frontier**: the files in `<effort>/to-do/` that are unblocked. First by number wins.
- **Claim**: `git mv` the file to `<effort>/in-progress/` before any work.
- **Resolve**: append the answer under `## Answer`, `git mv` to `<effort>/done/`, then append a
  gist + link to the map's Decisions-so-far.

**Gotcha:** moving a file between folders breaks every `([source](../to-do/...))` link in
`action-items/`, and it 404s silently on the dashboard. Fix the links in the same commit as the
move.

## What does NOT go in AI-docs

`CONTEXT.md` and `docs/adr/` live in **this** repo — they describe the code, not a plan for
changing it. Plans, specs, designs, decisions and action items go in AI-docs. See `domain.md`.
