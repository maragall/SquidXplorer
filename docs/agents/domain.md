# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the
codebase.

## Before exploring, read these

- **`CONTEXT.md`** at the repo root
- **`docs/adr/`** — read ADRs that touch the area you're about to work in

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest
creating them upfront. The `/domain-modeling` skill (reached via `/grill-with-docs` and
`/improve-codebase-architecture`) creates them lazily when terms or decisions actually get
resolved.

## File structure

Single-context repo:

```
/
├── CONTEXT.md
├── docs/adr/
│   ├── 0001-<decision-slug>.md
│   └── 0002-<decision-slug>.md
└── squidxplorer/
```

## These live in this repo, not in AI-docs

Plans, specs, designs and action items belong in `Cephla-Lab/AI-docs` under `SquidXplorer/` (see
`issue-tracker.md`). `CONTEXT.md` and `docs/adr/` are the exception: they describe the code as it
is, not a plan for changing it, and agents need them beside the code they're reading.

## Existing contracts that act as undeclared ADRs

Predating `docs/adr/`, these files already record binding decisions. Read them before working in
the areas they govern, and treat a contradiction the same way you'd treat an ADR conflict:

- **`docs/rendering-contract.md`** — the lazy multiscale pyramid, 2D vs 3D abstractions, the GPU
  texture cap, and the rule that nothing silently downsamples
- **`docs/plate-contract.md`** — the on-disk contract, its stable and optional halves
- **`docs/DESIGN.md`** — the v1 object model (Acquisition, AcquisitionImage, OperationStack,
  Iterator, ReadAcquisition, PlateView, ArrayViewer). Note this is partly **aspirational**: some
  names in it have no class in the code. Verify against the tree before relying on one.

Decisions also sit inside module docstrings (for example the argument at `squidxplorer/reader.py`
for a structural `Protocol` with no base class). When one of those turns out to be load-bearing,
promote it to `docs/adr/` rather than leaving it where only a reader of that file will find it.

## Use the glossary's vocabulary

When your output names a domain concept (in a ticket title, a refactor proposal, a hypothesis, a
test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary
explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal — either you're inventing
language the project doesn't use (reconsider) or there's a real gap (note it for
`/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently
overriding:

> _Contradicts ADR-0007 (event-sourced orders) — but worth reopening because…_
