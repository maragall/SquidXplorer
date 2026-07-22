# Nautilus — TODO: an agent that turns a repo link into an operator

Julio, 2026-07-22:

> "Remember the Nautilus agent that we could train for user to give it a repo link and it
> implementing the tool as an operator. Maybe Nautilus is just a white labeled instance of Claude,
> if that's possible. Definitely write that as a todo in the code base because it would save us
> time if we have data reader ready, all the metadata models, and ground this because I think it's
> awesome."

**Status: TODO. Nothing is implemented.** This file exists so the idea is written down where it
cannot slip, and — more usefully — so the parts of it that ARE already true are recorded, because
most of the hard work turns out to be host-side and most of that is done.

---

## The idea, stated precisely

A user pastes a GitHub URL for an image-processing tool. The agent reads that repo, writes the
adapter that registers it as an operator in SquidHCS, runs it against a real acquisition, and
reports whether it worked — with evidence, on real pixels.

The value is not "an LLM writes code". It is that **the adapter is the only thing that ever needs
writing**, because everything on our side of the boundary already exists. That claim is testable,
so the rest of this file tests it.

---

## Why this is closer than it sounds: what the host already provides

An agent writing a plugin needs the host to answer four questions. We answer three today.

### 1. "What is the data?" — ANSWERED

`squidmip/reader.py` reads four Squid output layouts behind one interface (`open_reader(path)`):
individual TIFFs, multi-page TIFF stacks, OME-TIFF, and OME-NGFF Zarr (HCS, per-FOV, and 6D).
An adapter never parses a directory layout. It asks for pixels.

`squidmip/_acquisition.py` is a frozen pydantic `Acquisition` model, validated ONCE at the reader
boundary, with `require_pixel_size_um()` / `require_dz_um()`. So an adapter does not guess whether
`pixel_size_um` is present — it asks, and gets a named refusal if it is missing.

**Gap:** RGB/colour acquisitions are still refused by name (`reader.py:113`). See `docs/SCOPE.md`.

### 2. "What shape must my function be?" — ANSWERED for two kinds

`squidmip/_engine.py` types operators by the axis they CONSUME:

```
consumes = frozenset({"z"})   z-reducer   stack -> plane    (T, C,  1, Y, X)    mip, reference
consumes = frozenset()        plane-op    plane -> plane    (T, C, Nz, Y, X)    decon, bgsub
```

`consumes` alone decides the grouping — there is no per-operator branch anywhere in the engine.
That is exactly the property an agent needs: a new operator declares what it eats, and the engine
already knows how to stream it over a plate with bounded memory.

The segmenter registry (`add_segmenter(name, fn, requires=(), blurb="")`) is the same shape one
level down: the operator is named for WHAT IT PRODUCES (`spot`), never for the algorithm, and
`requires=` names the optional dependency so a missing package is a **named refusal** rather than
a silently absent menu entry.

**Gap:** there is no `consumes={"t"}` (time) operator yet, and no RESULT type for anything that is
not pixels — which is why background subtraction is invisible in napari and why gallery view has
nowhere to land. An agent cannot register a tool whose output shape the host has no concept of.
**This is the single biggest blocker and it is worth fixing for its own sake.**

### 3. "Did it work?" — PARTLY ANSWERED

`squidmip/_oracle.py` is the model to copy. It grades a stitcher WITHOUT containing one: cut a
known image into tiles at known positions, hand them over, measure how far off they are put back.
That is a machine-checkable acceptance criterion, which is what an agent needs to know it
succeeded rather than merely to run without raising.

**Gap:** there is no equivalent oracle for segmentation, deconvolution or background subtraction,
and no WORKBENCH — Julio: "make a workbench module where we can test our implementations of new
tools live. I just give you the paths to the testing data, which varies, and then we plug in
different repos as SquidHCS plugins." A named-dataset workbench and a regression fixture are the
same artifact.

### 4. "How do I drive the app?" — NOT ANSWERED

There is a CLI and a GUI that share the engine, but no single named command surface. Julio: "make
sure that our GUI has an amazing API so that AI can interact with it... The logger and the API and
the CLI are amazing to design really well and deeply so that the agent can program our tool
cosmically."

An agent cannot drive a tool whose actions are only reachable by clicking. See `docs/SCOPE.md`
Core v.

---

## "Maybe Nautilus is just a white-labelled instance of Claude"

Plausible, and the deployment question is being checked separately rather than assumed here. The
honest summary of what matters for the DESIGN, independent of which product is used:

- The agent must run **headless and programmatically** — no terminal UI — because it lives inside
  a Qt application.
- It must accept **custom tools**, so "run this operator on this acquisition and show me the
  result" is a first-class action rather than the agent shelling out and parsing text.
- It must be **permission-restricted**: an agent that writes adapter code needs a sandbox, and it
  must never be able to write to the user's ACQUISITION DATA. Datasets are read-only in this
  project for a reason — a copy once filled the machine to zero bytes.
- Whatever it is called in the UI, the attribution and terms constraints of the underlying model
  provider apply. That is a product/legal question, not an engineering one, and it should be
  answered before the name ships, not after.

---

## The realistic failure modes, so they are designed for rather than discovered

1. **The repo does not do what its README says.** The adapter is written, it runs, the output is
   subtly wrong. Mitigation: an ORACLE per operator kind. Without one, "it ran" is the only
   available verdict, and that is how a scrambled mosaic passes.
2. **Dependency hell.** Half these tools pin conflicting versions of torch/numpy. Mitigation:
   `requires=` plus lazy imports, and a named refusal. Never import a plugin's dependency at
   `import squidmip`.
3. **The tool assumes a different data model** — single FOV, single channel, 8-bit, a specific
   axis order. Our unit is a REGION (a mosaic of FOVs), never a single field. The adapter's job is
   precisely this translation, and it is where an agent will most often get it wrong.
4. **Silent partial failure.** A plugin that skips a well and returns is the defect shape this
   codebase has paid for repeatedly. Any generated adapter must be held to the same rule as
   hand-written code: no `except Exception: pass`, a refusal names itself.

---

## The order of work this implies

Nautilus is not the next task. But three things on its critical path are worth doing anyway, for
reasons that have nothing to do with agents:

1. **A result type for non-pixel operator output** (labels, points, tables). Unblocks background
   subtraction in napari, gallery view, and Fractal-style feature tables.
2. **The workbench + named test datasets.** This is the regression-fixture work Julio already
   asked for, wearing a different hat.
3. **One command surface** shared by GUI, CLI and script. This is Core v.2.

Do those and Nautilus becomes "write the adapter", which is the part an agent is actually good at.
Skip them and Nautilus is an agent guessing at an interface that does not exist.

---

## The loop: CSAT survey -> agentic fix -> re-run

Julio, 2026-07-22:

> "Make sure that the Nautilus generation of the repo and a customer satisfaction questionnaire
> that they answer after every iteration and it recurses and fixes itself is possible. It will
> happen very soon. And there are already loops out there where you can deploy an agentic pipeline
> for adding post-processing repos."

> "Make sure that that is written somewhere in the codebase so that we don't lose the Nautilus
> idea of CSAT survey w/ an agentic loop to adopt new post-processing operators."

The shape:

```
    user pastes a repo URL
            |
            v
    agent writes the adapter  ------------------.
            |                                    |
            v                                    |
    run against REAL fixtures (the workbench)    |
            |                                    |
            v                                    |
    show the result to the scientist             |
            |                                    |
            v                                    |
    CSAT questionnaire: did this do what you     |
    expected? what is wrong with it?             |
            |                                    |
            +---- not satisfied ------------------'
            |
            v
       satisfied -> the adapter is kept as an operator
```

The questionnaire is the interesting part, and it is not decoration. An agent that writes an
adapter can tell whether the code RAN; it cannot tell whether the output is scientifically right.
The scientist can. So the survey is the ORACLE for everything an automated check cannot decide --
which is most of what matters about a segmentation or a deconvolution.

**Design constraints, so the loop cannot lie:**

* Each iteration is **versioned and reversible**. A loop that mutates one adapter in place
  destroys the evidence of what the scientist was actually judging.
* The survey response is stored WITH the adapter version, the dataset it ran on, and the measured
  wall clock and peak RSS. "It looked wrong" is not actionable; "it looked wrong on this dataset,
  at this version, and took 4 minutes" is.
* A rejected iteration must record WHY in a form the next iteration can read. Otherwise the loop
  is a random walk with a friendly interface.
* The loop must be able to STOP and say "I cannot do this" by name. An agentic pipeline that
  cannot fail visibly will keep producing plausible adapters forever.

This depends on the same three things Nautilus does (the result type, the workbench, one command
surface) plus a fourth: a per-operator record of **wall clock and memory footprint**.

Julio on that last point: "part of the logger is having a timer that times the wallclock run of
each operator, it also measures memory footprint. Like this is great for integrating new
implementations and variants of the operator, assuming you're using the registry and relations to
scale to n algorithms."

That is the same measurement the CSAT loop needs to make a rejection actionable, and the same one
the log panel needs to show the user something is happening. One measurement, three consumers.

## Shipping the dependencies

Julio: "We'll have to scale dependencies, but maybe we just ship it like FIJI with the toolkit
ready and they can update for more tools, and only if necessary an ImageJ version."

The Fiji model -- ship a working, curated toolkit; let the user add more; keep the door open to a
heavier runtime only when a tool genuinely needs one -- fits what is already true here: every
operator's dependency is declared (`requires=`), imported lazily, and refused BY NAME when absent.
A curated default set plus opt-in extras is that model expressed in Python packaging.

## Where Nautilus actually runs: the user's own agent

Julio, 2026-07-22 (explicitly NOT in scope for this commit, recorded so it is not lost):

> "the Nautilus basically connects to the user's Claude Code, Codex, etc in their terminal. Ask
> 'Nautilus' and he will get in touch with your agents. And it's great because then after the CSA
> survey goes green then it writes a branch that a watchdog can then look out for and merge, given
> that my infrastructure is robust enough."

This is a better answer than embedding an agent in our process, for three reasons worth writing
down:

1. **The credentials are already theirs.** No API key handling, no per-seat model billing, no
   question about whose account ran what. The scientist's own agent, on the scientist's machine.
2. **The blast radius is a BRANCH, not our app.** The loop's output is a git branch. Nothing is
   adopted until something merges it, so a bad adapter is a branch nobody merged rather than a
   broken viewer.
3. **The watchdog is a policy, not a program.** "Merge when CSAT is green and the gate passes" is
   a rule that can be read, argued with, and tightened. An agent that writes directly into the
   running app has no such seam.

The dependency is stated in his own words -- "given that my infrastructure is robust enough" -- and
that is the honest gate. A watchdog merging agent-written branches is only as safe as the test
suite and the acceptance oracles it merges against. Which is why the three prerequisites above
(result type, workbench, one command surface) and a real oracle per operator kind are not optional
groundwork: they are the thing that makes the watchdog safe rather than reckless.

**Status: NOT IN SCOPE. Recorded as intent.**
