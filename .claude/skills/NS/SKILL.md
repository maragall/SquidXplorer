---
name: NS
description: Append a quick product-requirement note to NEXT_STEPS.md, the running hit list of things we want SquidXplorer to do.
---

Append the user's note to `NEXT_STEPS.md` at the repo root.

Invoked as `/NS "the note"`. The quoted text is the ask, in the user's words.

## Instructions

1. Read `NEXT_STEPS.md`. If it does not exist, create it with the preamble that explains it is
   a product hit list and **not** `TODOS.md` (deferred engineering work from plan-eng-reviews).

2. **Check the existing list first.** If an entry already covers this, say so and offer to
   amend that entry instead of adding a near-duplicate. A hit list nobody trusts to be
   de-duplicated stops getting read.

3. Append a new item at the end of `## Hit list`, in this shape:

   ```markdown
   - [ ] **<short title>** — <the note, in the user's words>
     <sub>added YYYY-MM-DD · <who></sub>
   ```

   - **Title**: 2-6 words naming the thing wanted. Derive it; do not ask.
   - **Body**: what they said. Tighten the grammar, keep the meaning and the emphasis.
   - **Date**: today's actual date, absolute. Never "today" or "this week".
   - **Who**: the user, unless they name someone else.

4. **Add context only if you already have it in this session** — a file and line, a commit
   hash, a measured number, a blocking dependency. One or two sentences at most. Do NOT go
   research the codebase to enrich a note; the point of `/NS` is that it is cheap to file.
   A bare one-line entry is a success, not a failure.

5. Do not commit. The user batches these.

6. Confirm by printing the appended entry, nothing else.

## What does not belong here

- **Design.** This file holds the *ask*. Reasoning, trade-offs and code-level context belong
  in `TODOS.md` or an IMA ticket once the item has survived a design discussion.
- **Bugs you found yourself** while working. Those are your responsibility to raise in the
  conversation, not to quietly file where they may not be read for weeks.
- **Anything already an IMA ticket.** Point at the ticket instead.
