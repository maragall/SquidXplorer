# TODOS

## FOV composition (stitch) — deferred from IMA-187

**What:** Implement `compose(projected_fovs)` — stitch N per-FOV MIPs into one well image
using per-FOV `position` / `overlap`, memory-bounded (stream FOVs: MIP each, discard z-stack,
keep the 2D MIP).

**Why:** Turns the stubbed N>1 path (`NotImplementedError`) into a real feature when You Yan
needs up to 4 FOVs/well. Without it the axis is wired but non-functional past N=1.

**Pros:** Completes the forward-design; unblocks multi-FOV acquisitions; the axis + geometry
schema from IMA-187 make it a clean drop-in with no API change.

**Cons:** Real work — tile registration/blending at seams; needs the geometry representation
validated against a real multi-FOV Squid dataset.

**Context:** MIP-then-stitch ordering is fixed (memory bound for 1536wp requires streaming
FOVs, not holding all z-stacks). The `metadata.fovs` schema `{index, position:(x,y), overlap}`
locked in IMA-187 is what compose consumes.

**Depends on / blocked by:** IMA-187 (axis + geometry schema), IMA-188 (project seam), and a
real 2–4 FOV Squid sample to validate stitch geometry. Revisit when multi-FOV is scheduled.
