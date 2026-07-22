"""The memory budget's rules.

Julio: "the guard for memory is interesting because we have to maximize performance and that
doesn't happen by hardcoding guards", and "There are so many variations of desktop configuration."

The property under test: the budget SCALES with the machine, is never so small that the cache
thrashes, is never unbounded, and a human can override it.
"""

from __future__ import annotations

from squidmip._budget import (
    CEILING_BYTES,
    DEFAULT_FRACTION,
    ENV_VAR,
    FLOOR_BYTES,
    cache_budget,
    describe,
)

GIB = 1 << 30
MIB = 1 << 20


def test_the_budget_SCALES_with_the_machine():
    """THE point. A fixed 256 MiB is ~5% of available on a 16 GB laptop and ~0.5% on the CSO's
    96 GB demo machine: simultaneously too timid to use the machine and too blunt to protect a
    small one."""
    small = cache_budget(available=8 * GIB, env={})
    large = cache_budget(available=64 * GIB, env={})
    assert large > small * 4, (
        f"the budget barely moved between an 8 GB and a 64 GB machine ({small} -> {large}); "
        "that is a hardcoded guard wearing a function"
    )


def test_it_never_thrashes_on_a_small_machine():
    """Below the floor a fused-plane cache holds fewer than two 5731x4793 planes (54.9 MB each)
    and evicts the one it is about to be asked for again."""
    assert cache_budget(available=64 * MIB, env={}) >= FLOOR_BYTES
    assert cache_budget(available=0, env={}) == FLOOR_BYTES


def test_it_is_never_unbounded_however_big_the_machine():
    """Bounded memory is a project principle, not a consequence of small hardware."""
    assert cache_budget(available=512 * GIB, env={}) == CEILING_BYTES


def test_a_human_override_WINS_and_is_not_silently_capped():
    """Precedence: explicit override > measurement > default -- the same rule the contrast model
    uses, and for the same reason: the person who typed a number knows something the process does
    not. Capping it silently would make the control a lie.
    """
    assert cache_budget(env={ENV_VAR: "128"}, available=64 * GIB) == 128 * MIB
    huge = cache_budget(env={ENV_VAR: "16384"}, available=8 * GIB)
    assert huge == 16384 * MIB, "the override was silently clamped"


def test_a_nonsense_override_falls_back_to_MEASURING_rather_than_to_zero():
    """A typo in an env var must not disable the cache, and must not crash the app."""
    got = cache_budget(env={ENV_VAR: "not-a-number"}, available=32 * GIB)
    assert got == int(32 * GIB * DEFAULT_FRACTION)
    assert cache_budget(env={ENV_VAR: "-5"}, available=32 * GIB) > 0
    assert cache_budget(env={ENV_VAR: ""}, available=32 * GIB) > 0


def test_it_sizes_off_AVAILABLE_not_total():
    """A viewer that sizes off TOTAL RAM on a laptop already running a browser and an IDE wins the
    allocation and loses the machine. Swapping a 550 MB fused plane is far worse than re-reading
    it, and the scientist's other windows are not our memory to spend.

    Pinned as a signature property: `available` is what the function takes.
    """
    import inspect

    assert "available" in inspect.signature(cache_budget).parameters
    # ...and the real reader asks for available, not total.
    src = inspect.getsource(__import__("squidmip._budget", fromlist=["x"]).available_bytes)
    assert ".available" in src and ".total" not in src


def test_the_budget_can_be_EXPLAINED_to_a_human():
    """A budget nobody can see is a magic number. This line goes in the log panel."""
    line = describe(512 * MIB)
    assert "512" in line and "MiB" in line
