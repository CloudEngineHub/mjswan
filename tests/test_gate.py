"""Draw-selected env sets under `GatedIndexing`: kept whole and gated, as mjlab's dynamic
indexing would select them, and refused wherever the gate cannot follow."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mjswan.compile.gate import GatedIds, GatedIndexing  # noqa: E402

MASK = [True, False, True]


def _narrowing_body(x: torch.Tensor, flags: torch.Tensor, mask: torch.Tensor) -> None:
    """mjlab's idioms: a masked id set, a guard, reads and writes through it."""
    ids = torch.arange(3)[mask]
    if len(ids) > 0:
        x[ids, 0] = x[ids, 0].abs() + 1.0
        x[ids, 1] = 0.0
    flags[ids] = True
    picked = flags.nonzero(as_tuple=False).flatten()
    x[picked, :] = x[picked, :] * 2.0


@pytest.mark.parametrize("mask", [MASK, [False] * 3, [True] * 3])
def test_it_computes_what_dynamic_indexing_does(mask):
    mask = torch.tensor(mask)
    start = torch.tensor([[-1.0, 2.0], [3.0, -4.0], [5.0, 6.0]])
    want_x, want_flags = start.clone(), torch.zeros(3, dtype=torch.bool)
    _narrowing_body(want_x, want_flags, mask)

    x, flags = start.clone(), torch.zeros(3, dtype=torch.bool)
    with GatedIndexing():
        _narrowing_body(x, flags, mask)

    assert torch.equal(x, want_x)
    assert torch.equal(flags, want_flags)


def test_the_set_keeps_every_env_and_carries_the_mask():
    with GatedIndexing():
        ids = torch.arange(3)[torch.tensor(MASK)]
        narrowed = ids[torch.tensor([False, True, True])]
    assert isinstance(ids, GatedIds)
    assert len(ids) == 3
    assert ids.gate.tolist() == MASK
    assert narrowed.gate.tolist() == [False, False, True]


@pytest.mark.parametrize(
    "use",
    [
        lambda x, ids: ids + 1,
        lambda x, ids: x[:, ids],
        lambda x, ids: x[torch.tensor(MASK).nonzero()],
    ],
    ids=["arithmetic", "second-axis", "unflattened-nonzero"],
)
def test_a_use_the_gate_cannot_follow_is_refused(use):
    x = torch.ones(3, 3)
    with GatedIndexing(), pytest.raises(ValueError, match="draw selected"):
        use(x, torch.arange(3)[torch.tensor(MASK)])
