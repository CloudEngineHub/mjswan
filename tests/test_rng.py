"""How `rand` carries each sampler a term draws from.

The runtime draws every `rand` element uniformly in its range, so a draw that is not
uniform rides as the uniforms behind it and the graph maps them back. `DrawRecorder`
stores a live draw that way; `ReplayRng` must turn it back into the same value.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")

from mjlab.envs.mdp.dr._types import resolve_distribution  # noqa: E402
from mjlab.utils.lab_api.math import (  # noqa: E402
    sample_gaussian,
    sample_log_uniform,
    sample_uniform,
)

from mjswan.compile.rng import DrawRecorder, ReplayRng  # noqa: E402


def _record(body):
    with DrawRecorder(body) as rec:
        out = body()
    return out, rec


def _replay(body, rand):
    with ReplayRng(body, torch.as_tensor(rand, dtype=torch.float32)):
        return body()


def test_a_uniform_draw_rides_as_itself():
    def body():
        lower, upper = torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 2.0])
        return sample_uniform(lower, upper, (3, 2), "cpu")

    out, rec = _record(body)
    assert rec.rand_vector.equal(out.reshape(-1))
    assert rec.rand_ranges == [[-1.0, 1.0], [0.0, 2.0]] * 3
    assert _replay(body, rec.rand_vector).equal(out)


def test_a_log_uniform_draw_rides_as_its_log():
    def body():
        return sample_log_uniform(0.5, 2.0, (4,), "cpu")

    out, rec = _record(body)
    assert torch.allclose(rec.rand_vector, out.log())
    assert rec.rand_ranges == [pytest.approx([math.log(0.5), math.log(2.0)])] * 4
    assert torch.allclose(_replay(body, rec.rand_vector), out)


def test_a_gaussian_draw_rides_as_two_uniforms_per_value():
    def body():
        return sample_gaussian(1.0, 0.1, (5,), "cpu")

    out, rec = _record(body)
    assert rec.rand_dim == 10
    assert rec.rand_ranges == [[0.0, 1.0]] * 10
    assert torch.allclose(_replay(body, rec.rand_vector), out, atol=1e-5)


def test_a_gaussian_over_tensor_bounds_takes_their_shape():
    """mjlab's `torch.normal(mean, std)` ignores `size` once the bounds are tensors."""

    def body():
        mean, std = torch.tensor([1.0, 2.0]), torch.tensor([0.1, 0.2])
        return sample_gaussian(mean, std, (7,), "cpu")

    out, rec = _record(body)
    assert out.shape == (2,)
    assert rec.rand_dim == 4
    assert torch.allclose(_replay(body, rec.rand_vector), out, atol=1e-5)


def test_an_integer_draw_rides_as_itself_and_floors_back():
    def body():
        return torch.randint(2, 5, (6,))

    out, rec = _record(body)
    assert rec.rand_vector.equal(out.float())
    assert rec.rand_ranges == [[2.0, 5.0]] * 6
    # The runtime draws in [2, 5): anywhere in an integer's unit interval picks it.
    assert _replay(body, out.float() + 0.99).equal(out)
    assert _replay(body, torch.full((6,), 5.0)).equal(torch.full((6,), 4))


def test_mjlabs_dr_distributions_are_caught_in_their_own_module():
    """They call the samplers by name from `dr._types`, not from the term's globals."""

    def body():
        lower, upper = torch.tensor([0.5]), torch.tensor([2.0])
        return resolve_distribution("log_uniform").sample(lower, upper, (3,), "cpu")

    out, rec = _record(body)
    assert torch.allclose(rec.rand_vector, out.log())
    assert torch.allclose(_replay(body, rec.rand_vector), out)


def test_draws_concatenate_in_call_order():
    def body():
        return sample_uniform(0.0, 1.0, (2,), "cpu"), torch.randint(0, 3, (1,))

    (first, second), rec = _record(body)
    assert rec.rand_vector.tolist() == [*first.tolist(), float(second)]
    replayed = _replay(body, rec.rand_vector)
    assert replayed[0].equal(first)
    assert replayed[1].equal(second)


def test_an_in_place_draw_lands_in_the_tensor():
    """mjlab's velocity command reuses one buffer, `r.uniform_(lo, hi)` per field."""

    def body():
        r = torch.empty(2)
        first = r.uniform_(-1.0, 2.0).clone()
        second = r.normal_(0.5, 0.1).clone()
        return first, second, r

    (first, second, buffer), rec = _record(body)
    assert rec.rand_dim == 2 + 4
    assert rec.rand_ranges[:2] == [[-1.0, 2.0]] * 2
    replayed = _replay(body, rec.rand_vector)
    assert torch.allclose(replayed[0], first)
    assert torch.allclose(replayed[1], second, atol=1e-5)
    assert torch.allclose(replayed[2], buffer, atol=1e-5)


def test_lower_bound_draws_sit_at_each_range_low_end():
    """What a trace takes its example from: every `draw <= p` selection passes."""

    def body():
        r = torch.empty(1)
        return (
            sample_uniform(-1.0, 1.0, (2,), "cpu"),
            sample_log_uniform(0.5, 2.0, (1,), "cpu"),
            sample_gaussian(1.5, 0.1, (1,), "cpu"),
            torch.randint(3, 7, (1,)),
            r.uniform_(0.0, 1.0) <= 0.0,
        )

    with DrawRecorder(body, at_lower_bounds=True) as rec:
        uniform, log_uniform, gaussian, integer, selected = body()
    assert uniform.tolist() == [-1.0, -1.0]
    assert log_uniform.tolist() == pytest.approx([0.5])
    assert gaussian.tolist() == pytest.approx([1.5])
    assert integer.tolist() == [3]
    assert selected.tolist() == [True]
    assert rec.rand_vector.tolist() == pytest.approx(
        [-1.0, -1.0, math.log(0.5), 0.0, 0.0, 3.0, 0.0]
    )
