"""Every traced graph output keeps the name it was exported under.

Layer: L2 (torch + onnx + onnxruntime, no mjlab or MuJoCo).

An output that constant-folds to an initializer comes out of the TorchScript exporter
named after the initializer (``"8"``), while every consumer looks outputs up by the
requested name: `run_parity` refused the run, and the browser's `entityWrite` skipped
the write without a word (#140).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

import onnx  # noqa: E402
from torch import nn  # noqa: E402

from mjswan.compile.export import _export_onnx  # noqa: E402

OUTPUTS = ["e__joint_state__position", "e__joint_state__velocity"]


class _Park(nn.Module):
    """A fixed position from a baked buffer and zero velocity: no input is read."""

    def __init__(self):
        super().__init__()
        self.register_buffer("default", torch.tensor([[0.0, 0.3]]))

    def forward(self, rand):
        pos = self.default[:, [0]].clone() + 0.05
        return pos, torch.zeros_like(pos)


class _Hold(nn.Module):
    """Reads a live input, so nothing folds."""

    def forward(self, joint_pos, rand):
        pos = joint_pos[:, [0]].clone()
        return pos, torch.zeros_like(pos)


def _export(module, example, inputs):
    return _export_onnx(
        module,
        example,
        input_names=inputs,
        output_names=OUTPUTS,
        batch_axis=[*inputs[:-1], *OUTPUTS],
        opset=17,
    )


def test_a_constant_folded_output_keeps_its_name():
    onnx_bytes = _export(_Park(), (torch.zeros(0),), ["rand"])
    graph = onnx.load_from_string(onnx_bytes).graph
    assert [o.name for o in graph.output] == OUTPUTS

    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    position, velocity = session.run(OUTPUTS, {})
    assert position.reshape(-1).tolist() == pytest.approx([0.05])
    assert velocity.tolist() == [[0.0]]


def test_an_output_computed_from_an_input_keeps_its_name():
    module, example = _Hold(), (torch.zeros(1, 2), torch.zeros(0))
    onnx_bytes = _export(module, example, ["sim__qpos", "rand"])
    graph = onnx.load_from_string(onnx_bytes).graph
    assert [o.name for o in graph.output] == OUTPUTS
