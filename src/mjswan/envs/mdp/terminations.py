"""Custom termination registry.

mjswan reimplements none of mjlab's termination functions: a task's real function
object, ``time_out`` included, is traced to ONNX at build time. This module carries only
the ``TerminationBinding`` escape hatch, for a term that cannot be traced at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TerminationBinding:
    """A hand-written TS termination class, bound to an mjlab termination name.

    Attributes:
        ts_name: Class the ``.ts`` file exports, and the name the browser's
            ``Terminations`` registry resolves.
        defaults: Default parameters merged into the JSON config entry.
        ts_src: Absolute path to the ``.ts`` file exporting ``ts_name``, injected into
            the bundle at build time. Required — mjswan ships no built-in TS classes,
            so without it the build fails.
    """

    ts_name: str
    defaults: dict = field(default_factory=dict)
    ts_src: str | None = None


_custom_registry: dict[str, TerminationBinding] = {}
"""Maps an mjlab termination function name to its override.

Populated via :func:`register_termination`; consulted by the mjlab adapter when
the config's own ``func`` needs replacing."""


def register_termination(mjlab_name: str, sentinel: TerminationBinding) -> None:
    """Bind one mjlab termination to a hand-written TS class.

    Call before :meth:`~mjswan.Builder.build`, so the adapter resolves the name
    and the builder injects ``ts_src`` into the browser bundle.

    Args:
        mjlab_name: The mjlab termination function name (e.g. ``"illegal_contact"``).
        sentinel: A :class:`TerminationBinding` whose ``ts_src`` implements it.
    """
    _custom_registry[mjlab_name] = sentinel


__all__ = [
    "TerminationBinding",
    "register_termination",
    "_custom_registry",
]
