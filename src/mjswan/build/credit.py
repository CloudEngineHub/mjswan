"""A scene's ``credits`` and a project's ``work``: the license files the build writes
beside them (ADR 0007), listed because a static host lists no directory, and each skin's
own terms. A file's text stays in the file; its entry names it and says what it is."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..license import copyright_line, identify_license, restriction_label
from ..license.vrm import vrm_license

if TYPE_CHECKING:
    from ..project import ProjectConfig
    from ..scene import SceneConfig


def _license_fields(name: str, data: bytes) -> dict[str, str]:
    text = data.decode("utf-8", "replace")
    spdx = identify_license(text).spdx
    restriction = restriction_label(spdx)
    holder = copyright_line(text)
    return {
        "license": name,
        "spdx": spdx,
        **({"restriction": restriction} if restriction else {}),
        **({"copyright": holder} if holder else {}),
    }


def scene_credits(scene: SceneConfig) -> list[dict[str, Any]]:
    """One entry per third-party component, its paths relative to the scene directory.

    A ``skin-<id>`` component belongs to that skin. A VRM's terms come from its own
    ``meta``, summarised for display; the file itself is bundled unmodified.
    """
    skins = {f"skin-{skin.config.id}": skin for skin in scene.skins}
    entries: dict[str, dict[str, Any]] = {}
    for attribution in scene.attributions:
        component = attribution.component
        entry: dict[str, Any] = {"component": component}
        if component in skins:
            entry["skin"] = skins[component].config.id
        if attribution.license is not None:
            entry.update(_license_fields(f"LICENSE.{component}", attribution.license))
        if attribution.notice is not None:
            entry["notice"] = f"NOTICE.{component}"
        entries[component] = entry
    for component, skin in skins.items():
        if skin.file.vrm_version is None:
            continue
        terms = vrm_license(skin.file.meta, skin.file.vrm_version)
        entry = entries.setdefault(
            component, {"component": component, "skin": skin.config.id}
        )
        entry["vrm"] = {
            **({"name": terms.name} if terms.name else {}),
            "authors": list(terms.authors),
            **({"license": terms.license} if terms.license else {}),
            **(
                {"other_license_url": terms.other_license_url}
                if terms.other_license_url
                else {}
            ),
        }
    return list(entries.values())


def work_entry(project: ProjectConfig) -> dict[str, Any] | None:
    """The work's own ``LICENSE`` and ``NOTICE``, relative to the project directory."""
    if project.license is None and project.notice is None:
        return None
    return {
        **(
            _license_fields("LICENSE", project.license)
            if project.license is not None
            else {}
        ),
        **({"notice": "NOTICE"} if project.notice is not None else {}),
    }


__all__ = ["scene_credits", "work_entry"]
