"""Go1 with a fox or a pug skin

The Go1 velocity tasks from ``main.py``, flat and rough, on their own, with a fox or a
pug (the Skin menu switches) drawn in the Go1's place, the Go1 itself undrawn. Each
part of the animal turns as the Go1 body it maps to does, and each leg is then bent so
its paw lands where the Go1's foot is; the fox's tail swings, simulated by MuJoCo in a
model of its own. Physics, collisions and the policy are the Go1's, exactly as in
``main.py``.

    uv run python examples/demo/go1_avatar.py

Both are rigged glTF quadrupeds, bundled as they are: ``humanoid`` names their bones as
a VRM's would be (the front legs as arms, the hind legs as legs), and ``rest_pose`` is
the Go1's standing keyframe, the pose each animal's own rest pose stands for. The fox
is the Khronos glTF sample Fox, downloaded; the pug is ``assets/pug.glb``, committed, as
its source has no stable URL (``assets/make_pug.py`` made it). Run it as a script (not
``python -m``): it imports this directory's ``main.py``.

Needs `pip install 'mjswan[hf,mjlab]'`.
"""

import os
import sys
import urllib.error
from pathlib import Path

from main import HF_REPO, add_mjlab_task

import mjswan
from mjswan.source import hf

#: The scene list, in order: the same Go1 and skins on flat ground and on rough terrain.
TASKS = ("Mjlab-Velocity-Flat-Unitree-Go1", "Mjlab-Velocity-Rough-Unitree-Go1")

# The Fox's mesh is CC0, by PixelMannen; its rig and animation are CC BY 4.0, by
# @tomkranis, which asks for the credit below wherever it is shown.
FOX_URL = (
    "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/"
    "15cd02d304e54cb42a4db4eefe719f76190f4b68/Models/Fox/glTF-Binary/Fox.glb"
)
FOX_NOTICE = (
    "Fox: low-poly model by PixelMannen (CC0 1.0), rigged and animated by @tomkranis "
    "(CC BY 4.0, https://creativecommons.org/licenses/by/4.0/), glTF conversion by "
    "@AsoboStudio and @scurest. "
    "https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/Fox\n"
)

#: VRM humanoid bone -> the Fox's node. Its front legs are already named as arms.
FOX_HUMANOID = {
    "hips": "b_Hip_01",
    "spine": "b_Spine01_02",
    "chest": "b_Spine02_03",
    "neck": "b_Neck_04",
    "head": "b_Head_05",
    "leftUpperArm": "b_LeftUpperArm_09",
    "leftLowerArm": "b_LeftForeArm_010",
    "leftHand": "b_LeftHand_011",
    "rightUpperArm": "b_RightUpperArm_06",
    "rightLowerArm": "b_RightForeArm_07",
    "rightHand": "b_RightHand_08",
    "leftUpperLeg": "b_LeftLeg01_015",
    "leftLowerLeg": "b_LeftLeg02_016",
    "leftFoot": "b_LeftFoot01_017",
    "leftToes": "b_LeftFoot02_018",
    "rightUpperLeg": "b_RightLeg01_019",
    "rightLowerLeg": "b_RightLeg02_020",
    "rightFoot": "b_RightFoot01_021",
    "rightToes": "b_RightFoot02_022",
}

#: The Go1 body each bone follows: front legs for arms, hind legs for legs. The Go1 has
#: no neck, so the animal's spine, neck and head ride its trunk.
GO1_BONES = {
    **{bone: "trunk" for bone in ("hips", "spine", "chest", "neck", "head")},
    **{
        f"{side}{bone}": f"{leg}_{body}"
        for side, front, rear in (("left", "FL", "RL"), ("right", "FR", "RR"))
        for bone, leg, body in (
            ("UpperArm", front, "thigh"),
            ("LowerArm", front, "calf"),
            ("Hand", front, "calf"),
            ("UpperLeg", rear, "thigh"),
            ("LowerLeg", rear, "calf"),
            ("Foot", rear, "calf"),
            ("Toes", rear, "calf"),
        )
    },
}

FOX = mjswan.SkinConfig(
    source=FOX_URL,
    bones=GO1_BONES,
    humanoid=FOX_HUMANOID,
    # Its shoulders and hips then sit 0.37 m apart, as the Go1's front and rear hips do.
    scale=0.0078,
    rest_pose="init_state",
    springs=[
        mjswan.SpringChain(
            root="b_Tail01_012",
            stiffness=1.5,
            drag=0.3,
            gravity=0.1,
            hit_radius=0.03,
            tip=0.16,
            # The pelvis and the hind thighs, which the tail must not swing through.
            colliders=(
                mjswan.SpringCollider("b_Hip_01", radius=0.06),
                *(
                    mjswan.SpringCollider(thigh, radius=0.035, tail=(0.147, 0.0, 0.0))
                    for thigh in ("b_LeftLeg01_015", "b_RightLeg01_019")
                ),
            ),
        )
    ],
    license="CC-BY-4.0",
    notice=FOX_NOTICE,
)

PUG_GLB = Path(__file__).parent / "assets" / "pug.glb"
PUG_NOTICE = (
    "Pug: from Farm Animals Animated by Quaternius (CC0 1.0), "
    "https://quaternius.itch.io/lowpoly-animated-animals\n"
    "Made forward-kinematic and scaled by mjswan's examples/demo/assets/make_pug.py.\n"
)

#: VRM humanoid bone -> the Pug's bone. Its hips bone is mid-body, as the Go1's trunk
#: origin is; it has no toes.
PUG_HUMANOID = {
    "hips": "Body",
    "spine": "Shoulders",
    "neck": "Neck",
    "head": "Head",
    "leftUpperArm": "FrontUpLeg.L",
    "leftLowerArm": "FrontLowLeg.L",
    "leftHand": "FrontFoot.L",
    "rightUpperArm": "FrontUpLeg.R",
    "rightLowerArm": "FrontLowLeg.R",
    "rightHand": "FrontFoot.R",
    "leftUpperLeg": "BackUpLeg.L",
    "leftLowerLeg": "BackLowLeg.L",
    "leftFoot": "BackFoot.L",
    "rightUpperLeg": "BackUpLeg.R",
    "rightLowerLeg": "BackLowLeg.R",
    "rightFoot": "BackFoot.R",
}

PUG = mjswan.SkinConfig(
    source=str(PUG_GLB),
    bones={bone: GO1_BONES[bone] for bone in PUG_HUMANOID},
    humanoid=PUG_HUMANOID,
    rest_pose="init_state",
    # Toon shading, its colours balanced against the scene's lights as a VRM's are: the
    # shaded side 60% of the lit one, the two blending softly, as the source's do.
    material=mjswan.ToonShading(shade=0.6, toony=0.5),
    # MuJoCo's lights, shared out, leave the beige darker than the source's; this lifts
    # it back, short of clipping.
    brightness=1.25,
    license="CC0-1.0",
    notice=PUG_NOTICE,
)


def setup_builder() -> mjswan.Builder:
    """The Go1 velocity scenes of ``main.py``, with the pug, and the fox when it downloads."""
    builder = mjswan.Builder(base_path=os.getenv("MJSWAN_BASE_PATH", "/"))
    project = builder.add_project(name="Go1 Avatar")
    repo_onnx = hf.list_repo_onnx(HF_REPO)
    for task_id in TASKS:
        scene = add_mjlab_task(project, task_id, repo_onnx)
        for name, skin in (("Fox", FOX), ("Pug", PUG)):
            try:
                scene.add_skin(name, skin)
            except urllib.error.URLError as exc:
                print(
                    f"Running the Go1 without the {name.lower()}: {exc}",
                    file=sys.stderr,
                )
    return builder


def main() -> None:
    """Build and launch.

    Environment variables:
        MJSWAN_BASE_PATH: Base path for deployment (default: '/')
        MJSWAN_NO_LAUNCH: Set to '1' to skip launching the browser
    """
    # Its own output directory, so main.py's dist/ is left alone.
    app = setup_builder().build("dist_go1_avatar")
    if os.getenv("MJSWAN_NO_LAUNCH") != "1":
        app.launch()


if __name__ == "__main__":
    main()
