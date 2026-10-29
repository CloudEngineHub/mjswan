# A pick and a placement preview for hosts

> Status: **Accepted**. Adds `engine.pick` and `engine.preview` to the headless engine's API
> (ADR 0004). mjswan Cloud's Studio, which places primitive geoms by pointing at the scene,
> is the first host.

## Context

The headless engine gives a host a camera pose (`camera.get`) and nothing else about what
is on screen: no scene, no raycaster, no picking. Cloud's Studio puts a box, a sphere or a
capsule where the pointer is, so it re-implements the camera's projection, draws an HTML
outline over the canvas as a stand-in for the geom, and intersects the pointer's ray with
the plane z = 0 on its own. The outline is not the geom (it only approximates a capsule
or a box seen at an angle), it ignores whatever stands on the floor, and the copied math
has to follow the engine's camera conventions from release to release.

The engine has what Studio rebuilds: MuJoCo casts rays against the model (`mj_ray`, which
the height scan and the XR grounding already use), and `core/scene/scene.ts` builds a mesh
for every geom type.

## Decision

### 1. `engine.pick(clientX, clientY)`: what lies under a point of the canvas

Client CSS pixels, as a `PointerEvent` gives them, so a host passes its event's numbers as
they are. The engine casts `mj_ray` from the camera through that point and returns the
first geom it meets, or null:

- `point` and `normal`, in MuJoCo world axes (x forward, y left, z up), as `CameraView`
  is. The normal is the surface's, turned to face the camera.
- `geom` and `body`, the ids of what was hit.

MuJoCo's ray, not three.js's: it answers as the simulation would, for an infinite plane
(drawn as a quad in clip space, which a three.js ray cannot meet) and a height field
alike, and it sees nothing that is only drawn (the mode gizmos, the preview, a skin, a
splat). It casts against the groups the viewer draws (0 to 2), so it does not land on a
collision-only geom no one sees. Unlike the pointer modes it counts the world body's geoms,
since the floor is where most things are put.

### 2. `engine.preview`: one primitive geom, drawn and never simulated

`preview.show({ type, size, pos, quat?, rgba? })` draws a geom that is not in the model:

- `type` is one of `plane`, `sphere`, `capsule`, `ellipsoid`, `cylinder` and `box`, and
  `size` is MuJoCo's for that type. A plane is a finite one.
- `pos`, and `quat` (w, x, y, z), are in MuJoCo world axes. The default `quat` is the
  identity.
- `rgba` defaults to MuJoCo's geom grey.

It is built by the same geometry and material code as the model's geoms
(`core/scene/primitive.ts`, split out of `scene.ts` for both), so what a host previews is
what it would add. `show` again moves and reshapes the one on show, rebuilding its
geometry only for a new type or size; `preview.hide()` removes it.

It is parented to the three.js scene, as the mode gizmos are: a scene load keeps it,
`camera.frame()` does not frame it, and the pointer modes skip it. It casts a shadow,
which says where it would rest. It is live: the model is never touched.

### 3. Hosts feature-detect them

An engine before this one has neither, so a host checks for `pick` and keeps a fallback of
its own (Studio keeps its outline).

## Not decided

- Projecting a point to the screen (`project`), which Studio's size handles would use in
  place of their own camera math.
- A preview of a mesh geom, or of more than one geom at a time.
- Picking and previewing in an XR session.

## Consequences

- Studio drops its ray and ground-plane math for placement, and can put a geom on top of
  another one.
- The engine owns one more object in its scene and two `mj_ray` output buffers, which it
  frees with itself.
