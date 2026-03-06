# Platonic Geometry and Acoustic Functions

Reference mapping for object-level translation from organogram symbols to 3D form.

## Geometry to Behavior

- sphere (or near-spherical): diffuse projection, isotropic resonance.
- cube / box: modal clustering, strong axial modes, stable assembly.
- tetrahedron: pointed transients, directional emphasis.
- octahedron: balanced nodal distribution, mid-band articulation.
- dodecahedron: distributed cavity pathways, complex diffusion.
- icosahedron: dense face network, broad scattering behavior.

## Trimesh Primitive Suggestions

- sphere-like: `trimesh.creation.icosphere`
- cube-like: `trimesh.creation.box`
- conic radiation: `trimesh.creation.cone`
- tube/air column: `trimesh.creation.cylinder`
- hybrid couplers: concatenate multiple primitives with clear transforms

## Organogram Coupling Rule

For each major organogram element:
1. assign one geometric primitive,
2. define the acoustic role (resonator, interface, diffuser, actuator),
3. encode dimensions consistent with material and acoustic target.
