# Hypothesis: incremental rendering driven by uncertainty

## 0. The starting idea (H0)

> Stop computing images and maintain a *knowledge* of the light instead.
> Every ray is a measurement that updates what we know about the light in the scene.

Three building blocks:

1. **Light cache with uncertainty**: 3D cells, each storing an estimated radiance and a variance.
2. **Rays allocated to uncertainty**: every frame, the ray budget goes to the least certain visible cells.
3. **Dependency graph "like a compiler"**: each cell remembers which *objects* its rays touched. When an object moves,
   only the cells that depended on it are invalidated.

Promise: a static scene converges and ends up costing nothing, and a dynamic scene costs in proportion to what changes.

## 1. Questioning it

### 1.1 "Objects touched" is a wrong dependency (false negatives)

A footprint of the *touched* objects captures only half of the changes:

| Change | Did past rays touch the object? | Detected by H0? |
|---|---|---|
| An object leaves (disocclusion) | yes | ✅ |
| An object **arrives** on a path that was free (new shadow) | **no**: the ray went *past* it | ❌ |
| A lamp **switches on** | **no**: a lamp that is off is never sampled | ❌ |
| A lamp switches off | yes (shadow rays toward it) | ✅ |

What determines the light is not the objects touched: it is **the space crossed**. Proposal H1: each cell remembers the
set of *coarse regions of space* its paths crossed (bounce segments **and** shadow rays, voxelised with a DDA). A change
is described by a region (the swept AABB, old ∪ new, of a moving object). For a lamp switching on, a few **probe rays
are sent from the lamp** and the regions they touch are taken: a *forward* propagation of the change, the idea of
*Selective Photon Tracing* (Dmitriev 2002) applied to a camera-side cache.

### 1.2 Binary invalidation costs too much (false positives)

A binary spatial dependency is conservative. A cell at the back of room A of which *a single* path out of a thousand
went through the door to room B is marked as depending on B. If its history is thrown away when a lamp switches on in
B, a full reconvergence is paid for a change worth perhaps 0.1 %.

Proposal H2: the dependency does not decide, it **points to suspects**. A suspect cell keeps its estimate and receives
a small verification budget (≈ 8 fresh samples). A statistical test (z-test on luminance, old mean against fresh
samples) decides:

- **consistent**: merge and keep the history. Total cost: 8 rays.
- **inconsistent**: throw the history away and restart from the fresh samples.

Before the test, the history's weight is capped (N_cap = 64 samples). That way, a small change below the detection
threshold is still absorbed within a few dozen samples, instead of staying biased forever.

### 1.3 Cascades: truncate them by error, not by depth

Spatial dependencies already cover the first order (direct shadow, direct light) and a good part of the second (paths
bouncing in the changed area crossed it). Beyond that, H3: a **statistical safety net**. Each cell compares its last 32
samples with its long-term mean (z > 4: the history is thrown away), and an exploration floor gives every visible cell
at least 1 sample every 16 frames. The cascade is therefore propagated *as long as the error it induces exceeds the
noise*, which is the right stopping criterion.

### 1.4 A single currency: uncertainty

The three building blocks are unified:

- allocation minimises the expected image error. At a fixed budget, this is **Neyman** allocation:
  n_c ∝ √(screen footprint) · σ_c, computed greedily through the marginal gain f·σ²·(1/n − 1/(n+1));
- the variance is estimated with a **Bayesian prior** (median σ² of mature cells, weight κ = 4). Without it, a cell
  whose first 2 samples are 0 would never be sampled again;
- in "target quality" mode, only cells whose predicted relative error σ/√n exceeds τ are sampled. **This is the mode
  that makes the cost of a static scene tend toward ~0.**

### 1.5 What is not handled (yet)

- **Specular**: the cache is diffuse (outgoing radiance independent of the view). Glossy surfaces will have to be traced
  through (like NRC/SHaRC) or stored in a directional basis. Out of scope for v1.
- **Memory**: prototype = 512 region bits + 3 accumulators per cell (~200 B). A GPU version would use a hierarchical
  Bloom filter or sparse regions.
- **Discretisation bias**: a cell averages the radiance over 2 to 4 pixels, which softens sharp shadows. It is measured.

## 2. Hypothesis tested (H*)

> With a spatial radiance cache where **uncertainty is the only currency** (Neyman allocation, Bayesian prior),
> **spatial dependencies** (regions crossed) that *point to suspects*, and a **statistical verification** that decides
> on invalidation:
>
> 1. at an equal ray budget, the error on a dynamic sequence is lower than with (a) per-pixel accumulation with reset,
>    (b) exponential temporal accumulation, (c) a hysteresis cache in the style of SHaRC/DDGI, and (d) an adaptive cache
>    with global reset;
> 2. spatial dependencies beat "objects touched" dependencies (H0);
> 3. verification beats hard invalidation;
> 4. in target-quality mode, the per-frame cost of a static scene tends toward ~0, and after a change it is proportional
>    to the affected area.

## 3. Protocol

Scene: two rooms connected by a door ("doll's house" view). Sequence of 120 frames:

| Frames | Event |
|---|---|
| 0–39 | static |
| 40–63 | a box moves in room A (12 steps) |
| 64–79 | static |
| 80 | a powerful lamp **switches on** in room B |
| 100 | room A's ceiling light **switches off** |

Budget: 1 path per surface pixel per frame, for every method. Primary rays (G-buffer) are free and identical everywhere.
Metric: relMSE and RMSE against a 4096 spp per pixel reference, for each state of the scene.

Methods:

| Name | Description |
|---|---|
| `pt_reset` | per-pixel accumulation, full reset (oracle) at every change |
| `pt_ema` | exponential per-pixel accumulation α = 0.1 (TAA style) |
| `cache_ema` | spatial cache, uniform allocation, hysteresis (SHaRC/DDGI style) |
| `cache_reset` | cache + Neyman + global reset (oracle) at every change |
| `krc_nodep` | H* without the dependency graph (statistical net only) |
| `krc_obj` | H* with "objects touched" dependencies (H0) |
| `krc_hard` | H* with spatial dependencies but hard invalidation |
| `krc` | full **H\*** |

## 3 bis. Hypothesis revised after the experiment (H\*\*)

Building block 2 of H\* (verifying a suspect with a z-test on fresh samples) **failed**: with a path tracer's variance,
8 independent samples do not even detect a room going dark (see JOURNAL v1). Hence the replacement:

> **Do not test whether a cell changed: measure by how much, with correlated paths.** For a suspect cell, m paths are
> traced with the same seed in the scene before and the scene after:
>
> - a path that touches neither the changed area nor a changed lamp is *identical* in both scenes, so Δ = 0 exactly and
>   it is not retraced;
> - for a change of lamps only, the geometry is the same, and L_before = L_after + Σ w_j C_j is computed in *the same*
>   path;
> - the cache's history serves as a **control variate**: μ = x̄ − β(ȳ − μ_cache), with the optimal β ∈ [0, 1]. β = 1
>   corrects by Δ̂, β = 0 restarts from the fresh samples.
>
> The spatial dependency graph no longer decides anything: it **chooses which cells to spend these correlated paths
> on**. The cost of a change becomes O(cells whose paths cross the area), instead of O(visible cells).

This is the exact analogue of incremental compilation: the graph finds the units that may be affected, and the
"recompiler" (correlated paths) produces a diff that is applied, instead of rebuilding everything.

## 4. Related work

- Ward et al. 1988, *irradiance caching*: surface cache and interpolation.
- Walter et al. 1999, *Render Cache*: reusing samples across frames.
- Dmitriev et al. 2002, *Interactive Global Illumination using Selective Photon Tracing*: selectively retraces the
  photons whose paths touch the modified area. The direct ancestor of building block 3, on the light side.
- Binder et al. 2018/2019, *path space filtering* with jittered spatial hashing: our cell structure.
- Müller et al. 2017, *Practical Path Guiding*; Müller et al. 2021, *Neural Radiance Caching*.
- Majercik et al. 2019, *DDGI* (probes + hysteresis); NVIDIA SHaRC (spatial-hash cache).
- Schied et al. 2018, *A-SVGF*: history rejection through temporal gradients computed by reusing the previous frame's
  seeds. The screen-space analogue of our correlated paths; we apply it to a world cache, as a correction rather than a
  plain rejection.
- *Gradient-domain rendering* and *correlated sampling* (control variates, rendering the difference between two scenes):
  the theoretical basis of H\*\*.
- Brouillat et al. 2009, Marques et al. 2013, *Bayesian Monte Carlo* for global illumination.
- Lumen (Unreal Engine 5): a *surface cache* with budgeted updates.
