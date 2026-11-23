# Roadmap: from prototype to game engine

Goal: integrate KRC into a real-time game engine (typical budget for global illumination: 2 to 4 ms per frame on the
GPU, at 1080p to 1440p).

## Where things stand

| Building block | State |
|---|---|
| Algorithm (cache + uncertainty + correlated delta + weighted dependencies) | validated on a toy scene, CPU |
| Path tracing on the GPU | ✅ validated (statistically identical to the CPU) |
| Cache, allocation, events on the GPU | ✅ fully GPU pipeline, stable over 12,000 frames |
| Hardware rays (RT cores) | ✅ Vulkan ray query, BLAS/TLAS, two TLAS (before/after) |
| Real geometry (triangles, BVH) | ✅ Cornell box, Sponza (262 k triangles) |
| Comparison in ms against a standard cache | ✅ KRC v2 against the real SHaRC: 2 to 4× less error at equal time (Sponza), 20× (Cornell); ⚠️ static with a moving camera and minimum cost in SHaRC's favour |
| No bias on heavy-tailed radiance | ⚠️ two biases fixed; ~3 % residual in static, slow return after switching off |
| Moving camera | ✅ quality; ⚠️ minimum cost 3 ms at 1280×720 (KRC v2) |
| Specular | ✅ PBR materials (GGX), per-pixel reflections read from the cache, MIS (RESULTS §16–17) |

## Analysis (October 2026): optimising KRC, reducing noise, testing it in games

Drawn from the measurements of the three benches (Quake II RTX, Doom RT, NVIDIA's SDK; RESULTS §18 to §21). Ranked by
expected gain on the final image, for equal effort.

**Noise and correctness of the image**

1. **Per-pixel ratio for the denoiser's history** (1 day). Today S comes from the cell of the first hit: an average
   between its sunlit and shadowed parts, hence the error left after a change (sun patches in the living room; the
   Bistro at night darkened 6 % too much). At the first hit, each group's direct light is known exactly per pixel; its old
   value is its new one divided by r_g. Ratio = smoothed "after" sum / smoothed "before" sum (a ratio of sums, not a mean
   of ratios, otherwise noisy). That was the "direct-light ratio" that worked in Doom.
2. **Correlated paths for unknown groups** (2 to 3 days). A lamp switching on from zero, or lamps of one group changing
   differently, fall back on SHaRC's lag. Each sample of the update evaluates the lamp with its old and new intensity
   (same draw, MIS between the two laws so a switched-off lamp is still sampled); the difference, almost noise-free where
   nothing changes, corrects the cell's mean.
3. **Adaptive allocation of update paths** (2 days). SHaRC updates 1 pixel in 25, uniformly. KRC-A's moments already give
   the variance per cell: split the budget into 30 % uniform + 70 % Neyman, as in Doom (lower raw noise where it matters,
   without the dark areas of Neyman alone).
4. **Energy loss in the sample's NRD** (1 day of investigation). 9 to 15 % off-sky for every method, even without a cache.
   Fixing it improves every image and makes the comparison fair (a cache that is too bright is rewarded today). Leads:
   probabilistic diffuse/specular choice at the first bounce, albedo demodulation.
5. **Direct light with many lamps: ReSTIR DI** (NVIDIA RTXDI). With several lamps, the image's noise comes mostly from
   direct light (the sample draws 8 candidates without reuse). Not specific to KRC, but essential in a real game.
6. Abandoned: first-bounce guiding (no measured gain, +2 to +3 ms).

**Cost** (KRC-L +0.01 to +0.05 ms, KRC-A +0.24 ms on top of SHaRC's 3.85 ms, living room 1280×720)

1. Done: per-entry passes no longer walk the table writing (−2 ms).
2. Insertion done twice per vertex (SHaRC, then KRC to find the same cell): ask SHaRC for the index and the read-back
   decision, one probe less.
3. S smoothing pass (81 taps, full resolution): only after a first relight; for lamps that flicker all the time, run it at
   half resolution or make it separable.
4. Table size: 4 M entries in the sample; 1 M is often enough, and divides the cost of the per-entry passes by 4 (for
   SHaRC too).

**Testing in games**: KRC is now a SHaRC extension packaged as an engine-independent library (`krc_sdk/include`, guide in
docs/INTEGRATION.md: five hook points).
1. Quake II RTX (official SHaRC already plugged in): replace our home-made KRC with the library, so the three benches
   test the same code.
2. RTX Remix (`dxvk-remix`, open source): a single integration for Portal, Half-Life 2 and others; first check in its
   sources which cache it uses (NRC announced in 1.0).
3. Unreal Engine 5: heavier, later.

## Towards real games (after the showcase, RESULTS §17)

**Done (RESULTS §18)**: KRC and NVIDIA's official SHaRC SDK run in Quake II RTX, at equal time, on a real recorded game.
**Done (RESULTS §20)**: Doom RT (RTGL1), at SHaRC's cost; exact relighting by light groups and relit denoiser history: KRC
clearly ahead after a light change. To do: the end of convergence after a change (the unexplained part lingers), scenes
lit by emissive surfaces (sample them directly, guided by the cells), then Portal / RTX Remix.

KRC cannot be "injected" into a closed game like Cyberpunk 2077: its engine (REDengine, DX12) computes its own lighting
and offers no entry point to replace it (Cyberpunk's path tracing is NVIDIA's code, integrated by CD Projekt). The
realistic routes, in order:
1. **Bevy plugin** (Rust + wgpu, the same stack as this prototype): KRC as the global-illumination pass of an open game
   engine, with its G-buffers, lights and camera. That is the playable demo.
2. **RTX Remix** (`dxvk-remix` runtime, open source): it replaces the rendering of old DirectX 8/9 games (Portal,
   Half-Life 2, Morrowind…) with a path tracer, and its radiance cache is precisely SHaRC/NRC. Plugging KRC in instead
   would compare the two in real games.
3. **Unreal Engine 5** (sources available): as a rendering plugin, next to Lumen.
Common prerequisites: native HLSL/DX12 or Vulkan (port of the WGSL), a 2 to 4 ms budget at 1440p, many lights (ReSTIR for
direct), animated objects (updated BLAS).

## After the "game-like" test (RESULTS §14)

KRC v2 keeps the advantage in a game pipeline with a denoiser (1.4 to 5.7× at equal time on the sequences). Defects to
fix before a demo:
1. **Spots after a light change**: reduced (RESULTS §15: "total + direct share" cells, 32 correlated paths), not gone.
   Per-cell corrections still noisy (8 correlated paths at first). Leads: more correlated paths for lamps, denoise the
   field of corrections (deltas of neighbouring cells are correlated), sampling priority for corrected cells.
2. Dark drift after switching off (0.73).
3. Static Sponza: SHaRC slightly better (1.2 to 1.35×).
4. Then tests 2 to 4: lag and flicker with a moving camera, game scenes (many objects and lights), 1080p budget.

## Next decision (after KRC v2)

KRC v2 beats SHaRC on dynamic sequences. To go further:
1. Static with a moving camera: the static phase stays ≈2× in SHaRC's favour at 1280×720 (its voxels already exist off
   screen when a surface appears).
2. KRC's minimum cost (~3 ms at 1280×720) and event peaks (p95 5 ms).
3. A demo in an engine (Bevy).

### Earlier decision (before KRC v2)

1. Test SHaRC's "reactive lighting" mode: if KRC keeps its advantage on lamps, the lead goes on.
2. **Hybrid**: official SHaRC for the steady state, and only KRC's correlated corrections (dependency graph, correlated
   paths) applied to its voxels during events.
3. Failing that, reduce KRC's fixed cost: insertion merged with lookup, resolve limited to cells seen or sampled,
   eviction without tombstones.

## Steps, in order

1. **GPU-resident cache.** Hash table of cells in a GPU buffer (like SHaRC): accumulators (n, mean, M2), aged region
   bitset, vertex counters per super-region. Paths merge their results with `atomicAdd`/`atomicOr`. The CPU only receives
   aggregates (number of suspects, cost). Exit criterion: the `krc` run on the GPU reproduces the CPU metrics, and the
   per-frame cost is measured in ms.
2. **Allocation on the GPU.** Greedy Neyman allocation becomes a parallel pass: priority per cell, then a prefix sum to
   distribute the budget. The event (suspects, correlated paths) becomes a compute shader.
3. **Hardware rays.** Triangle meshes + BLAS/TLAS, ray query in the shader. Old and new scene = two TLAS (engines already
   keep the previous frame's transforms for motion vectors).
4. **Moving camera.** Cells are in world space, so they survive camera motion; new cells appearing (disocclusion) and a
   cell size depending on distance must be handled.
5. **Specular.** Trace through glossy surfaces up to the first diffuse vertex, then read the cache (like NRC/SHaRC).
6. **Engine integration.** A plugin for an open engine (Bevy in Rust with wgpu, or Godot), then comparison with the
   engine's native solution, at equal quality and cost.

## Known risks

- **Memory**: ~600 bytes per cell today (432-byte bitset + accumulators). For 1 M cells, that is 600 MB, too much. The
  exact bitset must become hierarchical or be replaced by the super-region counters alone (72 bytes), keeping the fine
  bitset for geometry only.
- **Cost of a light event**: even energy-weighted, switching on a strong lamp is still a real global change. It will have
  to be spread over several frames (capped budget, most visible cells first).
- **Heavy-tailed radiance** (small bright lamps, indirect): adaptive allocation and correlated paths are biased toward
  dark there (RESULTS §11). A robust allocation variance and explicit sampling of changed lamps will be needed.
- **Very dynamic scenes** (foliage, crowds): many small simultaneous changes; when the correlated delta stops paying off
  against plain hysteresis must be measured.
