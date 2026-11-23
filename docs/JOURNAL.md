# Development journal

(The commit numbers quoted refer to the original repository, where KRC started; this repository's history keeps the same
steps.)

## 2026-10-21 — Start

- Initial idea received (H0: cache + uncertainty + dependency graph by objects).
- State of the art: the closest ancestor of the "dependencies" building block is *Selective Photon Tracing* (Dmitriev et
  al., EGWR 2002), which retraces, on the light side, the photons of modified areas. Real-time caches (SHaRC, NRC, DDGI)
  handle dynamics through hysteresis or continuous learning, without explicit dependencies.
- Critique of H0 and reformulation as H*: see [HYPOTHESIS.md](HYPOTHESIS.md).
  1. Dependencies by *regions crossed* rather than by *objects touched* (false negatives: new shadow, lamp switched on).
  2. Dependencies point to suspects, a statistical test decides (against false positives).
  3. Cascades are truncated by error (statistical net), not by depth.
  4. A single currency: uncertainty (Neyman + Bayesian prior).
- Technical choices: Rust (CPU, zero dependencies, multithreaded with `std::thread::scope`), a scene of axis-aligned
  boxes, NEE on rectangular lamps, 192×96 px.

## v1 — first prototype (commit 23e1272)

- **Bug: prior starvation.** After an event, almost every cell is a suspect. The only "mature" cells left are faces
  turned toward the void (σ = 0), so the prior is 0, invalidated cells have zero gain and are never sampled again: the
  image freezes. Fix: prior computed over *all* mature cells, with a floor of 0.1. Exactly the trap the Bayesian prior
  was meant to avoid.
- **Bug: misaligned regions.** A 0.75 m region straddled both faces of the partition and created false dependencies
  between the rooms. Fix: 0.6 m regions, the partition falls on a boundary.
- **Negative result: the verification z-test is underpowered.** With 8 path-tracing samples (relative variance ~2 per
  sample), not even a room going dark is detected: the cell is wrongly confirmed and keeps the old lighting (`krc_ztest`,
  lamp off: 9.5e-2 against 6.5e-3 for hard invalidation). *Testing* whether a cell changed with independent samples is a
  dead end.

## v2 — measuring the change instead of testing it (commit c07d5dc)

- **Pivot (H\*\*): differential update through correlated paths.** The same paths (same seed) are retraced in the old and
  the new scene. A path that touches neither the changed area nor a changed lamp is *identical*: Δ = 0 exactly, without a
  second trace. The estimate is *corrected* (mean += Δ̂) instead of being thrown away. The true analogue of incremental
  compilation: a diff is applied.
- A "union" lamp pmf (independent of the on/off state) is needed so that both scenes consume the same random numbers.
- **Bug: shadow-ray dependency.** The segment up to the lamp was recorded, even when the ray was blocked. Fix: segment up
  to the occluder.
- **Control variate**: μ = x̄ − β(ȳ − μ_cache), with the optimal β estimated on the pairs. β = 1 gives the pure delta,
  β = 0 a fresh restart. In practice, the optimal β and β = 1 are statistically indistinguishable on this scene.
- Several seeds (5): the differences between delta variants are within the noise.
- **Important finding**: `krc_all` (delta on *every* cell, no graph) is as accurate as `krc`. The exact early exit of
  correlated paths is itself a dependency test, per path and exact. The graph only serves to reduce the *cost*:
  O(affected cells) instead of O(visible cells).

## v3 — scaling and lamp events (commit 3f0533c)

- Scene parameterised by the number of rooms (2, 4, 8), at constant pixel density.
- **Lamp-only changes**: the geometry is identical, so the old radiance follows from the same path:
  L_old = L + Σ_j w_j C_j, where C_j is the contribution lamp j would have if it were on. No second trace, and the cost of
  a lamp event is halved (20 k instead of 37.7 k paths). Exactness checked path by path against a full retrace.
- **Bug revealed by this change: the stopping rule of the adaptive m is wrong for lamps.** If 2 paths did not *select*
  the changed lamp, Δ = 0 exactly and the cell is declared unchanged: +7.8 % bias in room A. NEE selection is random and
  independent of the lamp's importance for the cell, so "Δ = 0 over 2 draws" proves nothing. Fix: M_DELTA paths for every
  suspect cell on a lamp change. (For geometry, stopping stays justified: a cell whose paths rarely touch the swept area
  has a small Δ, bounded by that frequency.)
- Diagnosis of the tail after A switches off: it is not a bias (|bias| < 0.6 %) but variance in room A, now dark.
  Hypotheses tested and refuted: underestimated per-sample variance (A), m₀ too small (F, weak effect), n_eff too large
  (weak effect).
- Method error, on my side: my quick aggregates read `metrics*.csv`, which held seeds left by an earlier version of the
  code. I cleaned up and reran everything; the numbers in RESULTS.md come from a clean run.

## v4 — dependency ageing, consolidated results (commit e307e72 and following)

- **Cascades measured**: at 8 rooms, switching on lamp B made 80 % of cells suspects. Binary dependencies accumulated
  over the whole history saturate: a single path through two doors is enough.
- **Ageing**: two alternating bitsets cover the last K to 2K samples (K = 64). A clear effect for geometry (suspects at
  N = 8: 1,070 against 1,837) and for A switching off, but not for B switching on.
- Diagnosis (debugging the event's regions): lamp B's probes really go through the door and touch the walls of rooms 0
  and 2. The graph is *qualitatively* right; it lacks the *magnitude*. Next step: energy-weighted dependencies.
- 150-frame warm-up for scaling. Without it, costs measured in target-quality mode at N = 8 were dominated by the initial
  convergence.
- Consolidated results: see [RESULTS.md](RESULTS.md).

## v5 — energy-weighted dependencies (commit 711c5d9)

- **Ground truth first**: signed change aggregated per room between references (N = 8). Lamp B on: rooms 0 to 3 = +4.1 %,
  +153 %, +165 %, +1.1 %; rooms 4 to 7 < 0.1 %. The binary graph made ~78 % of cells suspects *everywhere*. (A first
  computation as a mean |Δ| per pixel gave 2 to 5 % everywhere: that was the references' noise.)
- Lamp probes with magnitude: ΔL_R ≈ ρ·kΦ/(P·π·A_R).
- **Dead end 1**: a hashed sketch over 256 counters. A cell crosses 100 to 300 regions per window, collisions make every
  frequency high, so everything becomes a suspect.
- **Dead end 2**: *traversal* frequencies over exact super-regions. Rays escaping through the open front run along the
  other rooms without bouncing in them.
- **What works**: frequencies of path *vertices* (where light is picked up) per super-region (36 u8 counters per window),
  and the criterion "regions crossed AND Σ f_S·M_S > 1 % of the radiance". Suspects per room:
  100/100/94/53/8/1.5/0.3/0.6 %, which matches the physics.
- N = 8, lamp B on: 7,016 → 3,466 suspects, 56 k → 28 k paths (−51 %), equivalent error.

## v6 — GPU tracer (commit b3660fe)

- RTX 4070, wgpu 24 on Vulkan (hardware ray query available, not used yet).
- `trace.wgsl`: a faithful port of `pt.rs`, a `Tracer` interface shared by CPU and GPU.
- Validation: GPU/CPU relMSE = CPU/CPU noise; exact correlated paths on the GPU (≤ 2e-7); same dependency statistics.
- Throughput: pure compute reaches 466 M paths/s. Two overheads found and fixed: recreating buffers on every call
  (~85 ms), and building a 450-byte DepRec per result even without dependencies (900 MB for 2 M paths).
- **Architecture lesson**: with dependencies, the gain is only ×1.6, because reading back each path's dependencies costs
  more than tracing them. For real time, the cache must live on the GPU and nothing must come back to the CPU per path.

## v7 — profiling, the whole pipeline on the GPU (commits 3bfd29d → 5a5bbac)

- **Profile of a 1080p batch** (GPU timestamps + per-phase timers): without dependencies, the kernel takes 2.6 ms out of
  113 ms. Over 97 % of the time is CPU↔GPU plumbing (preparation, upload, readback, conversion). Hence the full port.
- **Memory**: binary bitset per cell at 0.6 / 1.2 / 2.4 m (864 / 108 / 16 bytes). Energy weighting does most of the
  sorting: L2 (16 bytes) costs only +14 to +33 % of paths per event, and the error stays within the bars. L2 is kept.
- **Fully GPU pipeline** (`krc_gpu.wgsl`, ~20 passes): G-buffer, hash table (two-step CAS, no spin-waiting), event,
  correlated paths in 2 rounds, allocation, tracing with atomic merging, resolve, display. The CPU only sends the scene
  and the event.
- Bugs found along the way: incomplete counter reset (2 groups instead of 4), budget under-used at 69 % (quota rounding,
  then extreme sensitivity of Σquota(λ) in the converged state: 3 sweeps of 32 candidates), dispatch > 65,535 groups.
- Measured optimisations: allocation by parallel sweeps (12× faster than a single-group bisection), indirect dispatch on
  the real number of cells, coarse DDA outside correlated paths, task interleaving against atomic contention.
- Validation: `gpu_krc` 4.19e-3 against 4.11e-3 for CPU `krc_energy_l2`; `gpu_sharc` equal to CPU `cache_ema`.

## v8 — long drift (commits 9705364, 39ad338)

- 12,000 frames (100 cycles of the sequence). No saturation, no table growth, debt back to 0.
- Two defects found and fixed:
  1. After a large combined change (cycle restart: box + 2 lamps), the error stayed 3× higher for ~40 frames: the n_eff of
     a corrected estimate was too optimistic. Cap N_POST = 64 (sweep 32/64/128/none).
  2. A slow drift (+3.1 % ± 1.3 % over 100 cycles, t = 2.4): history capped at 1,024 samples, which brings it back to
     −0.8 % ± 1.4 %.

## v9 — triangles and hardware ray tracing, comparison in ms (commits 6d2ae3f → …)

- wgpu 24 exposes Vulkan ray query (BLAS/TLAS) on the RTX 4070.
- Templated shader: a scene-specific grid and an interchangeable geometry block (boxes, or triangles through ray query on
  the current and previous TLAS). Lamp probes and references computed on the GPU (~1 G paths/s).
- Cornell box in triangles, same sequence: `tri_krc` 5.03e-3 ± 4e-5 against `tri_sharc` 3.91e-2 (7.8×).
- **Error/time comparison (1024×1024)**: at equal mean GPU time, KRC has 2.4 to 2.9× less error than SHaRC; at equal
  error, it is ~2× faster. But its event frames climb to 12–16 ms (p95 ~13 ms), against a constant cost for SHaRC.
- **Negative result: amortising lamp events** (correlated paths capped at 1 path/px, cells waiting) does reduce the peak
  (16.3 → 8.7 ms), but the error rises from 4.7e-3 to 1.19e-2: very wrong cells stay displayed for 2 frames. The p95 is
  set by the box's steps anyway. Option kept, disabled.

## v10 — Sponza, and two biases relMSE was hiding (commits c2293d4 → …)

- **Sponza** (Crytek glTF, 262 k triangles, scale 0.008): 4 lamps (a large one above the atrium, a small warm lamp in the
  gallery, two on each side), a 1.4 m box that moves, same 120-frame sequence. References at 16,384 spp on the GPU (the
  small lamps near the walls give fireflies at 4,096 spp).
- **Metric**: Sponza's raw relMSE is dominated by a few fireflies (one frame out of 117 is enough to decide the mean,
  standard deviation between seeds ~70 %). Added: relMSE without the worst 0.1 % of pixels (common practice in path
  guiding): standard deviation between seeds ~2 %, monotonic with the budget.
- **Bias found by looking at the images**: 20 frames after the large lamp switched off, the KRC image was 2× too dark and
  did not recover. relMSE rewarded it (with y² + 0.01 in the denominator, a dark scene favours a too-dark image). New
  indicator: total image luminance / reference. Command `krc bias`.
  1. **Allocation**: from an empty cache, static KRC drops to 0.68 within 4 frames (SHaRC 1.00; KRC 100 % uniform 0.98).
     A cell drawing a rare value (heavy tail: small bright lamp, indirect) sees its variance explode, is immediately
     oversampled, and the rare value is diluted in an n that grows *because it was seen*. A uniform defensive share does
     almost nothing (50 %: 0.74). Fix: relative allocation variance capped at the median → 0.93 at f4, 0.96–0.97 after.
  2. **Lamp events**: when the small lamp switched on, 81 % of suspects saw no difference on their 8 correlated paths and
     kept 1,024 samples without the lamp. The early exit is exact for paths, not for the cell's expectation. Fix: those
     cells keep their estimate with a capped weight; max weight of a correction 16 (instead of 64); variance after
     correction at least the history's.
- Price of the fixes: Sponza relMSE* 1.85e-2 → 2.47e-2 (the old value benefited from the bias), Cornell +8 %, box scene
  +13 % (4.14e-3 → 4.67e-3).
- **Sponza 1280×720 result**: at equal mean time, KRC has 2.7 to 2.9× less error than SHaRC; at equal error, it is 3.6×
  faster. Peaks: p95 9.6 ms against 4.5 ms.

## v11 — moving camera and the real SHaRC: the advantage narrows down to light changes (commits 39b9eeb → …)

- **Moving camera**: one view and one reference per frame; KRC evicts cells invisible for 60 frames (same rule as the
  SHaRC sample), free list, table rebuild. About 2 cells per pixel are needed: walking through Sponza at 1280×720, KRC
  keeps up to 1.35 M live cells (~460 MB), against a fixed 176 MB for SHaRC.
- **Official SHaRC** (NVIDIA RTXGI, v1.8.3) ported to WGSL, function by function, with the parameters of NVIDIA's sample.
  Validated: with generous settings, it converges to the reference (relMSE* 6.8e-4, lum 0.99, static Cornell).
- **Correction of an earlier conclusion**: the "2.7 to 2.9× less error than SHaRC at equal time" (v9, v10) were measured
  against my SHaRC-like cache, which traced 1 path per pixel with all of KRC's cell machinery. The real SHaRC updates 4 %
  of pixels and costs 0.8 to 3 ms where KRC costs 4 to 22.
- **Results against the real SHaRC**:
  - In static (fixed or moving camera), SHaRC is as good as or better than KRC for 3 to 25× less GPU time.
  - Over the full sequence, KRC has 3 to 18× less error, almost entirely thanks to light changes: after the large lamp
    switches off, SHaRC stays 7× too bright (lum 7.5) for dozens of frames, KRC is at 0.96.
  - SHaRC's "responsive lighting" mode, meant for this case, has not been tested yet: KRC's advantage on lamps remains to
    be confirmed against it.
- **KRC's cost with a moving camera**: 12.6 ms per frame at 0.25 path/px at 1280×720, of which 5.8 ms of table insertion
  (tombstones, 3.7 M insertions per frame) and 3.6 ms of resolve (every cell, visible or not, every frame). Fixed camera:
  3.9 ms.

## v12 — KRC v2: refining KRC to beat the real SHaRC (commits 7ecba68 → …)

Short tests (`krc duel`, `krc static`: 480×270 or 256², 3 seeds) to decide fast.

- **Fixed cost** divided by 3 with a moving camera (12.6 → 4.4 ms at 1280×720): insertion without useless CAS or shared
  counter, list of visible cells for allocation and resolve, sweeping eviction.
- **Taken from SHaRC, and useful**:
  - cache resampling: an ordinary path stops at the 1st or 2nd bounce on a cell, inheriting its dependencies (correlated
    paths stay complete);
  - larger cells (8 px instead of 4);
  - prior of a new cell from the neighbouring level (camera moving forward);
  - material demodulation.
- **KRC's own, and useful**: target quality on the GPU. In static, KRC stops sampling converged cells, while SHaRC spends
  its budget every frame.
- **Tried, without gain**: vertex reuse, off-screen support cells (memory ×3, overflow), K = 2, two-resolution cache,
  partial recording of dependencies. Kept as disabled options.
- **Result**: at equal GPU time, KRC v2 has 2 to 4× less error than SHaRC on Sponza (fixed or moving camera), including
  against a SHaRC tuned to react fast (5-frame accumulation), and 20 to 27× less on the Cornell box. Still in SHaRC's
  favour: static with a moving camera at high resolution (≈2×), and its lower minimum cost.

## v13 — Test 1 "game-like" (commit d6e9be2 → …)

- Identical pipeline for both methods: per-pixel final image, then a shared denoiser (simplified SVGF: reprojected
  temporal accumulation, history clamped by the neighbourhood's variance, three à-trous filters). KRC: per-pixel direct +
  indirect from the cells (cells now store only the indirect). SHaRC: its official Render pass.
- Without history clamping, the denoiser drags a switched-off lamp 2× too bright for 10 frames and erases KRC's
  advantage: clamping is essential (this is what A-SVGF and game denoisers do).
- At equal GPU time, final denoised image: KRC 1.4 to 3× less error on Sponza, 4.5 to 5.7× on the Cornell box. Static:
  KRC better on Cornell (2.5 to 3.6×), SHaRC slightly better on Sponza (1.2 to 1.35× with a fixed camera, level with a
  moving camera).
- **To the eye**, after Sponza's large lamp switches off: SHaRC too bright but clean; KRC more correct on average but
  **blotchy** (each cell corrected by 8 correlated paths keeps its own noise) and a bit too dark. That is the next defect
  to fix.

## v14 — fixing the defects of the "game-like" test

- **Cause of the blotches found**: in "indirect only" mode, a path that read a cell back at the next bounce recomputed
  that vertex's direct lighting with a single draw (the heavy tail of small lamps). Fix: cells keep the total radiance
  (read back the way SHaRC does) and the mean of their direct share; final image = per-pixel direct + (total − mean
  direct). During an event, the direct share is corrected by the same correlated paths.
- 32 correlated paths per cell for a lamp event (instead of 8).
- Tested without a clear gain: 12 and 16 px cells, no payback of events, reset on large changes, a 5-pass denoiser (too
  blurry without a luminance stop).
- Denoiser history clamp: 3 standard deviations helps KRC on Sponza but makes it slower on the Cornell box; comparisons
  made at both settings, for both methods.
- Result: blotches clearly reduced but not gone; dark drift after switching off 0.74–0.81 (denoiser setting); static
  Sponza still slightly in SHaRC's favour.

## v15 — showcase: textured, glossy Sponza, comparison video

- PBR materials: Sponza's real textures (69, 512² with mipmaps), Lambertian + GGX BRDF, an almost-mirror polished stone
  floor, satin stone, metal, alpha-masked foliage (transparent texels crossed), a rolling chrome sphere, a golden sphere,
  a sky above the roof that "sets", ACES tone mapping.
- Both methods in the same game pipeline: per-pixel reflection (KRC reads its cells at the reflected point, demodulated
  then multiplied back by the exact albedo: sharp texture, noise-free light; SHaRC follows its official lobe-footprint
  rule), a shared denoiser with a separate specular channel and an anti-firefly filter.
- Settings at equal time (~5.5 ms at 960×540): KRC 0.25 path/px into the cells + 2 direct estimates per pixel; SHaRC 1 pixel
  updated per 2×2 block, 3 paths/px.
- Fixes found by looking at the images: firefly clamping in cells at 100× the mean (20× darkened), 16 correlated paths
  per cell for a lamp (switch-on peak 19.7 → 12.6 ms).
- A false problem ruled out: the mirror sphere "misplaced" in the reference (its lower half reflects the floor and blends
  into it).

## v16 — KRC's white dots in the showcase

- Diagnosis per component (`krc showcasedbg f`: cell indirect only, direct + reflection only, full image, with and without
  the denoiser): the white blotches came from the **cells** (a cell drawing a huge rare value becomes a blotch of its size,
  which the denoiser spreads), not from direct lighting.
- An error on my side found along the way: after testing thresholds, cell firefly clamping had stayed disabled (1e9) in
  the previous video.
- Fixes: truncated mean of a pixel's 4 cells (without the brightest or the darkest: an isolated aberrant cell is never
  displayed); clamping at 100× the mean for a mature cell, 20× for a young one (< 64 samples, right after an event); path
  regularisation in game mode for both methods (beyond the first vertex, roughness ≥ 0.3: the diffuse → mirror → small
  lamp caustics, behind the cyan blotches on the blue curtain, disappear; the reference stays exact).
- 12 or 16 px cells: no better.

## v17 — showcase: banners, brightness and mirrors

Feedback by eye: "the flag isn't the same", "it's darker". Four causes, all in what surrounds the caches (denoiser,
reflections), three shared by both methods:

- **Blurry, dull banners** (both): the denoiser divides the light by the albedo before filtering, with a floor at 0.05.
  On a saturated red (green and blue ≈ 0.02), the floor left the texture in the filtered light: smeared, orange-tinted
  golden borders. Floor at 0.002: sharp borders, like the reference.
- **Image too dark** (both): the sky seen in glossy surfaces only arrived through lamp sampling, almost impossible for a
  narrow lobe (polished floor, mirror sphere). MIS (balance heuristic) at the first vertex between lamp sampling and the
  reflection ray: luminance 0.79 → 0.91 at f39, 0.98 to 1.05 afterwards (reference = 1).
- **Large white blotch on the rolling sphere** (both, revealed by the MIS): the unclamped specular history dragged the
  sky's reflection. 3×3 clamp (γ = 2) restored; it no longer darkens, because with MIS the reflected lamps are no longer
  rare values.
- **White dots in the mirror sphere** (KRC): off-screen reflections (scenery behind the camera, floor hidden by the
  sphere) were isolated paths, and the direct light at the reflected point a single shadow ray. New `krc_refl` pass,
  before the resolve: the reflection ray creates the cell at the hit point and feeds it (then 1 ray in 4 once mature, to
  follow the light); a non-metallic point is read entirely from its cell (direct included). KRC's sphere becomes clearly
  cleaner than SHaRC's.
- Cell size: 16 px instead of 8 in the showcase (fewer low-frequency blotches, and cheaper); the time saved goes to 0.4
  path/px.
- Measurement added: PSNR of the displayed images. relMSE* heavily penalises a bright pixel on a dark background of the
  reference (edge of a sky reflection) and hardly a missing reflection: with the sky finally present in mirrors, it rises
  for both methods while the image gets closer to the reference.

## v18 — in a real game: Quake II RTX

- Game: Quake II RTX (NVIDIA, open source, Vulkan path tracing), on a `krc` branch (patches in
  `third_party/q2rtx_krc_patches`; game data: version 1.8.1, 3 shareware levels). The console variable `pt_cache` picks
  the indirect lighting: 0 original game (one traced bounce), 1 SHaRC (NVIDIA's official SDK v1.8.3 in GLSL, unmodified,
  following its integration guide), 2 KRC. Everything else (direct, reflections, ASVGF denoiser, materials) is the
  game's.
- KRC in the game: world-space cells (~16 px on screen, per face), per-pixel Neyman allocation, compact task list,
  resampling on mature cells, and correlated paths on the game's animated lamps (the engine keeps each lamp's intensity
  from the previous frame: each sample evaluates before and after).
- Integrated denoiser (Skynizz's idea): the indirect is rebuilt by KRC (4 neighbouring cells, weighted by their
  confidence), without the game denoiser's history or filter, which re-accumulated the indirect and gave it back the lag
  the correlated paths had just removed. Small time gain here (0.06 ms: the denoiser filters the indirect at 1/3
  resolution; what costs is direct light and reflections at full resolution).
- KRC's cost: 1.48 → 0.54 ms per frame at 1280×720 (one insertion per pixel, marking without contended atomics, subgroup
  reductions, a task list for paths); 0.1 path/px for the same cost as SHaRC.
- Pitfalls: the engine's pause is ignored in timedemo (reference captured on the demo's end screen, fixed); the
  French-language MSVC compiler breaks Ninja's header tracking (UBO inconsistent between C files, black image); auto
  exposure mixed into the lag measurement (frozen during the test).

## v19 — reflections, a KRC-driven denoiser, NVIDIA options

- Places with reflections: `krc_view x y z pitch yaw` (a command added to the game's code); water and glass surfaces
  located in the maps (`tools/q2rtx/find_reflective.py`); two views of `demo1`: a red room with a water floor, an outdoor
  pool. Measured at the same instant for every method (game paused, exposure frozen): otherwise the rippling water made
  the PSNR vary by 3 dB from one session to the next (a first "KRC wins" was only that noise).
- An honest reference: `pt_cache 3`, full 8-bounce paths without a cache, accumulated; no depth of field, with true
  specular everywhere. The game's own reference traces only one bounce.
- Three KRC errors found by comparing: (1) the indirect provided with the normal as its direction (the denoiser's fake
  reflections too strong); (2) then a direction measured per cell (the idea of SHaRC's SH option); (3) first bounce drawn
  like the game (the "uniformish" law), the convention its spherical-harmonics reconstruction expects. Cell history 4096
  (correlated paths keep it up to date).
- KRC ↔ denoiser: a confidence (the cell's relative error σ/√n) decides where to filter the indirect; an "after / before"
  ratio of direct light (correlated paths) relights the denoiser's history. Tried and removed: throwing the history away
  where light changes (noise), keeping the relit history against the denoiser's gradients (the measured ratio is noisy
  with flickering lamps).
- SHaRC's NVIDIA options: no named presets in the SDK; FADE_ACCELERATION on (no measurable effect on the cut); 5-frame
  accumulation: more responsive, same plateau.

## v20 — Doom RT (PrBoom+ RT / RTGL1)

- Before diving in (Skynizz's request): RTGL1 v1.1.1's indirect is a two-bounce path per pixel, stored in spherical
  harmonics, denoised by A-SVGF; no cache. Doom's sector lamps are spherical lamps whose colour follows the sector's light
  level, with a stable identifier: ideal for exact relighting.
- Integration (a `krc` branch of RTGL1): official SHaRC, KRC (passes RtCache0..8), reference mode; tools driven by
  environment variables: session scenario (pause, mode, lamp cut, exposure, debug display), averaged captures, GPU time
  per step (`tools/doomrt/bench.py`, `indirect_bias.py`).
- **RTGL1 bug**: its gradient sample read the lamps' old colour back; the denoiser never saw an intensity change (about
  100 frames of trail in the original game). Fixed for every method.
- **Exact relighting by light groups** (Skynizz's idea): group = lamps of one sector with the same shape; the engine
  computes each group's "after / before" ratio, or declares it unknown if its lamps do not all change the same way. Cells
  follow their two main groups; the denoiser's history is relit by the same ratio (direct, indirect, specular at the
  reflection's hit point), and the gradient takes it into account.
- KRC defects found and fixed: indirect ratio measured on the mean (noisy, made the denoiser's history drift to white);
  **Neyman allocation alone is biased** (a cell that has not found a rare source no longer receives paths: dark areas at
  65 % of the reference; 30 % uniform + 70 % Neyman → 87 %); firefly clamping with a reserve; light direction per channel
  (colours do not arrive from the same side).
- Cost brought down to SHaRC's: direct light cached at reflection points (SHaRC's cone rule), compact view of cells,
  per-pixel read moved out of the ray shader (pass 8: −0.3 ms, KRC's code was taking registers there), pixel pass on one
  pixel per 2×2 block.
- Measurement pitfalls: RTGL1's emission convention (KRC_EMIT_PHYS); a reference averaged after the image's clamping
  (replaced by accumulation inside the engine); the pause state varies from one session to the next (compare only within
  one session).
- Tried without gain: history shortened by the unexplained change; a single cell at the reflection point; a 32,768-sample
  history.

## v21 — noise, bias, traced first bounce; NVIDIA bench ready

- **Noise measurement** (`tools/doomrt/noise.py`: 32 consecutive frames per method, frozen scene): KRC is not noisier than
  SHaRC (E1M1 grain 0.0042 against 0.0062); its lag came from **bias** (light leaking around edges, e.g. E1M3's panel).
- **Traced first bounce** (`KRC_HYBRID`: 0 cells at the pixel, 1 traced where neighbouring cells contradict each other
  beyond their uncertainty, 2 always, like SHaRC); KRC's cache read at the hit point, second bounce traced if there is no
  cell (off-screen point).
- Three defects found along the way: white noise instead of RTGL1's blue noise (4× more noise after the denoiser); a cell
  reused at a point by its raw value (weighted by 1/cos) instead of the received light (evaluating its harmonics with the
  normal); a lamp group not sent by the game treated as off (lasting darkening, frame +128 at 25 dB).
- Result, mode 2 (E1M1 / E1M3): bias 54.9 / 54.0 against SHaRC 54.8 / 54.0, lower grain on E1M1; reaction still better (+2
  to +4 dB up to +64). But E1M2 regresses (41.5 against 47.5 in mode 0), end of convergence below SHaRC (47.4 against 50.0
  at +250) and +0.7 ms GPU. The best mode depends on the map: to be decided.
- Reference bench: NVIDIA's RTXGI SDK (path tracer sample, SHaRC and NRC integrated by NVIDIA, Bistro scene provided)
  builds and runs here in DirectX 12.

## v22 — NVIDIA bench (RTXGI), ꟻLIP, adaptive cells, guiding

- **Reference bench**: the path tracer sample of NVIDIA's RTXGI SDK (on a `krc` branch), SHaRC and NRC integrated by
  NVIDIA, Bistro, living-room and bathroom scenes. Mode without a visible window or focus grab (patched Donut,
  `KRC_HIDDEN`), per-frame scenario, HDR captures, GPU time (`tools/rtxgi/`). Pitfall: the reference accumulated with
  jitter against NRD frames without jitter (every edge differs); everything now uses the same sampling.
- **ꟻLIP** (NVlabs/flip) in LDR and HDR, averaged over 8 denoised frames; plus the raw variance before NRD and the bias
  (mean of 512 frames).
- **KRC-A, adaptive cells** (Skynizz's idea) on SHaRC's table: a parent twice as large per fine cell, frame-to-frame
  variance, James-Stein shrinkage decided at resolve and smoothed. Cache bias reduced (living room: ꟻLIP 0.066 against
  0.074, relMSE −12 %) and raw noise slightly lower; but the frame denoised by NRD is worse (0.082 against 0.072): NRD
  converges to a different result, cause still unknown. +0.3 ms.
- **First-bounce guiding** (a von Mises-Fisher lobe learned per cell, 50 % defensive mixture, MIS): no effect on noise in
  these scenes (broad lighting, no small rare source), +2 to +3 ms (64-bit atomics). Negative result; to retry where the
  heavy tail exists (Doom, Bistro at night) and cheaper.
- **State of the art** (docs/STATE_OF_THE_ART.md): guiding in a hash table already exists (Real-Time Markov Chain Path
  Guiding, I3D 2025); so does the cache as a control variate with an unbiased correction (Neural Two-Level Monte Carlo,
  2025). Still KRC's own: exact correlated paths, relighting by light groups and of the denoiser's history, a guide frozen
  during an event, cell size by statistical test.

## v23 — NVIDIA bench: NRD understood, exact relighting, KRC library

- **NRD and KRC-A**: KRC-A without reading the parent gives SHaRC's image back; but the parent is not the cause: the
  sample's NRD loses 9 to 15 % of energy off-sky, even in pure path tracing, and SHaRC (+5 % off-sky) compensates. At
  equal exposure, KRC-A equals or beats SHaRC. REBLUR's anti-firefly: unrelated (toggle `nrdaf`).
- **Exact relighting in the NVIDIA bench** (the scene's lamps in 3 tracked groups): group shares per cell, tracked along
  SHaRC's paths with its weights (including when a vertex reads the cache back); a `krcRelight` pass before the update
  when the engine knows the ratio. Correct raw image from frame 0 (living room, Bistro at night).
- **NRD's history**: REBLUR's history confidence is not enough (history brought back to one frame, not to zero). What
  works: divide NRD's input by the product S of the pixel's cell factors and multiply its output back; its history comes
  out relit without touching it. S smoothed on screen (otherwise the cells show). Living room, sun cut: ꟻLIP 0.094 from
  frame 0 against 0.113 for SHaRC, which only catches up at +128.
- **Bistro at night**: 4 lamps added by the bench (`lampadd`). Pitfall: the sample samples only lamp 0 for direct light
  (`targetLight = 0`). After NRD, small gain and S darkens 6 % too much (per-cell ratio): next fix, a per-pixel ratio of
  direct light.
- **Cost**: +2 ms at the first try, brought down to +0.01–0.05 ms: the per-entry passes cleared the table's 4 M entries
  every frame; they now clean only at eviction (same fix for KRC-A: +0.24 ms).
- **Library**: `Libraries/Krc/include/KrcRelight.h` (GPU, SHaRC's conventions) and `KrcRelightHost.h` (CPU); the sample
  uses them; copied into `krc_sdk/include`, guide in docs/INTEGRATION.md. Analysis of the next steps: ROADMAP, "Analysis
  (October 2026)".

## v24 — towards a ReShade effect: a screen-space global illumination lab

- Skynizz's question: a universal system that adds ray tracing to any game, even DirectX 12 (Red Dead Redemption 2), with
  an ImGui menu. Answer: impossible as true path tracing (a DX12 game gives neither its geometry, nor its materials, nor
  its lamps; RTX Remix limits itself to DX8/9 for that reason). Possible on screen through ReShade (DX9 to DX12, Vulkan,
  OpenGL, built-in ImGui menu), which only sees the image and the depth. (This later became URTI, which does capture the
  geometry: <https://github.com/Skynizz/URTI>.)
- Lab in the NVIDIA bench (`tools/rtxgi/ssgi.py`, `Shaders/KrcScreenGI.hlsl`): the "game" is direct light only
  (`direct=1`, G-buffer written), the effect adds the indirect, reference = full path tracing. Effect: visibility mask
  (Therrien et al. 2023), GTAO measurement within the slice, reprojected temporal accumulation, à-trous, previous frame's
  radiance (several bounces). Our own addition: the **sky seen on screen** lights the unmasked directions (at infinity,
  the direction from the pixel is the camera's, in the slice's plane).
- First numbers (1280×720): living room (70 % of the indirect comes from the sky, partly off screen) ꟻLIP 0.563 → 0.513
  with the G-buffer, 0.547 with the image alone; Bistro by day 0.145 → 0.133. Cost 2 to 3 ms at full resolution: to move
  to half resolution.
- Fundamental limit: off-screen light; that is where KRC's world-space cache would come in.
