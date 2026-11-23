# KRC — Knowledge Radiance Cache

KRC is a research prototype of a real-time radiance cache for path-traced global illumination. It treats rendering as
**maintaining knowledge of the light** rather than computing images. Each cell of a world-space cache stores its
radiance and its uncertainty. Every frame, the ray budget goes where the doubt is. When something changes, only the
cells whose light really depended on it are updated.

The goal: a static scene converges and then costs almost nothing, and a dynamic scene costs in proportion to what
changes, not to its size.

## The idea, and what it became

The starting idea had three parts: a cache with uncertainty, rays allocated to uncertainty, and a dependency graph
"like a compiler" (each cell remembers which objects its rays touched). Measuring it changed the picture:

- **"Objects touched" is a wrong dependency.** A new shadow or a light switching on is never seen by rays that missed
  the object. Dependencies were reformulated as regions of space crossed by paths, weighted by energy (in the spirit of
  Selective Photon Tracing).
- **The graph is not what makes the difference.** The real gain comes from **differential updates through correlated
  paths**: the same random path is traced in the old and the new scene, and the difference corrects the cache, used as
  a control variate. The graph only reduces cost, and only for localised changes.
- **Statistics decide, not heuristics.** Dependencies point to suspects and a statistical test decides. Rays are
  allocated with Neyman allocation and a Bayesian prior; cell sizes are chosen by a statistical test.

The full reasoning and protocol: [docs/HYPOTHESIS.md](docs/HYPOTHESIS.md).

## Results

All numbers are measured, at equal budget, against a converged reference, with several seeds. Details, plots and
every negative result: [docs/RESULTS.md](docs/RESULTS.md).

- **Dynamic sequence (box scene, equal budget):** KRC divides the error by 3 compared with the best classic cache, and
  by 21 compared with accumulated path tracing (relMSE 4.1e-3 against 1.25e-2 and 8.9e-2).
- **Static scenes cost almost nothing:** in target-quality mode, KRC drops to about 0.01 path per pixel per frame once
  converged, which is the exploration floor.
- **Cost follows change, not scene size:** from 2 to 8 rooms, KRC's cost while a box moves goes *down* (16 k to 10 k
  paths per frame), while a cache that resets grows from 42 k to 165 k.
- **Light changes become local:** with energy-weighted dependencies, switching on a lamp in 8 rooms costs 28 k paths
  instead of 56 k, and no longer grows with the scene.
- **Against NVIDIA's real SHaRC (v1.8.3, ported to WGSL and validated), at equal GPU time:** KRC v2 has 2 to 4 times
  less error on Sponza in dynamic sequences, fixed or moving camera, and 20 to 27 times less on the Cornell box.
  **SHaRC stays better** on static scenes with a moving camera at 1280×720 (about 2×) and has a lower minimum cost
  (0.9 ms against 3 ms at 1280×720).
- **Fully on the GPU** (wgpu / Vulkan, hardware ray queries), validated against the CPU tracer and stable over
  12,000 frames.

## In real renderers

KRC was then tested outside its own prototype. The patches are in [third_party/](third_party/):

| Target | What was done |
|---|---|
| Quake II RTX | KRC and the official SHaRC as the indirect-light cache, KRC-driven denoiser, multi-bounce reflection reference |
| Doom RT (RTGL1) | KRC and SHaRC in the engine, noise and bias measurements, traced first bounce |
| NVIDIA RTXGI SDK (path tracer sample) | measurement bench with FLIP, adaptive cells (KRC-A), exact relighting by light groups, NRD history relit through the cache |

The relighting library used there is in [krc_sdk/include](krc_sdk/include) (`KrcRelight.h` for the GPU, following
SHaRC's conventions, and `KrcRelightHost.h` for the CPU). Integration guide: [docs/INTEGRATION.md](docs/INTEGRATION.md).

A ReShade effect, [reshade/KRC_GI.fx](reshade/KRC_GI.fx), came out of the last experiment: screen-space indirect
lighting with a half-resolution visibility mask and the sky seen on screen.

## Running it

Requirements: Rust (stable) and a Vulkan GPU with ray queries for the hardware ray-tracing modes.

```sh
cargo run --release -- all          # references and every method on the box scene
cargo run --release -- scale        # scaling with the number of rooms
python tools/plots.py               # plots in results/plots
sh tools/fetch_sponza.sh            # downloads Sponza (not in the repository)
cargo run --release -- sponza 480
```

## Repository layout

| Folder | Content |
|---|---|
| `src/` | the prototype: CPU and GPU path tracers, KRC, the SHaRC port, the scenes |
| `krc_sdk/` | the relighting library for engines |
| `third_party/` | patches for Quake II RTX, RTGL1 and the RTXGI SDK, SHaRC license |
| `tools/` | measurement benches and plots |
| `reshade/` | the KRC GI ReShade effect |
| `docs/` | hypothesis, results, state of the art, roadmap, integration, journal |
| `results/` | measurements and plots |

## State of the art

What KRC takes from recent work and what remains its own (correlated paths, relighting by light groups and of the
denoiser's history, statistical cell sizes) is listed in [docs/STATE_OF_THE_ART.md](docs/STATE_OF_THE_ART.md). The step-by-step
log of every experiment is [docs/JOURNAL.md](docs/JOURNAL.md).

## License

KRC's code is under the [MIT](LICENSE) license. `src/sharc.wgsl` is a port of NVIDIA's SHaRC and keeps the NVIDIA RTX
SDKs License ([third_party/sharc](third_party/sharc)). `reshade/ReShade*.fxh` come from crosire/reshade-shaders (CC0).
The patches in `third_party/` apply to projects that keep their own licenses.
