# State of the art relevant to KRC (October 2026)

What the best recent work does, what KRC takes from it, and what remains KRC's own. To be completed before any claim of
novelty.

## World-space radiance caches

- **SHaRC** (NVIDIA, RTXGI SDK v2, v1.8.3): a world-space hash table, cell size depending on the distance to the camera,
  updated by 1 pixel in 25, read at the hit point; optional spherical harmonics (light direction), blending of
  neighbouring levels when the camera moves. The baseline of our benches. <https://github.com/NVIDIA-RTX/RTXGI>
- **NRC** (Müller et al., SIGGRAPH 2021): a neural cache trained continuously; of little use for dielectric materials
  (glass). <https://arxiv.org/abs/2106.12372>
- **Neural Two-Level Monte Carlo / NIRC** (CGF 2025): a cache of *incident* radiance, and a two-level estimator: the
  cache gives the bulk, a rare unbiased correction removes its error. A direct cousin of our control variate (v2): a
  lead for correcting the cache's bias (leaks, clamping) instead of patching it. <https://arxiv.org/abs/2412.04634>
- **World-space ReSTIR** (GPUOpen, SIGGRAPH Asia 2021): reservoirs stored in a world-space hash table.
  <https://gpuopen.com/download/SA2021_WorldSpace_ReSTIR.pdf>
- **Bevy Solari** (2025): ReSTIR DI + ReSTIR GI + a world-space irradiance cache; the cache keeps the old light when lamps
  switch off (lag), a problem KRC addresses. <https://jms55.github.io/posts/2025-12-27-solari-bevy-0-18/>

## Path guiding

- **Real-Time Markov Chain Path Guiding** (Alber, Hanika, Dachsbacher, I3D 2025): unbiased real-time guiding, von
  Mises-Fisher mixtures stored in world-space hash tables, a path-traced Quake demo. Per-cell guiding is therefore not
  new in KRC. <https://cg.ivd.kit.edu/english/markov-chain-path-guiding.php>
- **Markov Chain Mixture Models for Real-Time Direct Illumination** (Dittebrandt et al., EGSR 2023): the screen-space
  version. <https://cg.ivd.kit.edu/english/mcmm.php>
- **RCPG** (product guiding through radiance cascades): <https://dl.acm.org/doi/10.1145/3840291>

## Measurement

- **ꟻLIP** (NVIDIA Research, HPG 2020): perceived error when alternating two images, LDR and HDR.
  <https://github.com/NVlabs/flip>

## What remains KRC's own (to be checked in the literature before claiming it)

- Correlated paths before / after a change of lamps, with an exact Δ = 0 where nothing changes.
- Exact relighting by light groups, applied to the denoiser's history too; for a closed denoiser (NRD), by normalising
  its input by the product of the cell's factors.
- A guide frozen during an event (same sampling law for the old and the new scene), so that correlated paths stay exact
  while being guided.
- Cell size decided by a statistical test (fine cell / parent, shrinkage).
