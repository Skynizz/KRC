# Integrating KRC into an engine or a game

For: an engine developer who already has a path tracer (or is integrating SHaRC). KRC sits **on top of SHaRC**: same
hash table, same passes, plus a few buffers and five hook points. An engine that already has SHaRC gets KRC in a day;
without SHaRC, first follow SHaRC's integration guide (NVIDIA's RTXGI SDK), then this document.

Reference implementation, built and measured: the path tracer sample of the RTXGI SDK, on a `krc` branch (files
`Samples/Pathtracer/Shaders/KrcRelight.hlsli`, `SharcResolve.hlsl`, `Pathtracer.hlsl`, `Denoiser.hlsl`, `KrcBench.cpp`;
the patches are in `third_party/rtxgi_krc_patches`). The library itself lives in `Libraries/Krc/include`, copied into this
repository under `krc_sdk/include`.

## What KRC brings

| Feature | File | Measured gain (NVIDIA bench) | Cost |
|---|---|---|---|
| Exact relighting of the cache by light groups | `KrcRelight.h` | correct raw image from frame 0 after a lamp change (SHaRC: ~128 frames) | +0.01 to +0.05 ms (living room, 1280×720) |
| Denoiser history relit (normalisation by S) | `KrcRelight.h` + smoothing pass | ꟻLIP after NRD 0.095 against 0.114 at frame 0 (living room, sun cut) | included above; smoothing only after a first relight |
| Adaptive cells (KRC-A) | `KrcAdaptive.hlsli` (sample) | cache bias −12 % (relMSE) | +0.24 ms |

## The five hook points

Everything goes through `KrcRelightParameters` (buffers and settings), like SHaRC with `SharcParameters`. Include
`KrcRelight.h` after `SharcCommon.h`.

1. **Buffers** (one entry per SHaRC table entry, cleared with it): `uint4` frame sums, `float4` smoothed shares, `float2`
   factors (frame k, product S). That is 40 bytes per entry: 160 MB for the sample's 4 M entries; 1 M entries is often
   enough (40 MB).
2. **SHaRC update** (update pass):
   - for each lamp evaluated at the vertex, `krcDirectGroups += KrcGroupSplit(p, lightIndex, contribution)`;
   - right before `SharcUpdateHit`: `KrcUpdateHit(sharc, p, sharcState, hitData, directLighting, krcDirectGroups, random)`
     with **the same random number** as `SharcUpdateHit` (draw it once into a variable);
   - right before `SharcUpdateMiss`: `KrcUpdateMiss(p, sharcState, skyRadiance)`.
3. **Resolve**: `bool v = KrcResolveEntry(i, sharc, p, accumulationFrameNum);` before `SharcResolveEntry`, then
   `KrcResolveEviction(i, sharc, p, v);` after.
4. **Relight pass**: a compute pass, one thread per entry, `KrcRelightEntry(i, sharc, p)`, **before the SHaRC update**,
   only on frames where `KrcRelightFrame::event` is true.
5. **CPU side**: `KrcRelightHost` (`KrcRelightHost.h`) receives each lamp's power and group every frame, and returns the
   ratios, `lightGroups`, `event` and `active`.

Groups: 0 to 2 tracked, 3 fixed. The engine chooses: for example 0 sun, 1 indoor lamps, 2 flickering lamps; in Doom, one
group per sector and lamp shape. A group whose lamps change by different ratios, or switch on from zero, is unknown that
frame (no relighting: SHaRC takes over, with its lag).

## Denoiser (optional, recommended)

NRD, DLSS RR and the others keep their own history, which the cache cannot touch. For it to follow the relighting:
1. at the first hit of each pixel, write `KrcGetRelightFactor(sharc, p, pos, normal).y` (the product S) into a texture, if
   `active`;
2. smooth that texture (the `krcScaleBlur` pass of `Denoiser.hlsl`: 9 × 9 taps spaced 4 pixels apart, in log space,
   stopped at edges), otherwise the cells show in the image;
3. divide the denoiser's input by the smoothed S, multiply its output by the same S.

Variant not kept: NRD's history confidence (`IN_DIFF_CONFIDENCE`); REBLUR only brings its history back to one frame, the
old lighting still weighs 50 % and then fades as 1/n.

## Pitfalls met

- NVIDIA's sample samples only lamp 0 for direct light by default (`targetLight = 0`): added lamps do nothing until it is
  set to −1.
- Any change to a CPU/GPU shared header (constants) must rebuild every file; the French-language MSVC compiler breaks
  Ninja's header tracking.
- Do not walk the whole table to write zeros every frame: most entries are empty, and 4 M writes cost more than the rest
  of KRC.
- GLSL (Quake II RTX, RTGL1): `KrcRelight.h` follows SHaRC's conventions (`RW_STRUCTURED_BUFFER`, `BUFFER_AT_OFFSET`) but
  uses HLSL syntax (`float3`, `lerp`, `saturate`, `InterlockedAdd`); it needs the same compatibility macros as the
  project uses for SHaRC.

## Where to test it

| Target | State | Effort |
|---|---|---|
| NVIDIA's path tracer sample (RTXGI) | done, automatic bench (`tools/rtxgi/`) | — |
| Quake II RTX (GLSL, official SHaRC) | original KRC integrated; the library still to plug in | 1 to 2 days |
| Doom RT (PrBoom+ RT / RTGL1, GLSL) | original KRC integrated, exact relighting | — |
| RTX Remix (`dxvk-remix`, open source) | to study: a single integration covers Portal, Half-Life 2, Morrowind…; its current cache is NRC, its sources should be checked for SHaRC too | 1 to 2 weeks |
| Unreal Engine 5 (sources available) | to study, heavier (reference path tracer, not real time) | several weeks |
