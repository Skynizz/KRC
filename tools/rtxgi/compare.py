"""Robust comparison of the methods in NVIDIA's sample path tracer (krc branch).

One invisible session per scene, everything on the same image (jitter off, like NRD):
  - reference: accumulated full paths;
  - for each method: 8 frames denoised by NRD (mean ꟻLIP: what the player sees),
    8 raw frames (noise before NRD), a mean of frames (bias of the cache alone), GPU time.

  python -I tools/rtxgi/compare.py --scene LivingRoom.scene.json --name living_room --methods sharc sharc_guide
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import img  # noqa: E402
import run  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

METHODS = {
    'sharc': ['tech=sharc', 'scale=50', 'guide=0', 'nrdaf=1'],
    'sharc_guide': ['tech=sharc', 'scale=50', 'guide=1', 'nrdaf=1'],
    'krca': ['tech=krc', 'scale=50', 'guide=0', 'krcdebug=0', 'nrdaf=1'],
    'krca_guide': ['tech=krc', 'scale=50', 'guide=1', 'nrdaf=1'],
    'krca_noparent': ['tech=krc', 'scale=50', 'guide=0', 'krcdebug=2', 'nrdaf=1'],  # all of KRC-A except reading the parent
    'sharc_af0': ['tech=sharc', 'scale=50', 'guide=0', 'nrdaf=0'],  # NRD's anti-firefly off
    'krca_af0': ['tech=krc', 'scale=50', 'guide=0', 'krcdebug=0', 'nrdaf=0'],
    'krcl': ['tech=sharc', 'scale=50', 'guide=0', 'relight=1', 'nrdconf=2', 'nrdaf=1'],  # SHaRC + exact relighting + relit NRD history
    'krcl_follow': ['tech=sharc', 'scale=50', 'guide=0', 'relight=1', 'nrdconf=0', 'nrdaf=1'],  # group shares only
    'path': ['tech=ref', 'guide=0', 'nrdaf=1'],   # full path tracing without a cache, same denoiser
}
FRAMES = 8


def plan(methods, ref_spp, warm, acc):
    cmds = ['20:jitter=0', '20:tech=ref', '20:den=acc', f'{20 + ref_spp}:capture=ref']
    f = 25 + ref_spp
    for name in methods:
        cmds += [f'{f}:{c}' for c in ['relight=0', 'nrdconf=0', 'krcdebug=0'] + METHODS[name] + ['den=nrd']]
        cmds.append(f'{f + warm - 110}:fps=100')
        cmds += [f'{f + warm + k}:capture={name}_nrd{k}' for k in range(FRAMES)]
        g = f + warm + FRAMES
        cmds.append(f'{g}:den=none')
        cmds += [f'{g + 10 + k}:capture={name}_raw{k}' for k in range(FRAMES)]
        g += 10 + FRAMES
        cmds.append(f'{g}:den=acc')
        cmds.append(f'{g + acc}:capture={name}_acc')
        f = g + acc + 5
    cmds.append(f'{f}:exit')
    return ';'.join(cmds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='bistro.scene.json')
    ap.add_argument('--name', default='bistro')
    ap.add_argument('--methods', nargs='+', default=['sharc', 'sharc_guide'])
    ap.add_argument('--ref', type=int, default=2048)
    ap.add_argument('--warm', type=int, default=250)
    ap.add_argument('--acc', type=int, default=512)
    ap.add_argument('--eval-only', action='store_true')
    a = ap.parse_args()

    out = f'F:/rtxgi/out/cmp_{a.name}'
    if not a.eval_only:
        if os.path.exists(f'{out}/fps.txt'):
            os.remove(f'{out}/fps.txt')
        print(run.run(out, plan(a.methods, a.ref, a.warm, a.acc), ['-scene', a.scene]))

    ref = img.read_pfm(f'{out}/ref.pfm')
    dst = os.path.join(ROOT, 'results', 'rtxgi', a.name)
    os.makedirs(dst, exist_ok=True)
    img.write_png(os.path.join(dst, 'reference.png'), img.tonemap(ref))
    lines = [l for l in open(f'{out}/fps.txt').read().splitlines() if l.strip()] if os.path.exists(f'{out}/fps.txt') else []
    gpu = {n: float(l.split('gpu_ms')[1].split()[0]) for n, l in zip(a.methods, lines)}

    res = {}
    for name in a.methods:
        nrd = [img.read_pfm(f'{out}/{name}_nrd{k}.pfm') for k in range(FRAMES)]
        raw = np.stack([img.tonemap(img.read_pfm(f'{out}/{name}_raw{k}.pfm')) for k in range(FRAMES)])
        acc = img.read_pfm(f'{out}/{name}_acc.pfm')
        flips = [img.flip(x, ref)[0] for x in nrd]
        r = {
            'nrd_flip': float(np.mean(flips)), 'nrd_flip_std': float(np.std(flips)),
            'nrd_psnr': float(np.mean([img.psnr(x, ref) for x in nrd])),
            'noise_raw': float(raw.std(0).mean()),
            'bias_flip': img.flip(acc, ref)[0], 'bias_relmse': img.rel_mse(acc, ref),
            'lum': float(acc.mean() / ref.mean()), 'gpu_ms': gpu.get(name, 0.0),
        }
        res[name] = r
        img.write_png(os.path.join(dst, f'{name}_nrd.png'), img.tonemap(nrd[0]))
        img.write_png(os.path.join(dst, f'{name}_raw.png'), raw[0])
        img.write_png(os.path.join(dst, f'{name}_flip.png'), img.flip(nrd[0], ref)[1])
        print(f'{name:12s} NRD FLIP {r["nrd_flip"]:.4f} +- {r["nrd_flip_std"]:.4f} PSNR {r["nrd_psnr"]:.2f} | '
              f'raw noise {r["noise_raw"]:.4f} | bias FLIP {r["bias_flip"]:.4f} relMSE {r["bias_relmse"]:.4f} '
              f'lum {r["lum"]:.3f} | GPU {r["gpu_ms"]:.2f} ms', flush=True)
    with open(os.path.join(dst, 'comparison.json'), 'w') as f:
        json.dump({'settings': vars(a), 'results': res}, f, indent=1)


if __name__ == '__main__':
    main()
