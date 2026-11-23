"""Lab for screen-space global illumination (KRC-SSGI, a future ReShade effect) in NVIDIA's path tracer.

One invisible session: reference (accumulated full paths), then the "game" (direct light
only, accumulated: noise-free, like a rasterised render), then the game + the effect, setting by
setting. Measured against the reference: ꟻLIP, luminance outside the sky, frame-to-frame noise,
GPU time (whole frame: the difference with the game alone is the cost of the effect).

  python -I tools/rtxgi/ssgi.py --scene LivingRoom.scene.json --name living_room
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

VARIANTS = {
    'game': ['ssgi=0'],                                             # no indirect lighting
    'ideal': ['ssgi=1', 'ssginormals=1', 'ssgialbedo=1'],           # G-buffer normals and albedo
    'reshade': ['ssgi=1', 'ssginormals=0', 'ssgialbedo=0'],         # colour + depth only
}
LAST = 8


def plan(variants, settings, ref_spp, frames):
    cmds = ['20:jitter=0', '20:tech=ref', '20:bounces=8', '20:direct=0', '20:ssgi=0', '20:den=acc', f'{20 + ref_spp}:capture=ref']
    f = 25 + ref_spp
    for name in variants:
        cmds += [f'{f}:{c}' for c in ['bounces=2', 'direct=1'] + settings + VARIANTS[name] + ['den=acc']]
        cmds.append(f'{f + frames - 150}:fps=100')
        cmds += [f'{f + frames + k}:capture={name}_{k}' for k in range(LAST)]
        f += frames + LAST + 5
    cmds.append(f'{f}:exit')
    return ';'.join(cmds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='LivingRoom.scene.json')
    ap.add_argument('--name', default='living_room')
    ap.add_argument('--variants', nargs='+', default=['game', 'ideal', 'reshade'])
    ap.add_argument('--set', nargs='*', default=[], help='shared ssgi*=... settings')
    ap.add_argument('--ref', type=int, default=2048)
    ap.add_argument('--frames', type=int, default=400)
    ap.add_argument('--eval-only', action='store_true')
    a = ap.parse_args()

    out = f'F:/rtxgi/out/ssgi_{a.name}'
    if not a.eval_only:
        if os.path.exists(f'{out}/fps.txt'):
            os.remove(f'{out}/fps.txt')
        print(run.run(out, plan(a.variants, a.set, a.ref, a.frames), ['-scene', a.scene]))

    L = lambda x: x[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
    ref = img.read_pfm(f'{out}/ref.pfm')
    mask = L(ref) < 5.0
    lines = [l for l in open(f'{out}/fps.txt').read().splitlines() if l.strip()] if os.path.exists(f'{out}/fps.txt') else []
    gpu = {n: float(l.split('gpu_ms')[1].split()[0]) for n, l in zip(a.variants, lines)}
    dst = os.path.join(ROOT, 'results', 'rtxgi', 'ssgi_' + a.name)
    os.makedirs(dst, exist_ok=True)
    img.write_png(os.path.join(dst, 'reference.png'), img.tonemap(ref))

    res = {}
    for name in a.variants:
        frames = [img.read_pfm(f'{out}/{name}_{k}.pfm') for k in range(LAST)]
        last = frames[-1]
        ldr = np.stack([img.tonemap(x) for x in frames])
        r = {
            'flip': img.flip(last, ref)[0], 'psnr': img.psnr(last, ref),
            'lum': float(L(last)[mask].mean() / L(ref)[mask].mean()),
            'noise': float(ldr.std(0).mean()), 'gpu_ms': gpu.get(name, 0.0),
        }
        res[name] = r
        img.write_png(os.path.join(dst, f'{name}.png'), img.tonemap(last))
        img.write_png(os.path.join(dst, f'{name}_flip.png'), img.flip(last, ref)[1])
        print(f'{name:10s} FLIP {r["flip"]:.4f} PSNR {r["psnr"]:.2f} | lum outside the sky {r["lum"]:.3f} | '
              f'noise {r["noise"]:.4f} | GPU {r["gpu_ms"]:.2f} ms', flush=True)
    with open(os.path.join(dst, 'ssgi.json'), 'w') as f:
        json.dump({'settings': vars(a), 'results': res}, f, indent=1)


if __name__ == '__main__':
    main()
