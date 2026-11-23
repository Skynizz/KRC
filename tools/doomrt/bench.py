"""Measurement bench in Doom RT (PrBoom+ RT, RTGL1 krc branch): one session per map, frozen by the
Pause key, where the methods run one after another on the same image.

For each method: warm-up, fps measurement, "before" image, a group of lamps switched off
(light=S), images at +1, +2, +4 ... frames, then the light comes back. The reference (mode 3,
full paths without a cache) is accumulated over hundreds of frames, before and after the cut.

  python -I tools/doomrt/bench.py --maps 1 2 3 --group half --scale 0.1
Outputs: results/doomrt/<name>/E1M<n>/ (selected PNG images) and results/doomrt/<name>/measurements.json
PSNRs are measured outside the player's weapon (identical for every method).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppm  # noqa: E402

GAME = 'F:/doom-rt-krc'
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NAMES = {0: 'game', 1: 'sharc', 2: 'krc'}
AFTER = [1, 2, 4, 8, 16, 32, 64, 128, 250]


def plan(methods, warm, scale, acc):
    """Script (KRC_SCRIPT), captures (KRC_CAPTURE) and the role of each capture."""
    script = ['100:pause', '101:expo']
    captures, roles = [], {}
    f = 110
    for m in methods:
        script.append(f'{f}:mode={m}')
        script.append(f'{f + warm - 210}:fps=200')
        before = f + warm - 1
        captures.append(before)
        roles[before] = (m, 'before')
        cut = f + warm
        script.append(f'{cut}:light={scale}')
        for k in AFTER:
            captures.append(cut + k)
            roles[cut + k] = (m, k)
        script.append(f'{cut + AFTER[-1] + 5}:light=1')
        f = cut + AFTER[-1] + 10
    # reference: full paths, accumulated (normal light, then cut)
    script.append(f'{f}:mode=3')
    captures.append(f'{f + 300}+{acc}')
    roles[f + 300] = (3, 'before')
    cut = f + 300 + acc + 5
    script.append(f'{cut}:light={scale}')
    captures.append(f'{cut + 300}+{acc}')
    roles[cut + 300] = (3, 'after')
    script.append(f'{cut + 300 + acc + 5}:light=1')
    return script, captures, roles


def run_session(mapnum, out, script, captures, env_extra, timeout):
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    env = dict(os.environ)
    env.update({
        'KRC_MODE': '0',
        'KRC_SCRIPT': ';'.join(script),
        'KRC_CAPTURE': ','.join(str(c) for c in captures),
        'KRC_CAPTURE_DIR': out,
        'KRC_EXIT_AFTER': '1',
    })
    env.update(env_extra)
    args = [os.path.join(GAME, 'prboom-plus.exe'), '-iwad', 'doom1.wad', '-warp', '1', str(mapnum), '-skill', '3',
            '-nomonsters', '-nomouse', '-nosound', '-nomusic', '-width', '1280', '-height', '720', '-window']
    # KRC_HIDDEN=1 (experiment): launched minimised without taking the focus, then the DLL moves the window
    # off screen. Does not work yet: the game no longer renders in that state.
    si = None
    if env.get('KRC_HIDDEN', '0') != '0':
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 7  # SW_SHOWMINNOACTIVE
    p = subprocess.Popen(args, cwd=GAME, env=env, startupinfo=si)
    try:
        p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        print(f'  E1M{mapnum}: timed out')


def fps_lines(out, methods):
    path = os.path.join(out, 'fps.txt')
    if not os.path.exists(path):
        return {}
    lines = open(path).read().split('\n')
    res = {}
    for m, line in zip(methods, [l for l in lines if l.strip()]):
        w = line.split()
        res[NAMES[m]] = {w[i]: float(w[i + 1]) for i in range(4, len(w) - 1, 2)}
    return res


def weapon_mask(shape):
    """Pixels outside the player's weapon (bottom centre of the screen), shared by every method."""
    h, w = shape[:2]
    mask = np.ones((h, w), bool)
    mask[int(h * 0.62):, int(w * 0.36):int(w * 0.66)] = False
    return mask


def analyse(out, roles, methods):
    img = lambda f: ppm.read(os.path.join(out, f'f{f:05d}.ppm'))
    ref = {k: img(f) for f, (m, k) in roles.items() if m == 3}
    mask = weapon_mask(ref['before'].shape)
    res = {}
    for m in methods:
        frames = {k: f for f, (mm, k) in roles.items() if mm == m}
        r = {'before': ppm.psnr(img(frames['before']), ref['before'], mask)}
        r['after'] = {k: ppm.psnr(img(frames[k]), ref['after'], mask) for k in AFTER}
        res[NAMES[m]] = r
    return res


def keep_images(out, dst, roles):
    """Useful images as PNG (before, +4, +32, +250 and references)."""
    os.makedirs(dst, exist_ok=True)
    for f, (m, k) in roles.items():
        if k in ('before', 'after', 4, 32, 250):
            name = f'{NAMES.get(m, "reference")}_{k}.png'
            subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', os.path.join(out, f'f{f:05d}.ppm'),
                            os.path.join(dst, name)], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maps', type=int, nargs='+', default=[1])
    ap.add_argument('--methods', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--group', default='all', choices=['all', 'half'])
    ap.add_argument('--scale', type=float, default=0.1)
    ap.add_argument('--warm', type=int, default=400)
    ap.add_argument('--acc', type=int, default=512)
    ap.add_argument('--name', default='cut')
    ap.add_argument('--env', nargs='*', default=[], help='extra settings, NAME=value')
    ap.add_argument('--timeout', type=int, default=900)
    a = ap.parse_args()

    script, captures, roles = plan(a.methods, a.warm, a.scale, a.acc)
    script.insert(2, f'102:group={a.group}')
    env = dict(e.split('=', 1) for e in a.env)
    results_dir = os.path.join(ROOT, 'results', 'doomrt', a.name)
    os.makedirs(results_dir, exist_ok=True)
    summary = {'settings': vars(a), 'maps': {}}
    for mp in a.maps:
        out = os.path.join(GAME, 'cap', a.name, f'E1M{mp}')
        print(f'E1M{mp} ...', flush=True)
        run_session(mp, out, script, captures, env, a.timeout)
        try:
            res = analyse(out, roles, a.methods)
        except FileNotFoundError as e:
            print('  missing capture:', e)
            continue
        res['fps'] = fps_lines(out, a.methods)
        summary['maps'][f'E1M{mp}'] = res
        keep_images(out, os.path.join(results_dir, f'E1M{mp}'), roles)
        for name, r in res.items():
            if name == 'fps':
                continue
            series = ' '.join(f'+{k}:{v:.1f}' for k, v in r['after'].items())
            t = res['fps'].get(name, {})
            print(f'  {name:6s} before {r["before"]:.2f} dB | after the cut {series} | GPU {t.get("gpu", 0):.2f} ms '
                  f'(cache {t.get("cache", 0):.2f}, indirect {t.get("indirect", 0):.2f}) {t.get("fps", 0):.0f} fps')
    with open(os.path.join(results_dir, 'measurements.json'), 'w') as f:
        json.dump(summary, f, indent=1)


if __name__ == '__main__':
    main()
