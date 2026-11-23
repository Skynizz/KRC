"""Images of the RTXGI bench: PFM reading (linear HDR), the sample's tone curve (Reinhard
on luminance, then sRGB), simple PSNR and ꟻLIP-like metrics, contact sheets and videos."""

import subprocess

import numpy as np


def read_pfm(path):
    with open(path, 'rb') as f:
        assert f.readline().strip() == b'PF'
        w, h = map(int, f.readline().split())
        scale = float(f.readline())
        data = np.frombuffer(f.read(), '<f4' if scale < 0 else '>f4')
    return data.reshape(h, w, 3)[::-1].astype(np.float64)


def tonemap(hdr, exposure=1.0):
    """Like Tonemapping.hlsl: exposure, Reinhard on luminance, clipping; then sRGB."""
    c = np.maximum(hdr * exposure, 0)
    lum = c @ np.array([0.299, 0.587, 0.114])
    k = np.where(lum > 0, (lum / (lum + 1)) / np.maximum(lum, 1e-12), 0)
    c = np.clip(c * k[..., None], 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def psnr(a, b):
    """PSNR of the displayed images (after the tone curve), in dB."""
    return 10 * np.log10(1.0 / max(((tonemap(a) - tonemap(b)) ** 2).mean(), 1e-12))


def rel_mse(a, b):
    """relMSE in HDR, without the worst 0.1 % of pixels (common practice in papers)."""
    e = (((a - b) ** 2) / (b ** 2 + 1e-2)).mean(-1).ravel()
    e.sort()
    return float(e[: int(len(e) * 0.999)].mean())


def write_png(path, ldr):
    img = (np.clip(ldr, 0, 1) * 255 + 0.5).astype(np.uint8)
    h, w, _ = img.shape
    p = subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                        '-s', f'{w}x{h}', '-i', '-', path], input=img.tobytes())
    assert p.returncode == 0


def error_map(a, b, gain=8.0):
    """Signed difference of the displayed images, grey = 0."""
    return np.clip((tonemap(a) - tonemap(b)) * gain + 0.5, 0, 1)


def flip(test, ref, hdr=False):
    """NVIDIA's ꟻLIP (NVlabs/flip): error perceived when flipping between the two images, 0 to 1 (lower =
    better). LDR: on the displayed images (the sample's tone curve); HDR: on the linear
    images, ꟻLIP sweeps the exposures itself. Returns (mean error, magma map)."""
    import flip_evaluator
    if hdr:
        a, b = np.maximum(ref, 0).astype(np.float32), np.maximum(test, 0).astype(np.float32)
        emap, mean, _ = flip_evaluator.evaluate(a, b, 'HDR', inputsRGB=False)
    else:
        a, b = tonemap(ref).astype(np.float32), tonemap(test).astype(np.float32)
        emap, mean, _ = flip_evaluator.evaluate(a, b, 'LDR')
    return float(mean), emap
