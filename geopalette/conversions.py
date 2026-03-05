"""
geopalette.conversions
~~~~~~~~~~~~~~~~~~~~~~
RGB ↔ color-space conversion functions for 2-D NumPy arrays (raster bands).

Every forward function has the signature::

    f(R, G, B) → tuple[np.ndarray, ...]

where R, G, B are 2-D uint8 arrays (0-255). Internally they are normalised
to [0, 1] float32.  The returned arrays are always float32.

The inverse ``lab_to_rgb`` converts CIELAB → linear RGB [0, 1].
"""

import numpy as np

# ---------------------------------------------------------------------------
# Hue-based spaces
# ---------------------------------------------------------------------------

def rgb_to_hsl(R, G, B):
    """RGB → HSL.  Returns H (0-360°), S (0-1), L (0-1)."""
    Rn = R.astype(np.float32) / 255.0
    Gn = G.astype(np.float32) / 255.0
    Bn = B.astype(np.float32) / 255.0

    cmax = np.maximum(np.maximum(Rn, Gn), Bn)
    cmin = np.minimum(np.minimum(Rn, Gn), Bn)
    delta = cmax - cmin

    L = (cmax + cmin) / 2.0

    H = np.zeros_like(cmax)
    mask = delta != 0
    mr = (cmax == Rn) & mask
    mg = (cmax == Gn) & mask
    mb = (cmax == Bn) & mask

    H[mr] = 60.0 * np.mod((Gn - Bn)[mr] / delta[mr], 6.0)
    H[mg] = (60.0 * (((Bn - Rn) / delta) + 2.0))[mg]
    H[mb] = (60.0 * (((Rn - Gn) / delta) + 4.0))[mb]

    S = np.zeros_like(cmax)
    non_zero = delta != 0
    S[non_zero] = delta[non_zero] / (1.0 - np.abs(2.0 * L[non_zero] - 1.0))

    return H.astype(np.float32), S.astype(np.float32), L.astype(np.float32)


def rgb_to_hsv(R, G, B):
    """RGB → HSV.  Returns H (0-360°), S (0-1), V (0-1)."""
    Rn = R.astype(np.float32) / 255.0
    Gn = G.astype(np.float32) / 255.0
    Bn = B.astype(np.float32) / 255.0

    cmax = np.maximum(np.maximum(Rn, Gn), Bn)
    cmin = np.minimum(np.minimum(Rn, Gn), Bn)
    delta = cmax - cmin

    H = np.zeros_like(cmax)
    mask = delta != 0
    mr = (cmax == Rn) & mask
    mg = (cmax == Gn) & mask
    mb = (cmax == Bn) & mask

    H[mr] = (60.0 * ((Gn - Bn) / delta))[mr] % 360.0
    H[mg] = (60.0 * (((Bn - Rn) / delta) + 2.0))[mg]
    H[mb] = (60.0 * (((Rn - Gn) / delta) + 4.0))[mb]

    S = np.zeros_like(cmax)
    nz = cmax != 0
    S[nz] = delta[nz] / cmax[nz]

    V = cmax

    return H.astype(np.float32), S.astype(np.float32), V.astype(np.float32)


def rgb_to_hsi(R, G, B):
    """RGB → HSI.  Returns H (0-360°), S (%), I (0-255)."""
    Rf = R.astype(np.float32)
    Gf = G.astype(np.float32)
    Bf = B.astype(np.float32)

    sum_rgb = Rf + Gf + Bf
    sum_rgb[sum_rgb == 0] = 1.0
    r = Rf / sum_rgb
    g = Gf / sum_rgb
    b = Bf / sum_rgb

    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    den[den == 0] = np.nan
    h_rad = np.arccos(np.clip(num / den, -1.0, 1.0))
    h_rad[b > g] = 2 * np.pi - h_rad[b > g]
    H = np.degrees(h_rad)
    H = np.nan_to_num(H)

    S = (1 - 3 * np.minimum(np.minimum(r, g), b)) * 100.0
    I = (Rf + Gf + Bf) / 3.0

    return H.astype(np.float32), S.astype(np.float32), I.astype(np.float32)


# ---------------------------------------------------------------------------
# CIE-based spaces
# ---------------------------------------------------------------------------

# Shared RGB → XYZ matrix (simplified, no gamma)
def _rgb_to_xyz_simple(Rn, Gn, Bn):
    """Linear RGB [0,1] → XYZ using simplified matrix."""
    X = 0.4887180 * Rn + 0.3106803 * Gn + 0.2006017 * Bn
    Y = 0.1762044 * Rn + 0.8129847 * Gn + 0.0108109 * Bn
    Z = 0.0000000 * Rn + 0.0102048 * Gn + 0.9897952 * Bn
    return X, Y, Z


def rgb_to_lab(R, G, B):
    """RGB → CIELAB.  Returns L* (0-100), a*, b*."""
    Rn = R.astype(np.float32) / 255.0
    Gn = G.astype(np.float32) / 255.0
    Bn = B.astype(np.float32) / 255.0

    X, Y, Z = _rgb_to_xyz_simple(Rn, Gn, Bn)

    Xr, Yr, Zr = 1.0, 1.0, 1.0
    eta = 0.008856
    kappa = 903.3 / 116.0
    add = 16.0 / 116.0
    onethird = 1.0 / 3.0

    xr = X / Xr
    yr = Y / Yr
    zr = Z / Zr

    fx = np.where(xr > eta, xr ** onethird, kappa * xr + add)
    fy = np.where(yr > eta, yr ** onethird, kappa * yr + add)
    fz = np.where(zr > eta, zr ** onethird, kappa * zr + add)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    return L.astype(np.float32), a.astype(np.float32), b.astype(np.float32)


def rgb_to_dlab(R, G, B):
    """RGB → CIELAB + DIN99.  Returns L*, a*, b*, L99, a99, b99."""
    L_lab, a_lab, b_lab = rgb_to_lab(R, G, B)

    L99 = 105.51 * np.log1p(0.0158 * L_lab)

    angle = np.deg2rad(16.0)
    e = a_lab * np.cos(angle) + b_lab * np.sin(angle)
    f = -a_lab * np.sin(angle) + b_lab * np.cos(angle)

    G_val = np.sqrt(e ** 2 + f ** 2)
    k_val = np.where(G_val != 0.0,
                     np.log1p(0.045 * G_val) / 0.045, 0.0)

    a99 = np.where(G_val != 0.0, k_val * (e / G_val), 0.0)
    b99 = np.where(G_val != 0.0, k_val * (f / G_val), 0.0)

    return (L_lab.astype(np.float32), a_lab.astype(np.float32),
            b_lab.astype(np.float32), L99.astype(np.float32),
            a99.astype(np.float32), b99.astype(np.float32))


def rgb_to_oklab(R, G, B):
    """RGB → Oklab.  Returns L, a, b."""
    Rn = R.astype(np.float32) / 255.0
    Gn = G.astype(np.float32) / 255.0
    Bn = B.astype(np.float32) / 255.0

    X = 0.4122214708 * Rn + 0.5363325363 * Gn + 0.0514459929 * Bn
    Y = 0.2119034982 * Rn + 0.6806995451 * Gn + 0.1073969566 * Bn
    Z = 0.0883024619 * Rn + 0.2817188376 * Gn + 0.6299787005 * Bn

    l = 0.8189330101 * X + 0.3618667424 * Y - 0.1288597137 * Z
    m = 0.0329845436 * X + 0.9293118715 * Y + 0.0361456387 * Z
    s = 0.0482003018 * X + 0.2643662691 * Y + 0.6338517070 * Z

    l_c = np.where(l >= 0, l ** (1 / 3), -(-l) ** (1 / 3))
    m_c = np.where(m >= 0, m ** (1 / 3), -(-m) ** (1 / 3))
    s_c = np.where(s >= 0, s ** (1 / 3), -(-s) ** (1 / 3))

    L = 0.2104542553 * l_c + 0.7936177850 * m_c - 0.0040720468 * s_c
    a = 1.9779984951 * l_c - 2.4285922050 * m_c + 0.4505937099 * s_c
    b = 0.0259040371 * l_c + 0.7827717662 * m_c - 0.8086757660 * s_c

    return L.astype(np.float32), a.astype(np.float32), b.astype(np.float32)


def rgb_to_luv(R, G, B):
    """RGB → CIELUV.  Returns L*, u*, v*."""
    Rn = R.astype(np.float32) / 255.0
    Gn = G.astype(np.float32) / 255.0
    Bn = B.astype(np.float32) / 255.0

    X, Y, Z = _rgb_to_xyz_simple(Rn, Gn, Bn)

    Xr, Yr, Zr = 1.0, 1.0, 1.0
    epsilon = 0.008856
    kappa = 903.3

    yr = Y / Yr
    L = np.where(yr > epsilon, 116.0 * np.cbrt(yr) - 16.0, kappa * yr)

    denom = X + 15.0 * Y + 3.0 * Z
    denom_ref = Xr + 15.0 * Yr + 3.0 * Zr
    u_prime = np.where(denom != 0,
                       4.0 * X / denom, 4.0 * Xr / denom_ref)
    v_prime = np.where(denom != 0,
                       9.0 * Y / denom, 9.0 * Yr / denom_ref)

    u_prime_r = 4.0 * Xr / denom_ref
    v_prime_r = 9.0 * Yr / denom_ref

    u_val = 13.0 * L * (u_prime - u_prime_r)
    v_val = 13.0 * L * (v_prime - v_prime_r)

    return L.astype(np.float32), u_val.astype(np.float32), v_val.astype(np.float32)


# ---------------------------------------------------------------------------
# Cylindrical (LCH) spaces
# ---------------------------------------------------------------------------

def rgb_to_lchab(R, G, B):
    """RGB → LCH(ab).  Returns L*, C, H (0-360°)."""
    L, a, b = rgb_to_lab(R, G, B)
    C = np.sqrt(a ** 2 + b ** 2)
    H_rad = np.arctan2(b, a)
    H_deg = np.degrees(H_rad)
    H_deg = np.where(H_deg < 0, H_deg + 360.0, H_deg)
    return L.astype(np.float32), C.astype(np.float32), H_deg.astype(np.float32)


def rgb_to_lchuv(R, G, B):
    """RGB → LCH(uv).  Returns L*, C, H (0-360°)."""
    L, u, v = rgb_to_luv(R, G, B)
    C = np.sqrt(u ** 2 + v ** 2)
    H_rad = np.arctan2(v, u)
    H_deg = np.degrees(H_rad)
    H_deg = np.where(H_deg < 0, H_deg + 360.0, H_deg)
    return L.astype(np.float32), C.astype(np.float32), H_deg.astype(np.float32)


# ---------------------------------------------------------------------------
# Other spaces
# ---------------------------------------------------------------------------

def rgb_to_xyY(R, G, B):
    """RGB → CIE xyY.  Returns x, y (chromaticity), Y (luminance)."""
    Rn = R.astype(np.float32) / 255.0
    Gn = G.astype(np.float32) / 255.0
    Bn = B.astype(np.float32) / 255.0

    X, Y, Z = _rgb_to_xyz_simple(Rn, Gn, Bn)

    sum_xyz = X + Y + Z
    mask = sum_xyz == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(mask, 1.0 / 3.0, X / sum_xyz)
        y = np.where(mask, 1.0 / 3.0, Y / sum_xyz)

    return x.astype(np.float32), y.astype(np.float32), Y.astype(np.float32)


def rgb_to_jch(R, G, B):
    """RGB → simplified JCH (CIECAM02-like).  Returns J, C, H (0-360°)."""
    Rn = R.astype(np.float32) / 255.0
    Gn = G.astype(np.float32) / 255.0
    Bn = B.astype(np.float32) / 255.0

    X = 0.4124564 * Rn + 0.3575761 * Gn + 0.1804375 * Bn
    Y = 0.2126729 * Rn + 0.7151522 * Gn + 0.0721750 * Bn

    J = Y * 100.0

    maxRGB = np.maximum(np.maximum(Rn, Gn), Bn)
    minRGB = np.minimum(np.minimum(Rn, Gn), Bn)
    C = (maxRGB - minRGB) * 100.0

    H = np.zeros_like(maxRGB)
    delta = maxRGB - minRGB
    mask = delta != 0
    mr = (maxRGB == Rn) & mask
    mg = (maxRGB == Gn) & mask
    mb = (maxRGB == Bn) & mask

    H[mr] = (60.0 * ((Gn - Bn) / delta))[mr]
    H[mg] = (60.0 * (((Bn - Rn) / delta) + 2.0))[mg]
    H[mb] = (60.0 * (((Rn - Gn) / delta) + 4.0))[mb]
    H = np.where(H < 0, H + 360.0, H)
    H = H * 0.9

    return J.astype(np.float32), C.astype(np.float32), H.astype(np.float32)


def rgb_to_ycbcr(R, G, B):
    """RGB → YCbCr (BT.601 full-range).  Returns Y (16-235), Cb, Cr (16-240)."""
    Rn = R.astype(np.float32) / 255.0
    Gn = G.astype(np.float32) / 255.0
    Bn = B.astype(np.float32) / 255.0

    r_val = Rn * 255.0
    g_val = Gn * 255.0
    b_val = Bn * 255.0

    Y = 16.0 + (65.481 * r_val + 128.553 * g_val + 24.966 * b_val) / 255.0
    Cb = 128.0 + (-37.797 * r_val - 74.203 * g_val + 112.0 * b_val) / 255.0
    Cr = 128.0 + (112.0 * r_val - 93.786 * g_val - 18.214 * b_val) / 255.0

    return Y.astype(np.float32), Cb.astype(np.float32), Cr.astype(np.float32)


def rgb_to_jzczhz(Jz, az, bz):
    """Jzazbz → JzCzHz (cylindrical).  Returns Jz, Cz, hz (0-360°)."""
    Cz = np.sqrt(az ** 2 + bz ** 2)
    max_chroma = np.nanmax(Cz)
    Cz = np.clip(Cz, 0, max_chroma)

    hz_rad = np.arctan2(bz, az)
    hz_deg = np.degrees(hz_rad)
    hz_deg = np.where(hz_deg < 0, hz_deg + 360.0, hz_deg)
    hz_deg = np.mod(hz_deg, 360.0)

    return Jz.astype(np.float32), Cz.astype(np.float32), hz_deg.astype(np.float32)


# ---------------------------------------------------------------------------
# Inverse conversions
# ---------------------------------------------------------------------------

def lab_to_rgb(L, a, b):
    """CIELAB → linear RGB [0, 1].  Returns R, G, B as float32.

    Uses the inverse of the simplified matrix used in ``rgb_to_lab``.
    Applies sRGB companding (gamma ≈ 2.4).
    """
    epsilon = 0.008856
    kappa = 903.3
    Xr, Yr, Zr = 1.0, 1.0, 1.0

    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    X = np.where(fx ** 3 > epsilon, fx ** 3,
                 (116.0 * fx - 16.0) / kappa)
    Y = np.where(L > kappa * epsilon, fy ** 3, L / kappa)
    Z = np.where(fz ** 3 > epsilon, fz ** 3,
                 (116.0 * fz - 16.0) / kappa)

    X *= Xr
    Y *= Yr
    Z *= Zr

    # Inverse of simplified matrix
    r_lin = 2.373 * X - 0.903 * Y - 0.471 * Z
    g_lin = -0.515 * X + 1.428 * Y + 0.0888 * Z
    b_lin = 0.00531 * X - 0.01475 * Y + 1.0122 * Z

    # sRGB companding
    R_out = np.where(r_lin <= 0.00313081,
                     12.92 * r_lin,
                     1.055 * np.power(np.clip(r_lin, 0, None), 1.0 / 2.4) - 0.055)
    G_out = np.where(g_lin <= 0.00313081,
                     12.92 * g_lin,
                     1.055 * np.power(np.clip(g_lin, 0, None), 1.0 / 2.4) - 0.055)
    B_out = np.where(b_lin <= 0.00313081,
                     12.92 * b_lin,
                     1.055 * np.power(np.clip(b_lin, 0, None), 1.0 / 2.4) - 0.055)

    return R_out.astype(np.float32), G_out.astype(np.float32), B_out.astype(np.float32)


# ---------------------------------------------------------------------------
# Registry & dispatcher
# ---------------------------------------------------------------------------

_CONVERSIONS = {
    "dlab":   (rgb_to_dlab,   ["L", "a", "b", "L99", "a99", "b99"]),
    "hsl":    (rgb_to_hsl,    ["H", "S", "L"]),
    "hsi":    (rgb_to_hsi,    ["H", "S", "I"]),
    "hsv":    (rgb_to_hsv,    ["H", "S", "V"]),
    "jch":    (rgb_to_jch,    ["J", "C", "H"]),
    "jzczhz": (rgb_to_jzczhz, ["Jz", "Cz", "hz"]),
    "lab":    (rgb_to_lab,    ["L", "a", "b"]),
    "lchab":  (rgb_to_lchab,  ["L", "C", "Hab"]),
    "lchuv":  (rgb_to_lchuv,  ["L", "C", "Huv"]),
    "luv":    (rgb_to_luv,    ["L", "u", "v"]),
    "oklab":  (rgb_to_oklab,  ["L", "a", "b"]),
    "xyY":    (rgb_to_xyY,    ["x", "y_ch", "Y_lum"]),
    "ycbcr":  (rgb_to_ycbcr,  ["Y", "Cb", "Cr"]),
}


def available_spaces():
    """Return sorted list of supported color space names."""
    return sorted(_CONVERSIONS.keys())


def convertbands(R, G, B, space):
    """Convert RGB bands to the chosen color space.

    Parameters
    ----------
    R, G, B : numpy.ndarray
        2-D arrays (uint8 or float).  Each function normalises internally.
    space : str
        Target color space (see ``available_spaces()``).

    Returns
    -------
    comps : tuple of numpy.ndarray
        Component arrays (float32).
    names : list of str
        Component names.
    """
    try:
        func, names = _CONVERSIONS[space]
    except KeyError:
        raise ValueError(
            f"Unknown space '{space}'. Available: {available_spaces()}"
        )
    comps = func(R, G, B)
    return comps, names
