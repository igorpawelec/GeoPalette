"""
geopalette.conversions
~~~~~~~~~~~~~~~~~~~~~~
RGB ↔ color-space conversion functions for 2-D NumPy arrays (raster bands).

Every forward function has the signature::

    f(R, G, B) → tuple[np.ndarray, ...]

where R, G, B are 2-D uint8 arrays (0-255). Internally they are normalised
to [0, 1] float32 and linearized (inverse sRGB companding) where required.

Perceptual spaces (Lab, Luv, Oklab, LCH, xyY, Jzazbz) use sRGB-linearized
input. Hue-based spaces (HSL, HSV, HSI) and YCbCr work on gamma-encoded
values directly, as per their definitions.
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# sRGB linearization
# ═══════════════════════════════════════════════════════════════════════

def _srgb_to_linear(c):
    """Inverse sRGB companding: gamma-encoded [0,1] → linear [0,1]."""
    c = c.astype(np.float64)
    return np.where(c <= 0.04045,
                    c / 12.92,
                    ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _linear_to_srgb(c):
    """sRGB companding: linear [0,1] → gamma-encoded [0,1]."""
    c = np.clip(c, 0.0, None).astype(np.float64)
    return np.where(c <= 0.0031308,
                    12.92 * c,
                    1.055 * c ** (1.0 / 2.4) - 0.055).astype(np.float32)


def _safe_cbrt(x):
    """Cube root that handles negative values without RuntimeWarning."""
    return np.sign(x) * np.abs(x).astype(np.float64) ** (1.0 / 3.0)


# ═══════════════════════════════════════════════════════════════════════
# RGB → XYZ (sRGB D65 standard)
# ═══════════════════════════════════════════════════════════════════════

def _linear_rgb_to_xyz(R_lin, G_lin, B_lin):
    """Linear RGB [0,1] → CIE XYZ (D65 illuminant, sRGB primaries)."""
    X = 0.4124564 * R_lin + 0.3575761 * G_lin + 0.1804375 * B_lin
    Y = 0.2126729 * R_lin + 0.7151522 * G_lin + 0.0721750 * B_lin
    Z = 0.0193339 * R_lin + 0.1191920 * G_lin + 0.9503041 * B_lin
    return X, Y, Z


def _rgb_to_linear(R, G, B):
    """uint8 RGB → linearized float32 RGB."""
    return (_srgb_to_linear(R.astype(np.float32) / 255.0),
            _srgb_to_linear(G.astype(np.float32) / 255.0),
            _srgb_to_linear(B.astype(np.float32) / 255.0))


def _rgb_to_xyz(R, G, B):
    """uint8 sRGB → CIE XYZ (D65). Full pipeline with linearization."""
    R_lin, G_lin, B_lin = _rgb_to_linear(R, G, B)
    return _linear_rgb_to_xyz(R_lin, G_lin, B_lin)


# D65 white point
_Xn, _Yn, _Zn = 0.95047, 1.00000, 1.08883


# ═══════════════════════════════════════════════════════════════════════
# Hue-based spaces (operate on gamma-encoded values)
# ═══════════════════════════════════════════════════════════════════════

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
    denom = 1.0 - np.abs(2.0 * L - 1.0)
    denom = np.where(denom == 0, 1.0, denom)
    S[non_zero] = delta[non_zero] / denom[non_zero]

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


# ═══════════════════════════════════════════════════════════════════════
# CIE-based spaces (require linearized RGB → XYZ)
# ═══════════════════════════════════════════════════════════════════════

def rgb_to_lab(R, G, B):
    """RGB → CIELAB (D65).  Returns L* (0-100), a*, b*."""
    X, Y, Z = _rgb_to_xyz(R, G, B)

    epsilon = 0.008856
    kappa = 903.3

    xr = X / _Xn
    yr = Y / _Yn
    zr = Z / _Zn

    fx = np.where(xr > epsilon, _safe_cbrt(xr), (kappa * xr + 16.0) / 116.0)
    fy = np.where(yr > epsilon, _safe_cbrt(yr), (kappa * yr + 16.0) / 116.0)
    fz = np.where(zr > epsilon, _safe_cbrt(zr), (kappa * zr + 16.0) / 116.0)

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
    """RGB → Oklab.  Returns L, a, b.

    Uses proper sRGB linearization before Oklab transform.
    """
    R_lin, G_lin, B_lin = _rgb_to_linear(R, G, B)

    # sRGB linear → LMS (Oklab-specific matrix)
    l = 0.4122214708 * R_lin + 0.5363325363 * G_lin + 0.0514459929 * B_lin
    m = 0.2119034982 * R_lin + 0.6806995451 * G_lin + 0.1073969566 * B_lin
    s = 0.0883024619 * R_lin + 0.2817188376 * G_lin + 0.6299787005 * B_lin

    l_c = _safe_cbrt(l)
    m_c = _safe_cbrt(m)
    s_c = _safe_cbrt(s)

    L = 0.2104542553 * l_c + 0.7936177850 * m_c - 0.0040720468 * s_c
    a = 1.9779984951 * l_c - 2.4285922050 * m_c + 0.4505937099 * s_c
    b = 0.0259040371 * l_c + 0.7827717662 * m_c - 0.8086757660 * s_c

    return L.astype(np.float32), a.astype(np.float32), b.astype(np.float32)


def rgb_to_luv(R, G, B):
    """RGB → CIELUV (D65).  Returns L*, u*, v*."""
    X, Y, Z = _rgb_to_xyz(R, G, B)

    epsilon = 0.008856
    kappa = 903.3

    yr = Y / _Yn
    L = np.where(yr > epsilon, 116.0 * _safe_cbrt(yr) - 16.0, kappa * yr)

    denom = X + 15.0 * Y + 3.0 * Z
    denom_ref = _Xn + 15.0 * _Yn + 3.0 * _Zn
    u_prime = np.where(denom != 0,
                       4.0 * X / denom, 4.0 * _Xn / denom_ref)
    v_prime = np.where(denom != 0,
                       9.0 * Y / denom, 9.0 * _Yn / denom_ref)

    u_prime_r = 4.0 * _Xn / denom_ref
    v_prime_r = 9.0 * _Yn / denom_ref

    u_val = 13.0 * L * (u_prime - u_prime_r)
    v_val = 13.0 * L * (v_prime - v_prime_r)

    return L.astype(np.float32), u_val.astype(np.float32), v_val.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# Cylindrical (LCH) spaces
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# Other spaces
# ═══════════════════════════════════════════════════════════════════════

def rgb_to_xyY(R, G, B):
    """RGB → CIE xyY.  Returns x, y (chromaticity), Y (luminance)."""
    X, Y, Z = _rgb_to_xyz(R, G, B)

    sum_xyz = X + Y + Z
    mask = sum_xyz == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(mask, 1.0 / 3.0, X / sum_xyz)
        y = np.where(mask, 1.0 / 3.0, Y / sum_xyz)

    return x.astype(np.float32), y.astype(np.float32), Y.astype(np.float32)


def rgb_to_jch(R, G, B):
    """RGB → simplified JCH (CIECAM02-like).  Returns J, C, H (0-360°)."""
    X, Y, Z = _rgb_to_xyz(R, G, B)

    J = Y * 100.0

    Rn = R.astype(np.float32) / 255.0
    Gn = G.astype(np.float32) / 255.0
    Bn = B.astype(np.float32) / 255.0

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
    Rf = R.astype(np.float32)
    Gf = G.astype(np.float32)
    Bf = B.astype(np.float32)

    Y = 16.0 + (65.481 * Rf + 128.553 * Gf + 24.966 * Bf) / 255.0
    Cb = 128.0 + (-37.797 * Rf - 74.203 * Gf + 112.0 * Bf) / 255.0
    Cr = 128.0 + (112.0 * Rf - 93.786 * Gf - 18.214 * Bf) / 255.0

    return Y.astype(np.float32), Cb.astype(np.float32), Cr.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# Jzazbz / JzCzHz (Perceptual Quantizer based, HDR-ready)
# ═══════════════════════════════════════════════════════════════════════

def rgb_to_jzazbz(R, G, B):
    """RGB → Jzazbz.  Returns Jz, az, bz.

    Based on Safdar et al. (2017) "Perceptually uniform color space
    for image signals including high dynamic range and wide gamut".
    """
    X, Y, Z = _rgb_to_xyz(R, G, B)
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    Z = Z.astype(np.float64)

    # Absolute luminance (assume SDR peak ~203 cd/m²)
    X_abs = X * 203.0
    Y_abs = Y * 203.0
    Z_abs = Z * 203.0

    # XYZ → LMS (modified Hunt-Pointer-Estevez)
    Lp = 0.41478972 * X_abs + 0.579999 * Y_abs + 0.01464800 * Z_abs
    Mp = -0.20151000 * X_abs + 1.120649 * Y_abs + 0.05310080 * Z_abs
    Sp = -0.01660080 * X_abs + 0.264800 * Y_abs + 0.66847990 * Z_abs

    # PQ transfer function (Perceptual Quantizer)
    c1 = 3424.0 / 4096.0
    c2 = 2413.0 / 128.0
    c3 = 2392.0 / 128.0
    n = 2610.0 / 16384.0
    p = 1.7 * 2523.0 / 32.0

    def _pq(x):
        x = np.clip(x / 10000.0, 0.0, None)
        xn = x ** n
        return ((c1 + c2 * xn) / (1.0 + c3 * xn)) ** p

    Lp_pq = _pq(np.abs(Lp))
    Mp_pq = _pq(np.abs(Mp))
    Sp_pq = _pq(np.abs(Sp))

    # Izazbz
    Iz = 0.5 * Lp_pq + 0.5 * Mp_pq
    az = 3.524000 * Lp_pq - 4.066708 * Mp_pq + 0.542708 * Sp_pq
    bz = 0.199076 * Lp_pq + 1.096799 * Mp_pq - 1.295875 * Sp_pq

    # Jz
    d0 = 1.6295499532821566e-11
    Jz = (1.0 + d0) * Iz / (1.0 + d0 * Iz) - d0

    return Jz.astype(np.float32), az.astype(np.float32), bz.astype(np.float32)


def rgb_to_jzczhz(R, G, B):
    """RGB → JzCzHz (cylindrical Jzazbz).  Returns Jz, Cz, hz (0-360°)."""
    Jz, az, bz = rgb_to_jzazbz(R, G, B)
    Cz = np.sqrt(az ** 2 + bz ** 2)
    hz = np.degrees(np.arctan2(bz, az))
    hz = np.where(hz < 0, hz + 360.0, hz)
    return Jz.astype(np.float32), Cz.astype(np.float32), hz.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# Inverse conversions
# ═══════════════════════════════════════════════════════════════════════

def lab_to_rgb(L, a, b):
    """CIELAB → sRGB [0, 1].  Returns R, G, B as float32."""
    epsilon = 0.008856
    kappa = 903.3

    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    X = np.where(fx ** 3 > epsilon, fx ** 3,
                 (116.0 * fx - 16.0) / kappa) * _Xn
    Y = np.where(L > kappa * epsilon, fy ** 3, L / kappa) * _Yn
    Z = np.where(fz ** 3 > epsilon, fz ** 3,
                 (116.0 * fz - 16.0) / kappa) * _Zn

    # XYZ → linear sRGB (inverse of D65 matrix)
    r_lin =  3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    g_lin = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    b_lin =  0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z

    R_out = _linear_to_srgb(r_lin)
    G_out = _linear_to_srgb(g_lin)
    B_out = _linear_to_srgb(b_lin)

    return R_out.astype(np.float32), G_out.astype(np.float32), B_out.astype(np.float32)


def oklab_to_rgb(L, a, b):
    """Oklab → sRGB [0, 1].  Returns R, G, B as float32."""
    l_c = L + 0.3963377774 * a + 0.2158037573 * b
    m_c = L - 0.1055613458 * a - 0.0638541728 * b
    s_c = L - 0.0894841775 * a - 1.2914855480 * b

    l = l_c ** 3
    m = m_c ** 3
    s = s_c ** 3

    R_lin = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    G_lin = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    B_lin = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    return (_linear_to_srgb(R_lin).astype(np.float32),
            _linear_to_srgb(G_lin).astype(np.float32),
            _linear_to_srgb(B_lin).astype(np.float32))


def hsv_to_rgb(H, S, V):
    """HSV → RGB [0, 1].  H in [0, 360], S and V in [0, 1].  Returns R, G, B float32."""
    H = H.astype(np.float32)
    S = S.astype(np.float32)
    V = V.astype(np.float32)

    C = V * S
    Hp = H / 60.0
    X = C * (1.0 - np.abs(np.mod(Hp, 2.0) - 1.0))
    m = V - C

    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)

    mask0 = (Hp >= 0) & (Hp < 1); R[mask0] = C[mask0]; G[mask0] = X[mask0]
    mask1 = (Hp >= 1) & (Hp < 2); R[mask1] = X[mask1]; G[mask1] = C[mask1]
    mask2 = (Hp >= 2) & (Hp < 3); G[mask2] = C[mask2]; B[mask2] = X[mask2]
    mask3 = (Hp >= 3) & (Hp < 4); G[mask3] = X[mask3]; B[mask3] = C[mask3]
    mask4 = (Hp >= 4) & (Hp < 5); R[mask4] = X[mask4]; B[mask4] = C[mask4]
    mask5 = (Hp >= 5) & (Hp < 6); R[mask5] = C[mask5]; B[mask5] = X[mask5]

    return ((R + m).astype(np.float32),
            (G + m).astype(np.float32),
            (B + m).astype(np.float32))


def hsl_to_rgb(H, S, L):
    """HSL → RGB [0, 1].  H in [0, 360], S and L in [0, 1].  Returns R, G, B float32."""
    H = H.astype(np.float32)
    S = S.astype(np.float32)
    L = L.astype(np.float32)

    C = (1.0 - np.abs(2.0 * L - 1.0)) * S
    Hp = H / 60.0
    X = C * (1.0 - np.abs(np.mod(Hp, 2.0) - 1.0))
    m = L - C / 2.0

    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)

    mask0 = (Hp >= 0) & (Hp < 1); R[mask0] = C[mask0]; G[mask0] = X[mask0]
    mask1 = (Hp >= 1) & (Hp < 2); R[mask1] = X[mask1]; G[mask1] = C[mask1]
    mask2 = (Hp >= 2) & (Hp < 3); G[mask2] = C[mask2]; B[mask2] = X[mask2]
    mask3 = (Hp >= 3) & (Hp < 4); G[mask3] = X[mask3]; B[mask3] = C[mask3]
    mask4 = (Hp >= 4) & (Hp < 5); R[mask4] = X[mask4]; B[mask4] = C[mask4]
    mask5 = (Hp >= 5) & (Hp < 6); R[mask5] = C[mask5]; B[mask5] = X[mask5]

    return ((R + m).astype(np.float32),
            (G + m).astype(np.float32),
            (B + m).astype(np.float32))


# ═══════════════════════════════════════════════════════════════════════
# Registry & dispatcher
# ═══════════════════════════════════════════════════════════════════════

_CONVERSIONS = {
    "dlab":   (rgb_to_dlab,   ["L", "a", "b", "L99", "a99", "b99"]),
    "hsl":    (rgb_to_hsl,    ["H", "S", "L"]),
    "hsi":    (rgb_to_hsi,    ["H", "S", "I"]),
    "hsv":    (rgb_to_hsv,    ["H", "S", "V"]),
    "jch":    (rgb_to_jch,    ["J", "C", "H"]),
    "jzazbz": (rgb_to_jzazbz, ["Jz", "az", "bz"]),
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
