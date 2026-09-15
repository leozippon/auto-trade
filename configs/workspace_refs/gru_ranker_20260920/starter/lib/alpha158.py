"""Alpha158 feature block (158 operators), verbatim from the graduated github_confirm artifact's lib/features.py.

Kept byte-for-byte below this docstring so the reported control `c_lgbm` is the
graduate's feature code, not a re-implementation. Layout: every series is an
(S, T) wide matrix (S symbols x T bars, trade_date ascending); outputs are (S, T)
matrices where out[:, b] uses bars b-d+1..b. Bars without enough history are
NaN; the per-date cross-section fill is done by robust_zscore (missing -> 0).
Column order ALPHA158: 9 K-line + 4 position + 29 rolling operators x 5 windows.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

EPS = 1e-12
WINDOWS = (5, 10, 20, 30, 60)
KLINE = ("KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2")
POS = ("OPEN0", "HIGH0", "LOW0", "VWAP0")
ROLLING = ("ROC", "MA", "STD", "BETA", "RSQR", "RESI", "MAX", "MIN", "QTLU", "QTLD",
           "RANK", "RSV", "IMAX", "IMIN", "IMXD", "CORR", "CORD", "CNTP", "CNTN",
           "CNTD", "SUMP", "SUMN", "SUMD", "VMA", "VSTD", "WVMA", "VSUMP", "VSUMN", "VSUMD")
ALPHA158 = list(KLINE) + list(POS) + ["%s_%d" % (n, d) for d in WINDOWS for n in ROLLING]

_ZSCORE_BATCH = 16  # rows per nanmedian batch; bounds temp memory during fit


def _win(a, d):
    """(S, T) -> strided view (S, T-d+1, d), window over the date axis."""
    return sliding_window_view(a, d, axis=1)


def _fin(win):
    """(S, T-d+1, d) -> (S, T-d+1) bool: the window is complete (every value finite).
    Operators whose kernels suppress NaN (argmax/argmin, partition, comparisons) must be
    masked with this so an incomplete window yields NaN, matching qlib's rolling semantics.
    Sum/max-based kernels propagate NaN on their own and need no mask."""
    return np.isfinite(win).all(axis=2)


def _shift(x):
    """(S, T) -> (S, T) with y[:, t] = x[:, t-1]; first column NaN."""
    y = np.empty_like(x)
    y[:, 0] = np.nan
    y[:, 1:] = x[:, :-1]
    return y


def _quantile(win, q, d):
    """Linear-interpolation quantile (pandas convention pos = q*(d-1)) along the last axis."""
    pos = q * (d - 1)
    lo = int(np.floor(pos))
    frac = pos - lo
    if frac <= 0.0:
        return np.partition(win, lo, axis=-1)[..., lo]
    p = np.partition(win, [lo, lo + 1], axis=-1)
    return p[..., lo] + frac * (p[..., lo + 1] - p[..., lo])


def _pearson(sx, sy, sxx, syy, sxy, d):
    """Rolling Pearson r from rolling sums (denominator <= 0 -> NaN)."""
    num = d * sxy - sx * sy
    den = np.sqrt(np.maximum(d * sxx - sx * sx, 0.0) * np.maximum(d * syy - sy * sy, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    return np.where(den > 0.0, r, np.nan)


def compute_158(W):
    """Build all 158 operator matrices from wide inputs.

    W must contain (S, T) float64 arrays O, H, L, C (qfq prices), V (qfq volume),
    VWAP (= amount/vol, raw; the qfq factor cancels in the ratio). Returns dict name -> (S, T).
    """
    O, H, L, C = W["O"], W["H"], W["L"], W["C"]
    V, VW = W["V"], W["VWAP"]
    S, T = C.shape
    out = {name: np.full((S, T), np.nan) for name in ALPHA158}

    with np.errstate(divide="ignore", invalid="ignore"):
        HL = H - L
        out["KMID"] = (C - O) / O
        out["KLEN"] = HL / O
        out["KMID2"] = (C - O) / (HL + EPS)
        oc_hi = np.maximum(O, C)
        oc_lo = np.minimum(O, C)
        out["KUP"] = (H - oc_hi) / O
        out["KUP2"] = (H - oc_hi) / (HL + EPS)
        out["KLOW"] = (oc_lo - L) / O
        out["KLOW2"] = (oc_lo - L) / (HL + EPS)
        out["KSFT"] = (2.0 * C - H - L) / O
        out["KSFT2"] = (2.0 * C - H - L) / (HL + EPS)
        out["OPEN0"] = O / C
        out["HIGH0"] = H / C
        out["LOW0"] = L / C
        out["VWAP0"] = VW / C

    # Precomputed shifted series (first column NaN where the predecessor bar is absent).
    Cprev = _shift(C)
    ret1 = np.full_like(C, np.nan)  # C/C.shift(1)
    ret1[:, 1:] = C[:, 1:] / Cprev[:, 1:]
    Vprev = _shift(V)
    vlog = np.full_like(V, np.nan)  # log(V/V.shift(1) + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        vlog[:, 1:] = np.where(Vprev[:, 1:] > 0.0, np.log1p(V[:, 1:] / Vprev[:, 1:]), np.nan)
    dc = np.full_like(C, np.nan)  # C - C.shift(1)
    dc[:, 1:] = C[:, 1:] - Cprev[:, 1:]
    dv = np.full_like(V, np.nan)  # V - V.shift(1)
    dv[:, 1:] = V[:, 1:] - Vprev[:, 1:]
    wser = np.abs(ret1) * V  # WVMA series
    P = np.arange(T, dtype=np.float64)
    logV1 = np.log1p(V)  # CORR series: log(V + 1)

    for d in WINDOWS:
        off = d - 1
        wC = _win(C, d)
        wH = _win(H, d)
        wL = _win(L, d)
        wV = _win(V, d)
        sumC = wC.sum(axis=2)
        sumC2 = (wC * wC).sum(axis=2)
        meanC = sumC / d
        maxH = wH.max(axis=2)
        minL = wL.min(axis=2)
        varC = np.maximum(sumC2 / d - meanC * meanC, 0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            out["MA_%d" % d][:, off:] = meanC / C[:, off:]
            out["STD_%d" % d][:, off:] = np.sqrt(varC) / C[:, off:]

        # Rolling OLS of C on the in-window time index t = 0..d-1 (oldest -> newest).
        # S_tc = sum_t t*y_t = sum(p*y_p) - (b-d+1)*sum(y),  p = absolute bar index,
        # b = window end. slope = (S_tc - d*mean_t*mean_y)/S_tt.
        sumPc = _win(C * P, d).sum(axis=2)
        win_start = np.arange(off, T, dtype=np.float64) - off  # 0..T-d
        S_tc = sumPc - win_start[None, :] * sumC
        mean_t = 0.5 * (d - 1)
        S_tt = d * (d * d - 1) / 12.0
        slope = (S_tc - d * mean_t * meanC) / S_tt
        ss_tot = sumC2 - d * meanC * meanC
        rsqr = np.where(ss_tot > 0.0, slope * slope * S_tt / np.maximum(ss_tot, 1e-300), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["BETA_%d" % d][:, off:] = slope / C[:, off:]
            out["RESI_%d" % d][:, off:] = ((C[:, off:] - meanC) - slope * (d - 1 - mean_t)) / C[:, off:]
        out["RSQR_%d" % d][:, off:] = rsqr

        finC = _fin(wC)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["MAX_%d" % d][:, off:] = maxH / C[:, off:]
            out["MIN_%d" % d][:, off:] = minL / C[:, off:]
            out["RSV_%d" % d][:, off:] = (C[:, off:] - minL) / (maxH - minL + EPS)
            qtu = _quantile(wC, 0.8, d) / C[:, off:]
            qtd = _quantile(wC, 0.2, d) / C[:, off:]
        out["QTLU_%d" % d][:, off:] = np.where(finC, qtu, np.nan)
        out["QTLD_%d" % d][:, off:] = np.where(finC, qtd, np.nan)

        cl = C[:, off:][:, :, None]  # (S, T-d+1, 1) broadcasts over the window axis
        rank = ((wC < cl).sum(axis=2) + 0.5 * (wC == cl).sum(axis=2)) / d
        out["RANK_%d" % d][:, off:] = np.where(finC, rank, np.nan)

        wHf = np.where(np.isfinite(wH), wH, -np.inf)
        wLf = np.where(np.isfinite(wL), wL, np.inf)
        amax = wHf.argmax(axis=2)
        amin = wLf.argmin(axis=2)
        finH = _fin(wH)
        finL = _fin(wL)
        out["IMAX_%d" % d][:, off:] = np.where(finH, amax, np.nan) / d
        out["IMIN_%d" % d][:, off:] = np.where(finL, amin, np.nan) / d
        out["IMXD_%d" % d][:, off:] = np.where(finH & finL, (amax - amin).astype(np.float64), np.nan) / d

        # CORR(C, log(V+1)) and CORD(C/C.shift(1), log(V/V.shift(1)+1))
        wy = _win(logV1, d)
        sy = wy.sum(axis=2)
        syy = (wy * wy).sum(axis=2)
        out["CORR_%d" % d][:, off:] = _pearson(sumC, sy, sumC2, syy, (wC * wy).sum(axis=2), d)
        wx = _win(ret1, d)
        wx2 = _win(vlog, d)
        sx2 = wx.sum(axis=2)
        sy2 = wx2.sum(axis=2)
        out["CORD_%d" % d][:, off:] = _pearson(
            sx2, sy2, (wx * wx).sum(axis=2), (wx2 * wx2).sum(axis=2), (wx * wx2).sum(axis=2), d)

        # CNTP/CNTN count strict up / strict down days; a flat day (dc == 0) is neither,
        # so it contributes 0, while a missing change (dc NaN) makes the window incomplete.
        up = np.where(np.isfinite(dc), (dc > 0.0).astype(np.float64), np.nan)
        dn = np.where(np.isfinite(dc), (dc < 0.0).astype(np.float64), np.nan)
        cntp = _win(up, d).sum(axis=2) / d
        cntn = _win(dn, d).sum(axis=2) / d
        out["CNTP_%d" % d][:, off:] = cntp
        out["CNTN_%d" % d][:, off:] = cntn
        out["CNTD_%d" % d][:, off:] = cntp - cntn

        sumpos = _win(np.maximum(dc, 0.0), d).sum(axis=2)
        sumabs = _win(np.abs(dc), d).sum(axis=2)
        sump = sumpos / (sumabs + EPS)
        sumn = (sumabs - sumpos) / (sumabs + EPS)
        out["SUMP_%d" % d][:, off:] = sump
        out["SUMN_%d" % d][:, off:] = sumn
        out["SUMD_%d" % d][:, off:] = sump - sumn

        sumV = wV.sum(axis=2)
        meanV = sumV / d
        sumV2 = (wV * wV).sum(axis=2)
        varV = np.maximum(sumV2 / d - meanV * meanV, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["VMA_%d" % d][:, off:] = meanV / (V[:, off:] + EPS)
            out["VSTD_%d" % d][:, off:] = np.sqrt(varV) / (V[:, off:] + EPS)
        sw = _win(wser, d)
        sumw = sw.sum(axis=2)
        meanw = sumw / d
        varw = np.maximum((sw * sw).sum(axis=2) / d - meanw * meanw, 0.0)
        out["WVMA_%d" % d][:, off:] = np.sqrt(varw) / (meanw + EPS)

        vpos = np.maximum(dv, 0.0)
        vneg = np.maximum(-dv, 0.0)
        vabs = np.abs(dv)
        svp = _win(vpos, d).sum(axis=2)
        svn = _win(vneg, d).sum(axis=2)
        dva_ = _win(vabs, d).sum(axis=2)
        vsump = svp / (dva_ + EPS)
        vsumn = svn / (dva_ + EPS)
        out["VSUMP_%d" % d][:, off:] = vsump
        out["VSUMN_%d" % d][:, off:] = vsumn
        out["VSUMD_%d" % d][:, off:] = vsump - vsumn

    for d in WINDOWS:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["ROC_%d" % d][:, d:] = C[:, :T - d] / C[:, d:]
    return out


def robust_zscore(X):
    """Cross-sectional RobustZScore per row: (x - median)/(1.4826*MAD), MAD = median|x-median|
    of the same row. MAD == 0 -> 0, missing -> 0, clip to [-3, 3]. Returns float32.

    Accepts (S, F) for one cross-section or (N, S, F) for many rows (fit path).
    Statistics use every symbol present in the row (full listed cross-section).
    All-NaN columns are neutralized to 0 first so np.nanmedian never sees an all-NaN slice.
    """
    X = np.ascontiguousarray(X)
    single = X.ndim == 2
    if single:
        X = X[None, :, :]
    N, S, F = X.shape
    Z = np.zeros((N, S, F), dtype=np.float32)
    for i0 in range(0, N, _ZSCORE_BATCH):
        xb = np.ascontiguousarray(X[i0:i0 + _ZSCORE_BATCH], dtype=np.float64)
        anyf = np.isfinite(xb).any(axis=1)  # (b, F)
        xb2 = np.where(anyf[:, None, :], xb, 0.0)  # broadcast over the symbol axis
        with np.errstate(invalid="ignore"):
            med = np.nanmedian(xb2, axis=1)  # (b, F)
            mad = np.nanmedian(np.abs(xb2 - med[:, None, :]), axis=1)
        den = 1.4826 * mad
        ok = (den > 0.0) & np.isfinite(den)
        zb = np.zeros_like(xb)
        with np.errstate(invalid="ignore", divide="ignore"):
            np.divide(xb2 - med[:, None, :], den[:, None, :], out=zb, where=ok[:, None, :])
        zb = np.clip(zb, -3.0, 3.0)
        zb[~np.isfinite(zb)] = 0.0
        Z[i0:i0 + _ZSCORE_BATCH] = zb.astype(np.float32)
    return Z[0] if single else Z
