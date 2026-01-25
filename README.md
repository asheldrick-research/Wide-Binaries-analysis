# Wide-Binaries-analysis
Empirical analysis of wide binary stars using Gaia astrometry. This repository contains data selection, kinematic diagnostics, and Monte Carlo null tests designed to assess the statistical robustness of observed velocity and acceleration trends, without assuming or fitting a theoretical model.
# Phase-Asymmetric Dynamics in Wide Binary Stars

This repository contains the full analysis code used in the paper:

> *Phase-Asymmetric Dynamics in Wide Binary Stars*  
> (Sheldrick et al., submitted)

The goal of this work is to test for phase-dependent dynamical asymmetries
in wide binary systems using Gaia-based catalogs, under strict selection-aware
null tests.

---

## Repository structure

---

## Data availability

Due to size and licensing constraints, Gaia and wide-binary FITS catalogs
are **not included** in this repository.

The analysis uses publicly available data products.  
Instructions for obtaining the required input catalogs are provided in:

All notebooks assume the user places the relevant FITS files in a local
runtime or mounted drive and will automatically search common paths.

---

## Reproducibility

All notebooks required to reproduce the figures and numerical results
in the paper are contained in the `notebooks/` directory.

Each notebook:
- loads raw catalog data,
- applies all selection cuts explicitly,
- computes the reported statistics,
- and produces the figures shown in the manuscript.

No manual intervention is required beyond specifying the FITS file location.

---

## Exploratory analyses

Additional notebooks in `notebooks/exploratory/` explore alternative
phase proxies and diagnostics. These are provided for transparency but are
**not used to support claims in the paper**.

---

## Requirements

The analysis was run in Google Colab using Python ≥3.10.

Key dependencies:
- numpy
- scipy
- astropy
- matplotlib

See `requirements.txt` for a complete list.

---

## License

This repository is released for academic transparency and reproducibility.
Please cite the accompanying paper if you use this code.

## Colab cells below

## Step 1A

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# ============================================================
# Wide binaries: escape-speed diagnostic (eta=v/v_esc)
# SELF-CONTAINED: reads FITS and builds aligned arrays fresh.
# Includes global-shuffle and stratified-shuffle nulls.
# ============================================================

FITS_PATH = "/content/widebin/all_columns_catalog_shift.fits.gz"  # edit if needed

# --- constants ---
G    = 6.67430e-11
MSUN = 1.98847e30
AU_M = 1.495978707e11

# --- selection knobs ---
MAX_SEP_AU = 5e4
RMAX       = 0.10

# --- binning ---
NBINS      = 12
MINN       = 60

# --- null controls ---
NTRIAL      = 200
SEED        = 123
NSEP_BINS   = 12
NPLX_BINS   = 10
MINN_STRATA = 20

# -----------------------------
# helpers
# -----------------------------
def km_s_from_pm_parallax(pm_masyr, plx_mas):
    # v_t[km/s] = 4.74047 * mu[mas/yr] / plx[mas]
    return 4.74047 * (pm_masyr / np.maximum(plx_mas, 1e-12))

def abs_MG(Gmag, plx_mas):
    # MG = G + 5*log10(plx/1000) + 5  (plx in mas)
    return Gmag + 5.0*np.log10(np.maximum(plx_mas, 1e-6)/1000.0) + 5.0

MG_anc = np.array([3.5, 5.0, 7.0, 9.5, 11.5])
M_anc  = np.array([1.6, 1.1, 0.75, 0.35, 0.20])  # Msun

def mass_from_MG(MG):
    MGc = np.clip(MG, MG_anc.min(), MG_anc.max())
    return np.interp(MGc, MG_anc, M_anc)

def binned_quantiles(x, y, bins, qs=(0.10, 0.50, 0.90), minN=60):
    idx = np.digitize(x, bins) - 1
    xc, N = [], []
    Q = {q: [] for q in qs}
    for i in range(len(bins) - 1):
        sel = (idx == i)
        if np.count_nonzero(sel) < minN:
            continue
        xc.append(np.sqrt(bins[i] * bins[i+1]))
        N.append(np.count_nonzero(sel))
        yy = y[sel]
        for q in qs:
            Q[q].append(np.quantile(yy, q))
    xc = np.asarray(xc)
    N  = np.asarray(N)
    for q in qs:
        Q[q] = np.asarray(Q[q])
    return xc, N, Q

def stratified_shuffle(arr, sep_AU, plx_mas, rng,
                       nsep_bins=12, nplx_bins=10, minn=20):
    arr = np.asarray(arr)
    out = arr.copy()

    ls = np.log10(np.maximum(sep_AU, 1e-30))
    lp = np.log10(np.maximum(plx_mas, 1e-30))

    sep_edges = np.quantile(ls, np.linspace(0, 1, nsep_bins + 1))
    plx_edges = np.quantile(lp, np.linspace(0, 1, nplx_bins + 1))

    isep = np.clip(np.digitize(ls, sep_edges) - 1, 0, nsep_bins - 1)
    iplx = np.clip(np.digitize(lp, plx_edges) - 1, 0, nplx_bins - 1)
    sid  = isep * nplx_bins + iplx

    for s in np.unique(sid):
        ii = np.where(sid == s)[0]
        if ii.size >= minn:
            out[ii] = out[ii][rng.permutation(ii.size)]
    return out

def run_null_trials(eta_obs, sep_AU, plx_mas, bins, xc_ref, rng,
                    mode="global", ntrial=200):
    meds = []
    for _ in range(ntrial):
        if mode == "global":
            eta_sh = rng.permutation(eta_obs)
        elif mode == "stratified":
            eta_sh = stratified_shuffle(
                eta_obs, sep_AU=sep_AU, plx_mas=plx_mas, rng=rng,
                nsep_bins=NSEP_BINS, nplx_bins=NPLX_BINS, minn=MINN_STRATA
            )
        else:
            raise ValueError("mode must be 'global' or 'stratified'")

        xc, _, Q = binned_quantiles(sep_AU, eta_sh, bins=bins, qs=(0.50,), minN=MINN)
        if len(xc) == len(xc_ref) and np.allclose(xc, xc_ref, rtol=0, atol=0):
            meds.append(Q[0.50])

    if len(meds) == 0:
        return None
    meds = np.asarray(meds)
    return (np.quantile(meds, 0.16, axis=0),
            np.quantile(meds, 0.50, axis=0),
            np.quantile(meds, 0.84, axis=0))

def dex_score(obs_med, null_med):
    obs = np.log10(np.maximum(obs_med, 1e-30))
    nul = np.log10(np.maximum(null_med, 1e-30))
    return float(np.nanmean(obs - nul))

# -----------------------------
# LOAD FITS (fresh)
# -----------------------------
with fits.open(FITS_PATH, memmap=True) as hdul:
    tab = hdul[1].data

# columns
sep_AU = np.asarray(tab["sep_AU"], dtype=float)
rch    = np.asarray(tab["R_chance_align"], dtype=float)

pmra1  = np.asarray(tab["pmra1"], dtype=float)
pmra2  = np.asarray(tab["pmra2"], dtype=float)
pmdec1 = np.asarray(tab["pmdec1"], dtype=float)
pmdec2 = np.asarray(tab["pmdec2"], dtype=float)

plx1   = np.asarray(tab["parallax1"], dtype=float)
plx2   = np.asarray(tab["parallax2"], dtype=float)
plx    = 0.5*(plx1 + plx2)

G1 = np.asarray(tab["phot_g_mean_mag1"], dtype=float)
G2 = np.asarray(tab["phot_g_mean_mag2"], dtype=float)
c1 = np.asarray(tab["bp_rp1"], dtype=float)
c2 = np.asarray(tab["bp_rp2"], dtype=float)

# v_rel proxy
dpm = np.sqrt((pmra1 - pmra2)**2 + (pmdec1 - pmdec2)**2)  # mas/yr
v_kms = km_s_from_pm_parallax(dpm, plx)

# mass proxy
MG1 = abs_MG(G1, plx1)
MG2 = abs_MG(G2, plx2)

ms1 = (c1 > 0.4) & (c1 < 2.5) & (MG1 > 2.0) & (MG1 < 12.5)
ms2 = (c2 > 0.4) & (c2 < 2.5) & (MG2 > 2.0) & (MG2 < 12.5)

m1 = mass_from_MG(MG1)
m2 = mass_from_MG(MG2)
Msys_msun = m1 + m2

# -----------------------------
# Base mask + cuts (ALL ALIGNED because all are full-length here)
# -----------------------------
base = np.isfinite(sep_AU) & np.isfinite(rch) & np.isfinite(v_kms) & np.isfinite(Msys_msun) & np.isfinite(plx)
base &= (sep_AU > 0) & (rch >= 0) & (v_kms > 0) & (Msys_msun > 0) & (plx > 0)

m = base & (sep_AU <= MAX_SEP_AU) & (rch < RMAX) & ms1 & ms2

print(f"Full rows: {len(sep_AU):,}")
print(f"Kept after masks: {np.sum(m):,}  (R<{RMAX}, sep<={MAX_SEP_AU:g} AU, MS-ish)")

# slice
sep_AU_m = sep_AU[m]
plx_mas  = plx[m]
v_m      = v_kms[m] * 1000.0  # m/s
M_m      = Msys_msun[m] * MSUN
s_m      = sep_AU_m * AU_M

# -----------------------------
# eta = v / v_esc
# -----------------------------
v_esc = np.sqrt(2.0 * G * M_m / np.maximum(s_m, 1e-30))
eta   = v_m / np.maximum(v_esc, 1e-30)

# -----------------------------
# bins in separation
# -----------------------------
lo = max(np.nanpercentile(sep_AU_m, 1), 1e2)
hi = min(np.nanpercentile(sep_AU_m, 99), MAX_SEP_AU)
bins = np.logspace(np.log10(lo), np.log10(hi), NBINS + 1)

xc, Nbin, Q = binned_quantiles(sep_AU_m, eta, bins=bins, qs=(0.10, 0.50, 0.90), minN=MINN)
print(f"Bins (total): {NBINS} | kept (N>={MINN}): {len(xc)} | total N in kept bins: {np.sum(Nbin):,}")

# -----------------------------
# nulls
# -----------------------------
rng = np.random.default_rng(SEED)

null_global = run_null_trials(eta, sep_AU_m, plx_mas, bins=bins, xc_ref=xc, rng=rng, mode="global",     ntrial=NTRIAL)
null_strat  = run_null_trials(eta, sep_AU_m, plx_mas, bins=bins, xc_ref=xc, rng=rng, mode="stratified", ntrial=NTRIAL)

# -----------------------------
# plot
# -----------------------------
plt.figure(figsize=(10,6))
plt.plot(xc, Q[0.50], marker="o", label="OBS median")
plt.fill_between(xc, Q[0.10], Q[0.90], alpha=0.15, label="OBS 10–90%")

if null_global is not None:
    gl, gm, gh = null_global
    plt.plot(xc, gm, ls="--", label="NULL median (global shuffle)")
    plt.fill_between(xc, gl, gh, alpha=0.10, label="NULL 16–84% (global)")

if null_strat is not None:
    sl, sm, sh = null_strat
    plt.plot(xc, sm, ls="--", label="NULL median (stratified shuffle)")
    plt.fill_between(xc, sl, sh, alpha=0.10, label="NULL 16–84% (strat)")

plt.axhline(1.0, ls="--", label="eta = 1 (escape boundary)")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Separation s [AU]")
plt.ylabel("eta = v_rel / v_esc")
plt.title(f"Wide binaries: escape-speed diagnostic (R<{RMAX}, sep<={MAX_SEP_AU:g} AU)")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.show()

# -----------------------------
# scores
# -----------------------------
if null_global is not None:
    _, gm, _ = null_global
    print(f"Score OBS - NULL(global): {dex_score(Q[0.50], gm):+.3f} dex")
else:
    print("Global null failed (try lowering MINN or NBINS).")

if null_strat is not None:
    _, sm, _ = null_strat
    print(f"Score OBS - NULL(strat) : {dex_score(Q[0.50], sm):+.3f} dex")
else:
    print("Stratified null failed (try lowering MINN, reducing NSEP_BINS/NPLX_BINS, or MINN_STRATA).")


## Step 1B

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# ============================================================
# Wide binaries: escape-speed diagnostic (eta=v/v_esc)
# SELF-CONTAINED: reads FITS and builds aligned arrays fresh.
# Includes global-shuffle and stratified-shuffle nulls.
# ============================================================

FITS_PATH = "/content/widebin/all_columns_catalog_shift.fits.gz"  # edit if needed

# --- constants ---
G    = 6.67430e-11
MSUN = 1.98847e30
AU_M = 1.495978707e11

# --- selection knobs ---
MAX_SEP_AU = 5e4
RMAX       = 0.10

# --- binning ---
NBINS      = 12
MINN       = 60

# --- null controls ---
NTRIAL      = 200
SEED        = 123
NSEP_BINS   = 12
NPLX_BINS   = 10
MINN_STRATA = 20

# -----------------------------
# helpers
# -----------------------------
def km_s_from_pm_parallax(pm_masyr, plx_mas):
    # v_t[km/s] = 4.74047 * mu[mas/yr] / plx[mas]
    return 4.74047 * (pm_masyr / np.maximum(plx_mas, 1e-12))

def abs_MG(Gmag, plx_mas):
    # MG = G + 5*log10(plx/1000) + 5  (plx in mas)
    return Gmag + 5.0*np.log10(np.maximum(plx_mas, 1e-6)/1000.0) + 5.0

MG_anc = np.array([3.5, 5.0, 7.0, 9.5, 11.5])
M_anc  = np.array([1.6, 1.1, 0.75, 0.35, 0.20])  # Msun

def mass_from_MG(MG):
    MGc = np.clip(MG, MG_anc.min(), MG_anc.max())
    return np.interp(MGc, MG_anc, M_anc)

def binned_quantiles(x, y, bins, qs=(0.10, 0.50, 0.90), minN=60):
    idx = np.digitize(x, bins) - 1
    xc, N = [], []
    Q = {q: [] for q in qs}
    for i in range(len(bins) - 1):
        sel = (idx == i)
        if np.count_nonzero(sel) < minN:
            continue
        xc.append(np.sqrt(bins[i] * bins[i+1]))
        N.append(np.count_nonzero(sel))
        yy = y[sel]
        for q in qs:
            Q[q].append(np.quantile(yy, q))
    xc = np.asarray(xc)
    N  = np.asarray(N)
    for q in qs:
        Q[q] = np.asarray(Q[q])
    return xc, N, Q

def stratified_shuffle(arr, sep_AU, plx_mas, rng,
                       nsep_bins=12, nplx_bins=10, minn=20):
    arr = np.asarray(arr)
    out = arr.copy()

    ls = np.log10(np.maximum(sep_AU, 1e-30))
    lp = np.log10(np.maximum(plx_mas, 1e-30))

    sep_edges = np.quantile(ls, np.linspace(0, 1, nsep_bins + 1))
    plx_edges = np.quantile(lp, np.linspace(0, 1, nplx_bins + 1))

    isep = np.clip(np.digitize(ls, sep_edges) - 1, 0, nsep_bins - 1)
    iplx = np.clip(np.digitize(lp, plx_edges) - 1, 0, nplx_bins - 1)
    sid  = isep * nplx_bins + iplx

    for s in np.unique(sid):
        ii = np.where(sid == s)[0]
        if ii.size >= minn:
            out[ii] = out[ii][rng.permutation(ii.size)]
    return out

def run_null_trials(eta_obs, sep_AU, plx_mas, bins, xc_ref, rng,
                    mode="global", ntrial=200):
    meds = []
    for _ in range(ntrial):
        if mode == "global":
            eta_sh = rng.permutation(eta_obs)
        elif mode == "stratified":
            eta_sh = stratified_shuffle(
                eta_obs, sep_AU=sep_AU, plx_mas=plx_mas, rng=rng,
                nsep_bins=NSEP_BINS, nplx_bins=NPLX_BINS, minn=MINN_STRATA
            )
        else:
            raise ValueError("mode must be 'global' or 'stratified'")

        xc, _, Q = binned_quantiles(sep_AU, eta_sh, bins=bins, qs=(0.50,), minN=MINN)
        if len(xc) == len(xc_ref) and np.allclose(xc, xc_ref, rtol=0, atol=0):
            meds.append(Q[0.50])

    if len(meds) == 0:
        return None
    meds = np.asarray(meds)
    return (np.quantile(meds, 0.16, axis=0),
            np.quantile(meds, 0.50, axis=0),
            np.quantile(meds, 0.84, axis=0))

def dex_score(obs_med, null_med):
    obs = np.log10(np.maximum(obs_med, 1e-30))
    nul = np.log10(np.maximum(null_med, 1e-30))
    return float(np.nanmean(obs - nul))

# -----------------------------
# LOAD FITS (fresh)
# -----------------------------
with fits.open(FITS_PATH, memmap=True) as hdul:
    tab = hdul[1].data

# columns
sep_AU = np.asarray(tab["sep_AU"], dtype=float)
rch    = np.asarray(tab["R_chance_align"], dtype=float)

pmra1  = np.asarray(tab["pmra1"], dtype=float)
pmra2  = np.asarray(tab["pmra2"], dtype=float)
pmdec1 = np.asarray(tab["pmdec1"], dtype=float)
pmdec2 = np.asarray(tab["pmdec2"], dtype=float)

plx1   = np.asarray(tab["parallax1"], dtype=float)
plx2   = np.asarray(tab["parallax2"], dtype=float)
plx    = 0.5*(plx1 + plx2)

G1 = np.asarray(tab["phot_g_mean_mag1"], dtype=float)
G2 = np.asarray(tab["phot_g_mean_mag2"], dtype=float)
c1 = np.asarray(tab["bp_rp1"], dtype=float)
c2 = np.asarray(tab["bp_rp2"], dtype=float)

# v_rel proxy
dpm = np.sqrt((pmra1 - pmra2)**2 + (pmdec1 - pmdec2)**2)  # mas/yr
v_kms = km_s_from_pm_parallax(dpm, plx)

# mass proxy
MG1 = abs_MG(G1, plx1)
MG2 = abs_MG(G2, plx2)

ms1 = (c1 > 0.4) & (c1 < 2.5) & (MG1 > 2.0) & (MG1 < 12.5)
ms2 = (c2 > 0.4) & (c2 < 2.5) & (MG2 > 2.0) & (MG2 < 12.5)

m1 = mass_from_MG(MG1)
m2 = mass_from_MG(MG2)
Msys_msun = m1 + m2

# -----------------------------
# Base mask + cuts (ALL ALIGNED because all are full-length here)
# -----------------------------
base = np.isfinite(sep_AU) & np.isfinite(rch) & np.isfinite(v_kms) & np.isfinite(Msys_msun) & np.isfinite(plx)
base &= (sep_AU > 0) & (rch >= 0) & (v_kms > 0) & (Msys_msun > 0) & (plx > 0)

m = base & (sep_AU <= MAX_SEP_AU) & (rch < RMAX) & ms1 & ms2

print(f"Full rows: {len(sep_AU):,}")
print(f"Kept after masks: {np.sum(m):,}  (R<{RMAX}, sep<={MAX_SEP_AU:g} AU, MS-ish)")

# slice
sep_AU_m = sep_AU[m]
plx_mas  = plx[m]
v_m      = v_kms[m] * 1000.0  # m/s
M_m      = Msys_msun[m] * MSUN
s_m      = sep_AU_m * AU_M

# -----------------------------
# eta = v / v_esc
# -----------------------------
v_esc = np.sqrt(2.0 * G * M_m / np.maximum(s_m, 1e-30))
eta   = v_m / np.maximum(v_esc, 1e-30)

# -----------------------------
# bins in separation
# -----------------------------
lo = max(np.nanpercentile(sep_AU_m, 1), 1e2)
hi = min(np.nanpercentile(sep_AU_m, 99), MAX_SEP_AU)
bins = np.logspace(np.log10(lo), np.log10(hi), NBINS + 1)

xc, Nbin, Q = binned_quantiles(sep_AU_m, eta, bins=bins, qs=(0.10, 0.50, 0.90), minN=MINN)
print(f"Bins (total): {NBINS} | kept (N>={MINN}): {len(xc)} | total N in kept bins: {np.sum(Nbin):,}")

# -----------------------------
# nulls
# -----------------------------
rng = np.random.default_rng(SEED)

null_global = run_null_trials(eta, sep_AU_m, plx_mas, bins=bins, xc_ref=xc, rng=rng, mode="global",     ntrial=NTRIAL)
null_strat  = run_null_trials(eta, sep_AU_m, plx_mas, bins=bins, xc_ref=xc, rng=rng, mode="stratified", ntrial=NTRIAL)

# -----------------------------
# plot
# -----------------------------
plt.figure(figsize=(10,6))
plt.plot(xc, Q[0.50], marker="o", label="OBS median")
plt.fill_between(xc, Q[0.10], Q[0.90], alpha=0.15, label="OBS 10–90%")

if null_global is not None:
    gl, gm, gh = null_global
    plt.plot(xc, gm, ls="--", label="NULL median (global shuffle)")
    plt.fill_between(xc, gl, gh, alpha=0.10, label="NULL 16–84% (global)")

if null_strat is not None:
    sl, sm, sh = null_strat
    plt.plot(xc, sm, ls="--", label="NULL median (stratified shuffle)")
    plt.fill_between(xc, sl, sh, alpha=0.10, label="NULL 16–84% (strat)")

plt.axhline(1.0, ls="--", label="eta = 1 (escape boundary)")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Separation s [AU]")
plt.ylabel("eta = v_rel / v_esc")
plt.title(f"Wide binaries: escape-speed diagnostic (R<{RMAX}, sep<={MAX_SEP_AU:g} AU)")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.show()

# -----------------------------
# scores
# -----------------------------
if null_global is not None:
    _, gm, _ = null_global
    print(f"Score OBS - NULL(global): {dex_score(Q[0.50], gm):+.3f} dex")
else:
    print("Global null failed (try lowering MINN or NBINS).")

if null_strat is not None:
    _, sm, _ = null_strat
    print(f"Score OBS - NULL(strat) : {dex_score(Q[0.50], sm):+.3f} dex")
else:
    print("Stratified null failed (try lowering MINN, reducing NSEP_BINS/NPLX_BINS, or MINN_STRATA).")



## Step 2

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# ============================================================
# Wide binaries: escape-speed diagnostic (eta=v/v_esc)
# SELF-CONTAINED: reads FITS and builds aligned arrays fresh.
# Includes global-shuffle and stratified-shuffle nulls.
# ============================================================

FITS_PATH = "/content/widebin/all_columns_catalog_shift.fits.gz"  # edit if needed

# --- constants ---
G    = 6.67430e-11
MSUN = 1.98847e30
AU_M = 1.495978707e11

# --- selection knobs ---
MAX_SEP_AU = 5e4
RMAX       = 0.10

# --- binning ---
NBINS      = 12
MINN       = 60

# --- null controls ---
NTRIAL      = 200
SEED        = 123
NSEP_BINS   = 12
NPLX_BINS   = 10
MINN_STRATA = 20

# -----------------------------
# helpers
# -----------------------------
def km_s_from_pm_parallax(pm_masyr, plx_mas):
    # v_t[km/s] = 4.74047 * mu[mas/yr] / plx[mas]
    return 4.74047 * (pm_masyr / np.maximum(plx_mas, 1e-12))

def abs_MG(Gmag, plx_mas):
    # MG = G + 5*log10(plx/1000) + 5  (plx in mas)
    return Gmag + 5.0*np.log10(np.maximum(plx_mas, 1e-6)/1000.0) + 5.0

MG_anc = np.array([3.5, 5.0, 7.0, 9.5, 11.5])
M_anc  = np.array([1.6, 1.1, 0.75, 0.35, 0.20])  # Msun

def mass_from_MG(MG):
    MGc = np.clip(MG, MG_anc.min(), MG_anc.max())
    return np.interp(MGc, MG_anc, M_anc)

def binned_quantiles(x, y, bins, qs=(0.10, 0.50, 0.90), minN=60):
    idx = np.digitize(x, bins) - 1
    xc, N = [], []
    Q = {q: [] for q in qs}
    for i in range(len(bins) - 1):
        sel = (idx == i)
        if np.count_nonzero(sel) < minN:
            continue
        xc.append(np.sqrt(bins[i] * bins[i+1]))
        N.append(np.count_nonzero(sel))
        yy = y[sel]
        for q in qs:
            Q[q].append(np.quantile(yy, q))
    xc = np.asarray(xc)
    N  = np.asarray(N)
    for q in qs:
        Q[q] = np.asarray(Q[q])
    return xc, N, Q

def stratified_shuffle(arr, sep_AU, plx_mas, rng,
                       nsep_bins=12, nplx_bins=10, minn=20):
    arr = np.asarray(arr)
    out = arr.copy()

    ls = np.log10(np.maximum(sep_AU, 1e-30))
    lp = np.log10(np.maximum(plx_mas, 1e-30))

    sep_edges = np.quantile(ls, np.linspace(0, 1, nsep_bins + 1))
    plx_edges = np.quantile(lp, np.linspace(0, 1, nplx_bins + 1))

    isep = np.clip(np.digitize(ls, sep_edges) - 1, 0, nsep_bins - 1)
    iplx = np.clip(np.digitize(lp, plx_edges) - 1, 0, nplx_bins - 1)
    sid  = isep * nplx_bins + iplx

    for s in np.unique(sid):
        ii = np.where(sid == s)[0]
        if ii.size >= minn:
            out[ii] = out[ii][rng.permutation(ii.size)]
    return out

def run_null_trials(eta_obs, sep_AU, plx_mas, bins, xc_ref, rng,
                    mode="global", ntrial=200):
    meds = []
    for _ in range(ntrial):
        if mode == "global":
            eta_sh = rng.permutation(eta_obs)
        elif mode == "stratified":
            eta_sh = stratified_shuffle(
                eta_obs, sep_AU=sep_AU, plx_mas=plx_mas, rng=rng,
                nsep_bins=NSEP_BINS, nplx_bins=NPLX_BINS, minn=MINN_STRATA
            )
        else:
            raise ValueError("mode must be 'global' or 'stratified'")

        xc, _, Q = binned_quantiles(sep_AU, eta_sh, bins=bins, qs=(0.50,), minN=MINN)
        if len(xc) == len(xc_ref) and np.allclose(xc, xc_ref, rtol=0, atol=0):
            meds.append(Q[0.50])

    if len(meds) == 0:
        return None
    meds = np.asarray(meds)
    return (np.quantile(meds, 0.16, axis=0),
            np.quantile(meds, 0.50, axis=0),
            np.quantile(meds, 0.84, axis=0))

def dex_score(obs_med, null_med):
    obs = np.log10(np.maximum(obs_med, 1e-30))
    nul = np.log10(np.maximum(null_med, 1e-30))
    return float(np.nanmean(obs - nul))

# -----------------------------
# LOAD FITS (fresh)
# -----------------------------
with fits.open(FITS_PATH, memmap=True) as hdul:
    tab = hdul[1].data

# columns
sep_AU = np.asarray(tab["sep_AU"], dtype=float)
rch    = np.asarray(tab["R_chance_align"], dtype=float)

pmra1  = np.asarray(tab["pmra1"], dtype=float)
pmra2  = np.asarray(tab["pmra2"], dtype=float)
pmdec1 = np.asarray(tab["pmdec1"], dtype=float)
pmdec2 = np.asarray(tab["pmdec2"], dtype=float)

plx1   = np.asarray(tab["parallax1"], dtype=float)
plx2   = np.asarray(tab["parallax2"], dtype=float)
plx    = 0.5*(plx1 + plx2)

G1 = np.asarray(tab["phot_g_mean_mag1"], dtype=float)
G2 = np.asarray(tab["phot_g_mean_mag2"], dtype=float)
c1 = np.asarray(tab["bp_rp1"], dtype=float)
c2 = np.asarray(tab["bp_rp2"], dtype=float)

# v_rel proxy
dpm = np.sqrt((pmra1 - pmra2)**2 + (pmdec1 - pmdec2)**2)  # mas/yr
v_kms = km_s_from_pm_parallax(dpm, plx)

# mass proxy
MG1 = abs_MG(G1, plx1)
MG2 = abs_MG(G2, plx2)

ms1 = (c1 > 0.4) & (c1 < 2.5) & (MG1 > 2.0) & (MG1 < 12.5)
ms2 = (c2 > 0.4) & (c2 < 2.5) & (MG2 > 2.0) & (MG2 < 12.5)

m1 = mass_from_MG(MG1)
m2 = mass_from_MG(MG2)
Msys_msun = m1 + m2

# -----------------------------
# Base mask + cuts (ALL ALIGNED because all are full-length here)
# -----------------------------
base = np.isfinite(sep_AU) & np.isfinite(rch) & np.isfinite(v_kms) & np.isfinite(Msys_msun) & np.isfinite(plx)
base &= (sep_AU > 0) & (rch >= 0) & (v_kms > 0) & (Msys_msun > 0) & (plx > 0)

m = base & (sep_AU <= MAX_SEP_AU) & (rch < RMAX) & ms1 & ms2

print(f"Full rows: {len(sep_AU):,}")
print(f"Kept after masks: {np.sum(m):,}  (R<{RMAX}, sep<={MAX_SEP_AU:g} AU, MS-ish)")

# slice
sep_AU_m = sep_AU[m]
plx_mas  = plx[m]
v_m      = v_kms[m] * 1000.0  # m/s
M_m      = Msys_msun[m] * MSUN
s_m      = sep_AU_m * AU_M

# -----------------------------
# eta = v / v_esc
# -----------------------------
v_esc = np.sqrt(2.0 * G * M_m / np.maximum(s_m, 1e-30))
eta   = v_m / np.maximum(v_esc, 1e-30)

# -----------------------------
# bins in separation
# -----------------------------
lo = max(np.nanpercentile(sep_AU_m, 1), 1e2)
hi = min(np.nanpercentile(sep_AU_m, 99), MAX_SEP_AU)
bins = np.logspace(np.log10(lo), np.log10(hi), NBINS + 1)

xc, Nbin, Q = binned_quantiles(sep_AU_m, eta, bins=bins, qs=(0.10, 0.50, 0.90), minN=MINN)
print(f"Bins (total): {NBINS} | kept (N>={MINN}): {len(xc)} | total N in kept bins: {np.sum(Nbin):,}")

# -----------------------------
# nulls
# -----------------------------
rng = np.random.default_rng(SEED)

null_global = run_null_trials(eta, sep_AU_m, plx_mas, bins=bins, xc_ref=xc, rng=rng, mode="global",     ntrial=NTRIAL)
null_strat  = run_null_trials(eta, sep_AU_m, plx_mas, bins=bins, xc_ref=xc, rng=rng, mode="stratified", ntrial=NTRIAL)

# -----------------------------
# plot
# -----------------------------
plt.figure(figsize=(10,6))
plt.plot(xc, Q[0.50], marker="o", label="OBS median")
plt.fill_between(xc, Q[0.10], Q[0.90], alpha=0.15, label="OBS 10–90%")

if null_global is not None:
    gl, gm, gh = null_global
    plt.plot(xc, gm, ls="--", label="NULL median (global shuffle)")
    plt.fill_between(xc, gl, gh, alpha=0.10, label="NULL 16–84% (global)")

if null_strat is not None:
    sl, sm, sh = null_strat
    plt.plot(xc, sm, ls="--", label="NULL median (stratified shuffle)")
    plt.fill_between(xc, sl, sh, alpha=0.10, label="NULL 16–84% (strat)")

plt.axhline(1.0, ls="--", label="eta = 1 (escape boundary)")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Separation s [AU]")
plt.ylabel("eta = v_rel / v_esc")
plt.title(f"Wide binaries: escape-speed diagnostic (R<{RMAX}, sep<={MAX_SEP_AU:g} AU)")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.show()

# -----------------------------
# scores
# -----------------------------
if null_global is not None:
    _, gm, _ = null_global
    print(f"Score OBS - NULL(global): {dex_score(Q[0.50], gm):+.3f} dex")
else:
    print("Global null failed (try lowering MINN or NBINS).")

if null_strat is not None:
    _, sm, _ = null_strat
    print(f"Score OBS - NULL(strat) : {dex_score(Q[0.50], sm):+.3f} dex")
else:
    print("Stratified null failed (try lowering MINN, reducing NSEP_BINS/NPLX_BINS, or MINN_STRATA).")

## Step 3 v3

import os, glob, numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# ============================================================
# STEP 3 v3 — Phase-sensitive (apo vs peri) that WON'T starve bins
#   Designed for ~1–2k pairs after cuts.
# ============================================================

# ------------------------- SETTINGS -------------------------
MAX_SEP_AU   = 5e4
RMAX         = 0.10

SEP_BINS     = 5          # fewer bins => more counts per bin
TAIL_Q       = 0.40       # bigger tails => more slow/fast points per bin
MIN_PER_BIN  = 50         # keep bins with >=50 total
MIN_TAIL_BIN = 8          # need only 8 slow and 8 fast

# reduced strata complexity (prevents tail starvation)
NSEP_STRATA  = 5
NPLX_STRATA  = 4
NM_STRATA    = 3
MIN_STRATUM  = 20

NTRIAL       = 300
SEED         = 123

MASS_EDGES   = [0.2, 0.8, 10.0]

# --------------------- CONSTANTS/HELPERS --------------------
AU_M  = 1.495978707e11
G     = 6.67430e-11
MSUN  = 1.98847e30

def km_s_from_pm_parallax(pm_masyr, plx_mas):
    return 4.74047 * (pm_masyr / np.maximum(plx_mas, 1e-12))

def abs_MG(Gmag, plx_mas):
    return Gmag + 5.0*np.log10(np.maximum(plx_mas, 1e-6)/1000.0) + 5.0

MG_anc = np.array([3.5, 5.0, 7.0, 9.5, 11.5])
M_anc  = np.array([1.6, 1.1, 0.75, 0.35, 0.20])

def mass_from_MG(MG):
    MGc = np.clip(MG, MG_anc.min(), MG_anc.max())
    return np.interp(MGc, MG_anc, M_anc)

def quant_edges(x, nb):
    qs = np.linspace(0, 1, nb+1)
    e = np.quantile(x, qs)
    for i in range(1, len(e)):
        if e[i] <= e[i-1]:
            e[i] = e[i-1] + 1e-12
    return e

def make_strata_id(sep_AU, plx_mas, Msys_Msun,
                   nsep=NSEP_STRATA, nplx=NPLX_STRATA, nm=NM_STRATA):
    ls = np.log10(np.maximum(sep_AU, 1e-30))
    lp = np.log10(np.maximum(plx_mas, 1e-30))
    lm = np.log10(np.maximum(Msys_Msun, 1e-30))
    sep_edges = quant_edges(ls, nsep)
    plx_edges = quant_edges(lp, nplx)
    m_edges   = quant_edges(lm, nm)
    isep = np.clip(np.digitize(ls, sep_edges) - 1, 0, nsep-1)
    iplx = np.clip(np.digitize(lp, plx_edges) - 1, 0, nplx-1)
    im   = np.clip(np.digitize(lm, m_edges)   - 1, 0, nm-1)
    return (isep * (nplx*nm)) + (iplx * nm) + im

def compute_phase_stats(sep_AU, vrel, eta, sid, sep_bins):
    slow = np.zeros_like(vrel, dtype=bool)
    fast = np.zeros_like(vrel, dtype=bool)

    for s in np.unique(sid):
        idx = np.where(sid == s)[0]
        if idx.size < max(MIN_STRATUM, int(2.0/TAIL_Q)):
            continue
        vv = vrel[idx]
        qlo = np.quantile(vv, TAIL_Q)
        qhi = np.quantile(vv, 1.0 - TAIL_Q)
        slow[idx] = vv <= qlo
        fast[idx] = vv >= qhi

    b = np.digitize(sep_AU, sep_bins) - 1
    xc, Aeta, dlogeta = [], [], []
    Ntot, Nslow, Nfast = [], [], []
    dropped = {"too_few_total":0, "too_few_tails":0, "bad_quant":0}

    for i in range(len(sep_bins)-1):
        inbin = (b == i)
        nt = int(np.count_nonzero(inbin))
        if nt < MIN_PER_BIN:
            dropped["too_few_total"] += 1
            continue

        sslow = inbin & slow
        sfast = inbin & fast
        ns = int(np.count_nonzero(sslow))
        nf = int(np.count_nonzero(sfast))
        if ns < MIN_TAIL_BIN or nf < MIN_TAIL_BIN:
            dropped["too_few_tails"] += 1
            continue

        es = eta[sslow]
        ef = eta[sfast]
        if not (np.all(np.isfinite(es)) and np.all(np.isfinite(ef))):
            dropped["bad_quant"] += 1
            continue

        q10s = np.quantile(es, 0.10)
        q10f = np.quantile(ef, 0.10)
        if (not np.isfinite(q10s)) or (not np.isfinite(q10f)) or (q10f <= 0) or (q10s <= 0):
            dropped["bad_quant"] += 1
            continue

        A = q10s / q10f
        dl = np.median(np.log10(es)) - np.median(np.log10(ef))

        xc.append(np.sqrt(sep_bins[i]*sep_bins[i+1]))
        Aeta.append(A)
        dlogeta.append(dl)
        Ntot.append(nt); Nslow.append(ns); Nfast.append(nf)

    return (np.asarray(xc), np.asarray(Aeta), np.asarray(dlogeta),
            np.asarray(Ntot), np.asarray(Nslow), np.asarray(Nfast), dropped)

def stratified_shuffle(v, sid, rng):
    out = v.copy()
    for s in np.unique(sid):
        idx = np.where(sid == s)[0]
        if idx.size >= MIN_STRATUM:
            out[idx] = out[idx][rng.permutation(idx.size)]
    return out

def run_null(sep_AU, eta_base, vrel_base, sid, sep_bins, xc_ref):
    rng = np.random.default_rng(SEED)
    v0 = np.maximum(vrel_base, 1e-30)

    A_trials = []
    D_trials = []

    for _ in range(NTRIAL):
        v_sh = stratified_shuffle(vrel_base, sid, rng)
        eta_sh = eta_base * (v_sh / v0)
        xc, A, D, *_ = compute_phase_stats(sep_AU, v_sh, eta_sh, sid, sep_bins)
        if len(xc) == len(xc_ref) and np.allclose(xc, xc_ref, rtol=0, atol=0):
            A_trials.append(A); D_trials.append(D)

    if len(A_trials) == 0:
        return None

    A_trials = np.asarray(A_trials)
    D_trials = np.asarray(D_trials)

    A16, A50, A84 = np.quantile(A_trials, [0.16, 0.50, 0.84], axis=0)
    D16, D50, D84 = np.quantile(D_trials, [0.16, 0.50, 0.84], axis=0)
    return (A16, A50, A84, A_trials, D16, D50, D84, D_trials)

# -------------------- AUTO-FIND FITS PATH --------------------
def find_fits():
    roots = ["/content"]
    if os.path.exists("/content/drive/MyDrive"):
        roots.append("/content/drive/MyDrive")
    pats = ["**/*.fits", "**/*.fits.gz"]
    cands = []
    for r in roots:
        for p in pats:
            cands += glob.glob(os.path.join(r, p), recursive=True)
    preferred = [p for p in cands if "all_columns_catalog_shift.fits" in os.path.basename(p)]
    return preferred[0] if preferred else (cands[0] if cands else None)

FITS_PATH = find_fits()
if FITS_PATH is None:
    raise RuntimeError("No FITS/FITS.GZ found in runtime (or Drive).")
print(f"Using FITS_PATH: {FITS_PATH}")

# -------------------------- LOAD FITS --------------------------
with fits.open(FITS_PATH, memmap=True) as hdul:
    tab = hdul[1].data

sep_AU = np.asarray(tab["sep_AU"], dtype=float)
rch    = np.asarray(tab["R_chance_align"], dtype=float)

pmra1  = np.asarray(tab["pmra1"], dtype=float)
pmra2  = np.asarray(tab["pmra2"], dtype=float)
pmdec1 = np.asarray(tab["pmdec1"], dtype=float)
pmdec2 = np.asarray(tab["pmdec2"], dtype=float)

plx1   = np.asarray(tab["parallax1"], dtype=float)
plx2   = np.asarray(tab["parallax2"], dtype=float)

G1 = np.asarray(tab["phot_g_mean_mag1"], dtype=float)
G2 = np.asarray(tab["phot_g_mean_mag2"], dtype=float)
c1 = np.asarray(tab["bp_rp1"], dtype=float)
c2 = np.asarray(tab["bp_rp2"], dtype=float)

dpm = np.sqrt((pmra1 - pmra2)**2 + (pmdec1 - pmdec2)**2)
plx = 0.5*(plx1 + plx2)
vrel = km_s_from_pm_parallax(dpm, plx)

MG1 = abs_MG(G1, plx1)
MG2 = abs_MG(G2, plx2)
ms1 = (c1 > 0.4) & (c1 < 2.5) & (MG1 > 2.0) & (MG1 < 12.5)
ms2 = (c2 > 0.4) & (c2 < 2.5) & (MG2 > 2.0) & (MG2 < 12.5)

m1 = mass_from_MG(MG1)
m2 = mass_from_MG(MG2)
Msys = m1 + m2

base = np.isfinite(sep_AU) & np.isfinite(rch) & np.isfinite(vrel) & np.isfinite(Msys) & np.isfinite(plx)
base &= (sep_AU > 0) & (plx > 0) & (vrel > 0) & (Msys > 0) & (rch >= 0)

m = base & (sep_AU <= MAX_SEP_AU) & (rch < RMAX) & ms1 & ms2

print(f"\nFull rows: {len(sep_AU):,}")
print(f"Kept after masks: {np.sum(m):,}  (R<{RMAX}, sep<={MAX_SEP_AU:g} AU, MS-ish)")

sep_m = sep_AU[m]
plx_m = plx[m]
v_m   = vrel[m]
M_m   = Msys[m]

s_m_m = sep_m * AU_M
M_kg  = M_m * MSUN
v_ms  = v_m * 1000.0
v_esc = np.sqrt(2.0*G*M_kg / np.maximum(s_m_m, 1e-30))
eta   = v_ms / np.maximum(v_esc, 1e-30)

# sep bins
slo = np.nanpercentile(sep_m, 2)
shi = np.nanpercentile(sep_m, 98)
sep_bins = np.logspace(np.log10(max(slo, 80.0)), np.log10(min(shi, MAX_SEP_AU)), SEP_BINS+1)

print("\n====================")
print("STEP-3 v3 (phase-sensitive via apo/peri tails)")
print("====================")

mass_edges = np.array(MASS_EDGES, dtype=float)
## Step 3 v3

import os, glob, numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# ============================================================
# STEP 3 v3 — Phase-sensitive (apo vs peri) that WON'T starve bins
#   Designed for ~1–2k pairs after cuts.
# ============================================================

# ------------------------- SETTINGS -------------------------
MAX_SEP_AU   = 5e4
RMAX         = 0.10

SEP_BINS     = 5          # fewer bins => more counts per bin
TAIL_Q       = 0.40       # bigger tails => more slow/fast points per bin
MIN_PER_BIN  = 50         # keep bins with >=50 total
MIN_TAIL_BIN = 8          # need only 8 slow and 8 fast

# reduced strata complexity (prevents tail starvation)
NSEP_STRATA  = 5
NPLX_STRATA  = 4
NM_STRATA    = 3
MIN_STRATUM  = 20

NTRIAL       = 300
SEED         = 123

MASS_EDGES   = [0.2, 0.8, 10.0]

# --------------------- CONSTANTS/HELPERS --------------------
AU_M  = 1.495978707e11
G     = 6.67430e-11
MSUN  = 1.98847e30

def km_s_from_pm_parallax(pm_masyr, plx_mas):
    return 4.74047 * (pm_masyr / np.maximum(plx_mas, 1e-12))

def abs_MG(Gmag, plx_mas):
    return Gmag + 5.0*np.log10(np.maximum(plx_mas, 1e-6)/1000.0) + 5.0

MG_anc = np.array([3.5, 5.0, 7.0, 9.5, 11.5])
M_anc  = np.array([1.6, 1.1, 0.75, 0.35, 0.20])

def mass_from_MG(MG):
    MGc = np.clip(MG, MG_anc.min(), MG_anc.max())
    return np.interp(MGc, MG_anc, M_anc)

def quant_edges(x, nb):
    qs = np.linspace(0, 1, nb+1)
    e = np.quantile(x, qs)
    for i in range(1, len(e)):
        if e[i] <= e[i-1]:
            e[i] = e[i-1] + 1e-12
    return e

def make_strata_id(sep_AU, plx_mas, Msys_Msun,
                   nsep=NSEP_STRATA, nplx=NPLX_STRATA, nm=NM_STRATA):
    ls = np.log10(np.maximum(sep_AU, 1e-30))
    lp = np.log10(np.maximum(plx_mas, 1e-30))
    lm = np.log10(np.maximum(Msys_Msun, 1e-30))
    sep_edges = quant_edges(ls, nsep)
    plx_edges = quant_edges(lp, nplx)
    m_edges   = quant_edges(lm, nm)
    isep = np.clip(np.digitize(ls, sep_edges) - 1, 0, nsep-1)
    iplx = np.clip(np.digitize(lp, plx_edges) - 1, 0, nplx-1)
    im   = np.clip(np.digitize(lm, m_edges)   - 1, 0, nm-1)
    return (isep * (nplx*nm)) + (iplx * nm) + im

def compute_phase_stats(sep_AU, vrel, eta, sid, sep_bins):
    slow = np.zeros_like(vrel, dtype=bool)
    fast = np.zeros_like(vrel, dtype=bool)

    for s in np.unique(sid):
        idx = np.where(sid == s)[0]
        if idx.size < max(MIN_STRATUM, int(2.0/TAIL_Q)):
            continue
        vv = vrel[idx]
        qlo = np.quantile(vv, TAIL_Q)
        qhi = np.quantile(vv, 1.0 - TAIL_Q)
        slow[idx] = vv <= qlo
        fast[idx] = vv >= qhi

    b = np.digitize(sep_AU, sep_bins) - 1
    xc, Aeta, dlogeta = [], [], []
    Ntot, Nslow, Nfast = [], [], []
    dropped = {"too_few_total":0, "too_few_tails":0, "bad_quant":0}

    for i in range(len(sep_bins)-1):
        inbin = (b == i)
        nt = int(np.count_nonzero(inbin))
        if nt < MIN_PER_BIN:
            dropped["too_few_total"] += 1
            continue

        sslow = inbin & slow
        sfast = inbin & fast
        ns = int(np.count_nonzero(sslow))
        nf = int(np.count_nonzero(sfast))
        if ns < MIN_TAIL_BIN or nf < MIN_TAIL_BIN:
            dropped["too_few_tails"] += 1
            continue

        es = eta[sslow]
        ef = eta[sfast]
        if not (np.all(np.isfinite(es)) and np.all(np.isfinite(ef))):
            dropped["bad_quant"] += 1
            continue

        q10s = np.quantile(es, 0.10)
        q10f = np.quantile(ef, 0.10)
        if (not np.isfinite(q10s)) or (not np.isfinite(q10f)) or (q10f <= 0) or (q10s <= 0):
            dropped["bad_quant"] += 1
            continue

        A = q10s / q10f
        dl = np.median(np.log10(es)) - np.median(np.log10(ef))

        xc.append(np.sqrt(sep_bins[i]*sep_bins[i+1]))
        Aeta.append(A)
        dlogeta.append(dl)
        Ntot.append(nt); Nslow.append(ns); Nfast.append(nf)

    return (np.asarray(xc), np.asarray(Aeta), np.asarray(dlogeta),
            np.asarray(Ntot), np.asarray(Nslow), np.asarray(Nfast), dropped)

def stratified_shuffle(v, sid, rng):
    out = v.copy()
    for s in np.unique(sid):
        idx = np.where(sid == s)[0]
        if idx.size >= MIN_STRATUM:
            out[idx] = out[idx][rng.permutation(idx.size)]
    return out

def run_null(sep_AU, eta_base, vrel_base, sid, sep_bins, xc_ref):
    rng = np.random.default_rng(SEED)
    v0 = np.maximum(vrel_base, 1e-30)

    A_trials = []
    D_trials = []

    for _ in range(NTRIAL):
        v_sh = stratified_shuffle(vrel_base, sid, rng)
        eta_sh = eta_base * (v_sh / v0)
        xc, A, D, *_ = compute_phase_stats(sep_AU, v_sh, eta_sh, sid, sep_bins)
        if len(xc) == len(xc_ref) and np.allclose(xc, xc_ref, rtol=0, atol=0):
            A_trials.append(A); D_trials.append(D)

    if len(A_trials) == 0:
        return None

    A_trials = np.asarray(A_trials)
    D_trials = np.asarray(D_trials)

    A16, A50, A84 = np.quantile(A_trials, [0.16, 0.50, 0.84], axis=0)
    D16, D50, D84 = np.quantile(D_trials, [0.16, 0.50, 0.84], axis=0)
    return (A16, A50, A84, A_trials, D16, D50, D84, D_trials)

# -------------------- AUTO-FIND FITS PATH --------------------
def find_fits():
    roots = ["/content"]
    if os.path.exists("/content/drive/MyDrive"):
        roots.append("/content/drive/MyDrive")
    pats = ["**/*.fits", "**/*.fits.gz"]
    cands = []
    for r in roots:
        for p in pats:
            cands += glob.glob(os.path.join(r, p), recursive=True)
    preferred = [p for p in cands if "all_columns_catalog_shift.fits" in os.path.basename(p)]
    return preferred[0] if preferred else (cands[0] if cands else None)

FITS_PATH = find_fits()
if FITS_PATH is None:
    raise RuntimeError("No FITS/FITS.GZ found in runtime (or Drive).")
print(f"Using FITS_PATH: {FITS_PATH}")

# -------------------------- LOAD FITS --------------------------
with fits.open(FITS_PATH, memmap=True) as hdul:
    tab = hdul[1].data

sep_AU = np.asarray(tab["sep_AU"], dtype=float)
rch    = np.asarray(tab["R_chance_align"], dtype=float)

pmra1  = np.asarray(tab["pmra1"], dtype=float)
pmra2  = np.asarray(tab["pmra2"], dtype=float)
pmdec1 = np.asarray(tab["pmdec1"], dtype=float)
pmdec2 = np.asarray(tab["pmdec2"], dtype=float)

plx1   = np.asarray(tab["parallax1"], dtype=float)
plx2   = np.asarray(tab["parallax2"], dtype=float)

G1 = np.asarray(tab["phot_g_mean_mag1"], dtype=float)
G2 = np.asarray(tab["phot_g_mean_mag2"], dtype=float)
c1 = np.asarray(tab["bp_rp1"], dtype=float)
c2 = np.asarray(tab["bp_rp2"], dtype=float)

dpm = np.sqrt((pmra1 - pmra2)**2 + (pmdec1 - pmdec2)**2)
plx = 0.5*(plx1 + plx2)
vrel = km_s_from_pm_parallax(dpm, plx)

MG1 = abs_MG(G1, plx1)
MG2 = abs_MG(G2, plx2)
ms1 = (c1 > 0.4) & (c1 < 2.5) & (MG1 > 2.0) & (MG1 < 12.5)
ms2 = (c2 > 0.4) & (c2 < 2.5) & (MG2 > 2.0) & (MG2 < 12.5)

m1 = mass_from_MG(MG1)
m2 = mass_from_MG(MG2)
Msys = m1 + m2

base = np.isfinite(sep_AU) & np.isfinite(rch) & np.isfinite(vrel) & np.isfinite(Msys) & np.isfinite(plx)
base &= (sep_AU > 0) & (plx > 0) & (vrel > 0) & (Msys > 0) & (rch >= 0)

m = base & (sep_AU <= MAX_SEP_AU) & (rch < RMAX) & ms1 & ms2

print(f"\nFull rows: {len(sep_AU):,}")
print(f"Kept after masks: {np.sum(m):,}  (R<{RMAX}, sep<={MAX_SEP_AU:g} AU, MS-ish)")

sep_m = sep_AU[m]
plx_m = plx[m]
v_m   = vrel[m]
M_m   = Msys[m]

s_m_m = sep_m * AU_M
M_kg  = M_m * MSUN
v_ms  = v_m * 1000.0
v_esc = np.sqrt(2.0*G*M_kg / np.maximum(s_m_m, 1e-30))
eta   = v_ms / np.maximum(v_esc, 1e-30)

# sep bins
slo = np.nanpercentile(sep_m, 2)
shi = np.nanpercentile(sep_m, 98)
sep_bins = np.logspace(np.log10(max(slo, 80.0)), np.log10(min(shi, MAX_SEP_AU)), SEP_BINS+1)

print("\n====================")
print("STEP-3 v3 (phase-sensitive via apo/peri tails)")
print("====================")

mass_edges = np.array(MASS_EDGES, dtype=float)
results = []

for j in range(len(mass_edges)-1):
    loM, hiM = mass_edges[j], mass_edges[j+1]
    sel = (M_m >= loM) & (M_m < hiM)
    lab = f"M∈[{loM:g},{hiM:g}) Msun"
    Nj = int(np.count_nonzero(sel))

    print(f"\n============================================================")
    print(lab)
    print(f"  N pairs in mass bin: {Nj:,}")
    if Nj < 500:
        print("  -> too few pairs, skipping")
        continue

    sep_j = sep_m[sel]
    plx_j = plx_m[sel]
    v_j   = v_m[sel]
    eta_j = eta[sel]
    Mj    = M_m[sel]

    sid_j = make_strata_id(sep_j, plx_j, Mj)

    xc, Aobs, Dobs, Ntot, Nslow, Nfast, dropped = compute_phase_stats(sep_j, v_j, eta_j, sid_j, sep_bins)
    print(f"  Requested sep bins: {SEP_BINS} | Kept bins: {len(xc)}")
    print(f"  Dropped bins: {dropped}")

    if len(xc) == 0:
        print("  -> Still no bins. If this happens, set: SEP_BINS=4, TAIL_Q=0.45, MIN_TAIL_BIN=6, strata=(4,3,2).")
        continue

    null = run_null(sep_j, eta_j, v_j, sid_j, sep_bins, xc_ref=xc)
    if null is None:
        print("  -> NULL failed (no accepted trials). Reduce strata counts further.")
        results.append((lab, xc, Aobs, Dobs, None))
        continue

    A16, A50, A84, Atr, D16, D50, D84, Dtr = null
    # two-sided empirical p-values around null median
    pA = np.mean(np.abs(Atr - A50[None,:]) >= np.abs(Aobs[None,:] - A50[None,:]), axis=0)
    pD = np.mean(np.abs(Dtr - D50[None,:]) >= np.abs(Dobs[None,:] - D50[None,:]), axis=0)

    print(f"  NULL accepted trials: {Atr.shape[0]}/{NTRIAL}")
    print("  Per-bin:")
    for k in range(len(xc)):
        print(f"    s~{xc[k]:6.0f} AU | Aobs={Aobs[k]:.3f} null_med={A50[k]:.3f} [{A16[k]:.3f},{A84[k]:.3f}] p≈{pA[k]:.3f}"
              f" | Δlogη={Dobs[k]:+.3f} null_med={D50[k]:+.3f} [{D16[k]:+.3f},{D84[k]:+.3f}] p≈{pD[k]:.3f}")

    results.append((lab, xc, Aobs, Dobs, (A16,A50,A84,D16,D50,D84)))

if len(results) == 0:
    print("\nNo mass bins produced usable results with these settings.")
    print("Next escalation: SEP_BINS=4, TAIL_Q=0.45, MIN_TAIL_BIN=6, strata=(4,3,2).")
    raise RuntimeError("STEP-3 v3 produced no usable results.")

# -------------------------- PLOTS --------------------------
plt.figure(figsize=(10,6))
for (lab, xc, Aobs, Dobs, null) in results:
    plt.plot(xc, Aobs, marker="o", label=f"OBS Aη {lab}")
    if null is not None:
        A16,A50,A84,_,_,_ = null
        plt.plot(xc, A50, ls="--", label=f"NULL med {lab}")
        plt.fill_between(xc, A16, A84, alpha=0.12)
plt.axhline(1.0, ls="--", label="Aη=1")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Separation s [AU]")
plt.ylabel("Aη(s)=Q10(η_slow)/Q10(η_fast)")
plt.title(f"STEP-3 v3 | R<{RMAX} | tail={TAIL_Q:.2f} | strata={NSEP_STRATA},{NPLX_STRATA},{NM_STRATA}")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
for (lab, xc, Aobs, Dobs, null) in results:
    plt.plot(xc, Dobs, marker="o", label=f"OBS Δlogη {lab}")
    if null is not None:
        _,_,_,D16,D50,D84 = null
        plt.plot(xc, D50, ls="--", label=f"NULL med {lab}")
        plt.fill_between(xc, D16, D84, alpha=0.12)
plt.axhline(0.0, ls="--", label="Δlogη=0")
plt.xscale("log")
plt.xlabel("Separation s [AU]")
plt.ylabel("Δlogη(s)=median(logη_slow)−median(logη_fast)")
plt.title(f"STEP-3 v3 | R<{RMAX} | tail={TAIL_Q:.2f} | strata={NSEP_STRATA},{NPLX_STRATA},{NM_STRATA}")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.show()

## KILL TEST 1

# ============================================================
# KILL TEST 1 — PAIR-BREAKING (CORRECT)
#   Break the coupling between v_rel and (sep, Msys) by shuffling v_rel
#   across pairs, then recompute eta = v/vesc using each pair's own vesc.
# ============================================================

import numpy as np
from astropy.io import fits
import os

print("\n==============================")
print("KILL TEST 1 — PAIR-BREAKING (CORRECT)")
print("==============================")

# ------------------------------------------------------------
# Locate FITS automatically (runtime first, then Drive)
# ------------------------------------------------------------
CANDIDATES = [
    "/content/widebin/all_columns_catalog_shift.fits.gz",
    "/content/drive/MyDrive/widebin/all_columns_catalog_shift.fits.gz",
]
FITS_PATH = None
for p in CANDIDATES:
    if os.path.exists(p):
        FITS_PATH = p
        break
if FITS_PATH is None:
    raise RuntimeError("Could not find wide-binary FITS in runtime or Drive.")

print(f"Using FITS_PATH: {FITS_PATH}")

# ----------------- CONSTANTS -----------------
AU_M  = 1.495978707e11
G     = 6.67430e-11
MSUN  = 1.98847e30

# ----------------- SETTINGS -----------------
MAX_SEP_AU   = 5e4
RMAX         = 0.10
TAIL_Q       = 0.30
MIN_PER_BIN  = 30
MIN_TAIL_BIN = 8
SEP_BINS     = 6

NSEP_STRATA  = 6
NPLX_STRATA  = 5
NM_STRATA    = 5
MIN_STRATUM  = 20

SEED = 12345
rng = np.random.default_rng(SEED)

# ----------------- HELPERS -----------------
def km_s_from_pm_parallax(pm_masyr, plx_mas):
    return 4.74047 * pm_masyr / np.maximum(plx_mas, 1e-12)

def abs_MG(Gmag, plx_mas):
    return Gmag + 5*np.log10(np.maximum(plx_mas,1e-6)/1000) + 5

MG_anc = np.array([3.5,5.0,7.0,9.5,11.5])
M_anc  = np.array([1.6,1.1,0.75,0.35,0.20])
def mass_from_MG(MG):
    return np.interp(np.clip(MG, MG_anc.min(), MG_anc.max()), MG_anc, M_anc)

def quant_edges(x, n):
    q = np.quantile(x, np.linspace(0,1,n+1))
    for i in range(1,len(q)):
        if q[i] <= q[i-1]:
            q[i] = q[i-1] + 1e-12
    return q

def make_strata_id(sep, plx, M):
    ls, lp, lm = np.log10(sep), np.log10(plx), np.log10(M)
    es = quant_edges(ls, NSEP_STRATA)
    ep = quant_edges(lp, NPLX_STRATA)
    em = quant_edges(lm, NM_STRATA)
    isep = np.digitize(ls, es)-1
    iplx = np.digitize(lp, ep)-1
    im   = np.digitize(lm, em)-1
    return isep*(NPLX_STRATA*NM_STRATA) + iplx*NM_STRATA + im

def compute_A(sep, eta, v, sid, bins):
    slow = np.zeros(len(v), bool)
    fast = np.zeros(len(v), bool)

    for s in np.unique(sid):
        idx = np.where(sid == s)[0]
        if len(idx) < MIN_STRATUM:
            continue
        ql, qh = np.quantile(v[idx], [TAIL_Q, 1-TAIL_Q])
        slow[idx] = v[idx] <= ql
        fast[idx] = v[idx] >= qh

    xc, A = [], []
    b = np.digitize(sep, bins) - 1

    for i in range(len(bins)-1):
        m = b == i
        if np.sum(m) < MIN_PER_BIN:
            continue
        if np.sum(m & slow) < MIN_TAIL_BIN or np.sum(m & fast) < MIN_TAIL_BIN:
            continue

        qs = np.quantile(eta[m & slow], 0.1)
        qf = np.quantile(eta[m & fast], 0.1)
        if not np.isfinite(qs) or not np.isfinite(qf) or qf <= 0:
            continue

        xc.append(np.sqrt(bins[i]*bins[i+1]))
        A.append(qs/qf)

    return np.array(xc), np.array(A)

# ----------------- LOAD DATA -----------------
with fits.open(FITS_PATH, memmap=True) as hdul:
    t = hdul[1].data

sep = np.asarray(t["sep_AU"], float)
rch = np.asarray(t["R_chance_align"], float)

pmra1, pmdec1 = np.asarray(t["pmra1"], float), np.asarray(t["pmdec1"], float)
pmra2, pmdec2 = np.asarray(t["pmra2"], float), np.asarray(t["pmdec2"], float)

plx1, plx2 = np.asarray(t["parallax1"], float), np.asarray(t["parallax2"], float)
plx = 0.5*(plx1 + plx2)

dpm = np.hypot(pmra1 - pmra2, pmdec1 - pmdec2)
vrel = km_s_from_pm_parallax(dpm, plx)

MG1 = abs_MG(np.asarray(t["phot_g_mean_mag1"], float), plx1)
MG2 = abs_MG(np.asarray(t["phot_g_mean_mag2"], float), plx2)
Msys = mass_from_MG(MG1) + mass_from_MG(MG2)

# ----------------- MASK -----------------
m = (
    np.isfinite(sep) & np.isfinite(rch) & np.isfinite(vrel) & np.isfinite(plx) & np.isfinite(Msys) &
    (sep > 0) & (sep <= MAX_SEP_AU) &
    (rch < RMAX) &
    (plx > 0) &
    (Msys >= 0.8)
)

sep, vrel, plx, Msys = sep[m], vrel[m], plx[m], Msys[m]
print(f"Pairs used: {len(sep):,}")

# ----------------- Compute vesc and eta -----------------
vesc = np.sqrt(2 * G * (Msys*MSUN) / (sep*AU_M))  # m/s
eta_obs = (vrel*1000.0) / np.maximum(vesc, 1e-30)

# ----------------- bins + strata -----------------
bins = np.logspace(
    np.log10(np.percentile(sep, 5)),
    np.log10(np.percentile(sep, 95)),
    SEP_BINS + 1
)
sid = make_strata_id(sep, plx, Msys)

# ----------------- OBS -----------------
xc_obs, A_obs = compute_A(sep, eta_obs, vrel, sid, bins)

# ----------------- Pair-break: shuffle vrel ONLY, recompute eta with same vesc -----------------
perm = rng.permutation(len(vrel))
v_pb = vrel[perm]
eta_pb = (v_pb*1000.0) / np.maximum(vesc, 1e-30)

xc_pb, A_pb = compute_A(sep, eta_pb, v_pb, sid, bins)

# ----------------- REPORT -----------------
print("\nOBSERVED A(s):")
for x,a in zip(xc_obs, A_obs):
    print(f"  s~{int(x):6d} AU | A={a:.3f}")

if len(A_pb) == 0:
    print("\n✅ RESULT: Pair-breaking DESTROYS the asymmetry.")
    print("   → Strong evidence the effect depends on real pair dynamics.")
else:
    print("\n🚨 RESULT: Asymmetry survives CORRECT pair-breaking!")
    for x,a in zip(xc_pb, A_pb):
        print(f"  s~{int(x):6d} AU | A_pairbreak={a:.3f}")


## Kill test 1B

# ============================================================
# KILL TEST 1b — PAIR-BREAKING MONTE CARLO (p-values)
#   Shuffle vrel across pairs many times, recompute eta=v/vesc,
#   recompute A(s). Compare A_obs to null distribution.
# ============================================================

import numpy as np
from astropy.io import fits
import os

print("\n=======================================")
print("KILL TEST 1b — PAIR-BREAKING (MONTE CARLO)")
print("=======================================")

# ---- Find FITS in runtime/Drive ----
CANDIDATES = [
    "/content/widebin/all_columns_catalog_shift.fits.gz",
    "/content/drive/MyDrive/widebin/all_columns_catalog_shift.fits.gz",
]
FITS_PATH = next((p for p in CANDIDATES if os.path.exists(p)), None)
if FITS_PATH is None:
    raise RuntimeError("Could not find wide-binary FITS in runtime or Drive.")
print(f"Using FITS_PATH: {FITS_PATH}")

# ----------------- CONSTANTS -----------------
AU_M  = 1.495978707e11
G     = 6.67430e-11
MSUN  = 1.98847e30

# ----------------- SETTINGS -----------------
MAX_SEP_AU   = 5e4
RMAX         = 0.10
TAIL_Q       = 0.30
MIN_PER_BIN  = 30
MIN_TAIL_BIN = 8
SEP_BINS     = 6

NSEP_STRATA  = 6
NPLX_STRATA  = 5
NM_STRATA    = 5
MIN_STRATUM  = 20

NTRIAL = 500
SEED   = 12345
rng = np.random.default_rng(SEED)

# ----------------- HELPERS -----------------
def km_s_from_pm_parallax(pm_masyr, plx_mas):
    return 4.74047 * pm_masyr / np.maximum(plx_mas, 1e-12)

def abs_MG(Gmag, plx_mas):
    return Gmag + 5*np.log10(np.maximum(plx_mas,1e-6)/1000) + 5

MG_anc = np.array([3.5,5.0,7.0,9.5,11.5])
M_anc  = np.array([1.6,1.1,0.75,0.35,0.20])
def mass_from_MG(MG):
    return np.interp(np.clip(MG, MG_anc.min(), MG_anc.max()), MG_anc, M_anc)

def quant_edges(x, n):
    q = np.quantile(x, np.linspace(0,1,n+1))
    for i in range(1,len(q)):
        if q[i] <= q[i-1]:
            q[i] = q[i-1] + 1e-12
    return q

def make_strata_id(sep, plx, M):
    ls, lp, lm = np.log10(sep), np.log10(plx), np.log10(M)
    es = quant_edges(ls, NSEP_STRATA)
    ep = quant_edges(lp, NPLX_STRATA)
    em = quant_edges(lm, NM_STRATA)
    isep = np.clip(np.digitize(ls, es)-1, 0, NSEP_STRATA-1)
    iplx = np.clip(np.digitize(lp, ep)-1, 0, NPLX_STRATA-1)
    im   = np.clip(np.digitize(lm, em)-1, 0, NM_STRATA-1)
    return isep*(NPLX_STRATA*NM_STRATA) + iplx*NM_STRATA + im

def compute_A(sep, eta, v, sid, bins):
    slow = np.zeros(len(v), bool)
    fast = np.zeros(len(v), bool)

    for s in np.unique(sid):
        idx = np.where(sid == s)[0]
        if len(idx) < MIN_STRATUM:
            continue
        ql, qh = np.quantile(v[idx], [TAIL_Q, 1-TAIL_Q])
        slow[idx] = v[idx] <= ql
        fast[idx] = v[idx] >= qh

    xc, A = [], []
    b = np.digitize(sep, bins) - 1

    for i in range(len(bins)-1):
        m = b == i
        if np.sum(m) < MIN_PER_BIN:
            continue
        if np.sum(m & slow) < MIN_TAIL_BIN or np.sum(m & fast) < MIN_TAIL_BIN:
            continue
        qs = np.quantile(eta[m & slow], 0.1)
        qf = np.quantile(eta[m & fast], 0.1)
        if not np.isfinite(qs) or not np.isfinite(qf) or qf <= 0:
            continue
        xc.append(np.sqrt(bins[i]*bins[i+1]))
        A.append(qs/qf)

    return np.array(xc), np.array(A)

# ----------------- LOAD -----------------
with fits.open(FITS_PATH, memmap=True) as hdul:
    t = hdul[1].data

sep = np.asarray(t["sep_AU"], float)
rch = np.asarray(t["R_chance_align"], float)

pmra1, pmdec1 = np.asarray(t["pmra1"], float), np.asarray(t["pmdec1"], float)
pmra2, pmdec2 = np.asarray(t["pmra2"], float), np.asarray(t["pmdec2"], float)

plx1, plx2 = np.asarray(t["parallax1"], float), np.asarray(t["parallax2"], float)
plx = 0.5*(plx1 + plx2)

dpm  = np.hypot(pmra1 - pmra2, pmdec1 - pmdec2)
vrel = km_s_from_pm_parallax(dpm, plx)

MG1 = abs_MG(np.asarray(t["phot_g_mean_mag1"], float), plx1)
MG2 = abs_MG(np.asarray(t["phot_g_mean_mag2"], float), plx2)
Msys = mass_from_MG(MG1) + mass_from_MG(MG2)

# mask (match your current high-mass run)
m = (
    np.isfinite(sep) & np.isfinite(rch) & np.isfinite(vrel) & np.isfinite(plx) & np.isfinite(Msys) &
    (sep > 0) & (sep <= MAX_SEP_AU) &
    (rch < RMAX) & (plx > 0) &
    (Msys >= 0.8)
)
sep, vrel, plx, Msys = sep[m], vrel[m], plx[m], Msys[m]
print(f"Pairs used: {len(sep):,}")

vesc = np.sqrt(2 * G * (Msys*MSUN) / (sep*AU_M))  # m/s
eta_obs = (vrel*1000.0) / np.maximum(vesc, 1e-30)

bins = np.logspace(np.log10(np.percentile(sep, 5)), np.log10(np.percentile(sep, 95)), SEP_BINS+1)
sid  = make_strata_id(sep, plx, Msys)

xc_obs, A_obs = compute_A(sep, eta_obs, vrel, sid, bins)
if len(A_obs) == 0:
    raise RuntimeError("No usable A(s) bins in OBS. Loosen bin/tail thresholds.")

# Monte Carlo null
A_trials = []
for k in range(NTRIAL):
    perm = rng.permutation(len(vrel))
    v_pb = vrel[perm]
    eta_pb = (v_pb*1000.0) / np.maximum(vesc, 1e-30)
    xc, A = compute_A(sep, eta_pb, v_pb, sid, bins)
    if len(xc) == len(xc_obs) and np.allclose(xc, xc_obs, atol=0, rtol=0):
        A_trials.append(A)

A_trials = np.asarray(A_trials)
print(f"Null accepted: {len(A_trials)}/{NTRIAL}")
if len(A_trials) < 50:
    print("⚠️ Low accepted count; if needed reduce strata counts or loosen tail/bin mins.")

n16 = np.quantile(A_trials, 0.16, axis=0)
n50 = np.quantile(A_trials, 0.50, axis=0)
n84 = np.quantile(A_trials, 0.84, axis=0)

print("\nPer-bin comparison:")
for i, x in enumerate(xc_obs):
    # one-sided p-value: null >= obs
    p = float(np.mean(A_trials[:, i] >= A_obs[i]))
    ratio = A_obs[i] / max(n50[i], 1e-30)
    dex = np.log10(max(ratio, 1e-30))
    print(f"  s~{int(x):6d} AU | Aobs={A_obs[i]:.3f} | null_med={n50[i]:.3f} [{n16[i]:.3f},{n84[i]:.3f}] "
          f"| Aobs/null={ratio:.2f} ({dex:+.3f} dex) | p≈{p:.3f}")


## Kill Test 2

# ============================================================
# KILL TEST 2 — ANGLE RANDOMISATION
#   Keep |Δμ| fixed per pair, randomise its direction.
#   This preserves speed distribution but removes phase/alignment structure.
# ============================================================

import numpy as np
from astropy.io import fits
import os

print("\n==============================")
print("KILL TEST 2 — ANGLE RANDOMISATION")
print("==============================")

CANDIDATES = [
    "/content/widebin/all_columns_catalog_shift.fits.gz",
    "/content/drive/MyDrive/widebin/all_columns_catalog_shift.fits.gz",
]
FITS_PATH = next((p for p in CANDIDATES if os.path.exists(p)), None)
if FITS_PATH is None:
    raise RuntimeError("Could not find wide-binary FITS in runtime or Drive.")
print(f"Using FITS_PATH: {FITS_PATH}")

AU_M  = 1.495978707e11
G     = 6.67430e-11
MSUN  = 1.98847e30

MAX_SEP_AU   = 5e4
RMAX         = 0.10
TAIL_Q       = 0.30
MIN_PER_BIN  = 30
MIN_TAIL_BIN = 8
SEP_BINS     = 6

NSEP_STRATA  = 6
NPLX_STRATA  = 5
NM_STRATA    = 5
MIN_STRATUM  = 20

NTRIAL = 300
SEED   = 2026
rng = np.random.default_rng(SEED)

def km_s_from_pm_parallax(pm_masyr, plx_mas):
    return 4.74047 * pm_masyr / np.maximum(plx_mas, 1e-12)

def abs_MG(Gmag, plx_mas):
    return Gmag + 5*np.log10(np.maximum(plx_mas,1e-6)/1000) + 5

MG_anc = np.array([3.5,5.0,7.0,9.5,11.5])
M_anc  = np.array([1.6,1.1,0.75,0.35,0.20])
def mass_from_MG(MG):
    return np.interp(np.clip(MG, MG_anc.min(), MG_anc.max()), MG_anc, M_anc)

def quant_edges(x, n):
    q = np.quantile(x, np.linspace(0,1,n+1))
    for i in range(1,len(q)):
        if q[i] <= q[i-1]:
            q[i] = q[i-1] + 1e-12
    return q

def make_strata_id(sep, plx, M):
    ls, lp, lm = np.log10(sep), np.log10(plx), np.log10(M)
    es = quant_edges(ls, NSEP_STRATA)
    ep = quant_edges(lp, NPLX_STRATA)
    em = quant_edges(lm, NM_STRATA)
    isep = np.clip(np.digitize(ls, es)-1, 0, NSEP_STRATA-1)
    iplx = np.clip(np.digitize(lp, ep)-1, 0, NPLX_STRATA-1)
    im   = np.clip(np.digitize(lm, em)-1, 0, NM_STRATA-1)
    return isep*(NPLX_STRATA*NM_STRATA) + iplx*NM_STRATA + im

def compute_A(sep, eta, v, sid, bins):
    slow = np.zeros(len(v), bool)
    fast = np.zeros(len(v), bool)
    for s in np.unique(sid):
        idx = np.where(sid == s)[0]
        if len(idx) < MIN_STRATUM:
            continue
        ql, qh = np.quantile(v[idx], [TAIL_Q, 1-TAIL_Q])
        slow[idx] = v[idx] <= ql
        fast[idx] = v[idx] >= qh

    xc, A = [], []
    b = np.digitize(sep, bins) - 1
    for i in range(len(bins)-1):
        m = b == i
        if np.sum(m) < MIN_PER_BIN:
            continue
        if np.sum(m & slow) < MIN_TAIL_BIN or np.sum(m & fast) < MIN_TAIL_BIN:
            continue
        qs = np.quantile(eta[m & slow], 0.1)
        qf = np.quantile(eta[m & fast], 0.1)
        if not np.isfinite(qs) or not np.isfinite(qf) or qf <= 0:
            continue
        xc.append(np.sqrt(bins[i]*bins[i+1]))
        A.append(qs/qf)
    return np.array(xc), np.array(A)

with fits.open(FITS_PATH, memmap=True) as hdul:
    t = hdul[1].data

sep = np.asarray(t["sep_AU"], float)
rch = np.asarray(t["R_chance_align"], float)

pmra1, pmdec1 = np.asarray(t["pmra1"], float), np.asarray(t["pmdec1"], float)
pmra2, pmdec2 = np.asarray(t["pmra2"], float), np.asarray(t["pmdec2"], float)

plx1, plx2 = np.asarray(t["parallax1"], float), np.asarray(t["parallax2"], float)
plx = 0.5*(plx1 + plx2)

dpm  = np.hypot(pmra1 - pmra2, pmdec1 - pmdec2)  # |Δμ|  (mas/yr)

MG1 = abs_MG(np.asarray(t["phot_g_mean_mag1"], float), plx1)
MG2 = abs_MG(np.asarray(t["phot_g_mean_mag2"], float), plx2)
Msys = mass_from_MG(MG1) + mass_from_MG(MG2)

m = (
    np.isfinite(sep) & np.isfinite(rch) & np.isfinite(dpm) & np.isfinite(plx) & np.isfinite(Msys) &
    (sep > 0) & (sep <= MAX_SEP_AU) &
    (rch < RMAX) & (plx > 0) &
    (Msys >= 0.8)
)
sep, dpm, plx, Msys = sep[m], dpm[m], plx[m], Msys[m]
print(f"Pairs used: {len(sep):,}")

# Observed vrel from |Δμ| and plx (direction doesn't matter here, but we treat this as OBS baseline)
vrel_obs = km_s_from_pm_parallax(dpm, plx)

vesc = np.sqrt(2 * G * (Msys*MSUN) / (sep*AU_M))  # m/s
eta_obs = (vrel_obs*1000.0) / np.maximum(vesc, 1e-30)

bins = np.logspace(np.log10(np.percentile(sep, 5)), np.log10(np.percentile(sep, 95)), SEP_BINS+1)
sid  = make_strata_id(sep, plx, Msys)

xc_obs, A_obs = compute_A(sep, eta_obs, vrel_obs, sid, bins)

# Angle randomisation: keep |Δμ| fixed, randomise components, then rebuild vrel (same magnitude!)
# This is a "geometry kill": if your signal came from preferred component structure, it should collapse.
A_trials = []
for _ in range(NTRIAL):
    th = rng.uniform(0, 2*np.pi, size=len(dpm))
    dpm_ra  = dpm*np.cos(th)
    dpm_dec = dpm*np.sin(th)
    dpm_new = np.hypot(dpm_ra, dpm_dec)  # == dpm
    vrel = km_s_from_pm_parallax(dpm_new, plx)
    eta  = (vrel*1000.0) / np.maximum(vesc, 1e-30)
    xc, A = compute_A(sep, eta, vrel, sid, bins)
    if len(xc) == len(xc_obs) and np.allclose(xc, xc_obs, atol=0, rtol=0):
        A_trials.append(A)

A_trials = np.asarray(A_trials)
print(f"Null accepted: {len(A_trials)}/{NTRIAL}")

n16 = np.quantile(A_trials, 0.16, axis=0)
n50 = np.quantile(A_trials, 0.50, axis=0)
n84 = np.quantile(A_trials, 0.84, axis=0)

print("\nPer-bin comparison (angle-rand null):")
for i, x in enumerate(xc_obs):
    p = float(np.mean(A_trials[:, i] >= A_obs[i]))
    ratio = A_obs[i] / max(n50[i], 1e-30)
    dex = np.log10(max(ratio, 1e-30))
    print(f"  s~{int(x):6d} AU | Aobs={A_obs[i]:.3f} | null_med={n50[i]:.3f} [{n16[i]:.3f},{n84[i]:.3f}] "
          f"| Aobs/null={ratio:.2f} ({dex:+.3f} dex) | p≈{p:.3f}")

