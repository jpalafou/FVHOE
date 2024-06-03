import numpy as np
from fvhoe.hydro import compute_conservatives, compute_fluxes, compute_sound_speed
from fvhoe.named_array import NamedNumpyArray


def advection_upwind(
    wl: NamedNumpyArray,
    wr: NamedNumpyArray,
    gamma: float,
    dim: str,
) -> NamedNumpyArray:
    """
    upwinding numerical fluxes, pressure is assumed to be 0
    args:
        wl (NamedArray) : primitive variables to the left of interface
        wr (NamedArray) : primitive variables to the right of interface
        gamma (float) : specific heat ratio
        dim (str) : "x", "y", "z"
    returns:
        out (NamedArray) : upwinding fluxes for conservative variables
    """
    # compute conservative variables
    ul = compute_conservatives(wl, gamma)
    ur = compute_conservatives(wr, gamma)

    # get hydro fluxes
    Fl = compute_fluxes(u=ul, w=wl, gamma=gamma, dim=dim, include_pressure=False)
    Fr = compute_fluxes(u=ur, w=wr, gamma=gamma, dim=dim, include_pressure=False)

    # assume velocity is continuous across interface
    v = getattr(wl, "v" + dim)[np.newaxis]  # velocity in dim-direction

    # upwind
    out = np.where(v > 0, Fl, np.where(v < 0, Fr, 0))
    out = ul.__class__(out, names=ul.variable_names)
    return out


def llf(
    wl: NamedNumpyArray, wr: NamedNumpyArray, gamma: float, dim: str
) -> NamedNumpyArray:
    """
    llf numerical fluxes
    Riemann Solvers and Numerical Methods for Fluid Dynamics by Toro
    Page 331
    args:
        wl (NamedArray) : primitive variables to the left of interface
        wr (array_like) : primitive variables to the right of interface
        gamma (float) : specific heat ratio
        dim (str) : "x", "y", "z"
    returns:
        out (NamedArray) : llf fluxes for conservative variables
    """

    # compute conservative variables
    ul = compute_conservatives(wl, gamma)
    ur = compute_conservatives(wr, gamma)

    # get hydro fluxes
    Fl = compute_fluxes(u=ul, w=wl, gamma=gamma, dim=dim)
    Fr = compute_fluxes(u=ur, w=wr, gamma=gamma, dim=dim)

    # get sound speeds
    sl = getattr(wl, "v" + dim) + compute_sound_speed(wl, gamma)
    sr = getattr(wr, "v" + dim) + compute_sound_speed(wr, gamma)
    smax = np.maximum(sl, sr)

    # llf
    out = 0.5 * (Fl + Fr) - 0.5 * smax * (ur - ul)
    out = ul.__class__(out, names=ul.variable_names)
    return out


def hllc(
    wl: NamedNumpyArray,
    wr: NamedNumpyArray,
    gamma: float,
    dim: str,
) -> NamedNumpyArray:
    """
    hllc numerical fluxes
    args:
        wl (NamedArray) : primitive variables to the left of interface
        wr (NamedArray) : primitive variables to the right of interface
        gamma (float) : specific heat ratio
        dir (str) : "x", "y", "z"
    returns:
        out (array_like) : hllc fluxes for conservative variables
    """

    # compute conservative variables
    ul = compute_conservatives(wl, gamma)
    ur = compute_conservatives(wr, gamma)

    # get hydro fluxes
    Fl = compute_fluxes(u=ul, w=wl, gamma=gamma, dim=dim)
    Fr = compute_fluxes(u=ur, w=wr, gamma=gamma, dim=dim)

    # sound speed
    cl = compute_sound_speed(wl, gamma)
    cr = compute_sound_speed(wr, gamma)

    # single out relevant velocities
    vl = getattr(wl, "v" + dim)
    vr = getattr(wr, "v" + dim)

    # pressure estimate
    rhobar = 0.5 * (wl.rho + wr.rho)
    cbar = 0.5 * (cl + cr)
    Ppvrs = 0.5 * (wl.P + wr.P) - 0.5 * (vr - vl) * rhobar * cbar
    Pstar = np.maximum(0, Ppvrs)

    # wave speed estimates
    ql = np.where(
        Pstar <= wl.P,
        1,
        np.sqrt(1 + ((gamma + 1) / (2 * gamma)) * (Pstar / wl.P - 1)),
    )
    qr = np.where(
        Pstar <= wr.P,
        1,
        np.sqrt(1 + ((gamma + 1) / (2 * gamma)) * (Pstar / wr.P - 1)),
    )
    Sl = vl - cl * ql
    Sr = vr + cr * qr
    Sstar_top = wr.P - wl.P + wl.rho * vl * (Sl - vl) - wr.rho * vr * (Sr - vr)
    Sstar_bottom = wl.rho * (Sl - vl) - wr.rho * (Sr - vr)
    Sstar = Sstar_top / Sstar_bottom

    # star conserved variables and fluxes
    ustarl = ul.copy()
    ustarl.rho[...] = 1
    ustarl.mx[...] = wl.vx if dim != "x" else Sstar
    ustarl.my[...] = wl.vy if dim != "y" else Sstar
    ustarl.mz[...] = wl.vz if dim != "z" else Sstar
    ustarl.E[...] = ul.E / wl.rho + (Sstar - vl) * (Sstar + wl.P / (wl.rho * (Sl - vl)))
    ustarl *= wl.rho * (Sl - vl) / (Sl - Sstar)
    ustarr = ur.copy()
    ustarr.rho[...] = 1
    ustarr.mx[...] = wr.vx if dim != "x" else Sstar
    ustarr.my[...] = wr.vy if dim != "y" else Sstar
    ustarr.mz[...] = wr.vz if dim != "z" else Sstar
    ustarr.E[...] = ur.E / wr.rho + (Sstar - vr) * (Sstar + wr.P / (wr.rho * (Sr - vr)))
    ustarr *= wr.rho * (Sr - vr) / (Sr - Sstar)
    Fstarl = Fl + Sl * (ustarl - ul)
    Fstarr = Fr + Sr * (ustarr - ur)

    # HLLC flux
    out = Fl
    out = np.where(np.logical_and(Sl <= 0, 0 <= Sstar), Fstarl, out)
    out = np.where(np.logical_and(Sstar <= 0, 0 <= Sr), Fstarr, out)
    out = np.where(0 >= Sr, Fr, out)

    return out
