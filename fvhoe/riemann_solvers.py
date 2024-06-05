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
    sl = np.abs(getattr(wl, "v" + dim)) + compute_sound_speed(wl, gamma)
    sr = np.abs(getattr(wr, "v" + dim)) + compute_sound_speed(wr, gamma)
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


def hllc2(
    wl: NamedNumpyArray, wr: NamedNumpyArray, gamma: float, dim: str
) -> NamedNumpyArray:
    """
    hllc numerical fluxes (Romain variation)
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

    # sound speed
    cl = compute_sound_speed(wl, gamma)
    cr = compute_sound_speed(wr, gamma)

    # left state
    dl = wl.rho
    vl = getattr(wl, "v" + dim)
    pl = wl.P
    el = ul.E
    # right state
    dr = wr.rho
    vr = getattr(wr, "v" + dim)
    pr = wr.P
    er = ur.E
    # sound speed
    cl = np.sqrt(gamma * pl / dl)
    cr = np.sqrt(gamma * pr / dr)
    # waves speed
    sl = np.minimum(vl, vr) - np.maximum(cl, cr)
    sr = np.maximum(vl, vr) + np.maximum(cl, cr)
    dcl = dl * (vl - sl)
    dcr = dr * (sr - vr)
    # star state velocity and pressure
    vstar = (dcl * vl + dcr * vr + pl - pr) / (dcl + dcr)
    pstar = (dcl * pr + dcr * pl + dcl * dcr * (vl - vr)) / (dcl + dcr)
    # left and right star states
    dstarl = dl * (sl - vl) / (sl - vstar)
    dstarr = dr * (sr - vr) / (sr - vstar)
    estarl = ((sl - vl) * el - pl * vl + pstar * vstar) / (sl - vstar)
    estarr = ((sr - vr) * er - pr * vr + pstar * vstar) / (sr - vstar)
    # sample godunov state
    dg = np.where(sl > 0, dl, np.where(vstar > 0, dstarl, np.where(sr > 0, dstarr, dr)))
    vg = np.where(sl > 0, vl, np.where(vstar > 0, vstar, np.where(sr > 0, vstar, vr)))
    pg = np.where(sl > 0, pl, np.where(vstar > 0, pstar, np.where(sr > 0, pstar, pr)))
    eg = np.where(sl > 0, el, np.where(vstar > 0, estarl, np.where(sr > 0, estarr, er)))
    # compute godunov flux
    out = ul.copy()
    out.rho[...] = dg * vg
    out.mx[...] = (
        dg * vg * vg + pg
        if dim == "x"
        else dg * vg * np.where(vg > 0, wl.vx, np.where(vg < 0, wr.vx, 0))
    )
    out.my[...] = (
        dg * vg * vg + pg
        if dim == "y"
        else dg * vg * np.where(vg > 0, wl.vy, np.where(vg < 0, wr.vy, 0))
    )
    out.mz[...] = (
        dg * vg * vg + pg
        if dim == "z"
        else dg * vg * np.where(vg > 0, wl.vz, np.where(vg < 0, wr.vz, 0))
    )
    out.E[...] = (eg + pg) * vg
    return out
