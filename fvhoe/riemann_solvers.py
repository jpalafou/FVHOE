from fvhoe.hydro import (
    compute_conservatives,
    compute_fluxes,
    compute_sound_speed,
    HydroState,
)
import numpy as np
from typing import Tuple

_hs = HydroState(ndim=1)


def advection_upwind(
    hs: HydroState,
    riemann_problem: Tuple[np.ndarray, np.ndarray],
    gamma: float,
    dim: str,
    csq_floor: float,
) -> np.ndarray:
    """
    upwinding numerical fluxes, pressure is assumed to be 0
    args:
        hs (HydroState) : HydroState object
        riemann_problem (Tuple[array_like, array_like]) : primitive variables to the left and right of interface
        gamma (float) : specific heat ratio
        dim (str) : "x", "y", or "z"
        csq_floor (float) : floor on square of returned sound speed
    returns:
        out (array_like) : upwinding fluxes for conservative variables
    """
    # compute conservative variables for left and right states
    wl, wr = riemann_problem
    ul = compute_conservatives(hs, wl, gamma)
    ur = compute_conservatives(hs, wr, gamma)

    # get hydro fluxes
    Fl = compute_fluxes(hs=hs, w=wl, u=ul, gamma=gamma, dim=dim, include_pressure=False)
    Fr = compute_fluxes(hs=hs, w=wr, u=ur, gamma=gamma, dim=dim, include_pressure=False)

    # assume velocity is continuous across interface
    v = wl[_hs("v" + dim)][np.newaxis]  # velocity in dim-direction

    # upwind
    out = np.where(v > 0, Fl, np.where(v < 0, Fr, 0))
    return out


def llf(
    hs: HydroState,
    riemann_problem: Tuple[np.ndarray, np.ndarray],
    gamma: float,
    dim: str,
    csq_floor: float,
) -> np.ndarray:
    """
    llf numerical fluxes
    Riemann Solvers and Numerical Methods for Fluid Dynamics by Toro
    Page 331
    args:
        hs (HydroState) : HydroState object
        riemann_problem (Tuple[array_like, array_like]) : primitive variables to the left and right of interface
        gamma (float) : specific heat ratio
        dim (str) : "x", "y", or "z"
        csq_floor (float) : floor on square of returned sound speed
    returns:
        out (array_like) : llf fluxes for conservative variables
    """
    # compute conservative variables for left and right states
    wl, wr = riemann_problem
    ul = compute_conservatives(hs, wl, gamma)
    ur = compute_conservatives(hs, wr, gamma)

    # get hydro fluxes
    Fl = compute_fluxes(hs=hs, w=wl, u=ul, gamma=gamma, dim=dim)
    Fr = compute_fluxes(hs=hs, w=wr, u=ur, gamma=gamma, dim=dim)

    # get sound speeds
    sl = np.abs(wl[_hs("v" + dim)]) + compute_sound_speed(
        wl, gamma, csq_floor=csq_floor
    )
    sr = np.abs(wr[_hs("v" + dim)]) + compute_sound_speed(
        wr, gamma, csq_floor=csq_floor
    )
    smax = np.maximum(sl, sr)

    # llf
    out = 0.5 * (Fl + Fr) - 0.5 * smax * (ur - ul)
    return out


def hllc(
    hs: HydroState,
    riemann_problem: Tuple[np.ndarray, np.ndarray],
    gamma: float,
    dim: str,
    csq_floor: float,
) -> np.ndarray:
    """
    hllc numerical fluxes (David variation)
    args:
        hs (HydroState) : HydroState object
        riemann_problem (Tuple[array_like, array_like]) : primitive variables to the left and right of interface
        gamma (float) : specific heat ratio
        dim (str) : "x", "y", or "z"
        csq_floor (float) : floor on square of returned sound speed
    returns:
        out (array_like) : hllc fluxes for conservative variables
    """
    # compute conservative variables for left and right states
    wl, wr = riemann_problem
    ul = compute_conservatives(hs, wl, gamma)
    ur = compute_conservatives(hs, wr, gamma)

    # sound speed
    cl = compute_sound_speed(wl, gamma, csq_floor=csq_floor)
    cr = compute_sound_speed(wr, gamma, csq_floor=csq_floor)
    cmax = np.maximum(cl, cr)

    # single out relevant quantities
    vl, vr = wl[_hs("v" + dim)], wr[_hs("v" + dim)]
    rhol, rhor = wl[_hs("rho")], wr[_hs("rho")]
    Pl, Pr = wl[_hs("P")], wr[_hs("P")]
    El, Er = ul[_hs("E")], ur[_hs("E")]

    # Compute HLL wave speed
    Sl = np.minimum(vl, vr) - cmax
    Sr = np.maximum(vl, vr) + cmax

    # Compute lagrangian sound speed
    rc_L = rhol * (vl - Sl)
    rc_R = rhor * (Sr - vr)

    # Compute acoustic star state
    v_star = (rc_L * vl + rc_R * vr + Pl - Pr) / (rc_L + rc_R)
    P_star = (rc_L * Pr + rc_R * Pl + rc_L * rc_R * (vl - vr)) / (rc_L + rc_R)

    # Left star region variables
    r_starL = rhol * (Sl - vl) / (Sl - v_star)
    E_starL = ((Sl - vl) * El - Pl * vl + P_star * v_star) / (Sl - v_star)

    # Right star region variables
    r_starR = rhor * (Sr - vr) / (Sr - v_star)
    E_starR = ((Sr - vr) * Er - Pr * vr + P_star * v_star) / (Sr - v_star)

    # sample godunov state
    r_gdv = np.where(
        Sl > 0, rhol, np.where(v_star > 0, r_starL, np.where(Sr > 0, r_starR, rhor))
    )
    v_gdv = np.where(
        Sl > 0, vl, np.where(v_star > 0, v_star, np.where(Sr > 0, v_star, vr))
    )
    P_gdv = np.where(
        Sl > 0, Pl, np.where(v_star > 0, P_star, np.where(Sr > 0, P_star, Pr))
    )
    E_gdv = np.where(
        Sl > 0, El, np.where(v_star > 0, E_starL, np.where(Sr > 0, E_starR, Er))
    )

    # HLLC flux
    out = np.empty_like(ul)
    out[hs("rho")] = r_gdv * v_gdv
    out[hs("mx")] = (
        r_gdv * v_gdv * v_gdv + P_gdv
        if dim == "x"
        else r_gdv
        * v_gdv
        * np.where(v_gdv > 0, wl[_hs("vx")], np.where(v_gdv < 0, wr[_hs("vx")], 0))
    )
    out[hs("my")] = (
        r_gdv * v_gdv * v_gdv + P_gdv
        if dim == "y"
        else r_gdv
        * v_gdv
        * np.where(v_gdv > 0, wl[_hs("vy")], np.where(v_gdv < 0, wr[_hs("vy")], 0))
    )
    out[hs("mz")] = (
        r_gdv * v_gdv * v_gdv + P_gdv
        if dim == "z"
        else r_gdv
        * v_gdv
        * np.where(v_gdv > 0, wl[_hs("vz")], np.where(v_gdv < 0, wr[_hs("vz")], 0))
    )
    out[hs("E")] = v_gdv * (E_gdv + P_gdv)

    # handle passive scalars
    if hs.includes_passives:
        out[hs("passive_scalars")] = (
            r_gdv
            * v_gdv
            * np.where(
                v_gdv > 0,
                wl[hs("passive_scalars")],
                np.where(v_gdv < 0, wr[hs("passive_scalars")], 0),
            )
        )

    return out
