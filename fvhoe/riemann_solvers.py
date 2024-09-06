from fvhoe.array_manager import get_array_slice as slc
from fvhoe.hydro import compute_conservatives, compute_fluxes, compute_sound_speed
import numpy as np


def advection_upwind(
    wl: np.ndarray,
    wr: np.ndarray,
    gamma: float,
    dim: str,
    csq_floor: float,
) -> np.ndarray:
    """
    upwinding numerical fluxes, pressure is assumed to be 0
    args:
        wl (array_like) : primitive variables to the left of interface
        wr (array_like) : primitive variables to the right of interface
        gamma (float) : specific heat ratio
        dim (str) : "x", "y", "z"
        csq_floor (float) : floor on square of returned sound speed
    returns:
        out (array_like) : upwinding fluxes for conservative variables
    """
    # compute conservative variables
    ul = compute_conservatives(wl, gamma)
    ur = compute_conservatives(wr, gamma)

    # get hydro fluxes
    Fl = compute_fluxes(u=ul, w=wl, gamma=gamma, dim=dim, include_pressure=False)
    Fr = compute_fluxes(u=ur, w=wr, gamma=gamma, dim=dim, include_pressure=False)

    # assume velocity is continuous across interface
    v = wl[slc("v" + dim)][np.newaxis]  # velocity in dim-direction

    # upwind
    out = np.where(v > 0, Fl, np.where(v < 0, Fr, 0))
    return out


def llf(
    wl: np.ndarray,
    wr: np.ndarray,
    gamma: float,
    dim: str,
    csq_floor: float,
) -> np.ndarray:
    """
    llf numerical fluxes
    Riemann Solvers and Numerical Methods for Fluid Dynamics by Toro
    Page 331
    args:
        wl (array_like) : primitive variables to the left of interface
        wr (array_like) : primitive variables to the right of interface
        gamma (float) : specific heat ratio
        dim (str) : "x", "y", "z"
        csq_floor (float) : floor on square of returned sound speed
    returns:
        out (array_like) : llf fluxes for conservative variables
    """

    # compute conservative variables
    ul = compute_conservatives(wl, gamma)
    ur = compute_conservatives(wr, gamma)

    # get hydro fluxes
    Fl = compute_fluxes(u=ul, w=wl, gamma=gamma, dim=dim)
    Fr = compute_fluxes(u=ur, w=wr, gamma=gamma, dim=dim)

    # get sound speeds
    sl = np.abs(wl[slc("v" + dim)]) + compute_sound_speed(
        wl, gamma, csq_floor=csq_floor
    )
    sr = np.abs(wr[slc("v" + dim)]) + compute_sound_speed(
        wr, gamma, csq_floor=csq_floor
    )
    smax = np.maximum(sl, sr)

    # llf
    out = 0.5 * (Fl + Fr) - 0.5 * smax * (ur - ul)
    return out


def hllc(
    wl: np.ndarray,
    wr: np.ndarray,
    gamma: float,
    dim: str,
    csq_floor: float,
) -> np.ndarray:
    """
    hllc numerical fluxes (David variation)
    args:
        wl (array_like) : primitive variables to the left of interface
        wr (array_like) : primitive variables to the right of interface
        gamma (float) : specific heat ratio
        dir (str) : "x", "y", "z"
        csq_floor (float) : floor on square of returned sound speed
    returns:
        out (array_like) : hllc fluxes for conservative variables
    """

    # compute conservative variables
    ul = compute_conservatives(wl, gamma)
    ur = compute_conservatives(wr, gamma)

    # sound speed
    cl = compute_sound_speed(wl, gamma, csq_floor=csq_floor)
    cr = compute_sound_speed(wr, gamma, csq_floor=csq_floor)
    cmax = np.maximum(cl, cr)

    # single out relevant quantities
    vl, vr = wl[slc("v" + dim)], wr[slc("v" + dim)]
    rhol, rhor = wl[slc("rho")], wr[slc("rho")]
    Pl, Pr = wl[slc("P")], wr[slc("P")]
    El, Er = ul[slc("E")], ur[slc("E")]

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
    out[slc("rho")] = r_gdv * v_gdv
    out[slc("mx")] = (
        r_gdv * v_gdv * v_gdv + P_gdv
        if dim == "x"
        else r_gdv
        * v_gdv
        * np.where(v_gdv > 0, wl[slc("vx")], np.where(v_gdv < 0, wr[slc("vx")], 0))
    )
    out[slc("my")] = (
        r_gdv * v_gdv * v_gdv + P_gdv
        if dim == "y"
        else r_gdv
        * v_gdv
        * np.where(v_gdv > 0, wl[slc("vy")], np.where(v_gdv < 0, wr[slc("vy")], 0))
    )
    out[slc("mz")] = (
        r_gdv * v_gdv * v_gdv + P_gdv
        if dim == "z"
        else r_gdv
        * v_gdv
        * np.where(v_gdv > 0, wl[slc("vz")], np.where(v_gdv < 0, wr[slc("vz")], 0))
    )
    out[slc("E")] = v_gdv * (E_gdv + P_gdv)
    return out
