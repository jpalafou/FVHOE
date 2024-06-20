import numpy as np
from fvhoe.hydro import compute_conservatives, compute_fluxes, compute_sound_speed
from fvhoe.named_array import NamedNumpyArray


def advection_upwind(
    wl: NamedNumpyArray,
    wr: NamedNumpyArray,
    gamma: float,
    dim: str,
    rho_P_sound_speed_floor: bool = False,
) -> NamedNumpyArray:
    """
    upwinding numerical fluxes, pressure is assumed to be 0
    args:
        wl (NamedArray) : primitive variables to the left of interface
        wr (NamedArray) : primitive variables to the right of interface
        gamma (float) : specific heat ratio
        dim (str) : "x", "y", "z"
        rho_P_sound_speed_floor (bool) : whether to apply a floor to density and pressure when computing sound speed
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
    wl: NamedNumpyArray,
    wr: NamedNumpyArray,
    gamma: float,
    dim: str,
    rho_P_sound_speed_floor: bool = False,
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
        rho_P_sound_speed_floor (bool) : whether to apply a floor to density and pressure when computing sound speed
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
    sl = np.abs(getattr(wl, "v" + dim)) + compute_sound_speed(
        wl, gamma, rho_P_floor=rho_P_sound_speed_floor
    )
    sr = np.abs(getattr(wr, "v" + dim)) + compute_sound_speed(
        wr, gamma, rho_P_floor=rho_P_sound_speed_floor
    )
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
    rho_P_sound_speed_floor: bool = False,
) -> NamedNumpyArray:
    """
    hllc numerical fluxes (David variation)
    args:
        wl (NamedArray) : primitive variables to the left of interface
        wr (NamedArray) : primitive variables to the right of interface
        gamma (float) : specific heat ratio
        dir (str) : "x", "y", "z"
        rho_P_sound_speed_floor (bool) : whether to apply a floor to density and pressure when computing sound speed
    returns:
        out (array_like) : hllc fluxes for conservative variables
    """

    # compute conservative variables
    ul = compute_conservatives(wl, gamma)
    ur = compute_conservatives(wr, gamma)

    # sound speed
    cl = compute_sound_speed(wl, gamma, rho_P_floor=rho_P_sound_speed_floor)
    cr = compute_sound_speed(wr, gamma, rho_P_floor=rho_P_sound_speed_floor)
    cmax = np.maximum(cl, cr)

    # single out relevant velocities
    vl = getattr(wl, "v" + dim)
    vr = getattr(wr, "v" + dim)

    # Compute HLL wave speed
    Sl = np.minimum(vl, vr) - cmax
    Sr = np.maximum(vl, vr) + cmax

    # Compute lagrangian sound speed
    rc_L = wl.rho * (vl - Sl)
    rc_R = wr.rho * (Sr - vr)

    # Compute acoustic star state
    v_star = (rc_L * vl + rc_R * vr + wl.P - wr.P) / (rc_L + rc_R)
    P_star = (rc_L * wr.P + rc_R * wl.P + rc_L * rc_R * (vl - vr)) / (rc_L + rc_R)

    # Left star region variables
    r_starL = wl.rho * (Sl - vl) / (Sl - v_star)
    E_starL = ((Sl - vl) * ul.E - wl.P * vl + P_star * v_star) / (Sl - v_star)

    # Right star region variables
    r_starR = wr.rho * (Sr - vr) / (Sr - v_star)
    E_starR = ((Sr - vr) * ur.E - wr.P * vr + P_star * v_star) / (Sr - v_star)

    # sample godunov state
    r_gdv = np.where(
        Sl > 0, wl.rho, np.where(v_star > 0, r_starL, np.where(Sr > 0, r_starR, wr.rho))
    )
    v_gdv = np.where(
        Sl > 0, vl, np.where(v_star > 0, v_star, np.where(Sr > 0, v_star, vr))
    )
    P_gdv = np.where(
        Sl > 0, wl.P, np.where(v_star > 0, P_star, np.where(Sr > 0, P_star, wr.P))
    )
    E_gdv = np.where(
        Sl > 0, ul.E, np.where(v_star > 0, E_starL, np.where(Sr > 0, E_starR, ur.E))
    )

    # HLLC flux
    out = ul.copy()
    out.rho = r_gdv * v_gdv
    out.mx = (
        r_gdv * v_gdv * v_gdv + P_gdv
        if dim == "x"
        else r_gdv * v_gdv * np.where(v_gdv > 0, wl.vx, np.where(v_gdv < 0, wr.vx, 0))
    )
    out.my = (
        r_gdv * v_gdv * v_gdv + P_gdv
        if dim == "y"
        else r_gdv * v_gdv * np.where(v_gdv > 0, wl.vy, np.where(v_gdv < 0, wr.vy, 0))
    )
    out.mz = (
        r_gdv * v_gdv * v_gdv + P_gdv
        if dim == "z"
        else r_gdv * v_gdv * np.where(v_gdv > 0, wl.vz, np.where(v_gdv < 0, wr.vz, 0))
    )
    out.E = v_gdv * (E_gdv + P_gdv)

    return out
