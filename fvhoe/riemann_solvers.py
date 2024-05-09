import numpy as np
from fvhoe.hydro import compute_conservatives, compute_fluxes, compute_sound_speed


def advection_upwind(
    wl: np.ndarray,
    wr: np.ndarray,
    gamma: float,
    dir: str,
) -> np.ndarray:
    """
    Riemann Solvers and Numerical Methods for Fluid Dynamics by Toro
    Page 331
    args:
        wl/r (array_like) : array of primitive variables to the left/right of the interface
            density, pressure, x-velocity, y-velocity, z-velocity
        gamma (float) : specific heat ratio
        dir (str) : "x", "y", "z"
    returns:
        out (array_like) : upwinding flux for constant advection
    """
    ul = compute_conservatives(wl, gamma)
    ur = compute_conservatives(wr, gamma)

    # single out velocities and momentums
    vslice = {"x": 2, "y": 3, "z": 4}[dir]
    v, _ = (
        wl[vslice, ...],
        wr[vslice, ...],
    )  # velocities (use left velocity since it should be constant in time and space, anyways)
    pl, pr = ul[vslice, ...], ur[vslice, ...]  # momentums

    # upwinding flux
    out = np.zeros_like(wl)
    out[0, ...] = np.where(v > 0, pl, np.where(v < 0, pr, 0))
    out[1, ...] = np.where(v > 0, v * ul[1, ...], np.where(v < 0, v * ur[1, ...], 0))
    out[vslice, ...] = np.where(v > 0, v * pl, np.where(v < 0, v * pr, 0))

    return out


def HLLC(
    wl: np.ndarray,
    wr: np.ndarray,
    gamma: float,
    dir: str,
) -> np.ndarray:
    """
    args:
        wl/r (array_like) : array of primitive variables to the left/right of the interface
            density, pressure, x-velocity, y-velocity, z-velocity
        gamma (float) : specific heat ratio
        dir (str) : "x", "y", "z"
    returns:
        out (array_like) : HLLC flux
    """
    ul = compute_conservatives(wl, gamma)
    ur = compute_conservatives(wr, gamma)

    # single out velocities and momentums
    vslice = {"x": 2, "y": 3, "z": 4}[dir]
    vl, vr = wl[vslice, ...], wr[vslice, ...]  # velocities
    pl, pr = ul[vslice, ...], ur[vslice, ...]  # momentums
    cl, cr = compute_sound_speed(wl, gamma), compute_sound_speed(
        wr, gamma
    )  # sound speeds

    # pressure estimate
    rho_bar = 0.5 * (wl[0, ...] + wl[0, ...])
    c_bar = 0.5 * (cl + cr)
    P_star = np.max(
        0, 0.5 * (wl[1, ...] + wr[1, ...]) - 0.5 * (vr - vl) * rho_bar * c_bar
    )

    # wave speed estimates
    ql = np.where(
        P_star <= wl[1, ...],
        1,
        np.sqrt(1 + ((gamma + 1) / (2 * gamma)) * (P_star / wl[1, ...] - 1)),
    )
    qr = np.where(
        P_star <= wr[1, ...],
        1,
        np.sqrt(1 + ((gamma + 1) / (2 * gamma)) * (P_star / wr[1, ...] - 1)),
    )
    Sl, Sr = vl - cl * ql, vr + cr * qr
    S_star_numerator = (
        wr[1, ...]
        - wl[1, ...]
        + wl[0, ...] * vl * (Sl - vl)
        - wr[0, ...] * vr * (Sr - vr)
    )
    S_star_denominator = wl[0, ...] * (Sl - vl) - wr[0, ...] * (Sr - vr)
    S_star = S_star_numerator / S_star_denominator

    # star conserved variables
    u_star_l = wl[0, ...] * ((Sl - vl) / (Sl - S_star)) * np.ones_like(wl)
    u_star_r = wr[0, ...] * ((Sr - vr) / (Sr - S_star)) * np.ones_like(wr)
    u_star_l[2, ...] *= wl[2, ...]
    u_star_r[2, ...] *= wr[2, ...]
    u_star_l[3, ...] *= wl[3, ...]
    u_star_r[3, ...] *= wr[3, ...]
    u_star_l[4, ...] *= wl[4, ...]
    u_star_r[4, ...] *= wr[4, ...]
    u_star_l_E = ul[1, ...] / wl[0, ...] + (S_star - vl) * (
        S_star + wl[1, ...] / (wl[0, ...] * (Sl - vl))
    )
    u_star_r_E = ur[1, ...] / wr[0, ...] + (S_star - vr) * (
        S_star + wr[1, ...] / (wr[0, ...] * (Sr - vr))
    )
    u_star_l[1, ...] *= u_star_l_E
    u_star_r[1, ...] *= u_star_r_E

    # HLLC flux
    Fl = compute_fluxes(u=ul, w=wl, gamma=gamma, dir=dir)
    Fr = compute_fluxes(u=ur, w=wr, gamma=gamma, dir=dir)
    F_star_l = Fl + Sl * (u_star_l - ul)
    F_star_r = Fr + Sr * (u_star_r - ur)
    out = Fl
    out = np.where(np.logical_and(Sl <= 0, 0 <= S_star), F_star_l, out)
    out = np.where(np.logical_and(S_star <= 0, 0 <= Sr), F_star_r, out)
    out = np.where(0 >= Sr, Fr, out)

    return out
