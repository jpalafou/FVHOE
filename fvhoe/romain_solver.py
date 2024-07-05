import numpy as np

gamma = 1.4


def prim_to_cons(w):
    u = 0.0 * w
    # density
    u[0] = w[0]
    # momentum
    u[1] = w[0] * w[1]
    # total energy
    u[2] = 0.5 * w[0] * w[1] ** 2 + w[2] / (gamma - 1)
    return u


def prim_to_flux(w):
    f = 0.0 * w
    # mass flux
    f[0] = w[0] * w[1]
    # momentum flux
    f[1] = w[0] * w[1] ** 2 + w[2]
    # total energy flux
    f[2] = (0.5 * w[0] * w[1] ** 2 + gamma * w[2] / (gamma - 1)) * w[1]
    return f


def cons_to_prim(u):
    w = 0.0 * u
    # density
    w[0] = u[0]
    # velocity
    w[1] = u[1] / u[0]
    # pressure
    w[2] = (gamma - 1) * (u[2] - 0.5 * w[0] * w[1] ** 2)
    return w


def set_ic(x, ic_type="sod test"):
    n = x.size
    d = np.zeros(n)
    v = np.zeros(n)
    p = np.zeros(n)
    if ic_type == "sod test":
        for i in range(0, n):
            if x[i] < 0.5:
                d[i] = 1
                v[i] = 0
                p[i] = 1
            else:
                d[i] = 0.125
                v[i] = 0
                p[i] = 0.1
    elif ic_type == "toro test1":
        for i in range(0, n):
            if x[i] < 0.3:
                d[i] = 1
                v[i] = 0.75
                p[i] = 1
            else:
                d[i] = 0.125
                v[i] = 0
                p[i] = 0.1
    elif ic_type == "toro test2":
        for i in range(0, n):
            if x[i] < 0.5:
                d[i] = 1
                v[i] = -2
                p[i] = 0.4
            else:
                d[i] = 1
                v[i] = 2
                p[i] = 0.4
    elif ic_type == "toro test3":
        for i in range(0, n):
            if x[i] < 0.5:
                d[i] = 1
                v[i] = 0
                p[i] = 1000
            else:
                d[i] = 1
                v[i] = 0
                p[i] = 0.01
    elif ic_type == "shu osher":
        for i in range(0, n):
            if x[i] < 0.1:
                d[i] = 3.857143
                v[i] = 2.629369
                p[i] = 10.33333
            else:
                d[i] = 1 + 0.2 * np.sin(5 * (x[i] * 10 - 5))
                v[i] = 0
                p[i] = 1
    elif ic_type == "supersonic blob":
        for i in range(0, n):
            if x[i] > 0.4 and x[i] < 0.6:
                d[i] = 10
                v[i] = 3 * np.sqrt(50 / 3)
                p[i] = 1
            else:
                d[i] = 0.1
                v[i] = 0
                p[i] = 1
    elif ic_type == "sedov":
        d[...] = 1
        v[...] = 0
        p[...] = 1e-5
        N = len(x)
        Pmax = 0.5 * N * (gamma - 1)
        p[N // 2] = Pmax
        p[N // 2 + 1] = Pmax

    else:
        print("Unknown IC type=", ic_type)
    # convert to conservative variables
    w = np.reshape([d, v, p], (3, n))
    u = prim_to_cons(w)
    return u


def set_bc(u, type="periodic"):
    if type == "periodic":
        u[:, 0] = u[:, -8]
        u[:, 1] = u[:, -7]
        u[:, 2] = u[:, -6]
        u[:, 3] = u[:, -5]
        u[:, -1] = u[:, 7]
        u[:, -2] = u[:, 6]
        u[:, -3] = u[:, 5]
        u[:, -4] = u[:, 4]
    elif type == "free":
        u[:, 0] = u[:, 4]
        u[:, 1] = u[:, 4]
        u[:, 2] = u[:, 4]
        u[:, 3] = u[:, 4]
        u[:, -1] = u[:, -5]
        u[:, -2] = u[:, -5]
        u[:, -3] = u[:, -5]
        u[:, -4] = u[:, -5]
    else:
        print("Unknown BC type")


def smooth_extrema(u, bc_type):
    # compute central first derivative
    du = 0.5 * (u[:, 2:] - u[:, :-2])
    uprime = np.pad(du, [(0, 0), (1, 1)])
    set_bc(uprime, bc_type)

    # compute left, right and central second derivative
    dlft = uprime[:, 1:-1] - uprime[:, :-2]
    drgt = uprime[:, 2:] - uprime[:, 1:-1]
    dmid = 0.5 * (dlft + drgt)

    # detect discontinuity on the left (alpha_left<1)
    with np.errstate(divide="ignore", invalid="ignore"):
        alfp = np.minimum(1, np.maximum(2 * dlft, 0) / dmid)
        alfm = np.minimum(1, np.minimum(2 * dlft, 0) / dmid)
    alfl = np.where(dmid > 0, alfp, alfm)
    alfl = np.where(dmid == 0, 1, alfl)

    # detect discontinuity on the right (alpha_right<1)
    with np.errstate(divide="ignore", invalid="ignore"):
        alfp = np.minimum(1, np.maximum(2 * drgt, 0) / dmid)
        alfm = np.minimum(1, np.minimum(2 * drgt, 0) / dmid)
    alfr = np.where(dmid > 0, alfp, alfm)
    alfr = np.where(dmid == 0, 1, alfr)

    # finalize smooth extrema marker (alpha=1)
    alf = np.minimum(alfl, alfr)
    alpha = np.pad(alf, [(0, 0), (1, 1)])
    set_bc(alpha, bc_type)

    return alpha


def trace(u, alpha, space=1):
    if space == 1:
        uleft = u[:, 3:-3]
        uright = u[:, 3:-3]

    if space == 2:
        uleft = (-u[:, 2:-4] + 4 * u[:, 3:-3] + u[:, 4:-2]) / 4
        uright = (u[:, 2:-4] + 4 * u[:, 3:-3] - u[:, 4:-2]) / 4

    if space == 3:
        uleft = (-u[:, 2:-4] + 5 * u[:, 3:-3] + 2 * u[:, 4:-2]) / 6
        uright = (2 * u[:, 2:-4] + 5 * u[:, 3:-3] - u[:, 4:-2]) / 6
        umiddle = (-u[:, 2:-4] + 26 * u[:, 3:-3] - u[:, 4:-2]) / 24

    if space == 4:
        uleft = (
            u[:, 1:-5] - 6 * u[:, 2:-4] + 20 * u[:, 3:-3] + 10 * u[:, 4:-2] - u[:, 5:-1]
        ) / 24
        uright = (
            -u[:, 1:-5]
            + 10 * u[:, 2:-4]
            + 20 * u[:, 3:-3]
            - 6 * u[:, 4:-2]
            + u[:, 5:-1]
        ) / 24
        umiddle = (-u[:, 2:-4] + 26 * u[:, 3:-3] - u[:, 4:-2]) / 24

    if space == 5:
        uleft = (
            2 * u[:, 1:-5]
            - 13 * u[:, 2:-4]
            + 47 * u[:, 3:-3]
            + 27 * u[:, 4:-2]
            - 3 * u[:, 5:-1]
        ) / 60
        uright = (
            -3 * u[:, 1:-5]
            + 27 * u[:, 2:-4]
            + 47 * u[:, 3:-3]
            - 13 * u[:, 4:-2]
            + 2 * u[:, 5:-1]
        ) / 60
        umiddle = (
            27 * u[:, 1:-5]
            - 348 * u[:, 2:-4]
            + 6402 * u[:, 3:-3]
            - 348 * u[:, 4:-2]
            + 27 * u[:, 5:-1]
        ) / 5760

    if space > 1:
        bigm = np.maximum(u[:, 2:-4], np.maximum(u[:, 3:-3], u[:, 4:-2]))
        smallm = np.minimum(u[:, 2:-4], np.minimum(u[:, 3:-3], u[:, 4:-2]))

        if space > 2:
            bigmj = np.maximum(umiddle, np.maximum(uleft, uright)) - u[:, 3:-3]
            smallmj = np.minimum(umiddle, np.minimum(uleft, uright)) - u[:, 3:-3]
        else:
            bigmj = np.maximum(uleft, uright) - u[:, 3:-3]
            smallmj = np.minimum(uleft, uright) - u[:, 3:-3]

        # compute limiter
        theta = np.minimum(
            1,
            np.minimum(
                abs(bigm - u[:, 3:-3]) / (abs(bigmj) + 1e-15),
                abs(smallm - u[:, 3:-3]) / (abs(smallmj) + 1e-15),
            ),
        )
        # apply smooth extrema detection
        aslp = np.minimum(np.minimum(alpha[:, 2:-4], alpha[:, 4:-2]), alpha[:, 3:-3])
        theta = np.where(aslp < 1, theta, 1)
        # apply limiter
        uleft = theta * (uleft - u[:, 3:-3]) + u[:, 3:-3]
        uright = theta * (uright - u[:, 3:-3]) + u[:, 3:-3]

    return uleft, uright


# define riemann solvers


def llf(wleft, wright):
    uleft = prim_to_cons(wleft)
    uright = prim_to_cons(wright)
    fleft = prim_to_flux(wleft)
    fright = prim_to_flux(wright)
    sleft = abs(wleft[1]) + np.sqrt(gamma * wleft[2] / wleft[0])
    sright = abs(wright[1]) + np.sqrt(gamma * wright[2] / wright[0])
    smax = np.maximum(sleft, sright)
    flux = 0.5 * (fleft + fright) - 0.5 * smax * (uright - uleft)
    return flux


def hllc(wleft, wright):
    flux = 0.0 * wleft
    uleft = prim_to_cons(wleft)
    uright = prim_to_cons(wright)
    # left state
    dl = wleft[0]
    vl = wleft[1]
    pl = wleft[2]
    el = uleft[2]
    # right state
    dr = wright[0]
    vr = wright[1]
    pr = wright[2]
    er = uright[2]
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
    flux[0] = dg * vg
    flux[1] = dg * vg * vg + pg
    flux[2] = (eg + pg) * vg
    return flux


# arrange riemann solvers into one flux function


def cmp_flux(u, bc_type="periodic", riemann_solver="llf", space=1):
    set_bc(u, bc_type)  # set boundary conditions

    w = cons_to_prim(u)  # convert to primitive variables

    cs = np.sqrt(gamma * w[2] / w[0])  # compute sound speed

    alpha = smooth_extrema(w, bc_type)  # compute smooth extrema detector

    wminus, wplus = trace(w, alpha, space=space)  # computed interface values

    # solve riemann problem

    wleft = wminus[:, :-1]  # left state for riemann problem
    wright = wplus[:, 1:]  # right state for riemann problem

    if riemann_solver == "llf":
        flux = llf(wleft, wright)

    if riemann_solver == "hll":
        raise NotImplementedError("hll")

    if riemann_solver == "hllc":
        flux = hllc(wleft, wright)

    if riemann_solver == "exact":
        raise NotImplementedError("exact")

    return flux, w, cs


def weno(
    tend=1,
    n=100,
    cfl=0.8,
    ic_type="sod test",
    bc_type="periodic",
    riemann_solver="llf",
    time=1,
    space=1,
):
    # set run parameters
    h = 1 / n
    nitermax = 100000
    print("cell=", n, " itermax=", nitermax)

    # set grid geometry
    xf = np.linspace(0, 1, n + 1)
    x = 0.5 * (xf[1:] + xf[:-1])

    # allocate permanent storage
    u = np.zeros([nitermax + 1, 3, n])

    # set initial conditions
    u[0] = set_ic(x, ic_type=ic_type)

    # allocate temporary workspace
    u1 = np.zeros([3, n + 8])
    u2 = np.zeros([3, n + 8])
    u3 = np.zeros([3, n + 8])
    u4 = np.zeros([3, n + 8])

    # init time and iteration counter
    t = 0
    niter = 1

    # main time loop
    while t < tend and niter <= nitermax:
        u1[:, 4:-4] = u[niter - 1]  # copy old solution

        flux, w, cs = cmp_flux(
            u1, bc_type=bc_type, riemann_solver=riemann_solver, space=space
        )
        k1 = -(flux[:, 1:] - flux[:, :-1]) / h
        dt = cfl * h / max(abs(w[1]) + cs)  # compute new time step

        if time == 1:  # forward Euler
            unew = u1[:, 4:-4] + k1 * dt

        if time == 2:  # RK2 or SSP RK2
            u2[:, 4:-4] = u1[:, 4:-4] + k1 * dt / 2
            #            u2[:,4:-4] = u1[:,4:-4] + k1 * dt # SSP
            flux, w, cs = cmp_flux(
                u2, bc_type=bc_type, riemann_solver=riemann_solver, space=space
            )
            k2 = -(flux[:, 1:] - flux[:, :-1]) / h
            unew = u1[:, 4:-4] + k2 * dt
        #            unew = u1[:,4:-4]/2 + (u2[:,4:-4] + k2 * dt)/2 # SSP

        if time == 3:  # RK3 or SSP RK3
            u2[:, 4:-4] = u1[:, 4:-4] + k1 * dt / 3
            #            u2[:,4:-4] = u1[:,4:-4] + k1 * dt # SSP
            flux, w, cs = cmp_flux(
                u2, bc_type=bc_type, riemann_solver=riemann_solver, space=space
            )
            k2 = -(flux[:, 1:] - flux[:, :-1]) / h
            u3[:, 4:-4] = u1[:, 4:-4] + k2 * 2 * dt / 3
            #            u3[:,4:-4] = 3/4*u1[:,4:-4] + 1/4*(u2:,[4:-4]+k2 * dt) # SSP
            flux, w, cs = cmp_flux(
                u3, bc_type=bc_type, riemann_solver=riemann_solver, space=space
            )
            k3 = -(flux[:, 1:] - flux[:, :-1]) / h
            unew = u1[:, 4:-4] + (k1 + 3 * k3) * dt / 4
        #            unew = 1/3*u1[:,4:-4] + 2/3*(u3[:,4:-4]+k3 * dt)  SSP

        if time == 4:  # RK4
            u2[:, 4:-4] = u1[:, 4:-4] + k1 * dt / 2
            flux, w, cs = cmp_flux(
                u2, bc_type=bc_type, riemann_solver=riemann_solver, space=space
            )
            k2 = -(flux[:, 1:] - flux[:, :-1]) / h
            u3[:, 4:-4] = u1[:, 4:-4] + k2 * dt / 2
            flux, w, cs = cmp_flux(
                u3, bc_type=bc_type, riemann_solver=riemann_solver, space=space
            )
            k3 = -(flux[:, 1:] - flux[:, :-1]) / h
            u4[:, 4:-4] = u1[:, 4:-4] + k3 * dt
            flux, w, cs = cmp_flux(
                u4, bc_type=bc_type, riemann_solver=riemann_solver, space=space
            )
            k4 = -(flux[:, 1:] - flux[:, :-1]) / h
            unew = u1[:, 4:-4] + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

        u[niter] = unew  # store new solution
        t = t + dt  # update time
        #        print(niter,t,dt)
        niter = niter + 1  # update iteration counter

    print("Done ", niter - 1, t)
    return u[0:niter]
