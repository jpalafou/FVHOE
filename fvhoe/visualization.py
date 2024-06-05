import numpy as np


def plot_1d_slice(
    solver,
    ax,
    param: str,
    t: float = None,
    x=None,
    y=None,
    z=None,
    verbose: bool = True,
    **kwargs,
) -> None:
    """
    plot a 1-dimensional slice by specifying t and two of three spatial dimensions x, y, and z
    args:
        solver (EulerSolver) : EulerSolver object
        ax (Axes) : Axes object
        param (str) : parameter to plot
        t (float) : time. nearest snapshot time is used if it is not in the list of snapshot times
        x (float) : x-coordinate. nearest x-coordinate is used if it is not in the list of x-coordinates
        y (float) : y-coordinate. nearest y-coordinate is used if it is not in the list of y-coordinates
        z (float) : z-coordinate. nearest z-coordinate is used if it is not in the list of z-coordinates
        verbose (bool) : print the exact used coordinates
        **kwargs : keyword arguments for matplotlib.pyplot.plot
    returns:
        None : modifies the input Axes object
    """
    if sum([x is None, y is None, z is None]) != 1:
        raise BaseException("One out of the three coordinates x-y-z must be None")
    t = max(solver.snapshot_times) if t is None else t
    n = np.argmin(np.abs(np.array(list(solver.snapshot_times)) - t))
    t = list(solver.snapshot_times)[n]
    if x is None:
        j, k = np.argmin(np.abs(solver.y - y)), np.argmin(np.abs(solver.z - z))
        y, z = solver.y[j], solver.z[k]
        x = solver.x
        x_for_plotting = solver.x
        y_for_plotting = getattr(solver.snapshots[n]["fv"], param)[:, j, k]
    elif y is None:
        i, k = np.argmin(np.abs(solver.x - x)), np.argmin(np.abs(solver.z - z))
        x, z = solver.x[i], solver.z[k]
        y = solver.y
        x_for_plotting = solver.y
        y_for_plotting = getattr(solver.snapshots[n]["fv"], param)[i, :, k]
    elif z is None:
        i, j = np.argmin(np.abs(solver.x - x)), np.argmin(np.abs(solver.y - y))
        x, y = solver.x[i], solver.y[j]
        z = solver.z
        x_for_plotting = solver.z
        y_for_plotting = getattr(solver.snapshots[n]["fv"], param)[i, j, :]
    if verbose:
        t_message = f"{t:.2f}"
        x_message = (
            f"{x:.2f}"
            if (isinstance(x, int) or isinstance(x, float))
            else f"[{x[0]:.2f}, {x[-1]:.2f}]"
        )
        y_message = (
            f"{y:.2f}"
            if (isinstance(y, int) or isinstance(y, float))
            else f"[{y[0]:.2f}, {y[-1]:.2f}]"
        )
        z_message = (
            f"{z:.2f}"
            if (isinstance(z, int) or isinstance(z, float))
            else f"[{z[0]:.2f}, {z[-1]:.2f}]"
        )
        print(f"t={t_message}, x={x_message}, y={y_message}, z={z_message}")
    return ax.plot(x_for_plotting, y_for_plotting, **kwargs)


def plot_2d_slice(
    solver,
    ax,
    param: str,
    t: float = None,
    x=None,
    y=None,
    z=None,
    verbose: bool = True,
    **kwargs,
) -> None:
    """
    plot a 2-dimensional slice by specifying t and one of three spatial dimensions x, y, and z
    args:
        solver (EulerSolver) : EulerSolver object
        ax (Axes) : Axes object
        param (str) : parameter to plot
        t (float) : time. nearest snapshot time is used if it is not in the list of snapshot times
        x (float) : x-coordinate. nearest x-coordinate is used if it is not in the list of x-coordinates
        y (float) : y-coordinate. nearest y-coordinate is used if it is not in the list of y-coordinates
        z (float) : z-coordinate. nearest z-coordinate is used if it is not in the list of z-coordinates
        verbose (bool) : print the exact used coordinates
        **kwargs : keyword arguments for matplotlib.pyplot.imshow
    returns:
        None : modifies the input Axes object
    """
    if sum([x is None, y is None, z is None]) != 2:
        raise BaseException("Two out of the three coordinates x-y-z must be None")
    t = max(solver.snapshot_times) if t is None else t
    n = np.argmin(np.abs(np.array(list(solver.snapshot_times)) - t))
    t = list(solver.snapshot_times)[n]
    if x is None and y is None:
        k = np.argmin(np.abs(solver.z - z))
        z = solver.z[k]
        x, y = solver.x, solver.y
        z_for_plotting = getattr(solver.snapshots[n]["fv"], param)[:, :, k]
        z_for_plotting = np.rot90(z_for_plotting, 1)
        horizontal_axis, vertical_axis = "x", "y"
        limits = (x[0], x[-1], y[0], y[-1])
    elif y is None and z is None:
        i = np.argmin(np.abs(solver.x - x))
        x = solver.x[i]
        y, z = solver.y, solver.z
        z_for_plotting = getattr(solver.snapshots[n]["fv"], param)[i, :, :]
        z_for_plotting = np.rot90(z_for_plotting, 1)
        horizontal_axis, vertical_axis = "y", "z"
        limits = (y[0], y[-1], z[0], z[-1])
    elif x is None and z is None:
        j = np.argmin(np.abs(solver.y - y))
        y = solver.y[j]
        z, x = solver.z, solver.x
        z_for_plotting = getattr(solver.snapshots[n]["fv"], param)[:, j, :]
        z_for_plotting = np.rot90(z_for_plotting, 1)
        horizontal_axis, vertical_axis = "x", "z"
        limits = (x[0], x[-1], z[0], z[-1])
    if verbose:
        t_message = f"{t:.2f}"
        x_message = (
            f"{x:.2f}"
            if (isinstance(x, int) or isinstance(x, float))
            else f"[{x[0]:.2f}, {x[-1]:.2f}]"
        )
        y_message = (
            f"{y:.2f}"
            if (isinstance(y, int) or isinstance(y, float))
            else f"[{y[0]:.2f}, {y[-1]:.2f}]"
        )
        z_message = (
            f"{z:.2f}"
            if (isinstance(z, int) or isinstance(z, float))
            else f"[{z[0]:.2f}, {z[-1]:.2f}]"
        )
        print(f"t={t_message}, x={x_message}, y={y_message}, z={z_message}")
        print(f"{horizontal_axis=}, {vertical_axis=}")
    return ax.imshow(z_for_plotting, extent=limits, **kwargs)
