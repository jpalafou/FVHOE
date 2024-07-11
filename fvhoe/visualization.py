from matplotlib import cm
from matplotlib.colors import LogNorm
import numpy as np
from typing import Tuple


def get_snapshot_list(s) -> list:
    """
    args:
        s : snapshot list or solver
    returns:
        snapshot list
    """
    if isinstance(s, list):
        return s
    elif hasattr(s, "snapshots"):
        # solver is an EulerSolver object
        return s.snapshots
    else:
        raise ValueError("s must be a snapshot list or a solver instance")


def get_indices_from_coordinates(
    snapshots: list,
    t: float = None,
    x: float = None,
    y: float = None,
    z: float = None,
) -> Tuple[int, tuple]:
    """
    get the indices of the nearest x, y, z, t values in the snapshots
    args:
        snapshots (list) : list of snapshots, each a dictionary with keys "x", "y", "z", "t", "w"
        t (float) : time
        x (float) : x-coordinate or None for slice of entire dimension
        y (float) : y-coordinate or None for slice of entire dimension
        z (float) : z-coordinate or None for slice of entire dimension
    returns:
        n (int) : snapshot index nearest to t
        (i, j, k) (tuple) : array slices nearest to x, y, z
    """
    if t is None:
        n = -1
    else:
        tarr = np.array([snapshots[i]["t"] for i in range(len(snapshots))])
        n = np.argmin(np.abs(tarr - t))
    if x is None:
        i = slice(None)
    else:
        i = int(np.argmin(np.abs(snapshots[n]["x"] - x)))
    if y is None:
        j = slice(None)
    else:
        j = int(np.argmin(np.abs(snapshots[n]["y"] - y)))
    if z is None:
        k = slice(None)
    else:
        k = int(np.argmin(np.abs(snapshots[n]["z"] - z)))
    return n, (i, j, k)


def xyzt_summary(
    snapshots: list,
    n: int = None,
    i: int = None,
    j: int = None,
    k: int = None,
) -> str:
    """
    get a summary of the nearest x, y, z, t values
    args:
        snapshots (list) : list of snapshots
        n (int) : snapshot index
        i (int) : array slice in x
        j (int) : array slice in y
        z (int) : array slice in z
    returns:
        out (str) : summary of the nearest t, x, y, z values
    """
    x = snapshots[n]["x"][i]
    y = snapshots[n]["y"][j]
    z = snapshots[n]["z"][k]
    tstr = str(snapshots[n]["t"])
    xstr = str(x) if isinstance(i, int) else f"[{x[0]}, {x[-1]}]"
    ystr = str(y) if isinstance(j, int) else f"[{y[0]}, {y[-1]}]"
    zstr = str(z) if isinstance(k, int) else f"[{z[0]}, {z[-1]}]"
    out = f"t={tstr}, x={xstr}, y={ystr}, z={zstr}"
    return out


def get_velocity_magnitude(w) -> np.ndarray:
    """
    args:
        w (NamedArray) : primitive variables
    returns:
        out (array_like) : velocity magnitude in 3D
    """
    out = np.sqrt(np.square(w.vx) + np.square(w.vy) + np.square(w.vz))
    return out


def plot_1d_slice(
    solver,
    ax,
    param: str,
    t: float = None,
    x: float = None,
    y: float = None,
    z: float = None,
    tol: float = 0,
    verbose: bool = True,
    **kwargs,
) -> None:
    """
    plot a 1-dimensional slice by specifying t and two of three spatial dimensions x, y, and z
    args:
        solver (EulerSolver) : EulerSolver object or snapshot list
        ax (Axes) : Axes object
        param (str) : parameter to plot
        t (float) : time. nearest snapshot time is used if it is not in the list of snapshot times
        x (float) : x-coordinate. nearest x-coordinate is used if it is not in the list of x-coordinates
        y (float) : y-coordinate. nearest y-coordinate is used if it is not in the list of y-coordinates
        z (float) : z-coordinate. nearest z-coordinate is used if it is not in the list of z-coordinates
        tol (float) : tolerance for trouble based on magnitude of NAD violation
        verbose (bool) : print the exact used coordinates
        **kwargs : keyword arguments for matplotlib.pyplot.plot
    returns:
        None : modifies the input Axes object
    """
    snapshots = get_snapshot_list(solver)
    n, slices = get_indices_from_coordinates(snapshots, t, x, y, z)

    # x-data
    if x is None:
        x_for_plotting = snapshots[n]["x"]
    elif y is None:
        x_for_plotting = snapshots[n]["y"]
    elif z is None:
        x_for_plotting = snapshots[n]["z"]
    else:
        raise BaseException("One out of the three coordinates x-y-z must be None")
    # y-data
    if param == "trouble":
        trouble = snapshots[n]["trouble"]
        NAD_mag = snapshots[n]["NAD violation magnitude"]
        source_array = np.where(NAD_mag > tol, trouble, 0)
    elif param == "v":
        source_array = get_velocity_magnitude(snapshots[n]["w"])
    else:
        source_array = getattr(snapshots[n]["w"], param)
    y_for_plotting = source_array[slices]
    # print summary
    if verbose:
        print(xyzt_summary(snapshots, n, *slices))
    # return plot
    out = ax.plot(x_for_plotting, y_for_plotting, **kwargs)
    return out


def plot_2d_slice(
    solver,
    ax,
    param: str,
    t: float = None,
    x: float = None,
    y: float = None,
    z: float = None,
    overlay_trouble: bool = False,
    tol: float = 0,
    cmap: str = "GnBu_r",
    trouble_color: str = "red",
    verbose: bool = True,
    log: bool = False,
    log_vmin: float = None,
    log_vmax: float = None,
    contour: bool = False,
    levels: int = 30,
    **kwargs,
) -> None:
    """
    plot a 2-dimensional slice by specifying t and one of three spatial dimensions x, y, and z
    args:
        solver (EulerSolver) : EulerSolver object or snapshot list
        ax (Axes) : Axes object
        param (str) : parameter to plot
        t (float) : time. nearest snapshot time is used if it is not in the list of snapshot times
        x (float) : x-coordinate. nearest x-coordinate is used if it is not in the list of x-coordinates
        y (float) : y-coordinate. nearest y-coordinate is used if it is not in the list of y-coordinates
        z (float) : z-coordinate. nearest z-coordinate is used if it is not in the list of z-coordinates
        overlay_trouble (bool) : overlay trouble on the plot
        tol (float) : tolerance for trouble based on magnitude of NAD violation
        cmap (str) : colormap
        trouble_color (str) : color for trouble
        verbose (bool) : print the exact used coordinates
        log (bool) : whether to use a log-scale color bar
        log_vmin (float) : minimum value for log-scale color bar
        log_vmax (float) : maximum value for log-scale color bar
        contour (bool) : whether to plot contour lines
        levels (int, list, or array_like) : number of contour levels
        **kwargs : keyword arguments for matplotlib.pyplot.imshow
    returns:
        None : modifies the input Axes object
    """
    snapshots = get_snapshot_list(solver)
    n, slices = get_indices_from_coordinates(snapshots, t, x, y, z)

    # xy data
    if (x, y) == (None, None):
        horizontal_axis, vertical_axis = "x", "y"
        x_for_plotting, y_for_plotting = snapshots[n]["x"], snapshots[n]["y"]
    elif (y, z) == (None, None):
        horizontal_axis, vertical_axis = "y", "z"
        x_for_plotting, y_for_plotting = snapshots[n]["y"], snapshots[n]["z"]
    elif (x, z) == (None, None):
        horizontal_axis, vertical_axis = "x", "z"
        x_for_plotting, y_for_plotting = snapshots[n]["x"], snapshots[n]["z"]
    else:
        raise BaseException("Two out of the three coordinates x-y-z must be None")
    limits = (
        x_for_plotting[0],
        x_for_plotting[-1],
        y_for_plotting[0],
        y_for_plotting[-1],
    )
    # get z data
    if param == "trouble" or overlay_trouble:
        trouble = snapshots[n]["trouble"][slices]
        NAD_mag = snapshots[n]["NAD violation magnitude"][slices]
        trouble_for_plotting = np.where(NAD_mag > tol, trouble, 0)
    if param == "trouble":
        source_array = snapshots[n]["trouble"]
    elif param == "v":
        source_array = get_velocity_magnitude(snapshots[n]["w"])
    else:
        source_array = getattr(snapshots[n]["w"], param)
    z_for_plotting = source_array[slices]
    # rotate
    X, Y = np.meshgrid(x_for_plotting, y_for_plotting, indexing="ij")
    X = np.rot90(X, 1)
    Y = np.rot90(Y, 1)
    z_for_plotting = np.rot90(z_for_plotting, 1)
    if overlay_trouble:
        trouble_for_plotting = np.rot90(trouble_for_plotting, 1)
    # define colormap
    colormap = getattr(cm, cmap)
    if overlay_trouble:
        z_for_plotting = np.where(trouble_for_plotting > 0, np.nan, z_for_plotting)
        colormap.set_bad(color=trouble_color)
    # print summary
    if verbose:
        print(xyzt_summary(snapshots, n, *slices))
        print(f"{horizontal_axis=}, {vertical_axis=}")
    # return plot
    if contour:
        if isinstance(levels, int):
            contour_levels = np.linspace(
                z_for_plotting.min(), z_for_plotting.max(), levels
            )
        elif isinstance(levels, list) or isinstance(levels, np.ndarray):
            contour_levels = np.asarray(levels)
        else:
            raise BaseException("levels must be an integer, list, or numpy array")
        out = ax.contour(X, Y, z_for_plotting, levels=contour_levels, **kwargs)
    else:
        norm = LogNorm(vmin=log_vmin, vmax=log_vmax) if log else None
        out = ax.imshow(
            z_for_plotting, norm=norm, extent=limits, cmap=colormap, **kwargs
        )
    return out


def sample_circular_average(
    solver,
    param: str,
    t: float = None,
    center: Tuple[float, float, float] = None,
    radii: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    take average of a parameter in 3D radial bins. should be used when snapshots are of cell-centered values
    args:
        solver (EulerSolver) : EulerSolver object or snapshot list
        param (str) : parameter to sample
            primitive/conservative variable ["rho", "P", ...]
            "v" hypotenuse of vx, vy, and vz
        t (float) : time. nearest snapshot time is used if it is not in the list of snapshot times
        center (Tuple[float, float, float]) 3D point, center of radial bins
        radii (array_like) : radii of bin interfaces
    returns:
        bin_average (array_like) : average of param in each bin
        r_average (array_like) : average radius in each bin
    """
    snapshots = get_snapshot_list(solver)
    n, _ = get_indices_from_coordinates(snapshots, t)
    X, Y, Z = np.meshgrid(
        snapshots[n]["x"], snapshots[n]["y"], snapshots[n]["z"], indexing="ij"
    )
    R = np.sqrt(
        np.square(X - center[0]) + np.square(Y - center[1]) + np.square(Z - center[2])
    )
    if param == "v":
        vx = snapshots[n]["w"].vx
        vy = snapshots[n]["w"].vy
        vz = snapshots[n]["w"].vz
        param_data = np.sqrt(np.square(vx) + np.square(vy) + np.square(vz))
    else:
        param_data = getattr(snapshots[n]["w"], param)
    r_average = np.empty_like(radii[:-1])
    bin_average = np.empty_like(radii[:-1])
    for i, (little_r, big_r) in enumerate(zip(radii[:-1], radii[1:])):
        inside_bin = np.logical_and(R > little_r, R <= big_r)
        sample_n = np.sum(np.where(inside_bin, 1, 0))
        r_sum = np.sum(np.where(inside_bin, R, 0))
        bin_sum = np.sum(np.where(inside_bin, param_data, 0))
        r_average[i] = r_sum / sample_n if sample_n > 0 else np.nan
        bin_average[i] = bin_sum / sample_n if sample_n > 0 else np.nan
    return bin_average, r_average
