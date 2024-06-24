from matplotlib import cm
import numpy as np
from typing import Tuple


def get_indices_from_coordinates(
    snapshots: list, x: float = None, y: float = None, z: float = None, t: float = None
) -> Tuple[Tuple[int, int, int, int], Tuple[float, float, float, float]]:
    """
    get the indices of the nearest x, y, z, t values in the snapshots
    args:
        snapshots (list) : list of snapshots, each a dictionary with keys "x", "y", "z", "t", "w"
        x (float) : x-coordinate
        y (float) : y-coordinate
        z (float) : z-coordinate
        t (float) : time
    returns:
        i, j, k, n (int) : indices of the nearest x, y, z, t values in the snapshots
    """
    if x is None and y is None and z is None:
        raise BaseException("One out of the three coordinates x-y-z must be None")
    if t is None:
        n = -1
        nearest_t = snapshots[-1]["t"]
    else:
        tarr = np.array([snapshots[i]["t"] for i in range(len(snapshots))])
        n = np.argmin(np.abs(tarr - t))
        nearest_t = snapshots[n]["t"]
    if x is None:
        i = nearest_x = None
    else:
        i = np.argmin(np.abs(snapshots[n]["x"] - x))
        nearest_x = snapshots[n]["x"][i]
    if y is None:
        j = nearest_y = None
    else:
        j = np.argmin(np.abs(snapshots[n]["y"] - y))
        nearest_y = snapshots[n]["y"][j]
    if z is None:
        k = nearest_z = None
    else:
        k = np.argmin(np.abs(snapshots[n]["z"] - z))
        nearest_z = snapshots[n]["z"][k]
    return (i, j, k, n), (nearest_x, nearest_y, nearest_z, nearest_t)


def xyzt_summary(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    z_arr: np.ndarray,
    x_nearest: float,
    y_nearest: float,
    z_nearest: float,
    t_nearest: float,
) -> str:
    """
    get a summary of the nearest x, y, z, t values
    args:
        x_arr (np.ndarray) : array of x values
        y_arr (np.ndarray) : array of y values
        z_arr (np.ndarray) : array of z values
        x_nearest (float) : nearest x value
        y_nearest (float) : nearest y value
        z_nearest (float) : nearest z value
        t_nearest (float) : nearest t value
    returns:
        out (str) : summary of the nearest x, y, z, t values
    """
    t_message = f"{t_nearest:.2f}"
    x_message = (
        f"{x_nearest:.2f}"
        if (isinstance(x_nearest, int) or isinstance(x_nearest, float))
        else f"[{x_arr[0]:.2f}, {x_arr[-1]:.2f}]"
    )
    y_message = (
        f"{y_nearest:.2f}"
        if (isinstance(y_nearest, int) or isinstance(y_nearest, float))
        else f"[{y_arr[0]:.2f}, {y_arr[-1]:.2f}]"
    )
    z_message = (
        f"{z_nearest:.2f}"
        if (isinstance(z_nearest, int) or isinstance(z_nearest, float))
        else f"[{z_arr[0]:.2f}, {z_arr[-1]:.2f}]"
    )
    out = f"t={t_message}, x={x_message}, y={y_message}, z={z_message}"
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
        solver (EulerSolver) : EulerSolver object
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
    if isinstance(solver, list):
        snapshots = solver
    else:
        # solver is an EulerSolver object
        snapshots = solver.snapshots

    if sum([x is None, y is None, z is None]) != 1:
        raise BaseException("One out of the three coordinates x-y-z must be None")
    # get the indices of the nearest x, y, z, t values in the snapshots
    (i, j, k, n), (xn, yn, zn, tn) = get_indices_from_coordinates(snapshots, x, y, z, t)
    xarr, yarr, zarr = snapshots[n]["x"], snapshots[n]["y"], snapshots[n]["z"]
    # get slice information
    if x is None:
        x_for_plotting = xarr
        slices = (slice(None), j, k)
    elif y is None:
        x_for_plotting = yarr
        slices = (i, slice(None), k)
    elif z is None:
        x_for_plotting = zarr
        slices = (i, j, slice(None))
    # y-data
    if param == "trouble":
        trouble = snapshots[n]["trouble"]
        NAD_mag = snapshots[n]["NAD violation magnitude"]
        source_array = np.where(NAD_mag > tol, trouble, 0)
    else:
        source_array = getattr(snapshots[n]["w"], param)
    y_for_plotting = source_array[slices]
    # print summary
    if verbose:
        print(xyzt_summary(xarr, yarr, zarr, xn, yn, zn, tn))
    # return plot
    out = ax.plot(x_for_plotting, y_for_plotting, **kwargs)
    return out


def plot_2d_slice(
    solver,
    ax,
    param: str,
    t: float = None,
    x=None,
    y=None,
    z=None,
    overlay_trouble: bool = False,
    tol: float = 0,
    cmap: str = "GnBu_r",
    trouble_color: str = "red",
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
        overlay_trouble (bool) : overlay trouble on the plot
        tol (float) : tolerance for trouble based on magnitude of NAD violation
        cmap (str) : colormap
        trouble_color (str) : color for trouble
        verbose (bool) : print the exact used coordinates
        **kwargs : keyword arguments for matplotlib.pyplot.imshow
    returns:
        None : modifies the input Axes object
    """
    if isinstance(solver, list):
        snapshots = solver
    else:
        # solver is an EulerSolver object
        snapshots = solver.snapshots

    if sum([x is None, y is None, z is None]) != 2:
        raise BaseException("Two out of the three coordinates x-y-z must be None")
    # get the indices of the nearest x, y, z, t values in the snapshots
    (i, j, k, n), (xn, yn, zn, tn) = get_indices_from_coordinates(snapshots, x, y, z, t)
    xarr, yarr, zarr = snapshots[n]["x"], snapshots[n]["y"], snapshots[n]["z"]
    # get slice information
    if x is None and y is None:
        slices = (slice(None), slice(None), k)
        horizontal_axis, vertical_axis = "x", "y"
        limits = (xarr[0], xarr[-1], yarr[0], yarr[-1])
    elif y is None and z is None:
        slices = (i, slice(None), slice(None))
        horizontal_axis, vertical_axis = "y", "z"
        limits = (yarr[0], yarr[-1], zarr[0], zarr[-1])
    elif x is None and z is None:
        slices = (i, j, slice(None))
        horizontal_axis, vertical_axis = "x", "z"
        limits = (xarr[0], xarr[-1], zarr[0], zarr[-1])
    # get z data
    if param == "trouble" or overlay_trouble:
        trouble = snapshots[n]["trouble"][slices]
        NAD_mag = snapshots[n]["NAD violation magnitude"][slices]
        trouble_for_plotting = np.where(NAD_mag > tol, trouble, 0)
    if param == "trouble":
        source_array = snapshots[n]["trouble"]
    else:
        source_array = getattr(snapshots[n]["w"], param)
    z_for_plotting = source_array[slices]
    # rotate
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
        print(xyzt_summary(xarr, yarr, zarr, xn, yn, zn, tn))
        print(f"{horizontal_axis=}, {vertical_axis=}")
    # return plot
    out = ax.imshow(z_for_plotting, extent=limits, cmap=colormap, **kwargs)
    return out
