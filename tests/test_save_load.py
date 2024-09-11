from fvhoe.boundary_conditions import BoundaryCondition
from fvhoe.initial_conditions import shock_1d
from fvhoe.solver import EulerSolver
import numpy as np
import os
import pytest
import shutil


@pytest.mark.parametrize("dir_already_exists", [False, True])
@pytest.mark.parametrize("n_snapshots", [None, 10])
def test_save_load(dir_already_exists: bool, n_snapshots: int):
    """
    Test saving and loading snapshots
    args:
        dir_already_exists (bool) : whether the snapshot directory already exists
        n_snapshots (int) : number of snapshots to save
    """
    # test directory
    snapshot_dir = "snapshots/test"
    if dir_already_exists:
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)
        with open(os.path.join(snapshot_dir, "test.txt"), "w") as f:
            f.write("test.txt")
    else:
        if os.path.exists(snapshot_dir):
            shutil.rmtree(snapshot_dir)

    # set up a simple 1D Euler problem
    N = 60
    p = 0
    solver_config = dict(
        w0=shock_1d(),
        nx=N,
        px=p,
        riemann_solver="llf",
        bc=BoundaryCondition(x="free"),
    )
    solver = EulerSolver(**solver_config)

    # solve and save snapshots
    solver.euler(
        0.245,
        downbeats=None if n_snapshots is None else np.linspace(0, 0.245, n_snapshots),
        snapshot_dir=snapshot_dir,
        overwrite=True,
    )

    # overwrite should have removed the dummy file
    if dir_already_exists and os.path.exists(os.path.join(snapshot_dir, "test.txt")):
        raise ValueError("Snapshot directory was not overwritten.")

    # load snapshots
    solver2 = EulerSolver(**solver_config)
    solver2.euler(
        0.245,
        downbeats=None if n_snapshots is None else np.linspace(0, 0.245, n_snapshots),
        snapshot_dir=snapshot_dir,
    )

    assert np.all(solver.snapshots[-1]["w"] == solver2.snapshots[-1]["w"])
