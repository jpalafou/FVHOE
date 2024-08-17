from fvhoe.boundary_conditions import BoundaryCondition
from fvhoe.initial_conditions import kelvin_helmholtz_2d
from fvhoe.solver import EulerSolver
import matplotlib.pyplot as plt
import numpy as np
import os


def EulerSolver_wrapper(
    project_pref: str,
    snapshot_parent_dir: str,
    summary_parent_dir: str = None,
    ic: callable = kelvin_helmholtz_2d,
    bc: BoundaryCondition = BoundaryCondition(),
    T: float = 1.0,
    integrator: int = -1,
    n_snapshots: int = 2,
    plot_kwargs: dict = dict(z=0.5),
    **kwargs,
) -> EulerSolver:
    """
    Wrapper for EulerSolver to simplify saving snapshots and summary
    args:
        project_pref (str) : project prefect. [ex] 'double-mach-reflection'
        snapshot_parent_dir (str) : snapshot parent directory
        summary_parent_dir (str) : summary parent directory. used for saving output from snapshot
            helper functions. if None, no snapshot_helper_function is used.
        ic (callable) : valid EulerSolver initial condition function
        bc (BoundaryCondition) : valid EulerSolver boundary condition object
        T (float) : simulation time
        integrator (int) : integrator order. if -1, EulerSolver.rkorder() is used.
        n_snapshots (int) : number of snapshots to record. must be at least 2 (start and stop).
        plot_kwargs (dict) : passed to density plot
        **kwargs : passed to EulerSolver
    returns:
        solver (EulerSolver) : executed solver
    """
    # set up paths for savings snapshots and summaries
    project_name = project_pref + "_"
    sorted_kwargs = {
        k: kwargs[k]
        for k in sorted(
            kwargs.keys(), key=lambda x: [(c.lower(), c.isupper()) for c in x]
        )
    }
    project_name += "_".join([f"{k}={v}" for k, v in sorted_kwargs.items()])
    project_name += f"_{T=}_{integrator=}_{n_snapshots=}"
    snapshot_dir = os.path.join(snapshot_parent_dir, project_name)
    summary_dir = (
        None
        if summary_parent_dir is None
        else os.path.join(summary_parent_dir, project_name)
    )

    # generate plot at snapshots
    def snapshot_plotter(s):
        # density map
        fig, ax = plt.subplots()
        ax.set_title(r"$\rho$")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        s.plot_2d_slice(ax, param="rho", t=s.t, **plot_kwargs)

        # save to summary directory
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        plt.savefig(
            os.path.join(summary_dir, f"t={s.t:.2f}.png"), dpi=300, bbox_inches="tight"
        )

    # set up solver
    solver = EulerSolver(
        w0=ic,
        bc=bc,
        snapshot_helper_function=(
            None if summary_parent_dir is None else snapshot_plotter
        ),
        **kwargs,
    )
    integrator_config = dict(
        T=T, snapshot_dir=snapshot_dir, downbeats=np.linspace(0, T, n_snapshots)
    )

    # run solver
    match integrator:
        case -1:
            solver.rkorder(**integrator_config)
        case 1:
            solver.euler(**integrator_config)
        case 2:
            solver.ssprk2(**integrator_config)
        case 3:
            solver.ssprk3(**integrator_config)
        case 4:
            solver.rk4(**integrator_config)
        case _:
            raise ValueError(f"Invalid integrator value '{integrator}'")

    return solver
