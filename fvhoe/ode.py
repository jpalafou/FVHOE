from abc import ABC, abstractmethod
import numpy as np
import os
import subprocess
import time
from tqdm import tqdm
from typing import Any, Tuple


def _hit_target(t: float, dt: float, target: float = None) -> None:
    """
    overwrite dt if it overshoots the target
    args:
        t (float) : time value
        dt (float) : time-step size to reach t + dt
        target (float) : target time
    """
    newdt = dt
    if target is not None and target - t < dt:
        newdt = target - t
    return newdt


class ODE(ABC):
    """
    u' = f(t, u)
    """

    def __init__(self, u0: Any, progress_bar: bool = True):
        """
        args:
            u0 (Any) : initial state
            progress_bar (bool) : whether to print out a progress bar
        """
        self.u = u0
        self.t = 0
        self.timestamps = [0]
        self.step_count = 0

        # snapshots
        self.snapshots = []
        self.snapshot_times = []

        # progress bar
        self.print_progress_bar = True if progress_bar else False

        # git commit details
        self.commit_details = self.get_commit_details()

    def get_commit_details(self) -> dict:
        """
        get commit details as dict
        """
        repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        try:
            # Navigate to the repository path and get commit details
            result = subprocess.run(
                ["git", "-C", repo_path, "log", "-1", "--pretty=format:%H|%an|%ai|%D"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            commit_info = result.stdout.strip().split("|")
            commit_hash = commit_info[0]
            author_name = commit_info[1]
            commit_date = commit_info[2]
            branch_name = (
                commit_info[3].split(",")[0].strip().split()[-1]
                if len(commit_info) > 3
                else None
            )

            return {
                "commit_hash": commit_hash,
                "author_name": author_name,
                "commit_date": commit_date,
                "branch_name": branch_name,
            }
        except subprocess.CalledProcessError as e:
            return {"error": f"An error occurred: {e.stderr.strip()}"}

    def integrate(
        self,
        T: float = None,
        n: int = None,
        exact: bool = True,
        downbeats: Any = None,
        log_every_step: bool = False,
        snapshot_dir: str = None,
        overwrite: bool = False,
    ) -> None:
        """
        args:
            T (float) : time to simulate until
            n (int) : number of iterations to evolve. if defined, all other arguments are ignored
            exact (bool) : avoid overshooting T
            downbeats (iterable[float]) : times to reach exactly and log a snapshot, if exact is True
            log_every_step (bool) : take a snapshot at every step
            snapshot_dir (str) : directory to save snapshots. if None, does not save
            overwrite (bool) : overwrite the snapshot directory if it exists
        """
        predetermined_step_count = T is None

        # if given n, perform a simple time evolution
        if predetermined_step_count:
            self.snapshot()
            clock_start = time.time()
            for _ in tqdm(range(n)):
                self.take_step()
            self.execution_time = time.time() - clock_start
            self.snapshot()
            return

        # if save is True, try to read snapshots
        save = snapshot_dir is not None
        if save:
            self.snapshot_dir = os.path.normpath(snapshot_dir)
            if not overwrite:
                if self.read_snapshots():
                    return

        # initialize progress bar
        self.progress_bar_action(action="setup", T=T)

        # establish keytimes
        if downbeats is None:
            downbeats = []
        elif isinstance(downbeats, np.ndarray):
            downbeats = downbeats.tolist()

        downbeats = sorted(set([0] + downbeats + [T]))[
            1:
        ]  # get unique values and ignore 0
        target_time = downbeats.pop(0) if exact else None

        # initial snapshot
        self.snapshot()

        # simulation loop
        clock_start = time.time()
        while self.t < T:
            self.take_step(target_time=target_time)
            self.progress_bar_action(action="update")
            # target time actions
            if self.t == target_time or self.t >= T or log_every_step:
                self.snapshot()
                if downbeats and self.t == target_time:
                    target_time = downbeats.pop(0)
        self.execution_time = time.time() - clock_start

        # clean up progress bar
        self.progress_bar_action(action="cleanup")

        # if save is True, write snapshots
        if save:
            self.write_snapshots(overwrite)

    def snapshot(self):
        """
        overwrite to log more data
        """
        self.snapshots.append(None)
        self.snapshot_times.append(self.t)

    def take_step(self, target_time: float = None):
        """
        integrate helper function
        """
        dt, self.u = self.stepper(self.t, self.u, target_time=target_time)
        self.t += dt
        self.timestamps.append(self.t)
        self.step_count += 1
        self.step_helper_function()

    def step_helper_function(self):
        """
        is called at the end of each step
        """
        pass

    def progress_bar_action(self, action: str, T: float = None):
        """
        modify progress bar
        args:
            action (str) : "setup", "update", "cleanup"
            t (float) : time to simulate until
        """
        if self.print_progress_bar:
            if action == "setup":
                bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]"
                self.progress_bar = tqdm(total=T, bar_format=bar_format)
            elif action == "update":
                self.progress_bar.n = self.t
                self.progress_bar.refresh()
            elif action == "cleanup":
                self.progress_bar.close()

    def read_snapshots(self) -> bool:
        """
        read snapshots if they exist
        returns:
            bool : whether snapshots were read
        """
        pass

    def write_snapshots(self, overwrite: bool = False):
        """
        overwrite to save snapshots
        args:
            overwrite (bool) : overwrite the snapshot directory if it exists
        """
        pass

    @abstractmethod
    def f(self, t: float, u) -> Tuple[float, Any]:
        """
        args:
            t (float) : time value
            u (any) : state
        returns:
            dt, dudt (tuple[float, Any]) : time-step size, velocity
        """
        pass

    def euler(self, *args, **kwargs) -> None:
        self.integrator = "euler"

        def stepper(t, u, target_time=None):
            dt, dudt = self.f(t, u)
            dt = _hit_target(t=t, dt=dt, target=target_time)
            unext = u + dt * dudt
            return dt, unext

        self.stepper = stepper
        self.integrate(*args, **kwargs)

    def ssprk2(self, *args, **kwargs) -> None:
        self.integrator = "ssprk2"

        def stepper(t, u, target_time=None):
            dt, k0 = self.f(t, u)
            dt = _hit_target(t=t, dt=dt, target=target_time)
            u1 = u + dt * k0
            _, k1 = self.f(t, u1)
            unext = 0.5 * u + 0.5 * (u1 + dt * k1)
            return dt, unext

        self.stepper = stepper
        self.integrate(*args, **kwargs)

    def ssprk3(self, *args, **kwargs) -> None:
        self.integrator = "ssprk3"

        def stepper(t, u, target_time=None):
            dt, k0 = self.f(t, u)
            dt = _hit_target(t=t, dt=dt, target=target_time)
            _, k1 = self.f(t + dt, u + dt * k0)
            _, k2 = self.f(t + 0.5 * dt, u + 0.25 * dt * k0 + 0.25 * dt * k1)
            unext = u + (1 / 6) * dt * (k0 + k1 + 4 * k2)
            return dt, unext

        self.stepper = stepper
        self.integrate(*args, **kwargs)

    def rk4(self, *args, **kwargs) -> None:
        self.integrator = "rk4"

        def stepper(t, u, target_time=None):
            dt, k0 = self.f(t, u)
            dt = _hit_target(t=t, dt=dt, target=target_time)
            _, k1 = self.f(t + 0.5 * dt, u + 0.5 * dt * k0)
            _, k2 = self.f(t + 0.5 * dt, u + 0.5 * dt * k1)
            _, k3 = self.f(t + dt, u + dt * k2)
            unext = u + (1 / 6) * dt * (k0 + 2 * k1 + 2 * k2 + k3)
            return dt, unext

        self.stepper = stepper
        self.integrate(*args, **kwargs)
