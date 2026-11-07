"""Optional QtMatplotlib diagnostics, with no Matplotlib rendering or imports."""

from collections import deque
from pathlib import Path
import numpy as np


def _qt_plotter(**kwargs):
    from utilsuite import QtMatplotlib

    return QtMatplotlib(**kwargs)


class LivePlots:
    def __init__(self, history=500):
        self.errors = deque(maxlen=history)
        self.actions = deque(maxlen=history)
        self.steps = deque(maxlen=history)
        self.count = 0
        self.plotter = _qt_plotter(win_title="SIT-LMPC Remastered ST prediction and control", reopen=False)
        try:
            self.plotter.plot(
                [], live=True, plot_num=0, plot_area=0, color="#59b4ff", name="ST position error"
            )
            self.plotter.plot(
                [], live=True, plot_num=1, plot_area=1, color="#ffb347", name="Steering rate"
            )
            self.plotter.plot(
                [], live=True, plot_num=2, plot_area=1, color="#84d68b", name="Acceleration"
            )
            self.plotter.ylabel("ST position error (m)", plot_area=0)
            self.plotter.ylabel("Normalized actions", plot_area=1)
            self.plotter.xlabel("Plot update", plot_area=0)
            self.plotter.xlabel("Plot update", plot_area=1)
            self.plotter.flush()
        except BaseException:
            self.plotter.close()
            raise

    def update(self, state, predicted, action):
        # Closing the diagnostics window must not restart it or stop the experiment.
        if not self.plotter.is_alive():
            return
        self.errors.append(float(np.linalg.norm(np.asarray(state)[:2] - np.asarray(predicted)[:2])))
        self.actions.append(np.array(action, copy=True))
        self.steps.append(self.count)
        self.count += 1
        x, actions = np.asarray(self.steps), np.asarray(self.actions)
        self.plotter.plot(
            x, np.asarray(self.errors), live=True, plot_num=0, plot_area=0, color="#59b4ff"
        )
        self.plotter.plot(x, actions[:, 0], live=True, plot_num=1, plot_area=1, color="#ffb347")
        self.plotter.plot(x, actions[:, 1], live=True, plot_num=2, plot_area=1, color="#84d68b")

    def close(self):
        self.plotter.close()


def save_summary(directory, laps):
    if not laps:
        return
    # Export in the helper's own offscreen Qt process, even when a simulator Qt
    # application already exists or no desktop display is available.
    with _qt_plotter(
        win_title="SIT-LMPC Remastered lap summary", offscreen=True, window_size=(1000, 750)
    ) as plotter:
        numbers = np.arange(1, len(laps) + 1)
        times = np.asarray([lap.laptime for lap in laps])
        plotter.plot(numbers, times, live=True, plot_area=0, color="#59b4ff")
        plotter.scatter(numbers, times, live=True, plot_area=0, s=8)
        plotter.xlabel("Lap", plot_area=0)
        plotter.ylabel("Lap time (s)", plot_area=0)
        plotter.title("Lap times", plot_area=0)
        palette = ("#59b4ff", "#ffb347", "#84d68b", "#dc8cff", "#ff7f8a", "#55d7cb")
        for index, lap in enumerate(laps):
            plotter.plot(
                lap.cartesian[:, 0],
                lap.cartesian[:, 1],
                live=True,
                plot_area=1,
                color=palette[index % len(palette)],
                name=f"Lap {index + 1}",
            )
        plotter.xlabel("x (m)", plot_area=1)
        plotter.ylabel("y (m)", plot_area=1)
        plotter.title("Trajectories", plot_area=1)
        plotter.aspect(plot_area=1)
        plotter.savefig(Path(directory) / "summary.png")
