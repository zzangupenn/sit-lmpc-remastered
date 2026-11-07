"""Arrow-key history in the existing Gym window, with unchanged debug plots."""

import time
import numpy as np

from sit_lmpc.planners.common import PlannerResult


def frame_result(frame):
    return PlannerResult(
        frame["action"], frame["trajectory"], frame.get("reference"),
        float(frame["safe_set_violation"]), float(frame["boundary_violation"]),
        tuple(frame["lambdas"]), frame["metrics"], frame.get("debug"), frame.get("risk"),
    )


class GymInsight:
    def __init__(self, recorder, simulator, keyboard, draw, *, renderer=None, plots=None,
                 emit=lambda *args, **kwargs: None):
        self.recorder, self.simulator, self.keyboard = recorder, simulator, keyboard
        self.draw, self.renderer, self.plots, self.emit = draw, renderer, plots, emit
        self.active = False
        self.last_hold = None
        self.finished = False

    def poll(self, *, hold=False, reason=None, index=None):
        """Return True to quit; returning from the newest frame resumes live flow."""
        if self.active:
            return False
        if self.finished:
            return True
        commands = self.keyboard.take_navigation()
        if "q" in commands:
            self.finished = True
            return True
        if not hold and "left" not in commands:
            return False
        frames = list(self.recorder.history)
        if not frames:
            if commands:
                self.emit("insight_unavailable", message="No completed insight steps yet.")
            return False
        if not getattr(self.simulator, "render_enabled", False):
            self.emit("insight_unavailable", message="Gym history requires a rendered experiment.")
            return False
        position = len(frames)-1 if index is None else index
        live_display = self.simulator.capture_render()
        live_overlays = self.renderer.snapshot() if self.renderer is not None else None
        self.active = self.keyboard.inspecting = True
        if hold:
            self.last_hold = (frames[-1]["step"], reason)
        if self.plots is not None:
            self.plots.history_active = True
        changed, previous_view = hold, None
        try:
            while True:
                for key in commands:
                    if key == "q":
                        self.finished = True
                        return True
                    if key == "left":
                        position = max(0, position-1)
                        changed = True
                    elif key == "right":
                        if position == len(frames)-1:
                            self.emit("insight_live", message="Continue experiment" if reason is None
                                      else f"Continue failure handling: {reason}")
                            return False
                        position += 1
                        changed = True
                    if changed:
                        self.emit("insight_step", position=position+1, total=len(frames), step=frames[position]["step"])
                if hold:
                    if not commands:
                        self.emit("insight_step", position=position+1, total=len(frames), step=frames[position]["step"])
                    hold = False
                view = self.keyboard.view()
                if self.keyboard.consume_close_request() and self.plots is not None:
                    self.plots.close()
                    changed = True
                if changed or view != previous_view:
                    frame = frames[position]
                    # A terminal nonfinite state cannot be sent to OpenGL.
                    shown = frame
                    if not np.isfinite(frame.get("next_state", frame["state"])).all():
                        shown = next((item for item in reversed(frames[:position])
                                      if np.isfinite(item.get("next_state", item["state"])).all()), None)
                        if shown is None and np.isfinite(frame["state"]).all():
                            shown = dict(frame, next_state=frame["state"], next_frenet=frame["frenet"],
                                         render_observation=None)
                        self.emit("insight_display_fallback", message="Terminal state is nonfinite; displaying last finite state.")
                    if shown is not None:
                        observation = shown.get("render_observation")
                        if observation is None:
                            observation = self.simulator.observation_for_frame(shown)
                        self.simulator.display_observation(observation)
                        if self.renderer is not None:
                            self.renderer.display(shown.get("overlays", dict(
                                trajectory=np.asarray(shown["trajectory"])[:, :2],
                                reference=np.empty((0, 2)) if shown.get("reference") is None
                                else np.asarray(shown["reference"])[:, :2])))
                    try:
                        self.draw(frame, view)
                    except Exception as error:
                        self.emit("insight_plot_error", message=str(error), view=view, step=frame["step"])
                    changed, previous_view = False, view
                self.simulator.render()
                if self.plots is not None:
                    self.plots.pump()
                if not self.simulator.window_open():
                    self.finished = True
                    return True
                time.sleep(.01)
                commands = self.keyboard.take_navigation()
        finally:
            self.simulator.display_observation(live_display)
            if self.renderer is not None and live_overlays is not None:
                self.renderer.display(live_overlays)
            self.active = self.keyboard.inspecting = False
            if self.plots is not None:
                self.plots.history_active = False


def draw_frame(recorder, frame, number, plots, track, config):
    """One dispatcher for native live-history and saved-history inspection."""
    if plots is None or number == 0:
        return
    step, result = frame["step"], frame_result(frame)
    if number == 1:
        if result.debug is not None:
            plots.debug(result, step, pause=False)
        else:
            plots.unavailable("Tracking warm-up has no learned-controller samples.")
        return
    data = recorder.view(frame, number)
    if data.get("unavailable"):
        plots.unavailable(data["unavailable"])
    elif number == 2:
        from sit_lmpc.data.laps import Lap

        laps = [Lap(**lap) for lap in data["laps"]]
        plots.summaries(laps, data["metrics"], track, config.planner.half_width, show=True, export=False)
    elif number == 3:
        plots.risk_distribution(data.get("diagnostic"), step, status=data.get("status"),
                                pause=False, half_width=config.planner.half_width)
    elif number == 4:
        plots.risk_validation(data, pause=False)
    elif number == 5:
        from sit_lmpc.diagnostics.model_comparison import Comparison

        plots.model_comparison(Comparison(**data["comparison"]), track, config.planner.half_width,
                               show=True, safe_set_states=data.get("safe_set_states"))
    elif number == 6:
        plots.safe_set_costs(data.get("diagnostic"), step, density=data.get("density"), pause=False)


def replay_gym(recording, frames, index, directory):
    from copy import deepcopy
    from sit_lmpc.config import load_config
    from sit_lmpc.experiments.runtime import configure
    from sit_lmpc.adapters.f1tenth import F1TenthSimulator, vehicle_parameters
    from sit_lmpc.diagnostics.insight import InsightRecorder
    from sit_lmpc.diagnostics.keyboard import KeyboardControls
    from sit_lmpc.diagnostics.events import console_event
    from sit_lmpc.diagnostics.rendering import TrajectoryRenderer
    from sit_lmpc.diagnostics.restored_plots import MatplotlibPlots
    from sit_lmpc.track import Track, map_path

    settings = deepcopy(recording["config"])
    settings["diagnostics"].update(render=True, keyboard_controls=True, keep_open_on_failure=False)
    settings["output"].update(quick_load=False, resume=None)
    config = load_config(overrides=settings)
    configure(config)
    track_path = map_path(config.simulation.map)
    track = Track.from_track_path(track_path)
    simulator = F1TenthSimulator(config.simulation, track_path, True,
                                 vehicle_parameters(config.simulation.friction, config.simulation.mass_ratio))
    keyboard = KeyboardControls(config.diagnostics)
    plots = None
    def emit(event, **fields):
        print(console_event(event, fields), flush=True)
    try:
        initial = next(frame["state"] for frame in frames if np.isfinite(frame["state"]).all())
        simulator.reset(initial)
        renderer = TrajectoryRenderer(simulator, track, config.planner.half_width)
        renderer.update(frame_result(frames[-1]))
        simulator.render()
        keyboard.start()
        simulator.bind_keyboard(keyboard)
        plots = MatplotlibPlots(directory, interactive=True)
        plots.keyboard_command = keyboard.focused_command
        recorder = InsightRecorder(directory, len(frames), config)
        recorder.history.extend(frames)
        history = GymInsight(recorder, simulator, keyboard,
            lambda frame, number: draw_frame(recorder, frame, number, plots, track, config),
            renderer=renderer, plots=plots, emit=emit)
        history.poll(hold=True, reason=recording["reason"], index=index)
    finally:
        keyboard.close()
        if plots is not None:
            plots.close()
        simulator.close()
