"""Matplotlib-like plotting using PyQtGraph in a separate, spawn-safe Qt process.

No Matplotlib or Qt imports occur in the calling process. Scripts that create
plots must use an if __name__ == '__main__' guard, as required by spawn.
"""

import multiprocessing as mp
import operator
import os
from pathlib import Path
from queue import Empty
import time
import traceback

import numpy as np


def _index(value, name, minimum=0):
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _xy(x, y):
    if x is None and y is None:
        raise ValueError("plot() requires either x or y")
    if y is None:
        y, x = x, None
    y = np.array(y, copy=True)
    if y.ndim != 1:
        raise ValueError("x and y must be one-dimensional")
    x = np.arange(len(y)) if x is None else np.array(x, copy=True)
    if x.ndim != 1 or len(x) != len(y):
        raise ValueError("x and y must be one-dimensional with equal lengths")
    return x, y


class QtMatplotlib:
    def __init__(
        self,
        timer_delay=1,
        win_title="QtMatplotlib",
        show_fps=False,
        n_cols=1,
        offscreen=False,
        startup_timeout=30,
        reopen=True,
        window_size=(600, 450),
    ):
        self.timer_delay = _index(timer_delay, "timer_delay", 1)
        self.n_cols = _index(n_cols, "n_cols", 1)
        width, height = window_size
        self.window_size = (_index(width, "window width", 1), _index(height, "window height", 1))
        if startup_timeout <= 0:
            raise ValueError("startup_timeout must be positive")
        self.startup_timeout = startup_timeout
        self.reopen = reopen
        self.win_title, self.show_fps, self.offscreen = win_title, show_fps, offscreen
        self.total_plot_num = 0
        self.window_exist = False
        self.process = None
        self._reply = None
        self._plots = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def is_alive(self):
        self._check_error()
        return self.process is not None and self.process.is_alive()

    def _check_error(self):
        if self._reply is not None and self._reply.poll():
            try:
                kind, value = self._reply.recv()
            except EOFError:
                return
            if kind == "error":
                raise RuntimeError(f"Qt plot process failed:\n{value}")

    def _wait_reply(self, expected, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._reply.poll(min(0.1, max(0, deadline - time.monotonic()))):
                try:
                    kind, value = self._reply.recv()
                except EOFError as error:
                    raise RuntimeError("Qt plot process exited before replying") from error
                if kind == "error":
                    raise RuntimeError(f"Qt plot process failed:\n{value}")
                if kind == expected:
                    return value
            if not self.process.is_alive():
                self._check_error()
                raise RuntimeError(f"Qt plot process exited (code {self.process.exitcode})")
        raise TimeoutError(f"Qt plot process did not acknowledge {expected!r} in {timeout}s")

    def _init_process(self):
        self.close()
        ctx = mp.get_context("spawn")
        self.queue = ctx.Queue()
        self._reply, child_reply = ctx.Pipe(duplex=False)
        self.process = ctx.Process(
            target=QtPlotterProcess().run,
            args=(
                self.queue,
                self.timer_delay,
                self.win_title,
                child_reply,
                self.show_fps,
                self.n_cols,
                self.offscreen,
                self.window_size,
            ),
            daemon=True,
        )
        try:
            self.process.start()
            child_reply.close()
            self._wait_reply("ready", self.startup_timeout)
        except BaseException:
            child_reply.close()
            self.close()
            raise
        self.window_exist = True

    def _ensure_process(self):
        self._check_error()
        if self.process is not None and not self.process.is_alive() and not self.reopen:
            return False
        if self.process is None or not self.process.is_alive():
            self._init_process()
        return True

    def _add_plot(self, plot_num, live, plot_type, plot_area=0, **kwargs):
        plot_area = _index(plot_area, "plot_area")
        if plot_num is not None:
            plot_num = _index(plot_num, "plot_num")
            if plot_num > self.total_plot_num:
                raise ValueError("Invalid plot_num")
        if not self._ensure_process():
            return None
        self.send_dict = {"add": [], "update": []}
        if plot_num is None or plot_num == self.total_plot_num:
            plot_num = self.total_plot_num
            self.total_plot_num += 1
            self._plots.append((plot_type, plot_area))
            self.send_dict["add"].append({"plot_type": plot_type, "plot_area": plot_area, **kwargs})
        elif plot_num >= self.total_plot_num or self._plots[plot_num] != (plot_type, plot_area):
            raise ValueError("plot_num must refer to the same plot type and area")
        return plot_num

    def _update_plot(self, live):
        self.queue.put(self.send_dict)
        if not live:
            # The default remains blocking until the user closes the window.
            try:
                self.process.join()
                self._check_error()
            finally:
                self.close()

    def plot(self, x=None, y=None, live=False, plot_num=None, name="", plot_area=0, **kwargs):
        x, y = _xy(x, y)
        pen = {"color": kwargs.get("color", "w"), "width": kwargs.get("linewidth", 2)}
        connect = kwargs.get("connect", "auto")
        if isinstance(connect, np.ndarray):
            connect = np.array(connect, copy=True)
        plot_num = self._add_plot(plot_num, live, "line", plot_area, pen=pen, name=name)
        if plot_num is None:
            return None
        self.send_dict["update"].append(
            {"plot_num": plot_num, "plot_type": "line", "x": x, "y": y, "pen": pen,
             "connect": connect}
        )
        self._update_plot(live)
        return plot_num

    def scatter(self, x, y, c=None, s=10, name="", live=False, plot_num=None, plot_area=0):
        x, y = _xy(x, y)
        colors = None
        if c is not None:
            colors = np.array(c, dtype=float, copy=True)
            if colors.ndim == 0:
                colors = np.full(len(x), colors.item())
            if colors.ndim != 1 or len(colors) != len(x) or not np.isfinite(colors).all():
                raise ValueError("scatter colors must be finite scalars, one per point")
        plot_num = self._add_plot(plot_num, live, "scatter", plot_area, size=s, name=name)
        if plot_num is None:
            return None
        self.send_dict["update"].append(
            {"plot_num": plot_num, "plot_type": "scatter", "x": x, "y": y, "colors": colors}
        )
        self._update_plot(live)
        return plot_num

    def _send_axis_update(self, plot_area=0, **kwargs):
        plot_area = _index(plot_area, "plot_area")
        if not self._ensure_process():
            return
        self.queue.put({"axis": kwargs, "plot_area": plot_area})

    def xlabel(self, label, plot_area=0):
        self._send_axis_update(plot_area, x_label=label)

    def ylabel(self, label, plot_area=0):
        self._send_axis_update(plot_area, y_label=label)

    def title(self, label, plot_area=0):
        self._send_axis_update(plot_area, title=label)

    def xlim(self, left, right, plot_area=0):
        self._send_axis_update(plot_area, xlim=(left, right))

    def ylim(self, bottom, top, plot_area=0):
        self._send_axis_update(plot_area, ylim=(bottom, top))

    def aspect(self, locked=True, plot_area=0):
        self._send_axis_update(plot_area, aspect=bool(locked))

    def flush(self, timeout=10):
        """Wait until all previously queued plot updates have been applied."""
        if not self.is_alive():
            raise RuntimeError("Qt plot window is closed")
        self.queue.put({"flush": True})
        self._wait_reply("flushed", timeout)

    def savefig(self, filename, width=1000, timeout=30):
        """Export the complete panel layout to a PNG, including offscreen plots."""
        width = _index(width, "width", 1)
        if not self.is_alive():
            raise RuntimeError("Qt plot window is closed")
        path = str(Path(filename).expanduser().resolve())
        self.queue.put({"export": path, "width": width})
        self._wait_reply("exported", timeout)
        return Path(path)

    def close(self, timeout=5):
        """Shut down the owned worker; safe to call repeatedly or after a crash."""
        if self.process is not None:
            if self.process.pid is not None:
                if self.process.is_alive():
                    self.queue.put({"close": True})
                    self.process.join(timeout)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout)
            self.process.close()
            self.queue.cancel_join_thread()
            self.queue.close()
            self._reply.close()
        self.process, self._reply = None, None
        self.window_exist = False
        self.total_plot_num = 0
        self._plots = []


class QtPlotterProcess:
    def run(
        self,
        queue,
        timer_delay,
        win_title,
        reply,
        show_fps,
        n_cols=1,
        offscreen=False,
        window_size=(600, 450),
    ):
        try:
            if offscreen:
                os.environ["QT_QPA_PLATFORM"] = "offscreen"
            os.environ["PYQTGRAPH_QT_LIB"] = "PyQt6"
            import pyqtgraph as pg
            from PyQt6 import QtCore, QtWidgets

            self.pg, self.queue, self.reply = pg, queue, reply
            self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            pg.setConfigOptions(background="#191919", foreground="w", antialias=True)
            self.win = pg.GraphicsLayoutWidget(title=win_title)
            self.win.resize(*window_size)
            self.win.show()
            self.n_cols, self.figures, self.plots, self.data = n_cols, {}, [], []
            self.figure = self._get_figure(0)
            # PyQtGraph ships this colormap; no Matplotlib dependency is needed.
            self.colormap = pg.colormap.get("viridis")
            self.label = pg.LabelItem(justify="left")
            self.win.addItem(self.label, row=0, col=n_cols)
            self.figure.scene().sigMouseMoved.connect(self._on_mouse_move)
            self.show_fps = show_fps
            if show_fps:
                self.fps_text = pg.TextItem(anchor=(1, 1), color=(150, 150, 150))
                self.figure.addItem(self.fps_text)
                self.fps_text.setZValue(1000)
                self.elapsed_timer = QtCore.QElapsedTimer()
                self.elapsed_timer.start()
                self.frame_count = 0
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.update_figure)
            self.timer.start(timer_delay)
            self.app.processEvents()
            reply.send(("ready", None))
            self.app.exec()
        except BaseException:
            reply.send(("error", traceback.format_exc()))
        finally:
            reply.close()

    def _get_figure(self, plot_area):
        if plot_area not in self.figures:
            figure = self.win.addPlot(row=plot_area // self.n_cols, col=plot_area % self.n_cols)
            figure.enableAutoRange("xy", True)
            figure.showGrid(x=True, y=True, alpha=0.3)
            figure.addLegend()
            self.figures[plot_area] = figure
        return self.figures[plot_area]

    def _apply_axis_config(self, config, plot_area=0):
        figure = self._get_figure(plot_area)
        for key, axis in (("x_label", "bottom"), ("y_label", "left")):
            if key in config:
                figure.setLabel(axis, config[key])
        if "title" in config:
            figure.setTitle(config["title"])
        if "xlim" in config:
            figure.setXRange(*config["xlim"], padding=0)
        if "ylim" in config:
            figure.setYRange(*config["ylim"], padding=0)
        if "aspect" in config:
            figure.setAspectLocked(config["aspect"])

    def _on_mouse_move(self, event):
        for figure in self.figures.values():
            if figure.sceneBoundingRect().contains(event):
                point = figure.vb.mapSceneToView(event)
                self.label.setText(f"x {point.x():.2f}, y {point.y():.2f}")
                break

    def _get_brushes(self, values):
        values = np.asarray(values, dtype=float)
        if not len(values):
            return []
        span = values.max() - values.min()
        normalized = np.full_like(values, 0.5) if span == 0 else (values - values.min()) / span
        return [self.pg.mkBrush(*color) for color in self.colormap.map(normalized, mode="byte")]

    def add_plot(self, plot_type, size=10, pen=None, name="", plot_area=0):
        if plot_type == "scatter":
            item = self.pg.ScatterPlotItem(size=size, brush=self.pg.mkBrush("w"), name=name or None)
        elif plot_type == "line":
            item = self.pg.PlotDataItem(
                pen=self.pg.mkPen(**(pen or {"color": "w"})), name=name or None
            )
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")
        self._get_figure(plot_area).addItem(item)
        self.plots.append(item)
        self.data.append([])

    def update_plot(self, plot_num, plot_type, **kwargs):
        item = self.plots[plot_num]
        if plot_type == "scatter":
            colors = kwargs.get("colors")
            brushes = self._get_brushes(colors) if colors is not None else self.pg.mkBrush("w")
            item.setData(x=kwargs["x"], y=kwargs["y"], brush=brushes)
        else:
            item.setData(x=kwargs["x"], y=kwargs["y"], connect=kwargs.get("connect", "auto"))
            if kwargs.get("pen") is not None:
                item.setPen(self.pg.mkPen(**kwargs["pen"]))
        self.data[plot_num] = (kwargs["x"], kwargs["y"])

    def update_figure(self):
        try:
            # Queue.empty() is unreliable across processes. Drain non-blockingly,
            # but bound each tick so mouse events and window closing stay responsive.
            for _ in range(200):
                try:
                    message = self.queue.get_nowait()
                except Empty:
                    break
                if message.get("close"):
                    self.win.close()
                    self.app.quit()
                    return
                if "axis" in message:
                    self._apply_axis_config(message["axis"], message.get("plot_area", 0))
                for addition in message.get("add", []):
                    self.add_plot(**addition)
                for update in message.get("update", []):
                    self.update_plot(**update)
                if message.get("flush"):
                    self.reply.send(("flushed", None))
                if "export" in message:
                    from pyqtgraph.exporters import ImageExporter

                    self.win.ci.layout.activate()
                    exporter = ImageExporter(self.win.scene())
                    exporter.parameters()["width"] = message["width"]
                    exporter.export(message["export"])
                    if not Path(message["export"]).is_file():
                        raise OSError(f"Could not export plot to {message['export']}")
                    self.reply.send(("exported", None))
            if self.show_fps:
                self.frame_count += 1
                elapsed = self.elapsed_timer.elapsed()
                if elapsed >= 1000:
                    self.fps_text.setText(f"FPS: {self.frame_count * 1000 / elapsed:.1f}")
                    self.elapsed_timer.restart()
                    self.frame_count = 0
        except BaseException:
            self.reply.send(("error", traceback.format_exc()))
            self.timer.stop()
            self.app.quit()
