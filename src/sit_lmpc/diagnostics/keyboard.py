"""Owned, thread-safe SIT keyboard shortcuts; plotting stays on the main thread."""

import os
import threading
from collections import deque


def interactive_available(config):
    """Keyboard views need a desktop display, not a simulator renderer."""
    return config.keyboard_controls and bool(
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    )


class KeyboardControls:
    def __init__(self, config, emit=lambda *args, **kwargs: None, *, labels=None, shortcut_help=None,
                 extra_keys=(), extra_key_chords=None):
        self.config, self.emit = config, emit
        self.labels, self.shortcut_help = labels, shortcut_help
        self.listener = None
        self.lock = threading.Lock()
        self.suspended = False
        self.debug = False
        self.summary = False
        self.risk = False
        self.validation = False
        self.comparison = False
        self.safe_set_cost = False
        self.validation_request = 0
        self.close_requested = False
        self.navigation = deque()
        self.extra_keys, self.extra_commands = frozenset(extra_keys), deque()
        self.extra_key_chords = dict(extra_key_chords or {})
        self.inspecting = False

    def command(self, key):
        if not self.config.keyboard_controls:
            return
        if key == "toggle_listening":
            self.toggle_listening()
            return
        if key in self.extra_keys:
            with self.lock:
                accepted = not self.suspended
                if accepted:
                    self.extra_commands.append(key)
            self.emit('keyboard_received', key=key, accepted=accepted,
                      message='Tracking control command')
            return
        if key in ("left", "right", "q"):
            with self.lock:
                if not self.suspended:
                    self.navigation.append(key)
            return
        if key not in ("0", "1", "2", "3", "4", "5", "6", "alt_r"):
            return
        with self.lock:
            accepted = not self.suspended
            if accepted:
                names = ("debug", "summary", "risk", "validation", "comparison", "safe_set_cost")
                previous_mode = tuple(getattr(self, name) for name in names)
                if key == "alt_r":
                    enabled = not self.debug
                    for name in names:
                        setattr(self, name, name == "debug" and enabled)
                else:
                    for number, name in enumerate(names, 1):
                        setattr(self, name, key == str(number))
                    if key == "4":
                        self.validation_request += 1
                current_mode = tuple(getattr(self, name) for name in names)
                if key == "0" or (any(previous_mode) and previous_mode != current_mode):
                    # Only request closure here; the main thread owns Matplotlib/Qt.
                    self.close_requested = True
                labels = self.labels or ("Step debug", "Trial summaries after the next trial", "Sampled risk inspection",
                          "Held-out risk validation", "Dynamics / model comparison",
                          "Safe-set cost comparison and NF cutoff")
                message = next((label for label, active in zip(labels, current_mode, strict=True) if active),
                               "Close debug windows and continue")
            else:
                message = "Ignored: shortcuts suspended; Ctrl+Alt+K resumes"
        # Acknowledge from the receiving thread, including during training.
        # Emit outside the mode lock; the main thread still owns plot changes.
        self.emit("keyboard_received", key="right Alt" if key == "alt_r" else key,
                  accepted=accepted, message=message)

    def toggle_listening(self):
        """Suspend ordinary shortcuts, retaining only the resume chord.

        Listener callbacks never touch Qt. Closing windows is latched for the
        main thread, including when a training job is currently running.
        """
        if not self.config.keyboard_controls:
            return
        with self.lock:
            self.suspended = not self.suspended
            suspended = self.suspended
            if suspended:
                self.debug = self.summary = self.risk = self.validation = self.comparison = False
                self.safe_set_cost = False
                self.close_requested = True
        self.emit(
            "keyboard_listening",
            enabled=not suspended,
            message=("Shortcuts suspended; experiment continues. Ctrl+Alt+K resumes."
                     if suspended else "Shortcuts enabled; select a plot with 1-6."),
        )

    def poll(self):
        with self.lock:
            debug, summary = self.debug, self.summary
        return debug, summary

    def should_pause(self):
        with self.lock:
            return self.config.keyboard_controls and not self.navigation and not self.inspecting and (
                self.debug or self.risk or self.comparison or self.safe_set_cost
            )

    def focused_command(self, key):
        # A running global listener owns key releases. Qt forwards only when
        # that listener is unavailable, preventing duplicate arrow movement.
        if self.listener is None:
            self.command(key)

    def take_navigation(self):
        with self.lock:
            commands = list(self.navigation)
            self.navigation.clear()
            return commands

    def take_extra_commands(self):
        with self.lock:
            commands = list(self.extra_commands)
            self.extra_commands.clear()
            return commands

    def has_navigation(self):
        with self.lock:
            return bool(self.navigation)

    def view(self):
        with self.lock:
            return next((number for number, name in enumerate(
                ("debug", "summary", "risk", "validation", "comparison", "safe_set_cost"), 1
            ) if getattr(self, name)), 0)

    def safe_set_cost_mode(self):
        with self.lock:
            return self.config.keyboard_controls and self.safe_set_cost

    def comparison_mode(self):
        with self.lock:
            return self.config.keyboard_controls and self.comparison

    def risk_mode(self):
        with self.lock:
            return self.config.keyboard_controls and self.risk

    def validation_mode(self):
        with self.lock:
            return self.config.keyboard_controls and self.validation

    def validation_selection(self):
        with self.lock:
            return self.config.keyboard_controls and self.validation, self.validation_request

    def consume_close_request(self):
        """Take a latched window-close request once, from the main thread."""
        with self.lock:
            requested = self.close_requested
            self.close_requested = False
        return requested

    def start(self):
        if not self.config.keyboard_controls or not interactive_available(self.config):
            return
        try:
            from pynput import keyboard

            modifiers = set()
            alt_used = False

            def toggle():
                nonlocal alt_used
                alt_used = True
                self.command("toggle_listening")

            hotkeys = [keyboard.HotKey(keyboard.HotKey.parse("<ctrl>+<alt>+k"), toggle)]

            def extra_chord(command):
                nonlocal alt_used
                alt_used = True
                self.command(command)

            for command, chord in self.extra_key_chords.items():
                hotkeys.append(keyboard.HotKey(keyboard.HotKey.parse(chord),
                    lambda command=command: extra_chord(command)))

            def press(key):
                canonical = listener.canonical(key)
                if canonical in (keyboard.Key.ctrl, keyboard.Key.alt):
                    modifiers.add(canonical)
                for hotkey in hotkeys:
                    hotkey.press(canonical)

            def release(key):
                nonlocal alt_used
                canonical = listener.canonical(key)
                for hotkey in hotkeys:
                    hotkey.release(canonical)
                modifiers.discard(canonical)
                if canonical == keyboard.Key.alt:
                    # A right Alt used in the chord is not the step-debug toggle.
                    if key == keyboard.Key.alt_r and not alt_used and keyboard.Key.ctrl not in modifiers:
                        self.command("alt_r")
                    alt_used = False
                elif not modifiers and getattr(key, "char", None) in ("0", "1", "2", "3", "4", "5", "6"):
                    self.command(key.char)
                elif not modifiers and key in (keyboard.Key.left, keyboard.Key.right):
                    self.command("left" if key == keyboard.Key.left else "right")
                elif not modifiers and getattr(key, "char", None) == "q":
                    self.command("q")
                elif not modifiers and 'enter' in self.extra_keys and key == keyboard.Key.enter:
                    self.command('enter')
                elif not modifiers and 'space' in self.extra_keys and key == keyboard.Key.space:
                    self.command('space')
                elif not modifiers and str(getattr(key, "char", '')).lower() in self.extra_keys:
                    command = key.char.lower()
                    # Chord-protected commands must not fire while typing in
                    # another window, or again when the chord key is released.
                    if command not in self.extra_key_chords:
                        self.command(command)

            listener = keyboard.Listener(on_press=press, on_release=release)
            self.listener = listener
            listener.start()
            self.emit(
                "keyboard_started",
                shortcuts=self.shortcut_help or "Left/Right: inspect history / return to live; Q: quit; Ctrl+Alt+K: suspend/resume shortcuts; right Alt / 1: step debug; 2: trial summaries; 3: sampled risk distribution; 4: held-out predictions after risk training; 5: sampled dynamics trajectories / model comparison; 6: terminal safe-set cost vs hull and NF cutoff; 0: off and close plots",
                pause=True,
            )
        except (ImportError, RuntimeError, OSError) as error:
            self.close()
            self.emit("keyboard_unavailable", message=str(error))

    def close(self):
        if self.listener is not None:
            self.listener.stop()
            try:
                self.listener.join(timeout=1)
            except Exception as error:
                self.emit("keyboard_error", message=str(error))
            self.listener = None
