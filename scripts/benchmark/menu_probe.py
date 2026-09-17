"""Benchmark-only input injection through the real menu loop and key handlers.

Terminal decoding is deliberately excluded: these are application timings, not
ConPTY/emulator measurements. Nothing here is imported by the installed CLI.
"""
import os
import sys
import time


def install():
    from jarv.command_input import TextInput
    from jarv.tui_app import AltScreenApp

    paint = AltScreenApp._paint
    service_input = AltScreenApp._service_input
    # This probe owns the markers and stops after the input's real repaint.
    os.environ.pop("JARV_BENCH_FIRST_PAINT", None)
    os.environ.pop("JARV_BENCH_STOP_AFTER_PAINT", None)

    def mark(name, *values):
        print(name, *values, file=sys.__stderr__, flush=True)

    def measured_paint(app):
        painted = paint(app)
        if not painted:
            return False
        if not hasattr(app, "_bench_keys"):
            now = time.perf_counter_ns()
            label = app.first_paint_label
            mark("JARV_FIRST_PAINT", label, time.time_ns())
            mark("BENCH_READY", now)
            # Open the settings/setup provider picker and move its selection;
            # type into heads-up and session search. Never commit a setting.
            keys = {
                "headsup": [TextInput("x")],
                "settings": ["ENTER", "UP"],
                "setup": ["ENTER", "UP"],
                "sessions": ["CTRL_F", TextInput("x")],
            }.get(label, ["DOWN"])
            app._bench_keys = iter(keys)
            app._bench_remaining = len(keys)
            app._bench_first_key = True
            app._key_available_fn = lambda: app._bench_remaining > 0
            app._read_key_fn = lambda: (next(app._bench_keys), 1)
        elif app._bench_remaining == 0:
            if app.first_paint_label == "headsup":
                assert app.editor["buffer"] == "x", app.editor
            elif app.first_paint_label in ("settings", "setup"):
                assert app.edit["selected_provider"] != app.edit["original_selected_provider"], app.edit
            elif app.first_paint_label == "sessions":
                assert app.search_active and app.search_query == "x"
            mark("BENCH_INPUT_PAINT", time.perf_counter_ns())
            mark("BENCH_MODULES", ",".join(sorted(sys.modules)))
            os._exit(0)
        return True

    def measured_input(app):
        handled = service_input(app)
        if handled:
            now = time.perf_counter_ns()
            if app._bench_first_key:
                mark("BENCH_FIRST_KEY", now)
                app._bench_first_key = False
            app._bench_remaining -= 1
            if app._bench_remaining == 0:
                mark("BENCH_INPUT_READY", now)
        return handled

    AltScreenApp._paint = measured_paint
    AltScreenApp._service_input = measured_input
