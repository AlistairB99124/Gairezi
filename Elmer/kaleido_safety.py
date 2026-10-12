"""Shared hard-timeout wrapper around plotly's Kaleido-based fig.write_image().

Kaleido 1.x launches a fresh headless-Chromium subprocess for every
fig.write_image() call and tears it down again afterwards. On Windows that
browser can occasionally become unresponsive to the DevTools protocol
(observed with real-time antivirus scanning delaying/intercepting a freshly
spawned chrome.exe); choreographer's Browser._close() then awaits
`Browser.close` with no timeout at all, which can block the calling process
indefinitely. Any caller rendering a plotly figure to an image file in this
repository should go through write_image_with_timeout() instead of calling
fig.write_image() directly, so a wedged Chromium can never hang an Elmer job.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
import subprocess
import sys

import plotly.graph_objects as go

# How long a single fig.write_image() call is allowed to take, including
# Kaleido's own browser startup/render/shutdown sequence, before it is
# considered wedged and forcibly abandoned.
WRITE_IMAGE_TIMEOUT_SECONDS = 60


def _kill_stray_chrome_processes() -> None:
    """Best-effort cleanup of any leftover headless Chromium process after a
    fig.write_image() call times out. Kaleido's own browser handle is leaked
    in that case (the render thread is abandoned, not cancelled), so nothing
    else will ever kill it."""
    if sys.platform != "win32":
        return
    for name in ("chrome.exe", "headless_shell.exe"):
        subprocess.run(
            ["taskkill", "/F", "/IM", name, "/T"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def write_image_with_timeout(fig: go.Figure, out_path: Path, *, scale: float = 1.2) -> None:
    """Call fig.write_image() with a hard timeout, raising TimeoutError
    (instead of hanging indefinitely) if Kaleido's headless-Chromium browser
    does not finish rendering and shutting down within the allotted time."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fig.write_image, str(out_path), scale=scale)
        try:
            future.result(timeout=WRITE_IMAGE_TIMEOUT_SECONDS)
        except FutureTimeoutError:
            _kill_stray_chrome_processes()
            raise TimeoutError(
                f"fig.write_image() to {out_path.name} did not complete within "
                f"{WRITE_IMAGE_TIMEOUT_SECONDS}s; the underlying Kaleido/Chromium "
                "browser subprocess appears to be unresponsive and has been force-killed."
            ) from None
