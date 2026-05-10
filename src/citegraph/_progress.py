"""Internal helper: iterate a list with an optional rich progress bar.

Stage loops in :mod:`citegraph.pipeline` and the PDF -> markdown loop in
:mod:`citegraph.pdf_to_markdown` all want the same UX (spinner, current
item label, count, elapsed, ETA) over a list whose body is per-item work.
This module exists so that affordance lives in one place.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TypeVar

T = TypeVar("T")


def iter_with_progress(
    items: list[T],
    *,
    show_progress: bool,
    description: str,
    item_label: Callable[[T], str] | None = None,
) -> Iterator[T]:
    """Yield ``items`` one at a time, optionally with a rich progress bar.

    Parameters
    ----------
    items:
        Sequence to iterate. ``len(items)`` sets the bar total.
    show_progress:
        When ``False`` (or ``items`` is empty), yields plainly with no
        bar — callers don't need to branch on the flag themselves.
    description:
        Initial bar description, shown until the first item starts.
    item_label:
        Optional callable producing a per-item description (e.g. the
        current filename). Long labels are truncated to keep the bar
        layout from jittering.
    """
    if not show_progress or not items:
        yield from items
        return

    # Imported lazily so callers that pass ``show_progress=False`` don't
    # pay the import cost — keeps test runs and headless CLI use cheap.
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=False,
    ) as progress:
        task = progress.add_task(description, total=len(items))
        for item in items:
            if item_label is not None:
                label = item_label(item)
                if len(label) > 50:
                    label = label[:47] + "..."
                progress.update(task, description=label)
            yield item
            progress.advance(task)
        progress.update(task, description="Done")
