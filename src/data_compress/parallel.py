"""Split a chunked recording into contiguous chunk ranges and run them in worker processes.

Every range covers whole chunks, so workers never write to the same Zarr chunk or
the same byte range of an output file. Results do not depend on the number of jobs.
"""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np
from tqdm import tqdm

MAX_DEFAULT_JOBS = 8
TASKS_PER_JOB = 4  # several smaller tasks per worker keep all workers busy to the end


def default_jobs() -> int:
    return max(1, min(os.cpu_count() or 1, MAX_DEFAULT_JOBS))


def resolve_jobs(jobs: int | None) -> int:
    if jobs is None:
        return default_jobs()
    if jobs < 1:
        raise ValueError(f"jobs must be at least 1, got {jobs}")
    return jobs


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def n_chunks(n_samples: int, chunk: int) -> int:
    return -(-n_samples // chunk)


def chunk_bounds(index: int, chunk: int, n_samples: int) -> tuple[int, int]:
    start = index * chunk
    return start, min(start + chunk, n_samples)


def chunk_ranges(n_chunks: int, jobs: int) -> list[tuple[int, int]]:
    """Split chunk indices [0, n_chunks) into contiguous [start, stop) ranges."""
    n_tasks = min(n_chunks, jobs * TASKS_PER_JOB)
    edges = np.linspace(0, n_chunks, n_tasks + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def map_chunk_ranges(
        worker: Callable[..., Any],
        n_chunks: int,
        *,
        jobs: int,
        desc: str,
        show_progress: bool,
        **kwargs: Any,
) -> list[Any]:
    """Call `worker(start_chunk, stop_chunk, **kwargs)` on each range; results follow range order."""
    ranges = chunk_ranges(n_chunks, jobs)
    bar = tqdm(total=n_chunks, unit="chunk", desc=desc, disable=not show_progress)
    try:
        if jobs == 1:
            results = []
            for start, stop in ranges:
                results.append(worker(start, stop, **kwargs))
                bar.update(stop - start)
            return results
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(worker, start, stop, **kwargs): (start, stop) for start, stop in ranges}
            try:
                for future in as_completed(futures):
                    future.result()  # re-raises the worker's error
                    start, stop = futures[future]
                    bar.update(stop - start)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
            return [future.result() for future in futures]
    finally:
        bar.close()


def map_items(
        worker: Callable[[Any], Any],
        items: list[Any],
        *,
        jobs: int,
        desc: str,
        show_progress: bool,
) -> list[Any]:
    """Call `worker(item)` for each item in a process pool; results follow item order."""
    bar = tqdm(total=len(items), unit="file", desc=desc, disable=not show_progress)
    try:
        if jobs == 1 or len(items) <= 1:
            results = []
            for item in items:
                results.append(worker(item))
                bar.update(1)
            return results
        with ProcessPoolExecutor(max_workers=min(jobs, len(items))) as pool:
            futures = [pool.submit(worker, item) for item in items]
            try:
                for future in as_completed(futures):
                    future.result()  # re-raises the worker's error
                    bar.update(1)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
            return [future.result() for future in futures]
    finally:
        bar.close()


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists; pass overwrite to replace it")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
