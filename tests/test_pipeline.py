"""
Tests on synthetic data, where the right answer is known in advance.

Run with:  python -m pytest tests -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import importlib.util


def _load(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


seg = _load("02_segment_measure")
tim = _load("03_timing")


def synthetic_movie(n_frames=61, size=200, onset_frame=20):
    """Three discs of known radius; the release channel switches on at a known frame."""
    yy, xx = np.mgrid[0:size, 0:size]
    centres = [(50, 50, 10), (50, 150, 12), (150, 100, 14)]

    dna = np.zeros((n_frames, size, size), dtype=float)
    release = np.zeros((n_frames, size, size), dtype=float)
    for cy, cx, r in centres:
        disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        dna[:, disc] = 5000.0
        release[:, disc] = 150.0
        release[onset_frame:, disc] = 9000.0
    dna += 300.0
    release += 150.0
    return dna, release, centres


def test_segmentation_finds_every_disc():
    dna, _, centres = synthetic_movie()
    labels, _, _ = seg.segment_cells(dna)
    assert labels.max() == len(centres)


def test_segmented_areas_match_the_discs():
    dna, _, centres = synthetic_movie()
    labels, _, _ = seg.segment_cells(dna)
    from skimage.measure import regionprops

    found = sorted(r.area for r in regionprops(labels))
    expected = sorted(np.pi * r ** 2 for _, _, r in centres)
    for got, want in zip(found, expected):
        # smoothing and closing round the edges, so allow 25%
        assert want * 0.75 < got < want * 1.25, f"area {got:.0f}, expected ~{want:.0f}"


def test_half_max_crossing_finds_a_known_step():
    t = np.arange(0, 61) * 5.0
    signal = np.where(np.arange(61) < 20, 100.0, 900.0)
    crossing = tim.half_max_crossing(t, signal, baseline=100.0,
                                     extreme=900.0, rising=True)
    assert crossing == pytest.approx(100.0)   # frame 20 x 5 min


def test_half_max_crossing_handles_a_fall():
    t = np.arange(0, 61) * 5.0
    signal = np.where(np.arange(61) < 30, 1000.0, 200.0)
    crossing = tim.half_max_crossing(t, signal, baseline=1000.0,
                                     extreme=200.0, rising=False)
    assert crossing == pytest.approx(150.0)   # frame 30 x 5 min


def test_half_max_returns_nan_when_nothing_happens():
    t = np.arange(0, 61) * 5.0
    flat = np.full(61, 100.0)
    assert np.isnan(tim.half_max_crossing(t, flat, 100.0, 100.0, rising=True))
