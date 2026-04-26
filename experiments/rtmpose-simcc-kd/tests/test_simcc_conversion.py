"""Tests for SimCC label generation."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from simcc_label_generator import (
    batch_generate_simcc,
    generate_simcc_label,
    transform_teacher_coords,
)


def test_generate_simcc_label_basic():
    coords = np.array([[0.5, 0.5]], dtype=np.float32)
    confidence = np.array([1.0], dtype=np.float32)

    simcc_x, simcc_y = generate_simcc_label(coords, confidence, num_x_bins=192, num_y_bins=256)

    assert simcc_x.shape == (1, 192)
    assert simcc_y.shape == (1, 256)

    peak_x = simcc_x[0].argmax()
    peak_y = simcc_y[0].argmax()
    assert 94 <= peak_x <= 97
    assert 126 <= peak_y <= 130

    assert abs(simcc_x[0].sum() - 1.0) < 1e-5
    assert abs(simcc_y[0].sum() - 1.0) < 1e-5


def test_generate_simcc_label_edge():
    coords = np.array([[0.0, 1.0]], dtype=np.float32)
    confidence = np.array([0.5], dtype=np.float32)

    simcc_x, simcc_y = generate_simcc_label(coords, confidence, num_x_bins=192, num_y_bins=256)

    peak_x = simcc_x[0].argmax()
    peak_y = simcc_y[0].argmax()
    assert peak_x == 0
    assert peak_y == 255


def test_batch_generate():
    coords = np.array([[[0.25, 0.75]], [[0.75, 0.25]]], dtype=np.float32)
    confidence = np.array([[0.8], [0.6]], dtype=np.float32)

    simcc_x, simcc_y = batch_generate_simcc(coords, confidence, num_x_bins=192, num_y_bins=256)

    assert simcc_x.shape == (2, 1, 192)
    assert simcc_y.shape == (2, 1, 256)

    for i in range(2):
        sx, sy = generate_simcc_label(coords[i], confidence[i], num_x_bins=192, num_y_bins=256)
        np.testing.assert_allclose(simcc_x[i, 0], sx[0], rtol=1e-5)
        np.testing.assert_allclose(simcc_y[i, 0], sy[0], rtol=1e-5)


def test_transform_teacher_coords():
    crop_coords = np.array([[0.5, 0.5]], dtype=np.float32)
    crop_params = np.array([0.0, 0.0, 384.0, 288.0, 384.0, 288.0], dtype=np.float32)

    img_coords = transform_teacher_coords(crop_coords, crop_params)

    np.testing.assert_allclose(img_coords[0], [0.5, 0.5], rtol=1e-5)


def test_transform_teacher_coords_scaled():
    crop_coords = np.array([[0.5, 0.5]], dtype=np.float32)
    crop_params = np.array([0.0, 0.0, 192.0, 144.0, 384.0, 288.0], dtype=np.float32)

    img_coords = transform_teacher_coords(crop_coords, crop_params)

    np.testing.assert_allclose(img_coords[0], [0.25, 0.25], rtol=1e-5)
