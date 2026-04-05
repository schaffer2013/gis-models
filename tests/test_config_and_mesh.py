from __future__ import annotations

import numpy as np

from gis_models.config import (
    BuildConfig,
    compute_z_scale_mm_per_m,
    compute_uniform_xy_scale,
    projected_distance_m,
    route_width_projected_m,
    utm_epsg_from_lon_lat,
)
from gis_models.cli import _raise_land_above_sea_level, _smooth_low_gradient_heights, _suppress_high_elevation_spikes
from gis_models.mesh import GridModel, LayerModel, build_layer_mesh, build_partition_mesh


def test_z_scale_matches_requested_ratio() -> None:
    config = BuildConfig(elevation_mm_per_1000_ft=12.0)
    assert config.z_mm_per_meter == 12.0 * 3.280839895013123 / 1000.0


def test_xy_scale_fits_larger_span() -> None:
    scale = compute_uniform_xy_scale((0.0, 0.0, 6000.0, 3000.0), 300.0, 300.0)
    assert scale == 0.05


def test_vertical_exaggeration_uses_xy_scale() -> None:
    config = BuildConfig(vertical_exaggeration=1.7)

    assert compute_z_scale_mm_per_m(config, 0.05) == 0.085


def test_route_width_conversion() -> None:
    assert route_width_projected_m(5.0, 0.05) == 100.0


def test_body_gap_conversion() -> None:
    assert projected_distance_m(0.2, 0.05) == 4.0


def test_utm_epsg_for_king_county_area() -> None:
    assert utm_epsg_from_lon_lat(-122.3, 47.6) == 32610


def test_low_gradient_smoothing_preserves_flat_surface() -> None:
    heights = np.full((3, 3), 12.5)
    valid_mask = np.ones((3, 3), dtype=bool)

    smoothed = _smooth_low_gradient_heights(heights, valid_mask)

    assert np.allclose(smoothed, heights)


def test_highland_spike_suppression_reduces_isolated_peak() -> None:
    heights = np.full((7, 7), 1000.0)
    heights[3, 3] = 1300.0
    valid_mask = np.ones((7, 7), dtype=bool)

    filtered = _suppress_high_elevation_spikes(heights, valid_mask)

    assert filtered[3, 3] < heights[3, 3]
    assert np.allclose(filtered[0, 0], heights[0, 0])


def test_land_raise_only_applies_above_sea_level() -> None:
    heights = np.array([
        [-10.0, 0.0, 5.0],
        [20.0, -2.0, 1.0],
    ])
    valid_mask = np.ones_like(heights, dtype=bool)

    raised = _raise_land_above_sea_level(heights, valid_mask, 300.0)

    assert np.allclose(
        raised,
        np.array([
            [0.0, 0.0, 305.0],
            [320.0, 0.0, 301.0],
        ]),
    )


def test_build_partition_mesh_is_watertight_for_single_cell() -> None:
    x_mm = np.array([0.0, 10.0])
    y_mm = np.array([0.0, 10.0])
    z_mm = np.array([[5.0, 5.0], [5.0, 5.0]])
    mask = np.array([[True]])
    mesh, stats = build_partition_mesh(GridModel(x_mm=x_mm, y_mm=y_mm, z_mm=z_mm, cell_mask=mask, base_z_mm=0.0))

    assert stats.cells == 1
    assert mesh.is_watertight
    assert len(mesh.faces) == 12


def test_build_partition_mesh_empty_mask_returns_empty_mesh() -> None:
    x_mm = np.array([0.0, 10.0])
    y_mm = np.array([0.0, 10.0])
    z_mm = np.array([[5.0, 5.0], [5.0, 5.0]])
    mask = np.array([[False]])
    mesh, stats = build_partition_mesh(GridModel(x_mm=x_mm, y_mm=y_mm, z_mm=z_mm, cell_mask=mask, base_z_mm=0.0))

    assert stats.cells == 0
    assert len(mesh.faces) == 0


def test_build_layer_mesh_supports_variable_bottom_surface() -> None:
    x_mm = np.array([0.0, 10.0])
    y_mm = np.array([0.0, 10.0])
    top_z_mm = np.array([[6.0, 6.0], [6.0, 6.0]])
    bottom_z_mm = np.array([[2.0, 2.0], [2.0, 2.0]])
    mask = np.array([[True]])

    mesh, stats = build_layer_mesh(
        LayerModel(
            x_mm=x_mm,
            y_mm=y_mm,
            top_z_mm=top_z_mm,
            bottom_z_mm=bottom_z_mm,
            cell_mask=mask,
        )
    )

    assert stats.cells == 1
    assert mesh.is_watertight
    assert mesh.bounds[0, 2] == 2.0
    assert mesh.bounds[1, 2] == 6.0
