from __future__ import annotations

import numpy as np

from gis_models.config import BuildConfig, compute_uniform_xy_scale, route_width_projected_m, utm_epsg_from_lon_lat
from gis_models.mesh import GridModel, build_partition_mesh


def test_z_scale_matches_requested_ratio() -> None:
    config = BuildConfig(elevation_mm_per_1000_ft=12.0)
    assert config.z_mm_per_meter == 12.0 * 3.280839895013123 / 1000.0


def test_xy_scale_fits_larger_span() -> None:
    scale = compute_uniform_xy_scale((0.0, 0.0, 6000.0, 3000.0), 300.0, 300.0)
    assert scale == 0.05


def test_route_width_conversion() -> None:
    assert route_width_projected_m(5.0, 0.05) == 100.0


def test_utm_epsg_for_king_county_area() -> None:
    assert utm_epsg_from_lon_lat(-122.3, 47.6) == 32610


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
