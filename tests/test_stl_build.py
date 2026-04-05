from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import trimesh
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, box

from gis_models.stl_build import STLBuildConfig, _fit_geometry_to_build_volume, build_nz_stl_route_model


def _write_rectangle_terrain_stl(path: Path) -> None:
    top = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [10.0, 0.0, 2.0],
            [10.0, 4.0, 3.0],
            [0.0, 4.0, 1.5],
        ],
        dtype=float,
    )
    bottom = top.copy()
    bottom[:, 2] = 0.0
    vertices = np.vstack([top, bottom])
    faces = np.asarray(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    transform = trimesh.transformations.rotation_matrix(np.deg2rad(32.0), [0.0, 0.0, 1.0])
    transform[:3, 3] = [40.0, -15.0, 0.0]
    mesh.apply_transform(transform)
    mesh.export(path)


def test_fit_geometry_to_build_volume_rotates_to_use_more_xy_space() -> None:
    geometry = box(0.0, 0.0, 10.0, 4.0)

    fit, metrics = _fit_geometry_to_build_volume(
        geometry,
        max_x_mm=6.0,
        max_y_mm=10.0,
        max_z_mm=None,
        base_thickness_mm=2.0,
        relief_span_source_units=0.0,
        angle_step_deg=90.0,
    )

    assert fit.rotation_deg == pytest.approx(90.0)
    assert metrics["used_x_mm"] == pytest.approx(4.0)
    assert metrics["used_y_mm"] == pytest.approx(10.0)


def test_build_nz_stl_route_model_end_to_end(tmp_path) -> None:
    terrain_path = tmp_path / "terrain.stl"
    boundary_path = tmp_path / "boundary.gpkg"
    route_path = tmp_path / "route.gpkg"
    output_dir = tmp_path / "out"

    _write_rectangle_terrain_stl(terrain_path)

    boundary = gpd.GeoDataFrame(geometry=[box(0.0, 0.0, 1000.0, 400.0)], crs="EPSG:2193")
    boundary.to_file(boundary_path, driver="GPKG")

    route = gpd.GeoDataFrame(geometry=[LineString([(100.0, 200.0), (900.0, 200.0)])], crs="EPSG:2193")
    route.to_file(route_path, driver="GPKG")

    metadata = build_nz_stl_route_model(
        STLBuildConfig(
            terrain_stl=terrain_path,
            boundary_file=boundary_path,
            route_file=route_path,
            output_dir=output_dir,
            area_name="North Island test slice",
            max_x_mm=80.0,
            max_y_mm=45.0,
            max_z_mm=15.0,
            base_thickness_mm=3.0,
            route_width_mm=4.0,
            route_thickness_mm=1.5,
            gap_xy_mm=0.6,
            gap_z_mm=0.2,
            water_distance_mm=2.0,
            grid_resolution=80,
        )
    )

    terrain_output = output_dir / "terrain_base.stl"
    route_output = output_dir / "route_insert.stl"
    assert terrain_output.exists()
    assert route_output.exists()
    assert metadata["route"]["route_cells"] > 0
    assert metadata["route"]["cavity_cells"] >= metadata["route"]["route_cells"]
    assert metadata["build_volume"]["used_x_mm"] <= 80.0 + 1e-6
    assert metadata["build_volume"]["used_y_mm"] <= 45.0 + 1e-6
    assert metadata["build_volume"]["used_z_mm"] <= 15.0 + 1e-6

    route_mesh = trimesh.load_mesh(route_output, force="mesh")
    grouped: dict[tuple[float, float], list[float]] = {}
    for x, y, z in np.asarray(route_mesh.vertices):
        grouped.setdefault((round(float(x), 4), round(float(y), 4)), []).append(float(z))

    z_differences = [max(values) - min(values) for values in grouped.values() if len(values) >= 2]
    assert z_differences
    assert min(z_differences) == pytest.approx(1.5, abs=1e-3)
