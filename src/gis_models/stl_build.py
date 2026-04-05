from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import trimesh
from pyproj import CRS
from shapely.geometry import MultiPolygon, Polygon

from gis_models.correlation import (
    DEFAULT_NZ_ANALYSIS_CRS,
    SimilarityFit,
    _apply_similarity_to_mesh,
    _fit_similarity_icp,
    _mesh_footprint,
    _sample_boundary_points,
    apply_similarity_to_geometry,
)
from gis_models.mesh import LayerModel, build_layer_mesh
from gis_models.sources import load_route_geometry, write_metadata
from gis_models.terrain import fill_masked_heights, rasterize_geometry_mask


@dataclass(slots=True)
class STLBuildConfig:
    terrain_stl: Path
    boundary_file: Path
    output_dir: Path
    route_file: Path | None = None
    area_name: str = "New Zealand terrain"
    boundary_layer: str | None = None
    max_x_mm: float = 300.0
    max_y_mm: float = 300.0
    max_z_mm: float | None = None
    base_thickness_mm: float = 4.0
    route_width_mm: float = 5.0
    route_thickness_mm: float = 3.0
    gap_xy_mm: float = 0.2
    gap_z_mm: float = 0.0
    water_distance_mm: float = 0.0
    grid_resolution: int = 1200
    analysis_crs: str = DEFAULT_NZ_ANALYSIS_CRS

    def to_metadata(self) -> dict:
        return {
            "terrain_stl": str(self.terrain_stl),
            "boundary_file": str(self.boundary_file),
            "output_dir": str(self.output_dir),
            "route_file": str(self.route_file) if self.route_file else None,
            "area_name": self.area_name,
            "boundary_layer": self.boundary_layer,
            "max_x_mm": self.max_x_mm,
            "max_y_mm": self.max_y_mm,
            "max_z_mm": self.max_z_mm,
            "base_thickness_mm": self.base_thickness_mm,
            "route_width_mm": self.route_width_mm,
            "route_thickness_mm": self.route_thickness_mm,
            "gap_xy_mm": self.gap_xy_mm,
            "gap_z_mm": self.gap_z_mm,
            "water_distance_mm": self.water_distance_mm,
            "grid_resolution": self.grid_resolution,
            "analysis_crs": self.analysis_crs,
        }


def _export_mesh(mesh: trimesh.Trimesh, path: Path, name: str) -> None:
    if len(mesh.faces) == 0:
        path.write_text(f"solid {name}\nendsolid {name}\n", encoding="utf-8")
        return
    mesh.export(path)


def _validate_config(config: STLBuildConfig) -> None:
    numeric_fields = {
        "max_x_mm": config.max_x_mm,
        "max_y_mm": config.max_y_mm,
        "base_thickness_mm": config.base_thickness_mm,
        "route_width_mm": config.route_width_mm,
        "route_thickness_mm": config.route_thickness_mm,
        "grid_resolution": float(config.grid_resolution),
    }
    for name, value in numeric_fields.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    if config.max_z_mm is not None and config.max_z_mm <= 0:
        raise ValueError("max_z_mm must be positive when provided")
    if config.gap_xy_mm < 0:
        raise ValueError("gap_xy_mm must be non-negative")
    if config.gap_z_mm < 0:
        raise ValueError("gap_z_mm must be non-negative")
    if config.water_distance_mm < 0:
        raise ValueError("water_distance_mm must be non-negative")


def _load_boundary_geometry(path: Path, *, layer: str | None, analysis_crs: str) -> Polygon | MultiPolygon:
    boundary = gpd.read_file(path, layer=layer)
    if boundary.empty:
        raise RuntimeError(f"Boundary file {path} did not contain any features")
    if boundary.crs is None:
        raise RuntimeError(f"Boundary file {path} does not declare a CRS")

    geometry = boundary.to_crs(analysis_crs).geometry.union_all()
    if geometry.is_empty:
        raise RuntimeError(f"Boundary file {path} did not contain usable geometry")
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        raise RuntimeError(f"Boundary file {path} must contain polygonal geometry")
    return geometry


def _top_surface_triangles(mesh: trimesh.Trimesh) -> np.ndarray:
    mesh.fix_normals()
    triangles = np.asarray(mesh.triangles, dtype=float)
    normals = np.asarray(mesh.face_normals, dtype=float)
    top_mask = normals[:, 2] > 1e-8
    top_triangles = triangles[top_mask]
    if len(top_triangles) == 0:
        raise RuntimeError("Terrain STL did not expose any upward-facing surface triangles")
    return top_triangles


def _source_relief_range(mesh: trimesh.Trimesh) -> tuple[float, float]:
    top_triangles = _top_surface_triangles(mesh)
    z_values = top_triangles[:, :, 2]
    return float(np.min(z_values)), float(np.max(z_values))


def _fit_geometry_to_build_volume(
    geometry: Polygon | MultiPolygon,
    *,
    max_x_mm: float,
    max_y_mm: float,
    max_z_mm: float | None,
    base_thickness_mm: float,
    relief_span_source_units: float,
    angle_step_deg: float = 1.0,
) -> tuple[SimilarityFit, dict]:
    sampled = _sample_boundary_points(geometry, count=2048)
    best_fit: SimilarityFit | None = None
    best_metrics: dict | None = None
    best_score: tuple[float, float, float] | None = None

    if max_z_mm is not None:
        allowed_relief_mm = max_z_mm - base_thickness_mm
        if allowed_relief_mm <= 0:
            raise ValueError("max_z_mm must be larger than base_thickness_mm")
    else:
        allowed_relief_mm = math.inf

    step_count = max(1, int(round(180.0 / angle_step_deg)))
    for angle_deg in np.linspace(0.0, 180.0, step_count, endpoint=False):
        angle_rad = math.radians(float(angle_deg))
        rotation = np.asarray(
            [
                [math.cos(angle_rad), math.sin(angle_rad)],
                [-math.sin(angle_rad), math.cos(angle_rad)],
            ],
            dtype=float,
        )
        rotated = sampled @ rotation
        min_xy = rotated.min(axis=0)
        max_xy = rotated.max(axis=0)
        span_x = float(max_xy[0] - min_xy[0])
        span_y = float(max_xy[1] - min_xy[1])
        if span_x <= 0 or span_y <= 0:
            continue

        scale_xy = min(max_x_mm / span_x, max_y_mm / span_y)
        if relief_span_source_units > 0:
            scale_xy = min(scale_xy, allowed_relief_mm / relief_span_source_units)
        if scale_xy <= 0:
            continue

        translation = -scale_xy * min_xy
        used_x = scale_xy * span_x
        used_y = scale_xy * span_y
        used_z = base_thickness_mm + scale_xy * relief_span_source_units
        score = (scale_xy, used_x * used_y, -abs(used_x - used_y))
        if best_score is None or score > best_score:
            best_score = score
            best_fit = SimilarityFit(
                scale_xy=float(scale_xy),
                rotation_deg=float(angle_deg),
                translation_x_mm=float(translation[0]),
                translation_y_mm=float(translation[1]),
                rms_error_mm=0.0,
                mean_distance_mm=0.0,
                p95_distance_mm=0.0,
            )
            best_metrics = {
                "used_x_mm": float(used_x),
                "used_y_mm": float(used_y),
                "used_z_mm": float(used_z),
                "max_x_mm": float(max_x_mm),
                "max_y_mm": float(max_y_mm),
                "max_z_mm": None if max_z_mm is None else float(max_z_mm),
            }

    if best_fit is None or best_metrics is None:
        raise RuntimeError("Could not fit the correlated NZ footprint into the requested build volume")
    return best_fit, best_metrics


def _grid_shape(bounds: tuple[float, float, float, float], resolution: int) -> tuple[int, int]:
    min_x, min_y, max_x, max_y = bounds
    span_x = max_x - min_x
    span_y = max_y - min_y
    if span_x <= 0 or span_y <= 0:
        raise ValueError("Bounds must have positive width and height")

    if span_x >= span_y:
        cols = resolution
        rows = max(1, int(round(resolution * span_y / span_x)))
    else:
        rows = resolution
        cols = max(1, int(round(resolution * span_x / span_y)))
    return rows, cols


def _corner_mask_from_cells(cell_mask: np.ndarray) -> np.ndarray:
    corners = np.zeros((cell_mask.shape[0] + 1, cell_mask.shape[1] + 1), dtype=bool)
    corners[:-1, :-1] |= cell_mask
    corners[:-1, 1:] |= cell_mask
    corners[1:, :-1] |= cell_mask
    corners[1:, 1:] |= cell_mask
    return corners


def _sample_top_surface(
    mesh: trimesh.Trimesh,
    *,
    bounds: tuple[float, float, float, float],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = shape
    min_x, min_y, max_x, max_y = bounds
    x_mm = np.linspace(min_x, max_x, cols + 1)
    y_mm = np.linspace(max_y, min_y, rows + 1)
    sampled_z = np.full((rows + 1, cols + 1), np.nan, dtype=float)

    top_triangles = _top_surface_triangles(mesh)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)

    for triangle in top_triangles:
        xy = triangle[:, :2]
        z = triangle[:, 2]
        tri_min_x = float(np.min(xy[:, 0]))
        tri_max_x = float(np.max(xy[:, 0]))
        tri_min_y = float(np.min(xy[:, 1]))
        tri_max_y = float(np.max(xy[:, 1]))

        if tri_max_x < min_x or tri_min_x > max_x or tri_max_y < min_y or tri_min_y > max_y:
            continue

        j0 = max(0, int(math.floor((tri_min_x - min_x) / span_x * cols)))
        j1 = min(cols, int(math.ceil((tri_max_x - min_x) / span_x * cols)))
        i0 = max(0, int(math.floor((max_y - tri_max_y) / span_y * rows)))
        i1 = min(rows, int(math.ceil((max_y - tri_min_y) / span_y * rows)))
        if i1 < i0 or j1 < j0:
            continue

        x_slice = x_mm[j0 : j1 + 1]
        y_slice = y_mm[i0 : i1 + 1]
        grid_x, grid_y = np.meshgrid(x_slice, y_slice)

        a = xy[0]
        b = xy[1]
        c = xy[2]
        ab = b - a
        ac = c - a
        denominator = ab[0] * ac[1] - ac[0] * ab[1]
        if abs(denominator) <= 1e-12:
            continue

        ap_x = grid_x - a[0]
        ap_y = grid_y - a[1]
        alpha = (ap_x * ac[1] - ac[0] * ap_y) / denominator
        beta = (ab[0] * ap_y - ap_x * ab[1]) / denominator
        gamma = 1.0 - alpha - beta
        inside = (alpha >= -1e-9) & (beta >= -1e-9) & (gamma >= -1e-9)
        if not inside.any():
            continue

        interpolated = z[0] + alpha * (z[1] - z[0]) + beta * (z[2] - z[0])
        candidate = np.where(inside, interpolated, np.nan)
        current = sampled_z[i0 : i1 + 1, j0 : j1 + 1]
        current[:] = np.fmax(current, candidate)

    return x_mm, y_mm, sampled_z


def build_nz_stl_route_model(config: STLBuildConfig) -> dict:
    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    boundary_geometry = _load_boundary_geometry(
        config.boundary_file,
        layer=config.boundary_layer,
        analysis_crs=config.analysis_crs,
    )

    source_mesh = trimesh.load_mesh(config.terrain_stl, force="mesh")
    source_footprint = _mesh_footprint(source_mesh)
    geo_to_source_fit = _fit_similarity_icp(boundary_geometry, source_footprint)
    source_boundary = apply_similarity_to_geometry(boundary_geometry, geo_to_source_fit)

    relief_min_source, relief_max_source = _source_relief_range(source_mesh)
    source_relief_span = relief_max_source - relief_min_source
    output_fit, build_metrics = _fit_geometry_to_build_volume(
        source_boundary,
        max_x_mm=config.max_x_mm,
        max_y_mm=config.max_y_mm,
        max_z_mm=config.max_z_mm,
        base_thickness_mm=config.base_thickness_mm,
        relief_span_source_units=source_relief_span,
    )

    if config.water_distance_mm > 0:
        source_boundary_buffer = source_boundary.buffer(config.water_distance_mm / output_fit.scale_xy)
        output_fit, build_metrics = _fit_geometry_to_build_volume(
            source_boundary_buffer,
            max_x_mm=config.max_x_mm,
            max_y_mm=config.max_y_mm,
            max_z_mm=config.max_z_mm,
            base_thickness_mm=config.base_thickness_mm,
            relief_span_source_units=source_relief_span,
        )
        source_boundary = source_boundary_buffer

    final_boundary = apply_similarity_to_geometry(source_boundary, output_fit)
    if config.water_distance_mm > 0:
        final_boundary = final_boundary.buffer(0)

    final_mesh = _apply_similarity_to_mesh(source_mesh, output_fit, scale_z_with_xy=True)

    rows, cols = _grid_shape(final_boundary.bounds, config.grid_resolution)
    grid_bounds = tuple(float(value) for value in final_boundary.bounds)
    x_mm, y_mm, sampled_surface_z = _sample_top_surface(final_mesh, bounds=grid_bounds, shape=(rows, cols))
    area_mask = rasterize_geometry_mask(final_boundary, grid_bounds, (rows, cols))
    area_corner_mask = _corner_mask_from_cells(area_mask)
    if not np.isfinite(sampled_surface_z[area_corner_mask]).any():
        raise RuntimeError("Could not sample the STL terrain surface across the requested NZ area")
    sampled_surface_z = fill_masked_heights(sampled_surface_z, area_corner_mask)

    baseline_z = float(np.nanmin(sampled_surface_z[area_corner_mask]))
    top_surface_z = config.base_thickness_mm + (sampled_surface_z - baseline_z)

    route_mask = np.zeros((rows, cols), dtype=bool)
    cavity_mask = np.zeros((rows, cols), dtype=bool)
    route_geometry_final = None
    cavity_geometry_final = None
    if config.route_file is not None:
        route_geometry = load_route_geometry(config.route_file, target_crs=CRS.from_user_input(config.analysis_crs))
        route_source = apply_similarity_to_geometry(route_geometry, geo_to_source_fit)
        route_final_centerline = apply_similarity_to_geometry(route_source, output_fit)
        route_geometry_final = route_final_centerline.buffer(config.route_width_mm / 2.0, cap_style=2, join_style=2)
        cavity_geometry_final = route_final_centerline.buffer(
            (config.route_width_mm + config.gap_xy_mm) / 2.0,
            cap_style=2,
            join_style=2,
        )
        route_geometry_final = route_geometry_final.intersection(final_boundary)
        cavity_geometry_final = cavity_geometry_final.intersection(final_boundary)
        if not route_geometry_final.is_empty:
            route_mask = rasterize_geometry_mask(route_geometry_final, grid_bounds, (rows, cols)) & area_mask
        if not cavity_geometry_final.is_empty:
            cavity_mask = rasterize_geometry_mask(cavity_geometry_final, grid_bounds, (rows, cols)) & area_mask

    cavity_corner_mask = _corner_mask_from_cells(cavity_mask)
    route_corner_mask = _corner_mask_from_cells(route_mask)
    cavity_depth_mm = config.route_thickness_mm + config.gap_z_mm
    if cavity_corner_mask.any():
        minimum_cavity_top = float(np.nanmin(top_surface_z[cavity_corner_mask]))
        if minimum_cavity_top <= cavity_depth_mm + 1e-6:
            raise ValueError(
                "route_thickness_mm + gap_z_mm exceeds the available terrain height above the base in the route cavity"
            )

    terrain_top_z = top_surface_z.copy()
    terrain_top_z[cavity_corner_mask] -= cavity_depth_mm
    terrain_bottom_z = np.zeros_like(terrain_top_z)
    route_bottom_z = top_surface_z - config.route_thickness_mm

    terrain_mesh, terrain_stats = build_layer_mesh(
        LayerModel(
            x_mm=x_mm,
            y_mm=y_mm,
            top_z_mm=terrain_top_z,
            bottom_z_mm=terrain_bottom_z,
            cell_mask=area_mask,
        )
    )
    route_mesh, route_stats = build_layer_mesh(
        LayerModel(
            x_mm=x_mm,
            y_mm=y_mm,
            top_z_mm=top_surface_z,
            bottom_z_mm=route_bottom_z,
            cell_mask=route_mask,
        )
    )

    terrain_path = config.output_dir / "terrain_base.stl"
    route_path = config.output_dir / "route_insert.stl"
    _export_mesh(terrain_mesh, terrain_path, "terrain_base")
    _export_mesh(route_mesh, route_path, "route_insert")

    metadata = {
        "workflow": "nz_stl_route",
        "inputs": config.to_metadata(),
        "analysis_crs": config.analysis_crs,
        "geospatial_fit": asdict(geo_to_source_fit),
        "output_fit": asdict(output_fit),
        "build_volume": build_metrics,
        "surface": {
            "baseline_z_mm": baseline_z,
            "relief_span_mm": float(np.nanmax(top_surface_z[area_corner_mask]) - np.nanmin(top_surface_z[area_corner_mask])),
            "grid_rows": rows,
            "grid_cols": cols,
        },
        "route": {
            "has_route": config.route_file is not None,
            "route_cells": int(route_mask.sum()),
            "cavity_cells": int(cavity_mask.sum()),
            "route_width_mm": config.route_width_mm,
            "route_thickness_mm": config.route_thickness_mm,
            "gap_xy_mm": config.gap_xy_mm,
            "gap_z_mm": config.gap_z_mm,
        },
        "outputs": {
            "terrain_base_stl": str(terrain_path),
            "route_insert_stl": str(route_path),
        },
        "terrain_mesh": asdict(terrain_stats),
        "route_mesh": asdict(route_stats),
        "notes": [
            "The supplied terrain STL is treated as the terrain source, then correlated to the true NZ boundary footprint with a 2D similarity fit.",
            "The final print orientation is chosen to maximize XY usage within max_x_mm and max_y_mm, while max_z_mm, when provided, limits the relief plus base thickness.",
            "The route cavity is widened on the terrain-body side by gap_xy_mm, while the route insert keeps the exact requested route_thickness_mm.",
        ],
    }
    if route_geometry_final is not None:
        metadata["route"]["final_route_geometry_empty"] = bool(route_geometry_final.is_empty)
    if cavity_geometry_final is not None:
        metadata["route"]["final_cavity_geometry_empty"] = bool(cavity_geometry_final.is_empty)

    write_metadata(config.output_dir / "metadata.json", metadata)
    return metadata
