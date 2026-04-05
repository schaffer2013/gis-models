from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import trimesh
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import transform, unary_union


DEFAULT_NZ_ANALYSIS_CRS = "EPSG:2193"


@dataclass(slots=True)
class SimilarityFit:
    scale_xy: float
    rotation_deg: float
    translation_x_mm: float
    translation_y_mm: float
    rms_error_mm: float
    mean_distance_mm: float
    p95_distance_mm: float


@dataclass(slots=True)
class ReferenceComponent:
    index: int
    area_mm2: float
    centroid_x_mm: float
    centroid_y_mm: float
    bounds_mm: tuple[float, float, float, float]
    z_min_mm: float
    z_max_mm: float
    vertices: int
    faces: int
    footprint: Polygon | MultiPolygon


def _rotation_matrix(angle_rad: float) -> np.ndarray:
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    return np.asarray([[cos_theta, sin_theta], [-sin_theta, cos_theta]], dtype=float)


def _largest_polygon(geometry: Polygon | MultiPolygon) -> Polygon:
    if isinstance(geometry, Polygon):
        return geometry
    if isinstance(geometry, MultiPolygon):
        return max(geometry.geoms, key=lambda geom: geom.area)
    raise TypeError(f"Unsupported geometry type: {type(geometry)!r}")


def _polygon_components(geometry: Polygon | MultiPolygon) -> list[Polygon]:
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return [polygon for polygon in geometry.geoms if not polygon.is_empty]
    raise TypeError(f"Unsupported geometry type: {type(geometry)!r}")


def _sample_boundary_points(geometry: Polygon | MultiPolygon, count: int = 2048) -> np.ndarray:
    polygons = _polygon_components(geometry)
    lengths = np.asarray([polygon.exterior.length for polygon in polygons], dtype=float)
    total_length = float(lengths.sum())
    if total_length <= 0:
        raise RuntimeError("Boundary geometry did not contain a usable exterior ring")

    raw_counts = lengths / total_length * count
    samples_per_polygon = np.maximum(1, np.floor(raw_counts).astype(int))
    shortfall = count - int(samples_per_polygon.sum())
    if shortfall > 0:
        order = np.argsort(raw_counts - samples_per_polygon)[::-1]
        for index in order[:shortfall]:
            samples_per_polygon[index] += 1

    sampled: list[tuple[float, float]] = []
    for polygon, sample_count in zip(polygons, samples_per_polygon, strict=True):
        ring = polygon.exterior
        distances = np.linspace(0.0, ring.length, sample_count, endpoint=False)
        sampled.extend((ring.interpolate(float(distance)).x, ring.interpolate(float(distance)).y) for distance in distances)
    return np.asarray(sampled, dtype=float)


def _fit_similarity_icp(
    source_geometry: Polygon | MultiPolygon,
    target_geometry: Polygon | MultiPolygon,
    *,
    sample_count: int = 512,
    angle_step_deg: float = 45.0,
    max_iterations: int = 100,
) -> SimilarityFit:
    source_points = _sample_boundary_points(source_geometry, count=sample_count)
    target_points = _sample_boundary_points(target_geometry, count=sample_count)
    target_tree = cKDTree(target_points)

    area_ratio = float(target_geometry.area / source_geometry.area)
    initial_scale = float(math.sqrt(area_ratio))

    def objective(params: np.ndarray) -> float:
        log_scale, angle_rad, tx, ty = params
        scale = math.exp(float(log_scale))
        rotation = _rotation_matrix(float(angle_rad))
        transformed = scale * (source_points @ rotation) + np.asarray([tx, ty], dtype=float)
        source_distances, _ = target_tree.query(transformed)
        transformed_tree = cKDTree(transformed)
        target_distances, _ = transformed_tree.query(target_points)
        return float(np.mean(source_distances**2) + np.mean(target_distances**2))

    best_result = None
    best_value = math.inf
    steps = max(int(round(360.0 / angle_step_deg)), 1)
    for angle_deg in np.linspace(0.0, 360.0, steps, endpoint=False):
        angle_rad = math.radians(float(angle_deg))
        rotation = _rotation_matrix(angle_rad)
        translation = target_points.mean(axis=0) - initial_scale * (source_points.mean(axis=0) @ rotation)
        result = minimize(
            objective,
            x0=np.asarray([math.log(initial_scale), angle_rad, translation[0], translation[1]], dtype=float),
            method="Powell",
            options={"maxiter": max_iterations, "xtol": 1e-4, "ftol": 1e-4},
        )
        if result.fun < best_value:
            best_value = float(result.fun)
            best_result = result

    if best_result is None:
        raise RuntimeError("Could not determine a similarity transform for the supplied geometries")

    log_scale, angle_rad, tx, ty = best_result.x
    scale = math.exp(float(log_scale))
    rotation = _rotation_matrix(float(angle_rad))
    transformed = scale * (source_points @ rotation) + np.asarray([tx, ty], dtype=float)
    source_distances, _ = target_tree.query(transformed)
    transformed_tree = cKDTree(transformed)
    target_distances, _ = transformed_tree.query(target_points)
    symmetric_distances = np.concatenate([source_distances, target_distances])
    rotation_deg = ((math.degrees(float(angle_rad)) + 180.0) % 360.0) - 180.0
    return SimilarityFit(
        scale_xy=scale,
        rotation_deg=float(rotation_deg),
        translation_x_mm=float(tx),
        translation_y_mm=float(ty),
        rms_error_mm=float(math.sqrt(best_value / 2.0)),
        mean_distance_mm=float(symmetric_distances.mean()),
        p95_distance_mm=float(np.percentile(symmetric_distances, 95)),
    )


def _mesh_footprint(mesh: trimesh.Trimesh) -> Polygon | MultiPolygon:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices_xy = np.asarray(mesh.vertices, dtype=float)[:, :2]
    triangles = [Polygon(vertices_xy[face]) for face in faces]
    footprint = unary_union(triangles)
    if footprint.is_empty:
        raise RuntimeError("Projected mesh footprint was empty")
    return footprint


def _load_reference_components(reference_stl: Path, *, min_area_mm2: float = 0.4) -> list[ReferenceComponent]:
    mesh = trimesh.load_mesh(reference_stl, force="mesh")
    components: list[ReferenceComponent] = []
    for index, component in enumerate(mesh.split(only_watertight=False)):
        footprint = _mesh_footprint(component)
        area_mm2 = float(footprint.area)
        if area_mm2 < min_area_mm2:
            continue
        bounds = tuple(float(value) for value in footprint.bounds)
        mesh_bounds = np.asarray(component.bounds, dtype=float)
        centroid = footprint.centroid
        components.append(
            ReferenceComponent(
                index=index,
                area_mm2=area_mm2,
                centroid_x_mm=float(centroid.x),
                centroid_y_mm=float(centroid.y),
                bounds_mm=bounds,
                z_min_mm=float(mesh_bounds[0, 2]),
                z_max_mm=float(mesh_bounds[1, 2]),
                vertices=len(component.vertices),
                faces=len(component.faces),
                footprint=footprint,
            )
        )

    if len(components) < 2:
        raise RuntimeError(f"Expected at least two usable components in {reference_stl}")
    return components


def _terrain_metadata_path(terrain_path: Path) -> Path:
    metadata_path = terrain_path.parent / "metadata.json"
    if not metadata_path.exists():
        raise RuntimeError(f"Expected metadata.json next to terrain STL {terrain_path}")
    return metadata_path


def _load_model_space_boundary(boundary_path: Path, terrain_path: Path) -> Polygon | MultiPolygon:
    metadata = json.loads(_terrain_metadata_path(terrain_path).read_text(encoding="utf-8"))
    bounds = metadata["boundary"]["bounds_m"]
    min_x = float(bounds["min_x"])
    min_y = float(bounds["min_y"])
    max_x = float(bounds["max_x"])
    max_y = float(bounds["max_y"])
    x0 = (min_x + max_x) / 2.0
    y0 = (min_y + max_y) / 2.0
    xy_scale = float(metadata["xy_scale_mm_per_m"])
    dem_crs = metadata["boundary"]["dem_crs"]

    geometry = gpd.read_file(boundary_path).to_crs(dem_crs).geometry.union_all()
    if geometry.is_empty:
        raise RuntimeError(f"Boundary file {boundary_path} did not contain usable geometry")

    return transform(lambda x, y, z=None: ((x - x0) * xy_scale, (y - y0) * xy_scale), geometry)


def _apply_similarity_to_mesh(mesh: trimesh.Trimesh, fit: SimilarityFit, *, scale_z_with_xy: bool) -> trimesh.Trimesh:
    transformed = mesh.copy()
    vertices = np.asarray(transformed.vertices, dtype=float).copy()
    xy = vertices[:, :2]
    rotation = _rotation_matrix(math.radians(fit.rotation_deg))
    translation = np.asarray([fit.translation_x_mm, fit.translation_y_mm], dtype=float)
    vertices[:, :2] = fit.scale_xy * (xy @ rotation) + translation
    if scale_z_with_xy:
        vertices[:, 2] *= fit.scale_xy
    transformed.vertices = vertices
    transformed.fix_normals()
    return transformed


def apply_similarity_to_geometry(geometry, fit: SimilarityFit):
    return transform(
        lambda x, y, z=None: (
            fit.scale_xy * (np.asarray(x) * math.cos(math.radians(fit.rotation_deg)) + np.asarray(y) * -math.sin(math.radians(fit.rotation_deg))) + fit.translation_x_mm,
            fit.scale_xy * (np.asarray(x) * math.sin(math.radians(fit.rotation_deg)) + np.asarray(y) * math.cos(math.radians(fit.rotation_deg))) + fit.translation_y_mm,
        ),
        geometry,
    )


def correlate_new_zealand_reference(
    *,
    reference_stl: Path,
    north_boundary: Path,
    south_boundary: Path,
    north_terrain: Path,
    south_terrain: Path,
    output_dir: Path,
    scale_z_with_xy: bool = True,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_components = _load_reference_components(reference_stl)
    largest_two = sorted(reference_components, key=lambda component: component.area_mm2, reverse=True)[:2]
    north_reference = max(largest_two, key=lambda component: component.centroid_y_mm)
    south_reference = min(largest_two, key=lambda component: component.centroid_y_mm)

    north_source = _load_model_space_boundary(north_boundary, north_terrain)
    south_source = _load_model_space_boundary(south_boundary, south_terrain)
    north_fit = _fit_similarity_icp(north_source, north_reference.footprint)
    south_fit = _fit_similarity_icp(south_source, south_reference.footprint)

    north_mesh = trimesh.load_mesh(north_terrain, force="mesh")
    south_mesh = trimesh.load_mesh(south_terrain, force="mesh")
    north_aligned = _apply_similarity_to_mesh(north_mesh, north_fit, scale_z_with_xy=scale_z_with_xy)
    south_aligned = _apply_similarity_to_mesh(south_mesh, south_fit, scale_z_with_xy=scale_z_with_xy)

    north_output = output_dir / "north-island-aligned.stl"
    south_output = output_dir / "south-island-aligned.stl"
    combined_output = output_dir / "nz-main-islands-aligned.stl"
    north_aligned.export(north_output)
    south_aligned.export(south_output)
    trimesh.util.concatenate([south_aligned, north_aligned]).export(combined_output)

    summary = {
        "reference_stl": str(reference_stl),
        "analysis_crs": DEFAULT_NZ_ANALYSIS_CRS,
        "scale_z_with_xy": scale_z_with_xy,
        "reference_components": [
            {
                **asdict(component),
                "footprint": None,
            }
            for component in sorted(reference_components, key=lambda component: component.area_mm2, reverse=True)
        ],
        "matches": {
            "north_island": {
                "boundary_file": str(north_boundary),
                "source_terrain": str(north_terrain),
                "reference_component_index": north_reference.index,
                "reference_centroid_xy_mm": [north_reference.centroid_x_mm, north_reference.centroid_y_mm],
                "fit": asdict(north_fit),
                "output_stl": str(north_output),
            },
            "south_island": {
                "boundary_file": str(south_boundary),
                "source_terrain": str(south_terrain),
                "reference_component_index": south_reference.index,
                "reference_centroid_xy_mm": [south_reference.centroid_x_mm, south_reference.centroid_y_mm],
                "fit": asdict(south_fit),
                "output_stl": str(south_output),
            },
        },
        "combined_output_stl": str(combined_output),
        "notes": [
            "The command uses the two largest footprint components in the supplied reference STL.",
            "Those two components are assigned to North and South Island by Y centroid so the northern component maps to North Island.",
            "Smaller reference components are preserved only in the JSON report for now; the aligned output STL contains the two correlated terrain islands.",
        ],
    }

    summary_path = output_dir / "correlation-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
