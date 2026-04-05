from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy import ndimage

from gis_models.config import (
    BuildConfig,
    compute_uniform_xy_scale,
    compute_z_scale_mm_per_m,
    projected_distance_m,
    route_width_projected_m,
)
from gis_models.correlation import correlate_new_zealand_reference
from gis_models.mesh import GridModel, build_partition_mesh
from gis_models.sources import fetch_water_polygons, load_boundary_geometry, load_route_from_kmz, write_metadata
from gis_models.stl_build import STLBuildConfig, build_nz_stl_route_model
from gis_models.terrain import (
    fetch_dem,
    fill_masked_heights,
    flatten_water,
    rasterize_geometry_mask,
    resample_array,
    union_geometries,
)


DEFAULT_KING_COUNTY_CONFIG = BuildConfig()
LOWLAND_SMOOTHING_SIGMA_CELLS = 1.0
LOWLAND_SMOOTHING_GRADIENT_PERCENTILE = 55.0
HIGHLAND_SPIKE_PERCENTILE = 70.0
HIGHLAND_SPIKE_MEDIAN_SIZE = 5
HIGHLAND_SPIKE_BASE_ALLOWANCE_M = 14.0
HIGHLAND_SPIKE_ALLOWANCE_GAIN = 0.08



def _grid_axes(bounds: tuple[float, float, float, float], rows: int, cols: int, xy_scale_mm_per_m: float) -> tuple[np.ndarray, np.ndarray]:
    min_x, min_y, max_x, max_y = bounds
    x = np.linspace(min_x, max_x, cols + 1)
    y = np.linspace(max_y, min_y, rows + 1)
    x0 = (min_x + max_x) / 2.0
    y0 = (min_y + max_y) / 2.0
    x_mm = (x - x0) * xy_scale_mm_per_m
    y_mm = (y - y0) * xy_scale_mm_per_m
    return x_mm, y_mm



def _corner_heights_from_cell_centers(cell_heights: np.ndarray) -> np.ndarray:
    rows, cols = cell_heights.shape
    padded = np.pad(cell_heights, 1, mode="edge")
    corners = np.zeros((rows + 1, cols + 1), dtype=float)
    for i in range(rows + 1):
        for j in range(cols + 1):
            window = padded[i : i + 2, j : j + 2]
            corners[i, j] = float(np.mean(window))
    return corners


def _smooth_low_gradient_heights(cell_heights: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    if LOWLAND_SMOOTHING_SIGMA_CELLS <= 0 or not valid_mask.any():
        return cell_heights

    grad_x = ndimage.sobel(cell_heights, axis=1, mode="nearest")
    grad_y = ndimage.sobel(cell_heights, axis=0, mode="nearest")
    gradient = np.hypot(grad_x, grad_y)
    gradient_scale = float(np.percentile(gradient[valid_mask], LOWLAND_SMOOTHING_GRADIENT_PERCENTILE))
    if gradient_scale <= 0:
        return cell_heights

    smoothed = ndimage.gaussian_filter(cell_heights, sigma=LOWLAND_SMOOTHING_SIGMA_CELLS, mode="nearest")
    weight = np.exp(-((gradient / gradient_scale) ** 2))
    weight *= valid_mask
    return weight * smoothed + (1.0 - weight) * cell_heights


def _suppress_high_elevation_spikes(cell_heights: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    if not valid_mask.any():
        return cell_heights

    valid_heights = cell_heights[valid_mask]
    highland_threshold = float(np.percentile(valid_heights, HIGHLAND_SPIKE_PERCENTILE))
    local_median = ndimage.median_filter(cell_heights, size=HIGHLAND_SPIKE_MEDIAN_SIZE, mode="nearest")
    delta = cell_heights - local_median

    filtered = cell_heights.copy()
    highland_mask = valid_mask & (cell_heights >= highland_threshold)
    if not highland_mask.any():
        return filtered

    allowance = HIGHLAND_SPIKE_BASE_ALLOWANCE_M + HIGHLAND_SPIKE_ALLOWANCE_GAIN * np.maximum(
        cell_heights - highland_threshold,
        0.0,
    )
    filtered[highland_mask] = local_median[highland_mask] + np.clip(
        delta[highland_mask],
        -allowance[highland_mask],
        allowance[highland_mask],
    )
    return filtered


def _raise_land_above_sea_level(cell_heights: np.ndarray, valid_mask: np.ndarray, land_raise_m: float) -> np.ndarray:
    if land_raise_m <= 0:
        return cell_heights

    raised = cell_heights.copy()
    land_mask = valid_mask & (raised > 0.0)
    raised[land_mask] += land_raise_m
    raised[valid_mask & ~land_mask] = np.maximum(raised[valid_mask & ~land_mask], 0.0)
    return raised



def _export_mesh(mesh, path: Path, name: str) -> None:
    if len(mesh.faces) == 0:
        path.write_text(f"solid {name}\nendsolid {name}\n", encoding="utf-8")
        return
    mesh.export(path)



def build_model(config: BuildConfig) -> dict:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.body_gap_mm < 0:
        raise ValueError("body_gap_mm must be non-negative")

    boundary, target_crs, boundary_source = load_boundary_geometry(
        boundary_file=config.boundary_file,
        boundary_layer=config.boundary_layer,
        state_fips=config.state_fips,
        county_fips=config.county_fips,
    )
    terrain_raw_m, raw_bounds, dem_crs = fetch_dem(
        boundary,
        resolution_m=config.dem_resolution_m,
        dem_file=config.dem_file,
    )
    analysis_boundary = boundary.to_crs(dem_crs)
    area_geom = analysis_boundary.geometry.iloc[0]
    raw_area_mask = rasterize_geometry_mask(area_geom, raw_bounds, terrain_raw_m.shape)
    terrain_raw_m = fill_masked_heights(terrain_raw_m, raw_area_mask)
    terrain_m = resample_array(terrain_raw_m, raw_bounds, dem_crs, (config.grid_resolution, config.grid_resolution))
    area_mask = rasterize_geometry_mask(area_geom, raw_bounds, terrain_m.shape)
    terrain_m = fill_masked_heights(terrain_m, area_mask)

    xy_scale = compute_uniform_xy_scale(raw_bounds, config.output_width_mm, config.output_height_mm)
    z_scale = compute_z_scale_mm_per_m(config, xy_scale)

    route_mask = np.zeros_like(area_mask)
    terrain_gap_mask = np.zeros_like(area_mask)
    route_feature = None
    route_width_projected = 0.0
    body_gap_projected = 0.0
    if config.kmz_file:
        route_feature = load_route_from_kmz(config.kmz_file, target_crs=target_crs)
        route_feature_analysis = gpd.GeoSeries([route_feature], crs=target_crs).to_crs(dem_crs).iloc[0]
        route_width_projected = route_width_projected_m(config.route_width_mm, xy_scale)
        body_gap_projected = projected_distance_m(config.body_gap_mm, xy_scale)
        route_buffer = route_feature_analysis.buffer(route_width_projected / 2.0, cap_style=2, join_style=2)
        route_mask = rasterize_geometry_mask(route_buffer, raw_bounds, terrain_m.shape) & area_mask
        terrain_gap_mask = route_mask
        if body_gap_projected > 0.0:
            terrain_gap_buffer = route_feature_analysis.buffer(
                (route_width_projected + body_gap_projected) / 2.0,
                cap_style=2,
                join_style=2,
            )
            terrain_gap_mask = rasterize_geometry_mask(terrain_gap_buffer, raw_bounds, terrain_m.shape) & area_mask

    water_mask = np.zeros_like(area_mask)
    if config.include_water:
        try:
            water_geoms = fetch_water_polygons(boundary)
            water_union = union_geometries(water_geoms)
            if not water_union.is_empty:
                water_union = gpd.GeoSeries([water_union], crs=boundary.crs).to_crs(dem_crs).iloc[0]
                water_mask = rasterize_geometry_mask(water_union, raw_bounds, terrain_m.shape) & area_mask
        except Exception as exc:  # pragma: no cover - network/runtime variability
            print(f"Warning: failed to fetch water polygons ({exc}). Continuing without water.")

    terrain_m = _raise_land_above_sea_level(terrain_m, area_mask, config.land_raise_m)
    if config.apply_terrain_filters:
        terrain_m = _smooth_low_gradient_heights(terrain_m, area_mask)
        terrain_m = _suppress_high_elevation_spikes(terrain_m, area_mask)
    terrain_mm = terrain_m * z_scale
    terrain_mm = flatten_water(terrain_mm, water_mask, config.water_drop_mm)
    corner_z_mm = _corner_heights_from_cell_centers(terrain_mm) + config.base_thickness_mm

    route_mask &= area_mask
    terrain_gap_mask &= area_mask
    terrain_only_mask = area_mask & ~terrain_gap_mask

    x_mm, y_mm = _grid_axes(raw_bounds, terrain_m.shape[0], terrain_m.shape[1], xy_scale)
    terrain_mesh, terrain_stats = build_partition_mesh(
        GridModel(x_mm=x_mm, y_mm=y_mm, z_mm=corner_z_mm, cell_mask=terrain_only_mask, base_z_mm=0.0)
    )
    route_mesh, route_stats = build_partition_mesh(
        GridModel(x_mm=x_mm, y_mm=y_mm, z_mm=corner_z_mm, cell_mask=route_mask, base_z_mm=0.0)
    )

    terrain_path = config.output_dir / "terrain_body.stl"
    route_path = config.output_dir / "route_body.stl"
    _export_mesh(terrain_mesh, terrain_path, "terrain_body")
    _export_mesh(route_mesh, route_path, "route_body")

    metadata = {
        "build": config.to_metadata(),
        "boundary": {
            **boundary_source,
            "target_crs": str(target_crs),
            "dem_crs": str(dem_crs),
            "bounds_m": {
                "min_x": raw_bounds[0],
                "min_y": raw_bounds[1],
                "max_x": raw_bounds[2],
                "max_y": raw_bounds[3],
            },
        },
        "xy_scale_mm_per_m": xy_scale,
        "route_width_projected_m": route_width_projected,
        "body_gap_projected_m": body_gap_projected,
        "area_cells": int(area_mask.sum()),
        "route_cells": int(route_mask.sum()),
        "terrain_gap_cells": int(terrain_gap_mask.sum()),
        "water_cells": int(water_mask.sum()),
        "terrain_mesh": asdict(terrain_stats),
        "route_mesh": asdict(route_stats),
    }
    metadata["build"]["z_mm_per_meter"] = z_scale
    metadata["build"]["z_scale_mode"] = (
        "vertical_exaggeration" if config.vertical_exaggeration is not None else "fixed_mm_per_1000_ft"
    )
    if route_feature is not None:
        metadata["route_length_m"] = float(gpd.GeoSeries([route_feature], crs=target_crs).length.iloc[0])

    write_metadata(config.output_dir / "metadata.json", metadata)
    return metadata



def _add_common_build_arguments(parser: argparse.ArgumentParser, *, king_defaults: bool) -> None:
    parser.add_argument("--area-name", default=DEFAULT_KING_COUNTY_CONFIG.area_name if king_defaults else "Custom area")
    parser.add_argument("--boundary-file", type=Path, help="Boundary file for any area (GeoJSON, GPKG, Shapefile, etc.).")
    parser.add_argument("--boundary-layer", help="Optional layer name when reading a multi-layer boundary file.")
    parser.add_argument("--dem-file", type=Path, help="Optional DEM raster file (for example a GeoTIFF) to use instead of downloading USGS 3DEP.")
    parser.add_argument("--state-fips", default=DEFAULT_KING_COUNTY_CONFIG.state_fips if king_defaults else None)
    parser.add_argument("--county-fips", default=DEFAULT_KING_COUNTY_CONFIG.county_fips if king_defaults else None)
    parser.add_argument("--kmz-file", type=Path, help="Optional KMZ/KML route file to inlay into the terrain.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_KING_COUNTY_CONFIG.output_dir if king_defaults else Path("out/custom-area"))
    parser.add_argument("--output-width-mm", type=float, default=DEFAULT_KING_COUNTY_CONFIG.output_width_mm)
    parser.add_argument("--output-height-mm", type=float, default=DEFAULT_KING_COUNTY_CONFIG.output_height_mm)
    parser.add_argument(
        "--elevation-mm-per-1000-ft",
        type=float,
        default=DEFAULT_KING_COUNTY_CONFIG.elevation_mm_per_1000_ft,
    )
    parser.add_argument("--route-width-mm", type=float, default=DEFAULT_KING_COUNTY_CONFIG.route_width_mm)
    parser.add_argument(
        "--body-gap-mm",
        type=float,
        default=DEFAULT_KING_COUNTY_CONFIG.body_gap_mm,
        help="Additional XY clearance removed from the terrain body around the route insert.",
    )
    parser.add_argument("--base-thickness-mm", type=float, default=DEFAULT_KING_COUNTY_CONFIG.base_thickness_mm)
    parser.add_argument("--water-drop-mm", type=float, default=DEFAULT_KING_COUNTY_CONFIG.water_drop_mm)
    parser.add_argument(
        "--land-raise-m",
        type=float,
        default=DEFAULT_KING_COUNTY_CONFIG.land_raise_m,
        help="Add extra elevation to terrain above sea level before scaling.",
    )
    parser.add_argument(
        "--terrain-filters",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_KING_COUNTY_CONFIG.apply_terrain_filters,
        help="Apply the built-in smoothing and spike suppression filters.",
    )
    parser.add_argument(
        "--vertical-exaggeration",
        type=float,
        default=DEFAULT_KING_COUNTY_CONFIG.vertical_exaggeration,
        help="If set, derive Z from the XY scale and multiply by this exaggeration factor.",
    )
    parser.add_argument("--grid-resolution", type=int, default=DEFAULT_KING_COUNTY_CONFIG.grid_resolution)
    parser.add_argument("--dem-resolution-m", type=float, default=DEFAULT_KING_COUNTY_CONFIG.dem_resolution_m)
    parser.add_argument("--include-water", action=argparse.BooleanOptionalAction, default=DEFAULT_KING_COUNTY_CONFIG.include_water)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gis-models", description="Generate printable terrain models from GIS data.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_nz_stl = subparsers.add_parser(
        "build-nz-stl-route",
        help="Correlate a supplied NZ terrain STL to a real-world boundary and split out a printable route insert.",
    )
    build_nz_stl.add_argument("--terrain-stl", type=Path, required=True)
    build_nz_stl.add_argument("--boundary-file", type=Path, required=True)
    build_nz_stl.add_argument("--boundary-layer")
    build_nz_stl.add_argument("--route-file", type=Path)
    build_nz_stl.add_argument("--output-dir", type=Path, required=True)
    build_nz_stl.add_argument("--area-name", default="New Zealand terrain")
    build_nz_stl.add_argument("--max-x-mm", type=float, required=True)
    build_nz_stl.add_argument("--max-y-mm", type=float, required=True)
    build_nz_stl.add_argument("--max-z-mm", type=float)
    build_nz_stl.add_argument("--base-thickness-mm", type=float, required=True)
    build_nz_stl.add_argument("--route-width-mm", type=float, default=5.0)
    build_nz_stl.add_argument("--route-thickness-mm", type=float, required=True)
    build_nz_stl.add_argument("--gap-xy-mm", type=float, default=0.2)
    build_nz_stl.add_argument("--gap-z-mm", type=float, default=0.0)
    build_nz_stl.add_argument("--water-distance-mm", type=float, default=0.0)
    build_nz_stl.add_argument("--grid-resolution", type=int, default=1200)
    build_nz_stl.add_argument("--analysis-crs", default="EPSG:2193")

    build_area = subparsers.add_parser(
        "build-area",
        help="Build a terrain model for any area from a boundary file or a U.S. county FIPS pair.",
    )
    _add_common_build_arguments(build_area, king_defaults=False)

    build_king = subparsers.add_parser(
        "build-king-county",
        help="Build the King County, WA example model using the default 300 x 300 mm / 12 mm per 1000 ft settings.",
    )
    _add_common_build_arguments(build_king, king_defaults=True)

    correlate_nz = subparsers.add_parser(
        "correlate-nz-reference",
        help="Align the existing North and South Island terrain STLs to a supplied all-islands New Zealand reference STL.",
    )
    correlate_nz.add_argument("--reference-stl", type=Path, required=True)
    correlate_nz.add_argument("--north-boundary", type=Path, required=True)
    correlate_nz.add_argument("--south-boundary", type=Path, required=True)
    correlate_nz.add_argument("--north-terrain", type=Path, required=True)
    correlate_nz.add_argument("--south-terrain", type=Path, required=True)
    correlate_nz.add_argument("--output-dir", type=Path, required=True)
    correlate_nz.add_argument(
        "--scale-z-with-xy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale terrain Z by the same factor as XY before writing the aligned STLs.",
    )

    return parser



def _config_from_args(args: argparse.Namespace, *, king_county_defaults: bool) -> BuildConfig:
    config = BuildConfig(
        area_name=args.area_name,
        boundary_file=args.boundary_file,
        boundary_layer=args.boundary_layer,
        dem_file=args.dem_file,
        state_fips=args.state_fips,
        county_fips=args.county_fips,
        output_width_mm=args.output_width_mm,
        output_height_mm=args.output_height_mm,
        elevation_mm_per_1000_ft=args.elevation_mm_per_1000_ft,
        route_width_mm=args.route_width_mm,
        body_gap_mm=args.body_gap_mm,
        base_thickness_mm=args.base_thickness_mm,
        water_drop_mm=args.water_drop_mm,
        land_raise_m=args.land_raise_m,
        apply_terrain_filters=args.terrain_filters,
        vertical_exaggeration=args.vertical_exaggeration,
        grid_resolution=args.grid_resolution,
        dem_resolution_m=args.dem_resolution_m,
        include_water=args.include_water,
        output_dir=args.output_dir,
        kmz_file=args.kmz_file,
    )
    if king_county_defaults and config.boundary_file is None:
        config.area_name = DEFAULT_KING_COUNTY_CONFIG.area_name
        config.state_fips = DEFAULT_KING_COUNTY_CONFIG.state_fips
        config.county_fips = DEFAULT_KING_COUNTY_CONFIG.county_fips
    return config



def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build-nz-stl-route":
        metadata = build_nz_stl_route_model(
            STLBuildConfig(
                terrain_stl=args.terrain_stl,
                boundary_file=args.boundary_file,
                boundary_layer=args.boundary_layer,
                route_file=args.route_file,
                output_dir=args.output_dir,
                area_name=args.area_name,
                max_x_mm=args.max_x_mm,
                max_y_mm=args.max_y_mm,
                max_z_mm=args.max_z_mm,
                base_thickness_mm=args.base_thickness_mm,
                route_width_mm=args.route_width_mm,
                route_thickness_mm=args.route_thickness_mm,
                gap_xy_mm=args.gap_xy_mm,
                gap_z_mm=args.gap_z_mm,
                water_distance_mm=args.water_distance_mm,
                grid_resolution=args.grid_resolution,
                analysis_crs=args.analysis_crs,
            )
        )
    elif args.command == "build-area":
        metadata = build_model(_config_from_args(args, king_county_defaults=False))
    elif args.command == "build-king-county":
        metadata = build_model(_config_from_args(args, king_county_defaults=True))
    elif args.command == "correlate-nz-reference":
        metadata = correlate_new_zealand_reference(
            reference_stl=args.reference_stl,
            north_boundary=args.north_boundary,
            south_boundary=args.south_boundary,
            north_terrain=args.north_terrain,
            south_terrain=args.south_terrain,
            output_dir=args.output_dir,
            scale_z_with_xy=args.scale_z_with_xy,
        )
        print(f"Wrote correlation outputs to {args.output_dir}")
        print(
            "Matched reference components: "
            f"north={metadata['matches']['north_island']['reference_component_index']}, "
            f"south={metadata['matches']['south_island']['reference_component_index']}"
        )
        return 0
    else:
        parser.error(f"Unknown command: {args.command}")
        return 2

    print(f"Wrote outputs to {args.output_dir}")
    if args.command == "build-nz-stl-route":
        print(
            "Route cells: "
            f"{metadata['route']['route_cells']}; cavity cells: {metadata['route']['cavity_cells']}"
        )
    else:
        print(f"Route cells: {metadata['route_cells']}; water cells: {metadata['water_cells']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
