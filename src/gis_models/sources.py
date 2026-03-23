from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union

from gis_models.config import utm_epsg_from_lon_lat

COUNTY_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_500k.zip"


def determine_projected_crs(boundary_wgs84: gpd.GeoDataFrame) -> CRS:
    centroid = boundary_wgs84.geometry.unary_union.centroid
    return CRS.from_epsg(utm_epsg_from_lon_lat(centroid.x, centroid.y))



def load_boundary_geometry(
    *,
    boundary_file: Path | None = None,
    boundary_layer: str | None = None,
    state_fips: str | None = None,
    county_fips: str | None = None,
) -> tuple[gpd.GeoDataFrame, CRS, dict]:
    if boundary_file is not None:
        boundary = gpd.read_file(boundary_file, layer=boundary_layer)
        if boundary.empty:
            raise RuntimeError(f"Boundary file {boundary_file} did not contain any features")
        source_name = str(boundary_file)
        source_kind = "boundary_file"
    elif state_fips and county_fips:
        counties = gpd.read_file(COUNTY_URL)
        boundary = counties[(counties["STATEFP"] == state_fips) & (counties["COUNTYFP"] == county_fips)]
        if boundary.empty:
            raise RuntimeError(f"Could not find county boundary for STATEFP={state_fips} COUNTYFP={county_fips}")
        source_name = f"county:{state_fips}:{county_fips}"
        source_kind = "census_county"
    else:
        raise ValueError("Provide either --boundary-file or both --state-fips and --county-fips")

    if boundary.crs is None:
        raise RuntimeError("Boundary source does not declare a CRS")

    boundary = boundary[["geometry"]].copy()
    dissolved = boundary.dissolve().reset_index(drop=True)
    boundary_wgs84 = dissolved.to_crs("EPSG:4326")
    target_crs = determine_projected_crs(boundary_wgs84)
    return boundary_wgs84.to_crs(target_crs), target_crs, {
        "source_kind": source_kind,
        "source_name": source_name,
    }



def load_route_from_kmz(path: Path, *, target_crs: CRS) -> LineString | MultiLineString:
    suffix = path.suffix.lower()
    if suffix == ".kml":
        text = path.read_text(encoding="utf-8")
    elif suffix == ".kmz":
        import zipfile

        with zipfile.ZipFile(path) as archive:
            kml_name = next((name for name in archive.namelist() if name.lower().endswith(".kml")), None)
            if not kml_name:
                raise RuntimeError(f"No KML document found in {path}")
            text = archive.read(kml_name).decode("utf-8")
    else:
        raise ValueError(f"Unsupported route format: {path.suffix}")

    from xml.etree import ElementTree as ET

    root = ET.fromstring(text)
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    coordinates_nodes = root.findall(".//kml:LineString/kml:coordinates", ns)
    if not coordinates_nodes:
        raise RuntimeError("The KMZ/KML file did not contain any LineString coordinates")

    line_strings: list[LineString] = []
    for node in coordinates_nodes:
        if not node.text:
            continue
        coords = []
        for raw_pair in node.text.strip().split():
            lon, lat, *_rest = raw_pair.split(",")
            coords.append((float(lon), float(lat)))
        if len(coords) >= 2:
            line_strings.append(LineString(coords))

    if not line_strings:
        raise RuntimeError("The KMZ/KML file did not contain any usable route coordinates")

    route_wgs84 = gpd.GeoSeries(line_strings, crs="EPSG:4326").to_crs(target_crs)
    return unary_union(route_wgs84.geometry.tolist())



def fetch_water_polygons(boundary_gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    try:
        import osmnx as ox
    except ImportError as exc:  # pragma: no cover - dependency issue only at runtime
        raise RuntimeError("osmnx is required for water downloads") from exc

    polygon_wgs84 = boundary_gdf.to_crs("EPSG:4326").geometry.iloc[0]
    target_crs = boundary_gdf.crs
    tags = {
        "natural": ["water", "bay"],
        "water": True,
        "waterway": ["riverbank"],
        "landuse": ["reservoir"],
    }
    features = ox.features_from_polygon(polygon_wgs84, tags)
    if features.empty:
        return gpd.GeoSeries([], crs=target_crs)

    water = features[features.geometry.notnull()].to_crs(target_crs)
    clipped = gpd.clip(water[["geometry"]], boundary_gdf[["geometry"]])
    return clipped.geometry.reset_index(drop=True)



def write_metadata(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
