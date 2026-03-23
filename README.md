# gis-models

Python tooling to generate a printable terrain solid from GIS data, with a separately printable route inlay that follows the terrain surface.

## Scope

The project is meant to be usable for **any area**, not just one hard-coded place.

- The **example/default preset** is **King County, Washington**.
- You can also build from:
  - a **boundary file** (`.geojson`, `.gpkg`, Shapefile, etc.), or
  - a **U.S. county FIPS pair** (`--state-fips` + `--county-fips`).

## Default King County example

The example preset targets **King County, WA**, fitted into a **300 mm × 300 mm** square with a vertical scale of **12 mm per 1000 ft** of elevation.

## What the pipeline does

- Loads an area boundary from a file or Census county geometry.
- Chooses a projected CRS automatically from the boundary centroid.
- Pulls a terrain DEM from USGS 3DEP via `py3dep`.
- Optionally loads a route from a `.kmz` or `.kml` file and buffers it to a printable route width.
- Optionally fetches OSM water polygons and flattens them into the terrain surface.
- Generates two watertight meshes:
  - `terrain_body.stl`
  - `route_body.stl`
- Writes a metadata JSON file describing the applied scales and source extents.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Example: King County preset

```bash
gis-models build-king-county \
  --kmz-file /path/to/roadtrip.kmz \
  --output-dir out/king-county \
  --route-width-mm 5 \
  --grid-resolution 700 \
  --include-water
```

## Example: any area from a boundary file

```bash
gis-models build-area \
  --area-name "New Zealand" \
  --boundary-file /path/to/new-zealand.geojson \
  --kmz-file /path/to/route.kmz \
  --output-dir out/new-zealand \
  --output-width-mm 300 \
  --output-height-mm 300 \
  --elevation-mm-per-1000-ft 12 \
  --route-width-mm 5 \
  --include-water
```

## Example: any U.S. county by FIPS

```bash
gis-models build-area \
  --area-name "Marin County, CA" \
  --state-fips 06 \
  --county-fips 041 \
  --output-dir out/marin-county
```

## Outputs

The command writes:

- `terrain_body.stl`
- `route_body.stl` (or an empty mesh file if no route cells are present)
- `metadata.json`

## Notes and limitations

- The model is generated from a rasterized surface grid. This keeps the route split robust and ensures the route inherits terrain elevation, but route edges will follow the raster resolution.
- The water overlay is intentionally best-effort. If OSM water features cannot be downloaded, the build continues without them.
- The route split is cell-based rather than a full CAD boolean. For print-ready inlays, increase `--grid-resolution` until the route edge fidelity looks good for your printer.
