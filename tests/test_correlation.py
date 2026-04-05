from __future__ import annotations

import numpy as np
import trimesh
from shapely.affinity import rotate, scale, translate
from shapely.geometry import Polygon

from gis_models.correlation import SimilarityFit, _apply_similarity_to_mesh, _fit_similarity_icp


def test_fit_similarity_icp_recovers_known_transform() -> None:
    source = Polygon([
        (0.0, 0.0),
        (4.0, 0.0),
        (4.0, 1.0),
        (1.5, 1.0),
        (1.5, 3.0),
        (0.0, 3.0),
    ])
    target = rotate(source, 31.0, origin=(0.0, 0.0))
    target = scale(target, xfact=2.4, yfact=2.4, origin=(0.0, 0.0))
    target = translate(target, xoff=17.5, yoff=-9.25)

    fit = _fit_similarity_icp(source, target, sample_count=256, angle_step_deg=45.0, max_iterations=60)

    assert abs(fit.scale_xy - 2.4) < 1e-3
    assert abs(fit.rotation_deg - 31.0) < 0.5
    assert abs(fit.translation_x_mm - 17.5) < 1e-2
    assert abs(fit.translation_y_mm + 9.25) < 1e-2
    assert fit.rms_error_mm < 0.05


def test_apply_similarity_to_mesh_scales_xy_and_z() -> None:
    mesh = trimesh.Trimesh(
        vertices=np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 4.0],
        ]),
        faces=np.asarray([[0, 1, 2]]),
        process=False,
    )
    fit = SimilarityFit(
        scale_xy=2.0,
        rotation_deg=90.0,
        translation_x_mm=10.0,
        translation_y_mm=-5.0,
        rms_error_mm=0.0,
        mean_distance_mm=0.0,
        p95_distance_mm=0.0,
    )

    transformed = _apply_similarity_to_mesh(mesh, fit, scale_z_with_xy=True)

    assert np.allclose(
        transformed.vertices,
        np.asarray([
            [10.0, -5.0, 0.0],
            [10.0, -3.0, 4.0],
            [8.0, -5.0, 8.0],
        ]),
    )
