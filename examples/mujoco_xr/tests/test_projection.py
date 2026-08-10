# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the app hands mjvGLCamera, and what comes back in the depth buffer.

mjr_render builds its own projection from the mjvGLCamera frustum fields, so the
app's share of the clip convention is those six numbers plus the depth
inversion on readback. Both are pinned here, neither needs a GPU, a headset or a
VizSession.
"""

import math

import pytest

from isaacteleop_examples.mujoco_xr import _mujoco_xr

NEAR = 0.05
FAR = 50.0

# A plausible asymmetric headset fov, in radians.
FOV = [math.radians(-45.0), math.radians(42.0), math.radians(48.0), math.radians(-46.0)]

CENTER, HALF_WIDTH, BOTTOM, TOP, F_NEAR, F_FAR = range(6)


def test_the_frustum_is_the_fov_projected_onto_the_near_plane():
    f = _mujoco_xr.frustum_from_fov(FOV, NEAR, FAR)
    assert f[CENTER] - f[HALF_WIDTH] == pytest.approx(NEAR * math.tan(FOV[0]))
    assert f[CENTER] + f[HALF_WIDTH] == pytest.approx(NEAR * math.tan(FOV[1]))
    assert f[TOP] == pytest.approx(NEAR * math.tan(FOV[2]))
    assert f[BOTTOM] == pytest.approx(NEAR * math.tan(FOV[3]))
    assert (f[F_NEAR], f[F_FAR]) == pytest.approx((NEAR, FAR), rel=1e-6)


def test_half_width_is_set_and_not_left_to_the_aspect_fallback():
    """The load-bearing assertion.

    render_gl3.c's setView reads ``frustum_width`` as a HALF-width and, when it
    is zero, derives the horizontal extent from the viewport aspect ratio
    instead. An XR view's horizontal fov is close to its vertical one times the
    buffer aspect but not equal, and the drift shows on a headset as
    world-locked geometry sliding sideways under head motion. mjvisualize.h
    comments the field "not used for rendering"; as of 3.11.0 that is wrong.
    """
    f = _mujoco_xr.frustum_from_fov(FOV, NEAR, FAR)
    assert f[HALF_WIDTH] > 0.0

    aspect_derived = 0.5 * (f[TOP] - f[BOTTOM])
    assert f[HALF_WIDTH] != pytest.approx(aspect_derived), (
        "this fov happens to be square, so the test cannot tell the fallback apart"
    )


def test_no_y_flip_in_the_frustum():
    """The flip to the y-down image XR wants happens once, on readback --
    mapping angleUp to the frustum's bottom here would apply it twice."""
    f = _mujoco_xr.frustum_from_fov(FOV, NEAR, FAR)
    assert f[TOP] > 0.0 > f[BOTTOM]


def test_symmetric_fov_centres_the_optical_axis():
    half = math.radians(40.0)
    f = _mujoco_xr.frustum_from_fov([-half, half, half, -half], NEAR, FAR)
    assert f[CENTER] == pytest.approx(0.0, abs=1e-9)
    assert f[HALF_WIDTH] == pytest.approx(NEAR * math.tan(half))


def test_a_default_constructed_fov_is_rejected_loudly():
    """A default-constructed viz::Fov is four ZEROS, and must never render.

    Zero half_width is exactly the value that turns mjr_render's aspect-ratio
    fallback on, so the frame would come back looking plausible with a fov that
    carries nothing. ``FrameInfo.views`` is filled by the runtime, so this is a
    runtime/session bug the app cannot prevent -- only refuse.
    """
    with pytest.raises(ValueError):
        _mujoco_xr.frustum_from_fov([0.0, 0.0, 0.0, 0.0], NEAR, FAR)


def test_near_far_are_validated():
    with pytest.raises(ValueError):
        _mujoco_xr.frustum_from_fov(FOV, 0.0, FAR)
    with pytest.raises(ValueError):
        _mujoco_xr.frustum_from_fov(FOV, FAR, NEAR)


def test_submitted_depth_is_standard_z_not_the_reverse_z_mujoco_writes():
    """near -> 0, far -> 1, monotonic between.

    mjr_render writes the opposite (glClipControl ZERO_TO_ONE, glDepthFunc
    GEQUAL, glClearDepth 0, and a scale/translate ahead of its glFrustum), and
    shaders in gl_readback.cpp subtract it from 1. This is the specification
    that subtraction implements, and the pair viz puts in
    XrCompositionLayerDepthInfoKHR.
    """
    assert _mujoco_xr.submitted_depth(NEAR, NEAR, FAR) == pytest.approx(0.0, abs=1e-6)
    assert _mujoco_xr.submitted_depth(FAR, NEAR, FAR) == pytest.approx(1.0, abs=1e-6)

    depths = [_mujoco_xr.submitted_depth(d, NEAR, FAR) for d in (NEAR, 0.5, 5.0, FAR)]
    assert depths == sorted(depths)
