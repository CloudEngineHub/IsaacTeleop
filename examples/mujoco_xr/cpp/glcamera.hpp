// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// One XR view's asymmetric fov -> the mjvGLCamera frustum fields MuJoCo's
// renderer builds its projection from. A free function with no GPU and no
// MuJoCo state so tests/test_projection.py can pin the convention headless.
//
// `frustum_width` is a HALF-width, and at 0 mjr_render derives the horizontal
// extent from the viewport aspect ratio instead (render_gl3.c setView). An XR
// view's horizontal fov is close to its vertical one times the buffer aspect but
// not equal, so always set it -- and reject a fov whose half-width comes out 0
// rather than let that fallback engage. mjvisualize.h calls the field "not used
// for rendering"; as of 3.11.0 that is wrong.

#include <array>
#include <cmath>
#include <stdexcept>

namespace mujoco_xr
{

// The six mjvGLCamera projection fields, in MuJoCo's own spelling and units:
// extents measured on the near plane, half_width symmetric about center.
struct Frustum
{
    float center = 0.0f;
    float half_width = 0.0f;
    float bottom = 0.0f;
    float top = 0.0f;
    float near_z = 0.0f;
    float far_z = 0.0f;
};

// fov_lrud: (angle_left, angle_right, angle_up, angle_down) in radians, the
// field order of viz::Fov and of XrFovf. angle_left and angle_down are normally
// negative.
//
// No y flip here: OpenGL clip space is y-up like the fov itself, and the flip to
// the y-down image XR wants happens once, on readback (gl_readback.cpp).
inline Frustum frustum_from_fov(const std::array<float, 4>& fov_lrud, float near_z, float far_z)
{
    if (!(near_z > 0.0f) || !(far_z > near_z))
    {
        throw std::invalid_argument("mujoco_xr: need 0 < near_z < far_z");
    }
    const float left = near_z * std::tan(fov_lrud[0]);
    const float right = near_z * std::tan(fov_lrud[1]);
    const float top = near_z * std::tan(fov_lrud[2]);
    const float bottom = near_z * std::tan(fov_lrud[3]);

    Frustum f;
    f.center = 0.5f * (right + left);
    f.half_width = 0.5f * (right - left);
    f.bottom = bottom;
    f.top = top;
    f.near_z = near_z;
    f.far_z = far_z;

    // A default-constructed viz::Fov is all zeros, which reaches here as a
    // degenerate frustum. Caught rather than passed on: a zero half_width turns
    // MuJoCo's aspect-ratio fallback on, which renders something plausible from
    // a fov that carries no information.
    if (!(f.half_width > 0.0f) || !(f.top > f.bottom))
    {
        throw std::invalid_argument(
            "mujoco_xr: degenerate fov -- angle_right must exceed angle_left and angle_up must exceed "
            "angle_down. An all-zero fov means FrameInfo.views was never filled.");
    }
    return f;
}

// What one view-space distance ahead of the eye becomes in the depth buffer we
// hand ProjectionLayer.submit(): standard Z, near -> 0, far -> 1, matching the
// nearZ/farZ viz puts in XrCompositionLayerDepthInfoKHR.
//
// NOT what MuJoCo writes: mjr_render is reverse Z (glClipControl ZERO_TO_ONE,
// GL_GEQUAL, glClearDepth(0)), so near -> 1. The two differ by exactly `1 - d`,
// which is the subtraction in gl_readback.cpp's fragment shader.
inline float submitted_depth(float distance, float near_z, float far_z)
{
    return far_z * (distance - near_z) / (distance * (far_z - near_z));
}

} // namespace mujoco_xr
