// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// MuJoCo's own renderer, driven once per eye into an offscreen framebuffer that
// Readback turns into CUDA pointers. Every geom type, the scene XML's materials
// and lights, shadows and reflections -- none of it lives here.
//
// The OpenGL context is NOT created here: `mujoco.GLContext` makes it on the
// Python side and must be current on this thread before the constructor runs.
// It must also be on the GPU viz picked; the constructor checks that and says
// so rather than rendering into another card's memory.
//
// C++ owns mjvScene/mjvOption/mjvCamera; Python owns mjModel/mjData/mj_step.
// render() must run on the mj_step thread, after it, and treats mjData as
// const.

#include "gl_readback.hpp"

#include <mujoco/mujoco.h>

#include <cstdint>
#include <vector>

namespace mujoco_xr
{

class SceneRenderer
{
public:
    struct Config
    {
        uint32_t width = 0;
        uint32_t height = 0;
        // Stereo only. A field because the render loop reads it, not because
        // mono is supported.
        uint32_t view_count = 2;
        // The same pair goes into VizSessionConfig.xr_near_z / xr_far_z and so
        // into XrCompositionLayerDepthInfoKHR. Drift between the depth we
        // submit and the range the runtime is told makes reprojection wrong,
        // and the symptom (world-locked geometry swimming under head motion) is
        // only visible on hardware. No default and no literal in cpp/.
        float near_z = 0.0f;
        float far_z = 0.0f;
    };

    SceneRenderer(const Config& config, const mjModel* model);
    ~SceneRenderer();

    SceneRenderer(const SceneRenderer&) = delete;
    SceneRenderer& operator=(const SceneRenderer&) = delete;

    // mjv_updateScene, exactly once per frame. Returns the geom count.
    int update_scene(const mjModel* model, mjData* data);

    // Draws every view and leaves the CUDA pointers mapped, so they are safe to
    // hand to ProjectionLayer.submit() the moment this returns.
    //
    // poses_xyz_qwxyz: view_count * 7 floats -- position then orientation
    //   (w, x, y, z), matching viz.Pose3D's spelling.
    // fovs_lrud: view_count * 4 floats -- angle_left, angle_right, angle_up,
    //   angle_down, in radians, matching viz.Fov's field order.
    void render(const std::vector<float>& poses_xyz_qwxyz, const std::vector<float>& fovs_lrud);

    // The mjvGLCamera frustum used for `view` on the last render(), flattened as
    // (center, half_width, bottom, top, near, far), so the app can assert the
    // convention per frame.
    std::vector<float> frustum(int view) const;

    const Readback& readback() const
    {
        return readback_;
    }
    uint32_t view_count() const
    {
        return config_.view_count;
    }
    int ngeom() const
    {
        return scene_.ngeom;
    }
    int maxgeom() const
    {
        return scene_.maxgeom;
    }

private:
    void destroy();

    Config config_;
    Readback readback_;

    mjvScene scene_{};
    mjvOption scene_option_{};
    mjvCamera camera_{};
    mjrContext context_{};
    bool scene_made_ = false;
    bool context_made_ = false;

    std::vector<mjvGLCamera> cameras_;
};

} // namespace mujoco_xr
