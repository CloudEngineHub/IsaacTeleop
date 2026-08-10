// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// MuJoCo's offscreen framebuffer -> the CUDA-linear buffers
// viz::ProjectionLayer.submit() consumes, with no host round trip. Stage by
// stage in README.md.
//
// Use cudaGraphicsGLRegisterBUFFER, never RegisterImage: RegisterImage takes no
// depth format and no multisampled renderbuffer, and mjrContext.offDepthStencil
// is both.

#include "gl.hpp"

#include <cstdint>
#include <vector>

namespace mujoco_xr
{

class Readback
{
public:
    Readback() = default;
    ~Readback();

    Readback(const Readback&) = delete;
    Readback& operator=(const Readback&) = delete;

    // `src_fbo` is mjrContext.offFBO, needed here and not only in capture():
    // glBlitFramebuffer rejects a depth blit between differing formats, and
    // MuJoCo picks DEPTH32F_STENCIL8 or DEPTH24_STENCIL8 depending on
    // ARB_depth_buffer_float, so the blit target is matched to what it chose.
    void create(uint32_t width, uint32_t height, uint32_t view_count, gl::GLuint src_fbo);
    void destroy();

    // Steps 1-3 for one view, reading from `src_fbo` (mjrContext.offFBO).
    // Unmaps that view's buffers first, so a pointer handed out by ptr() is
    // valid only until the next capture() of the same view.
    void capture(uint32_t view, gl::GLuint src_fbo);

    // Step 4. Call once after the last capture() of the frame.
    void map();

    void* color_ptr(uint32_t view) const;
    void* depth_ptr(uint32_t view) const;

    uint32_t width() const
    {
        return width_;
    }
    uint32_t height() const
    {
        return height_;
    }

private:
    struct View
    {
        gl::GLuint blit_fbo = 0;
        gl::GLuint blit_color = 0; // RGBA8 texture
        gl::GLuint blit_depth = 0; // DEPTH24_STENCIL8 texture
        gl::GLuint out_fbo = 0;
        gl::GLuint out_color = 0; // RGBA8 texture
        gl::GLuint out_depth = 0; // R32F texture
        gl::GLuint color_pbo = 0;
        gl::GLuint depth_pbo = 0;
        void* color_resource = nullptr; // cudaGraphicsResource_t
        void* depth_resource = nullptr;
        void* color_device_ptr = nullptr;
        void* depth_device_ptr = nullptr;
        bool mapped = false;
    };

    void build_program();
    // throw_on_error false on the teardown path, which is reached from a
    // destructor with the GL context possibly already gone.
    void unmap(View& v, bool throw_on_error);
    const View& at(uint32_t view) const;

    uint32_t width_ = 0;
    uint32_t height_ = 0;
    gl::GLuint program_ = 0;
    gl::GLuint vao_ = 0;
    std::vector<View> views_;
};

} // namespace mujoco_xr
