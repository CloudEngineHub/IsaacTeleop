// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "gl.hpp"

#include <dlfcn.h>
#include <stdexcept>
#include <string>

namespace mujoco_xr
{
namespace gl
{

#define MUJOCO_XR_GL(name, upper) PFNGL##upper##PROC name = nullptr;
#include "gl_functions.inc"
#undef MUJOCO_XR_GL

namespace
{

using ProcLoader = void* (*)(const char*);

bool loaded_ = false;

// dlopen with RTLD_NOLOAD first: the library we want is the one the process has
// ALREADY loaded, because that is the one `mujoco.GLContext` made a context on.
// Loading a second copy would resolve symbols against a dispatch table with no
// current context. Falls back to a plain dlopen so a caller that made its
// context some other way still works.
void* open_already_loaded(const char* soname)
{
    void* handle = dlopen(soname, RTLD_LAZY | RTLD_NOLOAD);
    if (handle == nullptr)
    {
        handle = dlopen(soname, RTLD_LAZY);
    }
    return handle;
}

// eglGetProcAddress / glXGetProcAddress, whichever this process has. Both
// return the libglvnd dispatch stub for the calling thread's current context,
// so either serves whatever MUJOCO_GL selected; EGL comes first because
// headless, the only mode this example runs in, is the EGL path.
ProcLoader find_proc_loader()
{
    static constexpr struct
    {
        const char* soname;
        const char* symbol;
    } kCandidates[] = {
        { "libEGL.so.1", "eglGetProcAddress" },
        { "libGLX.so.0", "glXGetProcAddressARB" },
        { "libGL.so.1", "glXGetProcAddressARB" },
        { "libGL.so.1", "glXGetProcAddress" },
    };
    for (const auto& candidate : kCandidates)
    {
        void* handle = open_already_loaded(candidate.soname);
        if (handle == nullptr)
        {
            continue;
        }
        if (void* sym = dlsym(handle, candidate.symbol))
        {
            return reinterpret_cast<ProcLoader>(sym);
        }
    }
    throw std::runtime_error(
        "mujoco_xr: found neither eglGetProcAddress nor glXGetProcAddress. The OpenGL "
        "context must be created (mujoco.GLContext) BEFORE the renderer, on this thread.");
}

// Deduces the pointer type from the target, so the caller repeats neither the
// PFNGL...PROC spelling nor a cast.
template <typename Fn>
void resolve(ProcLoader get_proc, Fn& out, const char* name)
{
    void* sym = get_proc(name);
    if (sym == nullptr)
    {
        throw std::runtime_error(std::string("mujoco_xr: OpenGL entry point ") + name +
                                 " is unavailable. Either no context is current on this thread, or it is older "
                                 "than OpenGL 3.3.");
    }
    out = reinterpret_cast<Fn>(sym);
}

} // namespace

void load()
{
    if (loaded_)
    {
        return;
    }
    const ProcLoader get_proc = find_proc_loader();

#define MUJOCO_XR_GL(name, upper) resolve(get_proc, name, "gl" #name);
#include "gl_functions.inc"
#undef MUJOCO_XR_GL

    loaded_ = true;
}

bool loaded()
{
    return loaded_;
}

void check(const char* what)
{
    GLenum first = GL_NO_ERROR;
    for (GLenum err = GetError(); err != GL_NO_ERROR; err = GetError())
    {
        if (first == GL_NO_ERROR)
        {
            first = err;
        }
    }
    if (first == GL_NO_ERROR)
    {
        return;
    }
    static const char kHex[] = "0123456789abcdef";
    std::string code = "0x";
    for (int shift = 12; shift >= 0; shift -= 4)
    {
        code.push_back(kHex[(first >> shift) & 0xF]);
    }
    throw std::runtime_error("mujoco_xr: OpenGL error " + code + " during " + what);
}

} // namespace gl
} // namespace mujoco_xr
