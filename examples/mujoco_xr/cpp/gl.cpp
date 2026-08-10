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

#define MUJOCO_XR_GL_DEFINE(ret, name, args) ret(*name) args = nullptr;
MUJOCO_XR_GL_FUNCTIONS(MUJOCO_XR_GL_DEFINE)
#undef MUJOCO_XR_GL_DEFINE

namespace
{

using ProcLoader = void* (*)(const char*);

// RTLD_NOLOAD first: we want the copy the process already loaded, since that is
// the one `mujoco.GLContext` made a context on -- a second copy resolves against
// a dispatch table with no current context. Plain dlopen as a fallback.
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
// so either serves whatever MUJOCO_GL selected; we try EGL first because
// headless (the only mode this example runs in) is the EGL path.
ProcLoader find_proc_loader()
{
    struct Candidate
    {
        const char* soname;
        const char* symbol;
    };
    static constexpr Candidate kCandidates[] = {
        { "libEGL.so.1", "eglGetProcAddress" },
        { "libGLX.so.0", "glXGetProcAddressARB" },
        { "libGL.so.1", "glXGetProcAddressARB" },
        { "libGL.so.1", "glXGetProcAddress" },
    };
    for (const Candidate& c : kCandidates)
    {
        void* handle = open_already_loaded(c.soname);
        if (handle == nullptr)
        {
            continue;
        }
        void* sym = dlsym(handle, c.symbol);
        if (sym != nullptr)
        {
            return reinterpret_cast<ProcLoader>(sym);
        }
    }
    throw std::runtime_error(
        "mujoco_xr: found neither eglGetProcAddress nor glXGetProcAddress. The OpenGL context must be "
        "created (mujoco.GLContext) BEFORE the renderer, on this thread.");
}

bool loaded = false;

} // namespace

void load()
{
    if (loaded)
    {
        return;
    }
    const ProcLoader get_proc = find_proc_loader();

    // Assigned through a void* rather than a reinterpret_cast per line: the
    // -Wall build rejects casting an object pointer straight to a function
    // pointer, and GetProcAddress is defined to return one anyway.
#define MUJOCO_XR_GL_LOAD(ret, name, args)                                                                             \
    {                                                                                                                  \
        void* sym = get_proc("gl" #name);                                                                              \
        if (sym == nullptr)                                                                                            \
        {                                                                                                              \
            throw std::runtime_error(std::string("mujoco_xr: OpenGL entry point gl" #name                              \
                                                 " is unavailable. Either no context is current on this thread, "      \
                                                 "or it is older than OpenGL 3.3."));                                  \
        }                                                                                                              \
        name = reinterpret_cast<ret(*) args>(sym);                                                                     \
    }
    MUJOCO_XR_GL_FUNCTIONS(MUJOCO_XR_GL_LOAD)
#undef MUJOCO_XR_GL_LOAD

    loaded = true;
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
    if (first != GL_NO_ERROR)
    {
        throw std::runtime_error(std::string("mujoco_xr: OpenGL error 0x") +
                                 [](GLenum e)
                                 {
                                     static const char* kHex = "0123456789abcdef";
                                     std::string s;
                                     for (int shift = 12; shift >= 0; shift -= 4)
                                     {
                                         s.push_back(kHex[(e >> shift) & 0xF]);
                                     }
                                     return s;
                                 }(first) +
                                 " during " + what);
    }
}

} // namespace gl
} // namespace mujoco_xr
