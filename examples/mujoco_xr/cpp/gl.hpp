// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The OpenGL entry points this module calls (gl_functions.inc), resolved at
// runtime against the context `mujoco.GLContext` created. Nothing here links
// libGL: glcorearb.h supplies the enums and the PFNGL...PROC typedefs but
// declares no function, because GL_GLEXT_PROTOTYPES is not defined.
//
// glcorearb.h costs no build dependency. It ships in the same package as
// <GL/gl.h>, which cuda_gl_interop.h includes unconditionally.

#include <GL/glcorearb.h>

namespace mujoco_xr
{
namespace gl
{

#define MUJOCO_XR_GL(name, upper) extern PFNGL##upper##PROC name;
#include "gl_functions.inc"
#undef MUJOCO_XR_GL

// Resolves every entry point above against the CURRENT context. Idempotent, so
// callers need not track whether it has run. Throws std::runtime_error naming
// the first function that could not be resolved, which in practice means either
// no context is current or the context is older than OpenGL 3.3.
void load();

// Whether load() has succeeded. Teardown paths need it: they are reached with
// nothing loaded when construction threw.
bool loaded();

// Throws std::runtime_error naming `what` if glGetError() is not GL_NO_ERROR.
// Drains the error queue either way, so one stale error cannot fail every later
// check.
void check(const char* what);

} // namespace gl
} // namespace mujoco_xr
