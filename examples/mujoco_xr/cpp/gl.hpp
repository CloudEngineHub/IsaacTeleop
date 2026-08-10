// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The handful of OpenGL entry points this module calls, resolved at runtime.
//
// No GL headers and no libGL on the link line: the context belongs to
// `mujoco.GLContext` (EGL, GLX or OSMesa, per MUJOCO_GL), and linking libGL here
// would add a second dispatch path to it. Everything used is core GL 3.3, well
// under the 4.5 mjr_render itself needs for glClipControl.

#include <cstddef>
#include <cstdint>

namespace mujoco_xr
{
namespace gl
{

// ── Types ─────────────────────────────────────────────────────────────────
using GLenum = unsigned int;
using GLbitfield = unsigned int;
using GLuint = unsigned int;
using GLint = int;
using GLsizei = int;
using GLfloat = float;
using GLchar = char;
using GLboolean = unsigned char;
using GLsizeiptr = std::ptrdiff_t;
using GLintptr = std::ptrdiff_t;

// ── Enums ─────────────────────────────────────────────────────────────────
// Spelled out rather than included, for the reason in the file comment. Values
// are from the OpenGL registry and are ABI, not choices.
constexpr GLenum GL_NO_ERROR = 0;
constexpr GLenum GL_FALSE = 0;
constexpr GLenum GL_TRUE = 1;
constexpr GLenum GL_TRIANGLES = 0x0004;
constexpr GLenum GL_UNSIGNED_BYTE = 0x1401;
constexpr GLenum GL_FLOAT = 0x1406;
constexpr GLenum GL_RED = 0x1903;
constexpr GLenum GL_RGBA = 0x1908;
constexpr GLenum GL_DEPTH_COMPONENT = 0x1902;
constexpr GLenum GL_DEPTH_STENCIL = 0x84F9;
constexpr GLenum GL_UNSIGNED_INT_24_8 = 0x84FA;
constexpr GLenum GL_FLOAT_32_UNSIGNED_INT_24_8_REV = 0x8DAD;
constexpr GLenum GL_RGBA8 = 0x8058;
constexpr GLenum GL_R32F = 0x822E;
constexpr GLenum GL_DEPTH24_STENCIL8 = 0x88F0;
constexpr GLenum GL_DEPTH32F_STENCIL8 = 0x8CAD;
constexpr GLenum GL_FRAMEBUFFER_ATTACHMENT_COMPONENT_TYPE = 0x8211;
constexpr GLenum GL_DEPTH_ATTACHMENT = 0x8D00;
constexpr GLenum GL_TEXTURE_2D = 0x0DE1;
constexpr GLenum GL_TEXTURE0 = 0x84C0;
constexpr GLenum GL_TEXTURE1 = 0x84C1;
constexpr GLenum GL_TEXTURE_MIN_FILTER = 0x2801;
constexpr GLenum GL_TEXTURE_MAG_FILTER = 0x2800;
constexpr GLenum GL_TEXTURE_WRAP_S = 0x2802;
constexpr GLenum GL_TEXTURE_WRAP_T = 0x2803;
constexpr GLenum GL_NEAREST = 0x2600;
constexpr GLenum GL_CLAMP_TO_EDGE = 0x812F;
constexpr GLenum GL_FRAMEBUFFER = 0x8D40;
constexpr GLenum GL_READ_FRAMEBUFFER = 0x8CA8;
constexpr GLenum GL_DRAW_FRAMEBUFFER = 0x8CA9;
constexpr GLenum GL_COLOR_ATTACHMENT0 = 0x8CE0;
constexpr GLenum GL_COLOR_ATTACHMENT1 = 0x8CE1;
constexpr GLenum GL_DEPTH_STENCIL_ATTACHMENT = 0x821A;
constexpr GLenum GL_FRAMEBUFFER_COMPLETE = 0x8CD5;
constexpr GLenum GL_COLOR_BUFFER_BIT = 0x00004000;
constexpr GLenum GL_DEPTH_BUFFER_BIT = 0x00000100;
constexpr GLenum GL_PIXEL_PACK_BUFFER = 0x88EB;
constexpr GLenum GL_STREAM_READ = 0x88E1;
constexpr GLenum GL_FRAGMENT_SHADER = 0x8B30;
constexpr GLenum GL_VERTEX_SHADER = 0x8B31;
constexpr GLenum GL_COMPILE_STATUS = 0x8B81;
constexpr GLenum GL_LINK_STATUS = 0x8B82;
constexpr GLenum GL_INFO_LOG_LENGTH = 0x8B84;
constexpr GLenum GL_DEPTH_TEST = 0x0B71;
constexpr GLenum GL_CULL_FACE = 0x0B44;
constexpr GLenum GL_BLEND = 0x0BE2;
constexpr GLenum GL_SCISSOR_TEST = 0x0C11;
constexpr GLenum GL_PACK_ALIGNMENT = 0x0D05;
constexpr GLenum GL_DRAW_FRAMEBUFFER_BINDING = 0x8CA6;

// ── Entry points ──────────────────────────────────────────────────────────
// Function pointers rather than declarations, filled by load(). `extern` so
// every TU shares one copy.
#define MUJOCO_XR_GL_FUNCTIONS(X)                                                                                      \
    X(void, Enable, (GLenum))                                                                                          \
    X(void, Disable, (GLenum))                                                                                         \
    X(GLenum, GetError, ())                                                                                            \
    X(void, Viewport, (GLint, GLint, GLsizei, GLsizei))                                                                \
    X(void, PixelStorei, (GLenum, GLint))                                                                              \
    X(void, GetIntegerv, (GLenum, GLint*))                                                                             \
    X(void, ReadPixels, (GLint, GLint, GLsizei, GLsizei, GLenum, GLenum, void*))                                       \
    X(void, DrawArrays, (GLenum, GLint, GLsizei))                                                                      \
    X(void, GenTextures, (GLsizei, GLuint*))                                                                           \
    X(void, DeleteTextures, (GLsizei, const GLuint*))                                                                  \
    X(void, BindTexture, (GLenum, GLuint))                                                                             \
    X(void, TexImage2D, (GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void*))                  \
    X(void, TexParameteri, (GLenum, GLenum, GLint))                                                                    \
    X(void, ActiveTexture, (GLenum))                                                                                   \
    X(void, GenFramebuffers, (GLsizei, GLuint*))                                                                       \
    X(void, DeleteFramebuffers, (GLsizei, const GLuint*))                                                              \
    X(void, BindFramebuffer, (GLenum, GLuint))                                                                         \
    X(void, FramebufferTexture2D, (GLenum, GLenum, GLenum, GLuint, GLint))                                             \
    X(GLenum, CheckFramebufferStatus, (GLenum))                                                                        \
    X(void, GetFramebufferAttachmentParameteriv, (GLenum, GLenum, GLenum, GLint*))                                     \
    X(void, BlitFramebuffer, (GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLbitfield, GLenum))             \
    X(void, DrawBuffers, (GLsizei, const GLenum*))                                                                     \
    X(void, ReadBuffer, (GLenum))                                                                                      \
    X(void, GenBuffers, (GLsizei, GLuint*))                                                                            \
    X(void, DeleteBuffers, (GLsizei, const GLuint*))                                                                   \
    X(void, BindBuffer, (GLenum, GLuint))                                                                              \
    X(void, BufferData, (GLenum, GLsizeiptr, const void*, GLenum))                                                     \
    X(void, GenVertexArrays, (GLsizei, GLuint*))                                                                       \
    X(void, DeleteVertexArrays, (GLsizei, const GLuint*))                                                              \
    X(void, BindVertexArray, (GLuint))                                                                                 \
    X(GLuint, CreateShader, (GLenum))                                                                                  \
    X(void, ShaderSource, (GLuint, GLsizei, const GLchar* const*, const GLint*))                                       \
    X(void, CompileShader, (GLuint))                                                                                   \
    X(void, GetShaderiv, (GLuint, GLenum, GLint*))                                                                     \
    X(void, GetShaderInfoLog, (GLuint, GLsizei, GLsizei*, GLchar*))                                                    \
    X(void, DeleteShader, (GLuint))                                                                                    \
    X(GLuint, CreateProgram, ())                                                                                       \
    X(void, AttachShader, (GLuint, GLuint))                                                                            \
    X(void, LinkProgram, (GLuint))                                                                                     \
    X(void, GetProgramiv, (GLuint, GLenum, GLint*))                                                                    \
    X(void, GetProgramInfoLog, (GLuint, GLsizei, GLsizei*, GLchar*))                                                   \
    X(void, DeleteProgram, (GLuint))                                                                                   \
    X(void, UseProgram, (GLuint))                                                                                      \
    X(GLint, GetUniformLocation, (GLuint, const GLchar*))                                                              \
    X(void, Uniform1i, (GLint, GLint))

#define MUJOCO_XR_GL_DECLARE(ret, name, args) extern ret(*name) args;
MUJOCO_XR_GL_FUNCTIONS(MUJOCO_XR_GL_DECLARE)
#undef MUJOCO_XR_GL_DECLARE

// Resolves every entry point above against the CURRENT context. Idempotent, so
// callers need not track whether it has run. Throws std::runtime_error naming
// the first function that could not be resolved, which in practice means either
// no context is current or the context is too old.
void load();

// Throws std::runtime_error naming `what` if glGetError() is not GL_NO_ERROR.
// Drains the error queue either way, so one stale error cannot fail every later
// check.
void check(const char* what);

} // namespace gl
} // namespace mujoco_xr
