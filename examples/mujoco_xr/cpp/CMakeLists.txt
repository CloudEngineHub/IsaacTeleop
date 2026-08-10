# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The pybind11 module `_mujoco_xr` -- MuJoCo's own OpenGL renderer, read back
# into CUDA-visible buffers for viz::ProjectionLayer.
#
# Protocol-only linkage: this module links no viz:: target. It hands back
# __cuda_array_interface__ objects, so linking viz would buy a dependency on its
# ABI for nothing. For the same reason the pybind11 need not be the one the root
# build FetchContents -- nothing pybind11-registered crosses this boundary. Do
# not pin them together.
#
# No OpenGL on the link line either: cpp/gl.hpp resolves what it calls through
# the platform GetProcAddress, against the context `mujoco.GLContext` created.
# Adding libGL here would introduce a second dispatch path for the same context.
#
# Inherited-scope inputs, all set by ../CMakeLists.txt:
#   _mujoco_library, _mujoco_include_dir, _mujoco_xr_root, _mujoco_xr_standalone

cmake_minimum_required(VERSION 3.20)

find_package(CUDAToolkit REQUIRED)

pybind11_add_module(mujoco_xr_py
    mujoco_xr_bindings.cpp
    gl.cpp
    gl_readback.cpp
    scene_renderer.cpp
    frames.hpp
    gl.hpp
    glcamera.hpp
    gl_readback.hpp
    scene_renderer.hpp
)

target_include_directories(mujoco_xr_py
    PRIVATE
        "${_mujoco_include_dir}"
)

target_link_libraries(mujoco_xr_py
    PRIVATE
        # cudart_static matches viz_core (src/viz/core/cpp/CMakeLists.txt): no
        # runtime libcudart.so dependency on a machine that has only the driver.
        CUDA::cudart_static
        # cuda_gl_interop.h is header-only against cudart, but the loader in
        # gl.cpp needs dlopen/dlsym.
        ${CMAKE_DL_LIBS}
        "${_mujoco_library}"
)

target_compile_options(mujoco_xr_py PRIVATE -Wall -Wextra)

set_target_properties(mujoco_xr_py PROPERTIES
    OUTPUT_NAME "_mujoco_xr"
    # No RPATH, deliberately -- do not "fix" this. The package's __init__.py
    # imports `mujoco` before this extension, so the wheel's already-loaded
    # library satisfies our NEEDED entry. An RPATH pointing at the build
    # interpreter's wheel would silently load a second libmujoco and hand
    # mjModel* pointers across two copies; without one, a mismatch is a clean
    # ImportError.
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH ""
)

# ==============================================================================
# Where the .so goes, and it is a different place in each configure
# ==============================================================================
if(_mujoco_xr_standalone)
    # CMAKE_INSTALL_PREFIX is scikit-build-core's platlib staging root, so the
    # DESTINATION must spell the namespace too: drop the `isaacteleop_examples/`
    # half and the .so lands outside the package.
    #
    # The only owner of a build-produced file in the wheel. What keeps the
    # package copy from also shipping a stale in-place .so is `sdist.exclude` in
    # ../pyproject.toml. The breakage is intermittent: a same-ABI wheel collides
    # by name and looks fine, only a cross-ABI build ships two.
    install(TARGETS mujoco_xr_py
        LIBRARY DESTINATION isaacteleop_examples/mujoco_xr
    )
else()
    # In-tree: drop the .so next to __init__.py so the package imports straight
    # from the source tree, which is what tests/conftest.py reaches by
    # prepending python/ to sys.path (same shape as
    # examples/camera_viz/codec/CMakeLists.txt).
    #
    # ${_mujoco_xr_root} rather than ${CMAKE_SOURCE_DIR}, which is this example's
    # own directory in the standalone configure; rule 2 forbids "../" here.
    # Covered by .gitignore's repo-wide `*.so`. This copies and never removes, so
    # a renamed module leaves a stale .so until the tree is cleaned.
    set_target_properties(mujoco_xr_py PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${_mujoco_xr_root}/python/isaacteleop_examples/mujoco_xr"
    )
endif()
