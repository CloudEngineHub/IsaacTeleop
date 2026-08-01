# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

add_executable(live_trackers_tests
    test_location_flags_diagnostic.cpp
    test_vendor_validation.cpp
)

target_link_libraries(live_trackers_tests PRIVATE
    deviceio::live_trackers
    deviceio::deviceio_trackers
    # Provides the MCAP_IMPLEMENTATION symbols (mcap::McapWriter::*) that
    # live_trackers' McapTrackerChannels instantiations reference; MCAP is
    # compiled once inside deviceio_session (see src/core/mcap/cpp/CMakeLists.txt).
    deviceio::deviceio_session
    Catch2::Catch2WithMain
)

message(STATUS "live_trackers_tests target enabled with Catch2")
# Set pre-exec rather than from a static initializer in the test TU: the enable
# flag latches on its first call, and inter-TU initialization order is
# unspecified, so a future namespace-scope diagnostic in any other source here
# could latch it first and silently make the assertions vacuous.
catch_discover_tests(live_trackers_tests
    PROPERTIES ENVIRONMENT "ISAAC_TELEOP_LOG_XR_LOCATION_FLAGS=1"
)

# The disabled path -- the one every production run takes -- needs its own
# process, because that same latched static means one binary cannot exercise
# both states. See the header comment in the source.
add_executable(live_trackers_location_flags_disabled_tests
    test_location_flags_disabled.cpp
)

# Only the diagnostic TU of live_trackers is referenced here, so unlike the
# sibling target above this one pulls in no tracker impl and no MCAP symbols.
target_link_libraries(live_trackers_location_flags_disabled_tests PRIVATE
    deviceio::live_trackers
    Catch2::Catch2WithMain
)

message(STATUS "live_trackers_location_flags_disabled_tests target enabled with Catch2")
catch_discover_tests(live_trackers_location_flags_disabled_tests)
