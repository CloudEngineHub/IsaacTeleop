// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The disabled path of the XrSpaceLocationFlags diagnostic (issue #731).
//
// This is the configuration that actually ships: the environment variable is
// unset in every production run, so "emits nothing, mutates nothing" is the
// property that has to hold, and it is the one the sibling test file cannot
// check. `location_flags_logging_enabled()` latches a function-local static on
// its first call, so the enabled and disabled states are not both reachable in
// one process -- hence a second binary rather than a second TEST_CASE.
//
// The variable is cleared here rather than left to the ambient environment so
// the assertion holds even for a developer who has it exported. Doing it from
// static initialisation is sound in this target specifically: it is the only TU
// that references the diagnostic at all, so no other TU can latch the flag
// first. (The enabled binary cannot use the same trick -- there the guarantee
// would have to hold across TUs, whose init order is unspecified, so it takes
// the enable from ctest's ENVIRONMENT property instead.)

#include <catch2/catch_test_macros.hpp>
#include <live_trackers/location_flags_diagnostic.hpp>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <streambuf>
#include <string>

using core::location_flags_logging_enabled;
using core::LocationFlagsDiagnostic;

namespace
{

struct ClearLoggingEnv
{
    ClearLoggingEnv()
    {
#ifdef _WIN32
        // MSVC has no unsetenv; assigning an empty value removes the variable
        // from the process environment, which is what the presence-based toggle
        // has to see. <stdlib.h> is included explicitly rather than relying on
        // <cstdlib> to pull the global _putenv_s in: nobody on this side can
        // build against MSVC, so the one unverified link is spelled out.
        _putenv_s("ISAAC_TELEOP_LOG_XR_LOCATION_FLAGS", "");
#else
        unsetenv("ISAAC_TELEOP_LOG_XR_LOCATION_FLAGS");
#endif
    }
};

const ClearLoggingEnv clear_logging_env{};

class CoutCapture
{
public:
    CoutCapture() : previous_(std::cout.rdbuf(buffer_.rdbuf()))
    {
    }

    ~CoutCapture()
    {
        std::cout.rdbuf(previous_);
    }

    CoutCapture(const CoutCapture&) = delete;
    CoutCapture& operator=(const CoutCapture&) = delete;

    std::string str() const
    {
        return buffer_.str();
    }

private:
    std::ostringstream buffer_;
    std::streambuf* previous_;
};

XrPosef identity_pose()
{
    XrPosef pose{};
    pose.orientation = { 0.0f, 0.0f, 0.0f, 1.0f };
    return pose;
}

} // namespace

TEST_CASE("an unset variable leaves the diagnostic off", "[location_flags]")
{
    REQUIRE_FALSE(location_flags_logging_enabled());
}

TEST_CASE("no entry point writes anything when disabled", "[location_flags]")
{
    LocationFlagsDiagnostic diag;
    CoutCapture capture;

    // Every kind of line, including the one-time banner, must stay unwritten --
    // and the mask edge between the two log() calls must not force one out.
    diag.log("[T]", "left", "grip", XR_SPACE_LOCATION_POSITION_VALID_BIT, identity_pose(), 0, 0);
    diag.log("[T]", "left", "grip", 0, identity_pose(), 1000, 0);
    diag.log_locate_failed("[T]", "left", "grip", XR_ERROR_VALIDATION_FAILURE, 2000, 0);
    diag.log_inactive("[T]", "left", "grip", 3000, 0);

    CHECK(capture.str().empty());
}
