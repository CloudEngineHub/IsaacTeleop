// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Unit tests for the opt-in XrSpaceLocationFlags diagnostic (issue #731).
//
// What is worth asserting here is what is NOT a restatement of the
// implementation:
//   1-3. The exact wire format of each line kind -- these strings are the
//        artifact pasted into the issue, and downstream greps depend on them.
//   4.   Vendor/extension bits reach `raw=` without leaking into the four
//        named fields.
//   5.   The sampler state machine: first call, suppression inside the window,
//        an immediate emit on any mask change, the heartbeat, the `suppressed`
//        accounting, and that a backwards clock does not wedge it.
//   5.   `dpos_m` semantics: measured from the last POSITION_VALID frame,
//        including frames the sampler elided, so a recovery line reports the
//        full lurch magnitude -- and is dropped across a gap in which no
//        measurement happened at all, so no gap is ever reported as a lurch.
//
// Deliberately absent: per-bit `flags & BIT` assertions. They restate the
// implementation and would pass against a copy of the bug.
//
// The diagnostic is enabled for this binary by the ctest ENVIRONMENT property
// in the sibling CMakeLists.txt, set before exec rather than from static
// initialisation in this TU (inter-TU init order is unspecified, and the enable
// flag latches on first call). Running this binary directly therefore needs
// ISAAC_TELEOP_LOG_XR_LOCATION_FLAGS in the environment; the first case below
// says so out loud rather than letting five cases fail obscurely. The disabled
// path cannot be reached from this process at all -- it has its own binary, see
// test_location_flags_disabled.cpp.

#include <catch2/catch_test_macros.hpp>
#include <live_trackers/location_flags_diagnostic.hpp>

#include <cstdint>
#include <iostream>
#include <sstream>
#include <streambuf>
#include <string>
#include <vector>

using core::format_location_flags;
using core::k_location_flags_heartbeat_ns;
using core::location_flags_logging_enabled;
using core::LocationFlagsDiagnostic;

namespace
{

// Redirects std::cout for its lifetime. The diagnostic writes there by design
// (evidence must survive a crash), so capturing the stream is the only way to
// assert on the emitted text.
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

// The one-time banner precedes the first emitted line, so tests select the
// data lines rather than comparing raw output. The version stamp is what
// separates a data line from the banner's prose.
std::vector<std::string> data_lines(const std::string& captured)
{
    std::vector<std::string> lines;
    std::istringstream in(captured);
    std::string line;
    while (std::getline(in, line))
    {
        if (line.find("xr_location_flags v=1 ") != std::string::npos)
        {
            lines.push_back(line);
        }
    }
    return lines;
}

XrPosef make_pose(float x, float y, float z)
{
    XrPosef pose{};
    // Values are chosen to be exactly representable in binary32 so the fixed
    // 4-decimal rendering is deterministic across platforms.
    pose.position = { x, y, z };
    pose.orientation = { 0.0f, 0.0f, 0.0f, 1.0f };
    return pose;
}

constexpr XrSpaceLocationFlags k_valid_only =
    XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;

constexpr XrSpaceLocationFlags k_all_four =
    k_valid_only | XR_SPACE_LOCATION_POSITION_TRACKED_BIT | XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT;

} // namespace

TEST_CASE("this binary runs with the diagnostic enabled", "[location_flags]")
{
    // Every other case in this file asserts on emitted text and is vacuous
    // without this. See the ENVIRONMENT property on catch_discover_tests.
    REQUIRE(location_flags_logging_enabled());
}

TEST_CASE("the banner never collides with the data-line selector", "[location_flags]")
{
    // `xr_location_flags v=1 ` is the contractual selector (see AGENTS.md), and
    // the banner is prose printed on the same stream. Prose that happens to
    // contain the selector silently doubles every downstream line count; it has
    // already happened once.
    LocationFlagsDiagnostic diag;
    CoutCapture capture;
    diag.log_inactive("[T]", "left", "grip", 0, 0);

    // One selector match, banner or no banner. ctest gives each case its own
    // process, so the once-per-process banner is in this capture and prose
    // containing the selector would push the count to 2. Running the binary
    // directly with a filter can let another case consume the banner first --
    // hence no assertion on the banner being *here*, which would fail on nothing
    // more than Catch2's case ordering.
    const auto lines = data_lines(capture.str());
    REQUIRE(lines.size() == 1);
    CHECK(lines[0].find("event=inactive") != std::string::npos);
}

TEST_CASE("sample line is byte-exact in the VALID-but-not-TRACKED case", "[location_flags]")
{
    LocationFlagsDiagnostic diag;
    CoutCapture capture;
    diag.log("[ControllerTracker]", "left", "grip", k_valid_only, make_pose(0.25f, -1.5f, 0.125f), 1234, 5678);

    const auto lines = data_lines(capture.str());
    REQUIRE(lines.size() == 1);
    CHECK(lines[0] ==
          "[ControllerTracker] xr_location_flags v=1 site=left/grip event=sample mono_ns=1234 "
          "xr_ns=5678 position_valid=1 orientation_valid=1 position_tracked=0 orientation_tracked=0 "
          "raw=0x3 pos=(0.2500,-1.5000,0.1250) quat=(0.0000,0.0000,0.0000,1.0000) dpos_m=0.0000 "
          "suppressed=0");
}

TEST_CASE("locate_failed line is byte-exact and carries no measurement", "[location_flags]")
{
    LocationFlagsDiagnostic diag;
    CoutCapture capture;
    diag.log_locate_failed("[HeadTracker]", "head", "view", XR_ERROR_VALIDATION_FAILURE, 42, 99);

    const auto lines = data_lines(capture.str());
    REQUIRE(lines.size() == 1);
    CHECK(lines[0] ==
          "[HeadTracker] xr_location_flags v=1 site=head/view event=locate_failed mono_ns=42 xr_ns=99 "
          "result=XR_ERROR_VALIDATION_FAILURE(-1) suppressed=0");
}

TEST_CASE("inactive line is byte-exact and carries no measurement", "[location_flags]")
{
    LocationFlagsDiagnostic diag;
    CoutCapture capture;
    diag.log_inactive("[ControllerTracker]", "right", "grip", 7, 8);

    const auto lines = data_lines(capture.str());
    REQUIRE(lines.size() == 1);
    CHECK(lines[0] ==
          "[ControllerTracker] xr_location_flags v=1 site=right/grip event=inactive mono_ns=7 xr_ns=8 "
          "suppressed=0");
}

TEST_CASE("an XrResult outside the xrLocateSpace set still prints its integer", "[location_flags]")
{
    LocationFlagsDiagnostic diag;
    CoutCapture capture;
    diag.log_locate_failed("[HeadTracker]", "head", "view", static_cast<XrResult>(-424242), 0, 0);

    const auto lines = data_lines(capture.str());
    REQUIRE(lines.size() == 1);
    CHECK(lines[0].find("result=XR_RESULT_OTHER(-424242)") != std::string::npos);
}

TEST_CASE("a locate failure does not clobber the remembered mask", "[location_flags]")
{
    LocationFlagsDiagnostic diag;
    CoutCapture capture;
    diag.log("[ControllerTracker]", "right", "aim", k_all_four, make_pose(0.0f, 0.0f, 0.0f), 0, 0);
    diag.log_locate_failed("[ControllerTracker]", "right", "aim", XR_ERROR_VALIDATION_FAILURE, 1000, 1000);
    // Emitted because it leaves the gap, not because the mask moved.
    diag.log("[ControllerTracker]", "right", "aim", k_all_four, make_pose(0.0f, 0.0f, 0.0f), 2000, 2000);
    // Same mask again, gap now closed, still inside the heartbeat window: back
    // to suppression, i.e. the failure did not fabricate a mask edge.
    diag.log("[ControllerTracker]", "right", "aim", k_all_four, make_pose(0.0f, 0.0f, 0.0f), 3000, 3000);

    const auto lines = data_lines(capture.str());
    REQUIRE(lines.size() == 3);
    CHECK(lines[1].find("event=locate_failed") != std::string::npos);
    CHECK(lines[2].find("event=sample") != std::string::npos);
}

TEST_CASE("vendor bits reach raw= without leaking into the named fields", "[location_flags]")
{
    const XrSpaceLocationFlags flags = k_all_four | (XrSpaceLocationFlags{ 1 } << 40);
    CHECK(format_location_flags(flags) ==
          "position_valid=1 orientation_valid=1 position_tracked=1 "
          "orientation_tracked=1 raw=0x1000000000f");

    // An unknown bit on its own must not set any named field.
    CHECK(format_location_flags(XrSpaceLocationFlags{ 1 } << 40) ==
          "position_valid=0 orientation_valid=0 position_tracked=0 orientation_tracked=0 raw=0x10000000000");
}

TEST_CASE("sampler emits on mask edges and heartbeats otherwise", "[location_flags]")
{
    LocationFlagsDiagnostic diag;
    const XrPosef pose = make_pose(0.0f, 0.0f, 0.0f);

    SECTION("unchanged mask inside the window is suppressed and counted")
    {
        CoutCapture capture;
        diag.log("[T]", "left", "grip", k_all_four, pose, 0, 0);
        for (int i = 1; i <= 3; ++i)
        {
            diag.log("[T]", "left", "grip", k_all_four, pose, i * 1000, 0);
        }

        auto lines = data_lines(capture.str());
        REQUIRE(lines.size() == 1);
        CHECK(lines[0].find("suppressed=0") != std::string::npos);

        // The heartbeat fires exactly at the interval, and reports the three
        // frames elided since the previous emitted line.
        diag.log("[T]", "left", "grip", k_all_four, pose, k_location_flags_heartbeat_ns, 0);
        lines = data_lines(capture.str());
        REQUIRE(lines.size() == 2);
        CHECK(lines[1].find("suppressed=3") != std::string::npos);
    }

    SECTION("a single-bit change emits immediately inside the window")
    {
        CoutCapture capture;
        diag.log("[T]", "left", "grip", k_all_four, pose, 0, 0);
        diag.log("[T]", "left", "grip", k_all_four, pose, 1000, 0);
        // Only POSITION_TRACKED drops -- the exact signature a collapsed
        // valid/tracked bool would hide.
        diag.log("[T]", "left", "grip", k_all_four & ~XrSpaceLocationFlags{ XR_SPACE_LOCATION_POSITION_TRACKED_BIT },
                 pose, 2000, 0);

        const auto lines = data_lines(capture.str());
        REQUIRE(lines.size() == 2);
        CHECK(lines[1].find("position_tracked=0 orientation_tracked=1") != std::string::npos);
        CHECK(lines[1].find("suppressed=1") != std::string::npos);
    }

    SECTION("bits outside the four-bit mask do not count as an edge")
    {
        CoutCapture capture;
        diag.log("[T]", "left", "grip", k_all_four, pose, 0, 0);
        diag.log("[T]", "left", "grip", k_all_four | (XrSpaceLocationFlags{ 1 } << 40), pose, 1000, 0);

        CHECK(data_lines(capture.str()).size() == 1);
    }

    SECTION("a backwards clock step does not wedge the heartbeat")
    {
        CoutCapture capture;
        diag.log("[T]", "left", "grip", k_all_four, pose, 10 * k_location_flags_heartbeat_ns, 0);
        // Without an explicit guard the elapsed-time test would stay negative
        // for ten seconds of wall clock and the site would go silent.
        diag.log("[T]", "left", "grip", k_all_four, pose, 0, 0);
        diag.log("[T]", "left", "grip", k_all_four, pose, 1000, 0);

        const auto lines = data_lines(capture.str());
        REQUIRE(lines.size() == 2);
        // Re-anchored on the backwards value: the third call is suppressed
        // rather than emitted, so the sampler is running again, not free-running.
        CHECK(lines[1].find("suppressed=0") != std::string::npos);
    }
}

TEST_CASE("dpos_m spans the whole interval since the last POSITION_VALID frame", "[location_flags]")
{
    SECTION("suppressed valid frames still advance the reference position")
    {
        LocationFlagsDiagnostic diag;
        CoutCapture capture;
        diag.log("[T]", "left", "grip", k_all_four, make_pose(0.0f, 0.0f, 0.0f), 0, 0);
        diag.log("[T]", "left", "grip", k_all_four, make_pose(0.0f, 0.0f, 1.0f), 1000, 0);
        diag.log("[T]", "left", "grip", k_all_four, make_pose(0.0f, 0.0f, 3.0f), 2000, 0);
        // Mask edge forces an emit; the reference is the suppressed frame at
        // z=3, not the last emitted one at z=0.
        diag.log("[T]", "left", "grip", k_valid_only, make_pose(0.0f, 0.0f, 3.5f), 3000, 0);

        const auto lines = data_lines(capture.str());
        REQUIRE(lines.size() == 2);
        CHECK(lines[1].find("dpos_m=0.5000") != std::string::npos);
        CHECK(lines[1].find("suppressed=2") != std::string::npos);
    }

    SECTION("an invalid interval is bridged, so the recovery line shows the full lurch")
    {
        LocationFlagsDiagnostic diag;
        CoutCapture capture;
        diag.log("[T]", "left", "grip", k_all_four, make_pose(0.0f, 0.0f, 0.0f), 0, 0);
        // Bits clear: pose is undefined per spec, so it must not become the
        // reference no matter what the runtime left in the struct.
        diag.log("[T]", "left", "grip", 0, make_pose(99.0f, 99.0f, 99.0f), 1000, 0);
        diag.log("[T]", "left", "grip", 0, make_pose(99.0f, 99.0f, 99.0f), 2000, 0);
        diag.log("[T]", "left", "grip", k_all_four, make_pose(0.0f, 0.0f, 2.0f), 3000, 0);

        const auto lines = data_lines(capture.str());
        REQUIRE(lines.size() == 3);
        CHECK(lines[1].find("dpos_m=0.0000") != std::string::npos);
        CHECK(lines[2].find("dpos_m=2.0000") != std::string::npos);
        CHECK(lines[2].find("suppressed=1") != std::string::npos);
    }
}

// The gap cases. An interval in which no measurement happened at all is not a
// degraded measurement: the operator may have set the controller down and picked
// it up a metre away. Printing that metre as a single-frame dpos_m would be
// shape-identical to the tracking lurch this whole diagnostic exists to size, so
// the gap has to be visible in the trace and the reference has to be dropped.
TEST_CASE("a gap with no measurement is marked and never reported as a lurch", "[location_flags]")
{
    SECTION("an inactive controller marks the gap and zeroes the recovery delta")
    {
        LocationFlagsDiagnostic diag;
        CoutCapture capture;
        diag.log("[T]", "right", "grip", k_all_four, make_pose(0.0f, 0.0f, 0.0f), 0, 0);
        diag.log_inactive("[T]", "right", "grip", 1000, 0);
        diag.log_inactive("[T]", "right", "grip", 2000, 0);
        // Picked up 0.9 m away, bits fully nominal. Without the gap marker this
        // is a textbook false positive.
        diag.log("[T]", "right", "grip", k_all_four, make_pose(0.0f, 0.0f, 0.9f), 3000, 0);

        const auto lines = data_lines(capture.str());
        REQUIRE(lines.size() == 3);
        CHECK(lines[1].find("event=inactive") != std::string::npos);
        // The mask is unchanged across the gap, so only prev_gap_ can force
        // this line out; it must not wait for the heartbeat.
        CHECK(lines[2].find("event=sample") != std::string::npos);
        CHECK(lines[2].find("dpos_m=0.0000") != std::string::npos);
        // The elided inactive frame is still accounted for.
        CHECK(lines[2].find("suppressed=1") != std::string::npos);
    }

    SECTION("a long inactive stretch heartbeats instead of flooding")
    {
        LocationFlagsDiagnostic diag;
        CoutCapture capture;
        diag.log_inactive("[T]", "left", "aim", 0, 0);
        for (int i = 1; i <= 3; ++i)
        {
            diag.log_inactive("[T]", "left", "aim", i * 1000, 0);
        }
        CHECK(data_lines(capture.str()).size() == 1);

        diag.log_inactive("[T]", "left", "aim", k_location_flags_heartbeat_ns, 0);
        const auto lines = data_lines(capture.str());
        REQUIRE(lines.size() == 2);
        CHECK(lines[1].find("event=inactive") != std::string::npos);
        CHECK(lines[1].find("suppressed=3") != std::string::npos);
    }

    SECTION("an inactive gap is not a mask edge on the far side")
    {
        LocationFlagsDiagnostic diag;
        CoutCapture capture;
        diag.log("[T]", "left", "grip", k_all_four, make_pose(0.0f, 0.0f, 0.0f), 0, 0);
        diag.log_inactive("[T]", "left", "grip", 1000, 0);
        diag.log("[T]", "left", "grip", k_all_four, make_pose(0.0f, 0.0f, 0.0f), 2000, 0);
        // Same mask again, now with prev_gap_ back to None: back to suppression,
        // i.e. the gap did not clobber the remembered mask.
        diag.log("[T]", "left", "grip", k_all_four, make_pose(0.0f, 0.0f, 0.0f), 3000, 0);

        CHECK(data_lines(capture.str()).size() == 3);
    }

    SECTION("a failed locate also drops the dpos_m reference")
    {
        LocationFlagsDiagnostic diag;
        CoutCapture capture;
        diag.log("[T]", "head", "view", k_all_four, make_pose(0.0f, 0.0f, 0.0f), 0, 0);
        // The call site throws here, so an arbitrary interval may pass before the
        // next sample lands.
        diag.log_locate_failed("[T]", "head", "view", XR_ERROR_TIME_INVALID, 1000, 0);
        diag.log("[T]", "head", "view", k_valid_only, make_pose(0.0f, 0.0f, 5.0f), 2000, 0);

        const auto lines = data_lines(capture.str());
        REQUIRE(lines.size() == 3);
        CHECK(lines[2].find("dpos_m=0.0000") != std::string::npos);
    }

    SECTION("a failed locate marks the gap, so the recovery sample always emits")
    {
        LocationFlagsDiagnostic diag;
        CoutCapture capture;
        diag.log("[T]", "head", "view", k_all_four, make_pose(0.0f, 0.0f, 0.0f), 0, 0);
        diag.log_locate_failed("[T]", "head", "view", XR_ERROR_TIME_INVALID, 1000, 0);
        // Unchanged mask and well inside the heartbeat window: only the gap
        // marker can force this line out, and without it the recovery -- the one
        // line whose dpos_m is deliberately zeroed -- would be missing from the
        // trace entirely.
        diag.log("[T]", "head", "view", k_all_four, make_pose(0.0f, 0.0f, 0.9f), 2000, 0);

        const auto lines = data_lines(capture.str());
        REQUIRE(lines.size() == 3);
        CHECK(lines[2].find("event=sample") != std::string::npos);
        CHECK(lines[2].find("dpos_m=0.0000") != std::string::npos);
    }

    SECTION("a stuck locate failure heartbeats instead of flooding")
    {
        LocationFlagsDiagnostic diag;
        CoutCapture capture;
        for (int i = 0; i <= 3; ++i)
        {
            diag.log_locate_failed("[T]", "head", "view", XR_ERROR_TIME_INVALID, i * 1000, 0);
        }
        CHECK(data_lines(capture.str()).size() == 1);

        diag.log_locate_failed("[T]", "head", "view", XR_ERROR_TIME_INVALID, k_location_flags_heartbeat_ns, 0);
        const auto lines = data_lines(capture.str());
        REQUIRE(lines.size() == 2);
        CHECK(lines[1].find("event=locate_failed") != std::string::npos);
        CHECK(lines[1].find("suppressed=3") != std::string::npos);
    }

    SECTION("a gap that changes kind emits at once")
    {
        LocationFlagsDiagnostic diag;
        CoutCapture capture;
        diag.log_inactive("[T]", "left", "grip", 0, 0);
        // `result=` is the whole payload of this line and the call site throws
        // right after it, so it may be the only one there is; sharing a plain
        // "in a gap" bool would have suppressed it for up to a heartbeat.
        diag.log_locate_failed("[T]", "left", "grip", XR_ERROR_SESSION_LOST, 1000, 0);

        const auto lines = data_lines(capture.str());
        REQUIRE(lines.size() == 2);
        CHECK(lines[1].find("result=XR_ERROR_SESSION_LOST") != std::string::npos);
    }
}
