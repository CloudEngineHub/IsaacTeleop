// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/live_trackers/location_flags_diagnostic.hpp"

#include <oxr_utils/oxr_funcs.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace core
{

namespace
{

// The mask an "edge" is defined on. Never collapse these into a single bool:
// a POSITION_TRACKED drop while ORIENTATION_TRACKED stays high is exactly the
// signature this diagnostic exists to catch.
constexpr XrSpaceLocationFlags k_edge_bits =
    XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT |
    XR_SPACE_LOCATION_POSITION_TRACKED_BIT | XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT;

// Fixed decimals for every metric quantity on the line. 1e-4 m is 0.1 mm and
// 1e-4 rad is ~0.006 deg, both far below what this diagnostic reasons about,
// and a fixed width keeps the lines diffable.
constexpr int k_precision = 4;

// Printed once per process, on the first emitted line, so a pasted excerpt is
// self-describing in the issue tracker. '#' prefixes keep it greppable-out.
void emit_banner_once()
{
    static const bool printed = []
    {
        std::cout << "# XrSpaceLocationFlags diagnostic, enabled by the presence of\n"
                     "#   ISAAC_TELEOP_LOG_XR_LOCATION_FLAGS -- the value is ignored, so \"=0\" still enables it.\n"
                     "#   It observes only; no tracker behavior changes when it is on.\n"
                     "# Nominal frames are elided: each site emits on any change in the four bits below and on\n"
                     "#   entering or leaving a gap, and otherwise heartbeats at ~1 Hz. suppressed=<n> counts the\n"
                     "#   frames that site elided since its own previous line -- it is one counter shared by every\n"
                     "#   event kind, so summing it per kind mis-bins them. On the line that ends a gap it counts\n"
                     "#   only the elided gap frames: the frame that opened the gap got its own line, so the gap\n"
                     "#   spans suppressed+1 frames.\n"
                     "# Every data line starts <tag> xr_location_flags v=<n> site=<side>/<space> event=<kind>;\n"
                     "#   this is layout v=1. Select data lines with: grep 'xr_location_flags v=1'\n"
                     "#   event=sample        a completed xrLocateSpace: bits, raw mask, pose, dpos_m.\n"
                     "#   event=locate_failed xrLocateSpace returned an error; result=<NAME>(<int>).\n"
                     "#   event=inactive      the controller was inactive, so no locate ran at all.\n"
                     "#   Only event=sample carries a measurement.\n"
                     "# Bits: *_VALID   = the matching pose field is meaningful this frame.\n"
                     "#       *_TRACKED = the runtime is actively tracking it, not extrapolating.\n"
                     "#   When a VALID bit is clear the matching pose field is undefined per the OpenXR\n"
                     "#   spec, so pos/quat on such a line carry no information -- some runtimes leave\n"
                     "#   garbage there rather than identity, so a non-unit quat is itself a signal.\n"
                     "# Watch for: position_valid=1 with position_tracked=0 -- a pose the pipeline accepts\n"
                     "#   as valid while the runtime is only extrapolating it.\n"
                     "# Units: positions in metres, quaternions in (x,y,z,w) order, both expressed in the\n"
                     "#   base space the host session supplied as OpenXRSessionHandles::space -- the frame\n"
                     "#   MCAP records. That is XR_REFERENCE_SPACE_TYPE_STAGE under the in-tree session,\n"
                     "#   but an embedder may pass any space; LOCAL is the common alternative and is\n"
                     "#   recentered by the runtime, which STAGE normally is not, making the reading rule\n"
                     "#   below more likely to apply.\n"
                     "# dpos_m (event=sample only): the distance from the position on the most recent frame\n"
                     "#   whose POSITION_VALID bit was set -- including frames the sampler elided. It is a\n"
                     "#   single-frame delta on a heartbeat line and spans the gap on a recovery line, and\n"
                     "#   is never a running total. dpos_m=0.0000 with position_valid=0 means \"not\n"
                     "#   measured\", not \"did not move\", and so does the first sample after an\n"
                     "#   event=inactive or event=locate_failed gap.\n"
                     "# mono_ns is the recording join key -- the same value MCAP writes as the monotonic\n"
                     "#   field of DeviceDataTimestamp. xr_ns is the XrTime passed to xrLocateSpace, logged\n"
                     "#   because a discontinuity in it is its own hypothesis; it is NOT a join key.\n"
                     "# Reading rule: a large dpos_m with all four bits nominal and no bit edge points at a\n"
                     "#   session-wide or base-space cause (e.g. recentering), not at occlusion."
                  << std::endl;
        return true;
    }();
    (void)printed;
}

// `result=` is the one field a reader cannot decode from the line itself, and
// it sits on the failure path this diagnostic exists to investigate. The spec
// declares 8 results for xrLocateSpace; the 2 success codes are omitted because
// the only call site is inside `if (XR_FAILED(result))` and cannot reach them.
// Anything else still prints its integer, which is what the thrown
// std::runtime_error carries too.
const char* xr_result_name(XrResult result)
{
    switch (result)
    {
    case XR_ERROR_VALIDATION_FAILURE:
        return "XR_ERROR_VALIDATION_FAILURE";
    case XR_ERROR_RUNTIME_FAILURE:
        return "XR_ERROR_RUNTIME_FAILURE";
    case XR_ERROR_HANDLE_INVALID:
        return "XR_ERROR_HANDLE_INVALID";
    case XR_ERROR_INSTANCE_LOST:
        return "XR_ERROR_INSTANCE_LOST";
    case XR_ERROR_SESSION_LOST:
        return "XR_ERROR_SESSION_LOST";
    case XR_ERROR_TIME_INVALID:
        return "XR_ERROR_TIME_INVALID";
    default:
        return "XR_RESULT_OTHER";
    }
}

// Common prefix of every line kind. `site=` is composed here so call sites
// never concatenate strings at runtime.
std::string format_line_header(
    const char* component_tag, const char* side, const char* space, const char* event, int64_t mono_ns, int64_t xr_ns)
{
    std::ostringstream oss;
    oss << component_tag << " xr_location_flags v=1 site=" << side << '/' << space << " event=" << event
        << " mono_ns=" << mono_ns << " xr_ns=" << xr_ns;
    return oss.str();
}

} // namespace

std::string format_location_flags(XrSpaceLocationFlags flags)
{
    std::ostringstream oss;
    oss << "position_valid=" << ((flags & XR_SPACE_LOCATION_POSITION_VALID_BIT) != 0)
        << " orientation_valid=" << ((flags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) != 0)
        << " position_tracked=" << ((flags & XR_SPACE_LOCATION_POSITION_TRACKED_BIT) != 0)
        << " orientation_tracked=" << ((flags & XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT) != 0) << " raw=0x"
        << std::hex << flags;
    return oss.str();
}

std::string format_pose(const XrPosef& pose)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(k_precision);
    oss << "pos=(" << pose.position.x << ',' << pose.position.y << ',' << pose.position.z << ") quat=("
        << pose.orientation.x << ',' << pose.orientation.y << ',' << pose.orientation.z << ',' << pose.orientation.w
        << ')';
    return oss.str();
}

bool location_flags_logging_enabled()
{
    static const bool enabled = std::getenv("ISAAC_TELEOP_LOG_XR_LOCATION_FLAGS") != nullptr;
    return enabled;
}

void LocationFlagsDiagnostic::log(const char* component_tag,
                                  const char* side,
                                  const char* space,
                                  XrSpaceLocationFlags flags,
                                  const XrPosef& pose,
                                  int64_t mono_ns,
                                  int64_t xr_ns)
{
    if (!location_flags_logging_enabled())
    {
        return;
    }

    const XrSpaceLocationFlags mask = flags & k_edge_bits;
    const int64_t elapsed_ns = mono_ns - last_emit_ns_;
    // An edge is the first line, any change in the four-bit mask, or the first
    // sample after a gap of either kind -- that last one so the recovery frame
    // (the one whose dpos_m is deliberately zeroed) is always in the trace.
    const bool edge = !primed_ || mask != prev_mask_ || prev_gap_ != Gap::None;
    // A backwards clock step must not wedge the heartbeat, so treat it as due.
    const bool heartbeat = !edge && (elapsed_ns >= k_location_flags_heartbeat_ns || elapsed_ns < 0);

    const bool position_valid = (flags & XR_SPACE_LOCATION_POSITION_VALID_BIT) != 0;
    float dpos_m = 0.0f;
    if (position_valid && has_prev_position_)
    {
        const float dx = pose.position.x - prev_position_.x;
        const float dy = pose.position.y - prev_position_.y;
        const float dz = pose.position.z - prev_position_.z;
        dpos_m = std::sqrt(dx * dx + dy * dy + dz * dz);
    }
    if (position_valid)
    {
        prev_position_ = pose.position;
        has_prev_position_ = true;
    }

    if (!edge && !heartbeat)
    {
        ++suppressed_;
        return;
    }

    emit_banner_once();

    std::ostringstream oss;
    oss << format_line_header(component_tag, side, space, "sample", mono_ns, xr_ns) << ' '
        << format_location_flags(flags) << ' ' << format_pose(pose) << std::fixed << std::setprecision(k_precision)
        << " dpos_m=" << dpos_m << " suppressed=" << suppressed_;
    // '\n', not std::endl: an edge storm (a TRACKED bit chattering at frame
    // rate across five sites) would otherwise put a blocking write() on the
    // serial update() loop during exactly the degradation being measured. The
    // heartbeat below flushes at ~1 Hz per site, which bounds crash loss to
    // about a second of buffered edges without ever dropping one.
    std::cout << oss.str() << '\n';
    if (heartbeat)
    {
        std::cout << std::flush;
    }

    prev_mask_ = mask;
    primed_ = true;
    prev_gap_ = Gap::None;
    last_emit_ns_ = mono_ns;
    suppressed_ = 0;
}

void LocationFlagsDiagnostic::log_locate_failed(
    const char* component_tag, const char* side, const char* space, XrResult result, int64_t mono_ns, int64_t xr_ns)
{
    if (!location_flags_logging_enabled())
    {
        return;
    }

    // Unconditional, not just on the emitted frames: the call site throws, so
    // the next sample may be an arbitrary interval later and no position
    // measured before the failure may be compared against one measured after.
    has_prev_position_ = false;

    const int64_t elapsed_ns = mono_ns - last_emit_ns_;
    // Entering the gap always emits, including from the *other* gap kind: an
    // inactive stretch that starts failing must report its `result=` at once,
    // and that first line may be the only one, since the call site throws.
    const bool edge = prev_gap_ != Gap::LocateFailed;
    const bool heartbeat = !edge && (elapsed_ns >= k_location_flags_heartbeat_ns || elapsed_ns < 0);
    if (!edge && !heartbeat)
    {
        ++suppressed_;
        return;
    }

    emit_banner_once();

    std::ostringstream oss;
    oss << format_line_header(component_tag, side, space, "locate_failed", mono_ns, xr_ns)
        << " result=" << xr_result_name(result) << '(' << static_cast<int32_t>(result) << ')'
        << " suppressed=" << suppressed_;
    // std::endl here, unlike the other two entry points: the call site throws
    // immediately after this returns, and an exception that escapes update()
    // terminates the process without flushing std::cout -- which would drop
    // exactly the line that explains the termination. The rate limit above
    // bounds how often this can block.
    std::cout << oss.str() << std::endl;

    // No previous-mask state is advanced here on purpose -- see the header.
    prev_gap_ = Gap::LocateFailed;
    last_emit_ns_ = mono_ns;
    suppressed_ = 0;
}

void LocationFlagsDiagnostic::log_inactive(
    const char* component_tag, const char* side, const char* space, int64_t mono_ns, int64_t xr_ns)
{
    if (!location_flags_logging_enabled())
    {
        return;
    }

    // Unconditional, not just on the emitted frames: the whole point is that no
    // position measured before the gap may be compared against one measured
    // after it.
    has_prev_position_ = false;

    const int64_t elapsed_ns = mono_ns - last_emit_ns_;
    const bool edge = prev_gap_ != Gap::Inactive;
    const bool heartbeat = !edge && (elapsed_ns >= k_location_flags_heartbeat_ns || elapsed_ns < 0);
    if (!edge && !heartbeat)
    {
        ++suppressed_;
        return;
    }

    emit_banner_once();

    std::ostringstream oss;
    oss << format_line_header(component_tag, side, space, "inactive", mono_ns, xr_ns) << " suppressed=" << suppressed_;
    // '\n' plus a heartbeat-only flush, exactly as in log() and for the same
    // reason: an isActive flap at frame rate produces an entry edge every other
    // frame, and flushing each one would put a blocking write() on the serial
    // update() loop, at every site at once.
    std::cout << oss.str() << '\n';
    if (heartbeat)
    {
        std::cout << std::flush;
    }

    // prev_mask_ is deliberately left alone: an inactive gap is not a mask
    // change, and clobbering it would fabricate an edge on the far side. The
    // far side emits regardless, because prev_gap_ forces it. `primed_` is left
    // alone too -- it means "prev_mask_ holds a mask", and no gap line sets one.
    prev_gap_ = Gap::Inactive;
    last_emit_ns_ = mono_ns;
    suppressed_ = 0;
}

} // namespace core
