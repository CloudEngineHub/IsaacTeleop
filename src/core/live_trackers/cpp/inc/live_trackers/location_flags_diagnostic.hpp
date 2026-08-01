// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <oxr_utils/oxr_funcs.hpp>

#include <cstdint>
#include <string>

namespace core
{

// Delete when `is_tracked` reaches MCAP (Phase 1 of issue #731). Exists only
// because the TRACKED bits are otherwise unobservable; VALID already is, via
// `is_valid`.
//
// This changes no tracker behavior. When the variable is unset every entry
// point returns after a single cached bool test, before touching any state.

// "position_valid=<0|1> orientation_valid=<0|1> position_tracked=<0|1>
//  orientation_tracked=<0|1> raw=0x<hex>". `raw` is the whole mask, so a
// vendor/extension bit (or a bug in the decoding above) stays visible.
std::string format_location_flags(XrSpaceLocationFlags flags);

// "pos=(x,y,z) quat=(x,y,z,w)", metres and (x,y,z,w) order, fixed 4 decimals.
std::string format_pose(const XrPosef& pose);

// Cached once on first call. Presence of the variable is the toggle.
bool location_flags_logging_enabled();

// Steady-state emission floor per site when the four-bit mask is unchanged.
constexpr int64_t k_location_flags_heartbeat_ns = 1'000'000'000;

// One instance per observed space (per controller side and per grip/aim, one
// for the head view). Not thread-safe: it carries the same single-threaded
// per-tracker `update()` assumption `last_update_time_` already carries.
class LocationFlagsDiagnostic
{
public:
    // Emits when the four-bit mask differs from the last emitted one, else at
    // most once per k_location_flags_heartbeat_ns. `mono_ns` must be the
    // tracker's `last_update_time_` (the MCAP join key), `xr_ns` the XrTime
    // handed to xrLocateSpace. `pose` must be the raw `XrSpaceLocation::pose`,
    // never a derived/zeroed copy.
    void log(const char* component_tag,
             const char* side,
             const char* space,
             XrSpaceLocationFlags flags,
             const XrPosef& pose,
             int64_t mono_ns,
             int64_t xr_ns);

    // xrLocateSpace failed, so this frame carries no measurement. Opens a gap
    // (see `Gap` below) and so is rate-limited exactly like `log_inactive`.
    // Carries no bits or pose and deliberately leaves the previous-mask state
    // untouched: a failed locate is not a measurement, and advancing the mask
    // here would fabricate a spurious edge on the next successful line. It does
    // drop the `dpos_m` reference, because the call site throws and an unknown
    // amount of time may pass before the next sample; carrying the reference
    // across would fabricate a lurch.
    void log_locate_failed(
        const char* component_tag, const char* side, const char* space, XrResult result, int64_t mono_ns, int64_t xr_ns);

    // The controller was reported inactive, so no locate happened at all.
    // Marks the gap in the trace and drops the `dpos_m` reference: a controller
    // set down and picked up a metre away would otherwise print that metre as a
    // single-frame delta, which is indistinguishable from a tracking lurch.
    void log_inactive(const char* component_tag, const char* side, const char* space, int64_t mono_ns, int64_t xr_ns);

private:
    // An interval in which no measurement happened at all. Both kinds are
    // rate-limited the same way -- entering the gap always emits, the frames
    // between are counted in `suppressed`, a heartbeat marks a long gap as
    // ongoing, and leaving it always emits, whether the next line is a sample
    // or the other kind of gap. The kind is tracked rather than a plain bool so
    // that a stretch of `inactive` frames which starts failing still reports its
    // `result=` on the first failing frame -- which, because the call site
    // throws, may be the only one there is.
    enum class Gap : std::uint8_t
    {
        None,
        Inactive,
        LocateFailed,
    };

    XrSpaceLocationFlags prev_mask_ = 0;
    // True once `prev_mask_` holds a mask from an emitted sample. Gap lines
    // never set it: a gap is not a mask.
    bool primed_ = false;
    // The gap the last emitted line for this site recorded, so the first line
    // out of that gap is always emitted rather than suppressed.
    Gap prev_gap_ = Gap::None;
    int64_t last_emit_ns_ = 0;
    int64_t suppressed_ = 0;
    // Last position seen with POSITION_VALID set, updated on suppressed frames
    // too so `dpos_m` on a recovery line spans the whole degraded interval.
    XrVector3f prev_position_{ 0.0f, 0.0f, 0.0f };
    bool has_prev_position_ = false;
};

} // namespace core
