// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tracker.hpp"

#include <schema/serialized.hpp>

namespace core
{

struct FullBodyPoseTracked;

// Abstract base interface for full body tracker implementations.
// Vendor-agnostic: every live/replay backend (native XR, pushed tensor, ...)
// implements this and produces the same Serialized<FullBodyPoseTracked> payload.
class IFullBodyTrackerImpl : public ITrackerImpl
{
public:
    virtual const Serialized<FullBodyPoseTracked>& get_body_pose() const = 0;
};

} // namespace core
