// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tracker.hpp"

#include <schema/serialized.hpp>

namespace core
{

struct HandPoseTracked;

// Abstract base interface for hand tracker implementations.
class IHandTrackerImpl : public ITrackerImpl
{
public:
    virtual const Serialized<HandPoseTracked>& get_left_hand() const = 0;
    virtual const Serialized<HandPoseTracked>& get_right_hand() const = 0;
};

} // namespace core
