// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Helpers for this repo's `Tracked` / `Record` wrapper tables.
//
// Every wrapper in `fbs/` has the same shape — a single `data` field holding the real
// payload (`HandPoseTracked { data: HandPose }`), optionally alongside a timestamp.
// That is a convention of these schemas, not a property of FlatBuffers, so it lives
// here rather than on `Serialized<T>`, which stays schema-agnostic.
//
// Including this header is the signal that a translation unit depends on that shape.

#pragma once

#include <schema/serialized.hpp>

#include <memory>
#include <utility>

namespace core
{

/*!
 * @brief The `data` field of a wrapper, or null when there is no payload.
 *
 * Collapses the two checks a consumer would otherwise write ("is the handle non-empty"
 * and "is its `data` set") into one, so the usual read is
 * `if (const auto* pose = payload(tracked))`. The return type follows the field, so it
 * is a table pointer for the single-payload wrappers and a vector pointer for the
 * batch ones.
 */
template <typename T>
auto payload(const Serialized<T>& wrapper)
{
    return wrapper ? wrapper->data() : nullptr;
}

/*!
 * @brief Encodes a payload into its wrapper, or yields an empty handle.
 *
 * The shape every single-payload tracker impl needs: a null payload (device inactive,
 * no sample yet, replay gap) becomes an absent handle rather than a buffer holding a
 * null `data`, so consumers test one thing.
 */
template <typename TrackedT, typename DataT>
Serialized<TrackedT> pack_tracked(std::shared_ptr<DataT> data)
{
    if (!data)
    {
        return Serialized<TrackedT>();
    }
    typename TrackedT::NativeTableType native;
    native.data = std::move(data);
    return pack<TrackedT>(native);
}

} // namespace core
