// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "schema_tracker_base.hpp"

#include <flatbuffers/flatbuffers.h>
#include <mcap/tracker_channels.hpp>
#include <schema/serialized.hpp>
#include <schema/tracked.hpp>

#include <memory>
#include <optional>

namespace core
{

/**
 * @brief Typed SchemaTracker that optionally records to MCAP.
 *
 * Wraps SchemaTrackerBase with FlatBuffer type knowledge so that each sample
 * read from the tensor can be automatically written to an MCAP channel.
 *
 * @tparam RecordT    FlatBuffer record wrapper (e.g. Generic3AxisPedalOutputRecord).
 * @tparam DataTableT FlatBuffer data table (e.g. Generic3AxisPedalOutput).
 * @tparam TrackedT   FlatBuffer Tracked wrapper (e.g. Generic3AxisPedalOutputTracked),
 *                    the type published to consumers.
 */
template <typename RecordT, typename DataTableT, typename TrackedT>
class SchemaTracker : public SchemaTrackerBase
{
public:
    using NativeDataT = typename DataTableT::NativeTableType;
    using Channels = McapTrackerChannels<RecordT, DataTableT>;

    /**
     * @param mcap_channels Non-owning pointer to the MCAP channel writer. Must outlive
     *        this SchemaTracker. Owned by the live tracker impl that also owns this
     *        SchemaTracker instance. Null when recording is disabled.
     * @param mcap_channel_index 0-based sub-channel index within mcap_channels
     *        used for per-sample recording.
     * @param mcap_channel_tracked_index If set, an additional write of only the final
     *        sample per update() call is made to this sub-channel index within the
     *        same mcap_channels. Unset to disable.
     */
    SchemaTracker(const OpenXRSessionHandles& handles,
                  SchemaTrackerConfig config,
                  Channels* mcap_channels = nullptr,
                  size_t mcap_channel_index = 0,
                  std::optional<size_t> mcap_channel_tracked_index = std::nullopt)
        : SchemaTrackerBase(handles, std::move(config)),
          mcap_channels_(mcap_channels),
          mcap_channel_index_(mcap_channel_index),
          mcap_channel_tracked_index_(mcap_channel_tracked_index)
    {
    }

    /**
     * @brief Read all pending samples; write each to MCAP if channels are set.
     *
     * Each sample is unpacked, repacked into a Record with its timestamp, and written
     * to the MCAP channel. When any sample was read, the last one is re-encoded as a
     * Tracked buffer and published through out_tracked.
     *
     * @param out_tracked Receives a freshly encoded snapshot of the final sample when
     *                    samples were read; left untouched when the collection is
     *                    present but produced nothing this tick (last-known sample is
     *                    retained); emptied when the collection is absent.
     * @throws std::runtime_error On critical OpenXR/tensor API failures propagated
     *         from SchemaTrackerBase.
     * @note Missing collection, temporary collection loss, and "no new sample"
     *       are treated as common non-fatal conditions and do not throw.
     */
    void update(Serialized<TrackedT>& out_tracked)
    {
        samples_.clear();
        bool present = read_all_samples(samples_);

        if (samples_.empty())
        {
            if (!present)
            {
                latest_.reset();
                out_tracked = Serialized<TrackedT>();
            }
            return;
        }

        DeviceDataTimestamp last_timestamp{};
        for (const auto& sample : samples_)
        {
            auto fb = flatbuffers::GetRoot<DataTableT>(sample.buffer.data());
            if (!fb)
            {
                continue;
            }

            if (!latest_)
            {
                latest_ = std::make_shared<NativeDataT>();
            }
            fb->UnPackTo(latest_.get());
            last_timestamp = sample.timestamp;

            // write() serializes synchronously and does not retain the shared_ptr,
            // so reusing latest_ across loop iterations is safe.
            if (mcap_channels_)
            {
                mcap_channels_->write(mcap_channel_index_, sample.timestamp, latest_);
            }
        }

        if (mcap_channel_tracked_index_ && mcap_channels_ && latest_)
        {
            mcap_channels_->write(*mcap_channel_tracked_index_, last_timestamp, latest_);
        }

        // Encode into a new buffer rather than over the previous one: consumers hold
        // Serialized snapshots, and a caller that read last frame must keep seeing last
        // frame's values.
        out_tracked = pack_tracked<TrackedT>(latest_);
    }

private:
    Channels* mcap_channels_;
    size_t mcap_channel_index_;
    std::optional<size_t> mcap_channel_tracked_index_;
    std::vector<SampleResult> samples_;
    // Last-known sample, reused as the unpack target across ticks. Internal scratch:
    // it feeds MCAP and the encoder, and never escapes this class.
    std::shared_ptr<NativeDataT> latest_;
};

} // namespace core
