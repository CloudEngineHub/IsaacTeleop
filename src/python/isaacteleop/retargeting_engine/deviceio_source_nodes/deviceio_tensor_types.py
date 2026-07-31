# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DeviceIO Tensor Types - Tracked wrapper objects from DeviceIO.

These tensor types represent the encoded payloads returned by DeviceIO trackers.
Each is a handle over the encoded payload: falsy when the device is inactive,
and otherwise read directly through the schema accessors.
"""

import warnings
from enum import IntEnum
from typing import Any
from ..interface.tensor_type import TensorType
from ..interface.tensor_group_type import TensorGroupType
from isaacteleop.schema import (
    HeadPose,
    HandPose,
    ControllerSnapshot,
    Generic3AxisPedalOutput,
    JointStateOutput,
    FullBodyPose,
    MessageChannelMessagesTracked,
)


class HeadPoseTrackedType(TensorType):
    """HeadPose wrapper type from DeviceIO HeadTracker."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def _check_instance_compatibility(self, other: TensorType) -> bool:
        if not isinstance(other, HeadPoseTrackedType):
            raise TypeError(f"Expected HeadPoseTrackedType, got {type(other).__name__}")
        return True

    def validate_value(self, value: Any) -> None:
        if not isinstance(value, HeadPose):
            raise TypeError(
                f"Expected HeadPose for '{self.name}', got {type(value).__name__}"
            )


class HandPoseTrackedType(TensorType):
    """HandPose wrapper type from DeviceIO HandTracker."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def _check_instance_compatibility(self, other: TensorType) -> bool:
        if not isinstance(other, HandPoseTrackedType):
            raise TypeError(f"Expected HandPoseTrackedType, got {type(other).__name__}")
        return True

    def validate_value(self, value: Any) -> None:
        if not isinstance(value, HandPose):
            raise TypeError(
                f"Expected HandPose for '{self.name}', got {type(value).__name__}"
            )


class ControllerSnapshotTrackedType(TensorType):
    """ControllerSnapshot wrapper type from DeviceIO ControllerTracker."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def _check_instance_compatibility(self, other: TensorType) -> bool:
        if not isinstance(other, ControllerSnapshotTrackedType):
            raise TypeError(
                f"Expected ControllerSnapshotTrackedType, got {type(other).__name__}"
            )
        return True

    def validate_value(self, value: Any) -> None:
        if not isinstance(value, ControllerSnapshot):
            raise TypeError(
                f"Expected ControllerSnapshot for '{self.name}', got {type(value).__name__}"
            )


class Generic3AxisPedalOutputTrackedType(TensorType):
    """Generic3AxisPedalOutput wrapper type from DeviceIO Generic3AxisPedalTracker."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def _check_instance_compatibility(self, other: TensorType) -> bool:
        if not isinstance(other, Generic3AxisPedalOutputTrackedType):
            raise TypeError(
                f"Expected Generic3AxisPedalOutputTrackedType, got {type(other).__name__}"
            )
        return True

    def validate_value(self, value: Any) -> None:
        if not isinstance(value, Generic3AxisPedalOutput):
            raise TypeError(
                f"Expected Generic3AxisPedalOutput for '{self.name}', got {type(value).__name__}"
            )


class JointStateOutputTrackedType(TensorType):
    """JointStateOutput wrapper type from DeviceIO JointStateTracker."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def _check_instance_compatibility(self, other: TensorType) -> bool:
        if not isinstance(other, JointStateOutputTrackedType):
            raise TypeError(
                f"Expected JointStateOutputTrackedType, got {type(other).__name__}"
            )
        return True

    def validate_value(self, value: Any) -> None:
        if not isinstance(value, JointStateOutput):
            raise TypeError(
                f"Expected JointStateOutput for '{self.name}', got {type(value).__name__}"
            )


class FullBodyPoseTrackedType(TensorType):
    """FullBodyPose wrapper type from DeviceIO FullBodyTracker.

    Vendor-agnostic: the full-body tracker produces the same FullBodyPose
    payload regardless of the live vendor (native XR, pushed tensor, ...).
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def _check_instance_compatibility(self, other: TensorType) -> bool:
        if not isinstance(other, FullBodyPoseTrackedType):
            raise TypeError(
                f"Expected FullBodyPoseTrackedType, got {type(other).__name__}"
            )
        return True

    def validate_value(self, value: Any) -> None:
        if not isinstance(value, FullBodyPose):
            raise TypeError(
                f"Expected FullBodyPose for '{self.name}', got {type(value).__name__}"
            )


class MessageChannelMessagesTrackedType(TensorType):
    """MessageChannelMessagesTracked wrapper type from DeviceIO MessageChannelTracker."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def _check_instance_compatibility(self, other: TensorType) -> bool:
        if not isinstance(other, MessageChannelMessagesTrackedType):
            raise TypeError(
                f"Expected MessageChannelMessagesTrackedType, got {type(other).__name__}"
            )
        return True

    def validate_value(self, value: Any) -> None:
        if not isinstance(value, MessageChannelMessagesTracked):
            raise TypeError(
                f"Expected MessageChannelMessagesTracked for '{self.name}', got {type(value).__name__}"
            )


class MessageChannelConnectionStatus(IntEnum):
    """Message channel connection states exposed by MessageChannelSource."""

    CONNECTING = 0
    CONNECTED = 1
    SHUTTING = 2
    DISCONNECTED = 3
    UNKNOWN = -1


class MessageChannelStatusType(TensorType):
    """Enum status for message channel connectivity."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def _check_instance_compatibility(self, other: TensorType) -> bool:
        if not isinstance(other, MessageChannelStatusType):
            raise TypeError(
                f"Expected MessageChannelStatusType, got {type(other).__name__}"
            )
        return True

    def validate_value(self, value: Any) -> None:
        if not isinstance(value, MessageChannelConnectionStatus):
            raise TypeError(
                f"Expected MessageChannelConnectionStatus for '{self.name}', got {type(value).__name__}"
            )


def DeviceIOHeadPoseTracked() -> TensorGroupType:
    """Tracked head pose from DeviceIO HeadTracker.

    Contains:
        head_tracked: HeadPose handle (empty when inactive)
    """
    return TensorGroupType("deviceio_head_pose", [HeadPoseTrackedType("head_tracked")])


def DeviceIOHandPoseTracked() -> TensorGroupType:
    """Tracked hand pose from DeviceIO HandTracker.

    Contains:
        hand_tracked: HandPose handle (empty when inactive)
    """
    return TensorGroupType("deviceio_hand_pose", [HandPoseTrackedType("hand_tracked")])


def DeviceIOControllerSnapshotTracked() -> TensorGroupType:
    """Tracked controller snapshot from DeviceIO ControllerTracker.

    Contains:
        controller_tracked: ControllerSnapshot handle (empty when inactive)
    """
    return TensorGroupType(
        "deviceio_controller_snapshot",
        [ControllerSnapshotTrackedType("controller_tracked")],
    )


def DeviceIOGeneric3AxisPedalOutputTracked() -> TensorGroupType:
    """Tracked pedal data from DeviceIO Generic3AxisPedalTracker.

    Contains:
        pedal_tracked: Generic3AxisPedalOutput handle (empty when inactive)
    """
    return TensorGroupType(
        "deviceio_generic_3axis_pedal_output",
        [Generic3AxisPedalOutputTrackedType("pedal_tracked")],
    )


def DeviceIOJointStateOutputTracked() -> TensorGroupType:
    """Tracked joint-state data from DeviceIO JointStateTracker.

    Contains:
        joint_state_tracked: JointStateOutput handle (empty when inactive)
    """
    return TensorGroupType(
        "deviceio_joint_state_output",
        [JointStateOutputTrackedType("joint_state_tracked")],
    )


def DeviceIOFullBodyPoseTracked() -> TensorGroupType:
    """Tracked full body pose data from DeviceIO FullBodyTracker.

    Contains:
        full_body_tracked: FullBodyPose handle (empty when inactive)
    """
    return TensorGroupType(
        "deviceio_full_body_pose",
        [FullBodyPoseTrackedType("full_body_tracked")],
    )


def DeviceIOMessageChannelMessagesTracked() -> TensorGroupType:
    """Tracked message wrapper from DeviceIO MessageChannelTracker."""
    return TensorGroupType(
        "deviceio_message_channel_messages_tracked",
        [MessageChannelMessagesTrackedType("messages_tracked")],
    )


def MessageChannelMessagesTrackedGroup() -> TensorGroupType:
    """Tracked batch of messages drained in one update."""
    return TensorGroupType(
        "message_channel_messages_tracked",
        [MessageChannelMessagesTrackedType("messages_tracked")],
    )


def MessageChannelStatusGroup() -> TensorGroupType:
    """Message channel connection status enum."""
    return TensorGroupType(
        "message_channel_status",
        [MessageChannelStatusType("status")],
    )


# Deprecated aliases resolved lazily via __getattr__ so accessing them emits a
# DeprecationWarning.
_DEPRECATED_ALIASES = {
    "FullBodyPosePicoTrackedType": "FullBodyPoseTrackedType",
    "DeviceIOFullBodyPosePicoTracked": "DeviceIOFullBodyPoseTracked",
}


def __getattr__(name: str):
    new_name = _DEPRECATED_ALIASES.get(name)
    if new_name is not None:
        warnings.warn(
            f"{name} is deprecated; use {new_name} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new_name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
