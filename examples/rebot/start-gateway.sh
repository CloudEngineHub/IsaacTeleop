#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Start the motorbridge WS gateway for the reBot B601-RS arm, then point
# Motorbridge Studio (https://motorbridge.github.io/motorbridge-studio/) at it.
#
# The gateway takes exclusive ownership of the CAN bus — stop it before running
# any motorbridge-cli command.
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# The gateway refuses a non-loopback bind unless this is set. The default is a
# placeholder: on a shared network, pass a real token.
: "${MOTORBRIDGE_WS_TOKEN:=0000}"
: "${BIND:=0.0.0.0:9002}"
export MOTORBRIDGE_WS_TOKEN

# Router mode starts fine without CAN, so a missing bus is a warning. Only an
# absent or explicitly-down interface is reported; CAN links often read
# "unknown" while perfectly usable.
can_state="$(cat /sys/class/net/can0/operstate 2>/dev/null || true)"
if [[ -z "$can_state" ]]; then
  echo "warning: can0 not found — no motors will respond (see SKILL.md, Step 2)" >&2
elif [[ "$can_state" == "down" ]]; then
  echo "warning: can0 is down — run: sudo ip link set can0 up" >&2
fi

echo "gateway listening on ws://$BIND (token required)"
exec uv run --project "$here" motorbridge-gateway --bind "$BIND" "$@"
