# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure helpers from the app that guard against silent-corruption bugs."""

import pytest

app = pytest.importorskip(
    "isaacteleop_examples.mujoco_xr.app", reason="isaacteleop is not on PYTHONPATH"
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0.011, 0.011),
        (0.0, 0.0),
        (-1.0, 0.0),  # clock went backwards
        (5.0, app.MAX_DT_S),  # a long stall
        (float("inf"), app.MAX_DT_S),
    ],
)
def test_clamp_dt(raw, expected):
    assert app._clamp_dt(raw) == expected


def test_clamp_dt_sends_nan_to_zero():
    """The whole reason the clamp is spelled with comparisons.

    ``min(max(nan, 0), 0.1)`` returns nan -- NaN passes through BOTH limits and
    reaches mj_step, which then poisons every qpos in the model. The comparison
    form sends it to 0 because ``nan > 0`` is False.
    """
    assert app._clamp_dt(float("nan")) == 0.0


def test_frame_clock_refuses_the_zeroed_timestamp():
    """Regression: the 50-step physics lurch at every session start.

    ``viz_session.cpp:255-256`` sets ``should_render = false`` AND
    ``predicted_display_time = 0`` together, on every frame before the session
    reaches kRunning. Sampling that zero as a clock reading makes the next real
    frame compute ``dt = t_now - 0``, which clamps to MAX_DT_S and steps the
    simulation 0.1 s inside a single display frame. _frame_clock must report
    "no sample here" instead, so the caller can skip it.
    """

    class _Info:
        predicted_display_time = 0

    assert app._frame_clock(_Info()) is None

    _Info.predicted_display_time = 2_000_000_000  # ns
    assert app._frame_clock(_Info()) == 2.0


def test_near_far_are_a_single_sane_pair():
    assert 0.0 < app.NEAR_Z < app.FAR_Z
    # viz defaults far to 100.0; an arm's-length scene does not want that
    # precision spent 50-100 m away.
    assert app.FAR_Z <= 100.0


class _Fov:
    angle_left = -0.7
    angle_right = 0.7
    angle_up = 0.7
    angle_down = -0.7


def _good_frustum():
    from isaacteleop_examples.mujoco_xr import _mujoco_xr

    return list(
        _mujoco_xr.frustum_from_fov(
            [_Fov.angle_left, _Fov.angle_right, _Fov.angle_up, _Fov.angle_down],
            app.NEAR_Z,
            app.FAR_Z,
        )
    )


def test_assert_frustum_accepts_what_the_renderer_builds():
    """The assertion has to pass on the real thing before its rejections mean
    anything -- float32 round-tripping alone could make it fire on every frame."""
    app._assert_frustum(_good_frustum(), _Fov(), app.NEAR_Z, app.FAR_Z)


def test_assert_frustum_rejects_a_zeroed_half_width():
    """Zero is the one wrong value mjr_render does not complain about: it turns
    the viewport-aspect fallback on."""
    f = _good_frustum()
    f[1] = 0.0
    with pytest.raises(AssertionError, match="degenerate frustum"):
        app._assert_frustum(f, _Fov(), app.NEAR_Z, app.FAR_Z)


def test_assert_frustum_rejects_clip_planes_that_drifted_from_viz():
    f = _good_frustum()
    f[5] = app.FAR_Z * 2.0
    with pytest.raises(AssertionError, match="clip planes drifted"):
        app._assert_frustum(f, _Fov(), app.NEAR_Z, app.FAR_Z)


def test_assert_frustum_rejects_a_frustum_that_does_not_match_its_fov():
    f = _good_frustum()
    f[0] += 0.01  # slide the optical axis without touching anything else
    with pytest.raises(AssertionError, match="frustum left"):
        app._assert_frustum(f, _Fov(), app.NEAR_Z, app.FAR_Z)
