"""无状态设备模拟量启停周期识别：滞环阈值 + 连续确认 + 采样断档截断。

输入按时间排序的模拟量读数，用开启上阈值 / 关闭下阈值构成滞环（hysteresis），
初始视为关闭态：

- 关闭态下连续 ``confirm_points`` 个读数 **≥ on_threshold** 才切换为开启，
  开启时刻取该连续序列的**首点**时刻；
- 开启态下连续 ``confirm_points`` 个读数 **≤ off_threshold** 才正常关闭，
  关闭时刻同样取该连续序列的**首点**时刻；
- 读数落在两阈值之间（``off_threshold < v < on_threshold``）时**保持原状态**
  并**清空待确认计数**（未确认完的连续序列作废，需重新累计）；
- 相邻读数间隔超过 ``max_gap_seconds``（含等号不断档）即视为采样断档：
  在断档前末点截断当前开启段（若处于开启态），清空全部待确认计数，
  状态重置为关闭并从断档后第一个读数重新开始；
- 序列扫描结束时仍处于开启态的开启段在**末点截断**。

每个开启段输出起止（正常关闭的终点为确认序列首点；截断段终点为截断末点）、
持续秒数与结束原因（normal / gap_truncated / end_truncated）。
时间比较一律换算为 epoch 整数微秒。
"""

from __future__ import annotations

from .models import parse_timestamp
from .resampling import epoch_microseconds, inclusive_gap_microseconds

_MICROSECONDS_PER_SECOND = 1_000_000

# 开启段结束原因代码（close_reason_detail 为对应中文解释）
REASON_NORMAL = "normal"
REASON_GAP_TRUNCATED = "gap_truncated"
REASON_END_TRUNCATED = "end_truncated"


def identify_run_cycles(
    timestamps: list[str],
    values: list[float],
    *,
    on_threshold: float,
    off_threshold: float,
    confirm_points: int,
    max_gap_seconds: float,
    series_id: str | None = None,
) -> dict:
    """识别启停周期。

    ``timestamps`` 必须已按时间排序且无重复（由接口层保证），
    且 ``on_threshold > off_threshold``（由模型校验保证）。
    返回各开启段（起止索引 / 时刻、持续秒数、结束原因）与汇总
    （总开启时长、正常启停次数、各类截断段数）。
    """
    max_gap_us = inclusive_gap_microseconds(max_gap_seconds)
    sample_us = [epoch_microseconds(parse_timestamp(ts)) for ts in timestamps]
    n = len(values)

    segments: list[dict] = []
    is_on = False  # 初始视为关闭
    high_count = 0  # 关闭态下连续 ≥ on_threshold 的读数计数
    low_count = 0  # 开启态下连续 ≤ off_threshold 的读数计数
    high_start = -1  # 当前高位连续序列首点索引（未确认的候选开启点）
    low_start = -1  # 当前低位连续序列首点索引（未确认的候选关闭点）
    open_start = -1  # 当前开启段起点索引

    def truncate_segment(
        end_index: int,
        reason: str,
        *,
        gap_before_time: str | None = None,
        gap_seconds: float | None = None,
    ) -> None:
        """在 end_index 截断当前开启段并登记结果。"""
        duration_seconds = (
            sample_us[end_index] - sample_us[open_start]
        ) / _MICROSECONDS_PER_SECOND
        if reason == REASON_GAP_TRUNCATED:
            detail = (
                f"采样断档截断：断档前末点 {timestamps[end_index]} 之后的相邻间隔 "
                f"{gap_seconds:g}s 超过 max_gap_seconds={max_gap_seconds:g}s，"
                "在断档前末点截断开启段并从关闭态重启"
            )
        else:  # end_truncated
            detail = (
                f"末尾截断：序列末尾（末点 {timestamps[end_index]}）仍处于开启状态，"
                "未观测到满足关闭确认的连续读数，在末点截断开启段"
            )
        segments.append(
            {
                "index": len(segments),
                "start_index": open_start,
                "end_index": end_index,
                "start_time": timestamps[open_start],
                "end_time": timestamps[end_index],
                "point_count": end_index - open_start + 1,
                "duration_seconds": duration_seconds,
                "close_reason": reason,
                "close_reason_detail": detail,
                "gap_before_time": gap_before_time,
                "gap_seconds": gap_seconds,
            }
        )

    for i in range(n):
        # 采样断档：在处理当前点之前，先与前一点比较
        if i > 0:
            delta_us = sample_us[i] - sample_us[i - 1]
            if delta_us > max_gap_us:
                if is_on:
                    truncate_segment(
                        i - 1,
                        REASON_GAP_TRUNCATED,
                        gap_before_time=timestamps[i],
                        gap_seconds=delta_us / _MICROSECONDS_PER_SECOND,
                    )
                # 清空计数并从关闭态重启
                is_on = False
                high_count = low_count = 0
                high_start = low_start = -1
                open_start = -1

        value = values[i]

        if not is_on:
            if value >= on_threshold:
                if high_count == 0:
                    high_start = i
                high_count += 1
                if high_count == confirm_points:
                    # 连续确认完成：开启时刻取该连续序列首点
                    is_on = True
                    open_start = high_start
                    low_count = 0
                    low_start = -1
            else:
                # < on_threshold（含落在两阈值之间与 ≤ off_threshold）：
                # 未开启，待确认高位计数清零
                high_count = 0
                high_start = -1
        else:
            if value <= off_threshold:
                if low_count == 0:
                    low_start = i
                low_count += 1
                if low_count == confirm_points:
                    # 连续确认完成：正常关闭，关闭时刻取该连续序列首点
                    duration_seconds = (
                        sample_us[low_start] - sample_us[open_start]
                    ) / _MICROSECONDS_PER_SECOND
                    segments.append(
                        {
                            "index": len(segments),
                            "start_index": open_start,
                            "end_index": low_start,
                            "start_time": timestamps[open_start],
                            "end_time": timestamps[low_start],
                            "point_count": low_start - open_start + 1,
                            "duration_seconds": duration_seconds,
                            "close_reason": REASON_NORMAL,
                            "close_reason_detail": (
                                f"正常关闭：自 {timestamps[low_start]} 起连续 "
                                f"{confirm_points} 个读数不高于关闭下阈值 "
                                f"{off_threshold:g}，关闭时刻取该连续序列首点"
                            ),
                            "gap_before_time": None,
                            "gap_seconds": None,
                        }
                    )
                    is_on = False
                    open_start = -1
                    high_count = 0
                    high_start = -1
            else:
                # > off_threshold（含落在两阈值之间与 ≥ on_threshold）：
                # 保持开启，待确认低位计数清零
                low_count = 0
                low_start = -1

    # 末尾仍开启：在末点截断
    if is_on:
        truncate_segment(n - 1, REASON_END_TRUNCATED)

    total_on_duration = sum(seg["duration_seconds"] for seg in segments)
    normal_count = sum(1 for seg in segments if seg["close_reason"] == REASON_NORMAL)
    gap_count = sum(
        1 for seg in segments if seg["close_reason"] == REASON_GAP_TRUNCATED
    )
    end_count = sum(
        1 for seg in segments if seg["close_reason"] == REASON_END_TRUNCATED
    )

    return {
        "series_id": series_id,
        "point_count": n,
        "on_threshold": on_threshold,
        "off_threshold": off_threshold,
        "confirm_points": confirm_points,
        "max_gap_seconds": max_gap_seconds,
        "segment_count": len(segments),
        "total_on_duration_seconds": total_on_duration,
        "normal_cycle_count": normal_count,
        "gap_truncated_count": gap_count,
        "end_truncated_count": end_count,
        "segments": segments,
    }
