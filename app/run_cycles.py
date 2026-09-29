"""无状态设备模拟量启停周期识别：上下阈值 + 连续确认点数 + 采样断档截断。

输入按时间排序的设备模拟量读数（如电机电流、泵出口压力、功率等），用施密特
风格的双阈值 + 连续确认点数识别开启 / 关闭状态机，初始视为**关闭**：

- 关闭态下连续 ``confirm_points`` 个读数 **≥ ``on_threshold``** 才**开启**，
  开启时刻取该连续序列的**首点**时刻；
- 开启态下连续 ``confirm_points`` 个读数 **≤ ``off_threshold``** 才**关闭**，
  关闭时刻取该连续序列的**首点**时刻（开启段在该点结束）；
- 读数落在两阈值之间（``off_threshold < v < on_threshold``）时**保持原状态**，
  并**清除待确认计数**（开启 / 关闭确认都要求严格连续）；
- 要求 ``on_threshold > off_threshold``，中间存在迟滞带，可抑制阈值附近抖动。

采样断档：相邻读数间隔**超过** ``max_gap_seconds`` 时，若当时处于开启态，
在**断档前末点**截断当前开启段（``gap_truncated``），随后清空待确认计数并
从**关闭态**重新开始（断档后第一个点按关闭态重新评估）；扫描结束时若仍处于
开启态，同样在**末点**截断（``end_truncated``）。只有经关闭确认正常收尾的
段才标 ``normal_close``。

时间比较一律换算为 epoch 整数微秒，``max_gap_seconds`` 含等号语义
（间隔恰好等于阈值不算断档）。
"""

from __future__ import annotations

from .models import parse_timestamp
from .resampling import epoch_microseconds, inclusive_gap_microseconds

_MICROSECONDS_PER_SECOND = 1_000_000

# 开启段收尾原因代码（reason_detail 为对应中文解释）
CLOSE_NORMAL = "normal_close"
CLOSE_GAP = "gap_truncated"
CLOSE_END = "end_truncated"


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
    """识别设备模拟量读数中的开启 / 关闭周期。

    ``timestamps`` 必须已按时间排序且无重复（由接口层保证）。
    返回各开启段（起止索引 / 时刻 / 读数、持续秒数、收尾原因）与汇总
    （总开启时长、正常启停次数、各收尾原因计数）。
    """
    max_gap_us = inclusive_gap_microseconds(max_gap_seconds)
    sample_us = [epoch_microseconds(parse_timestamp(ts)) for ts in timestamps]
    n = len(values)

    # 状态：is_on=False 初始关闭；pending_run_start 记录当前连续确认序列首点索引
    is_on = False
    pending_count = 0
    pending_run_start: int | None = None

    segments: list[dict] = []
    # 当前开启段：{"start_index", "start_time", "start_value", "start_us"}
    open_segment: dict | None = None
    total_open_us = 0

    def emit_segment(end_index: int, reason: str, detail: str) -> None:
        """把当前开启段以 end_index 点收尾并计入汇总。"""
        nonlocal open_segment, total_open_us
        assert open_segment is not None
        duration_us = sample_us[end_index] - open_segment["start_us"]
        total_open_us += duration_us
        segments.append(
            {
                "index": len(segments),
                "start_index": open_segment["start_index"],
                "end_index": end_index,
                "start_time": open_segment["start_time"],
                "end_time": timestamps[end_index],
                "start_value": open_segment["start_value"],
                "end_value": values[end_index],
                "point_count": end_index - open_segment["start_index"] + 1,
                "duration_seconds": duration_us / _MICROSECONDS_PER_SECOND,
                "close_reason": reason,
                "reason_detail": detail,
            }
        )
        open_segment = None

    def reset_pending() -> None:
        nonlocal pending_count, pending_run_start
        pending_count = 0
        pending_run_start = None

    for i in range(n):
        # 断档判定优先：相邻间隔超过 max_gap_seconds 即在断档前末点截断开启段，
        # 清空待确认计数并从关闭态重启（当前点随后按关闭态重新评估）
        if i > 0 and sample_us[i] - sample_us[i - 1] > max_gap_us:
            gap_seconds = (sample_us[i] - sample_us[i - 1]) / _MICROSECONDS_PER_SECOND
            if is_on:
                emit_segment(
                    i - 1,
                    CLOSE_GAP,
                    (
                        f"采样断档截断：{timestamps[i - 1]} → {timestamps[i]} 间隔 "
                        f"{gap_seconds:g}s 超过 max_gap_seconds={max_gap_seconds:g}s，"
                        f"开启段在断档前末点 {timestamps[i - 1]} 截断，状态重置为关闭"
                    ),
                )
                is_on = False
            reset_pending()

        v = values[i]
        if not is_on:
            if v >= on_threshold:
                if pending_count == 0:
                    pending_run_start = i
                pending_count += 1
                if pending_count >= confirm_points:
                    # 连续确认开启：开启时刻取连续序列首点
                    start = pending_run_start
                    assert start is not None
                    is_on = True
                    open_segment = {
                        "start_index": start,
                        "start_time": timestamps[start],
                        "start_value": values[start],
                        "start_us": sample_us[start],
                    }
                    reset_pending()
            else:
                # 未达上阈值（含迟滞带内与 ≤ 下阈值）：继续关闭，确认计数清零
                reset_pending()
        else:
            if v <= off_threshold:
                if pending_count == 0:
                    pending_run_start = i
                pending_count += 1
                if pending_count >= confirm_points:
                    # 连续确认关闭：关闭时刻取连续序列首点（开启段在该点结束）
                    start = pending_run_start
                    assert start is not None
                    emit_segment(
                        start,
                        CLOSE_NORMAL,
                        (
                            f"正常关闭：自 {timestamps[start]} 起连续 {confirm_points} 个读数"
                            f"不高于关闭下阈值 {off_threshold:g}（末点 {timestamps[i]} "
                            f"读数 {v:g}），关闭时刻取该连续序列首点"
                        ),
                    )
                    is_on = False
                    reset_pending()
            else:
                # 高于下阈值（含迟滞带内与 ≥ 上阈值）：保持开启，清除待确认计数
                reset_pending()

    # 末尾仍开启：在末点截断
    if is_on:
        emit_segment(
            n - 1,
            CLOSE_END,
            (
                f"末尾截断：截至末点 {timestamps[n - 1]} 未出现连续 {confirm_points} 个"
                f"不高于关闭下阈值 {off_threshold:g} 的读数，开启段在末点截断"
            ),
        )

    close_counts = {CLOSE_NORMAL: 0, CLOSE_GAP: 0, CLOSE_END: 0}
    for seg in segments:
        close_counts[seg["close_reason"]] += 1

    return {
        "series_id": series_id,
        "point_count": n,
        "on_threshold": on_threshold,
        "off_threshold": off_threshold,
        "confirm_points": confirm_points,
        "max_gap_seconds": max_gap_seconds,
        "segment_count": len(segments),
        "total_open_seconds": total_open_us / _MICROSECONDS_PER_SECOND,
        "normal_cycle_count": close_counts[CLOSE_NORMAL],
        "gap_truncated_count": close_counts[CLOSE_GAP],
        "end_truncated_count": close_counts[CLOSE_END],
        "segments": segments,
    }
