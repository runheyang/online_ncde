"""系统级 streaming benchmark 的纯调度与统计测试。"""
from __future__ import annotations

import unittest
from dataclasses import dataclass

from evoocc.streaming.system_benchmark_loop import (
    FrameTiming,
    build_dual_system_schedule,
    index_scene_frames_by_token,
    summarize_system_benchmark,
)


@dataclass
class _Meta:
    sample_token: str
    scene_name: str
    timestamp_us: int


def _make_scene(scene_name: str, count: int, start_index: int = 0):
    out = []
    for i in range(count):
        token = f"{scene_name}-{i:03d}"
        out.append(
            (
                start_index + i,
                _Meta(
                    sample_token=token,
                    scene_name=scene_name,
                    timestamp_us=i * 500_000,
                ),
            )
        )
    return out


class SystemScheduleTest(unittest.TestCase):
    def test_five_second_schedule_and_scene_reset(self):
        fast_scene_a = _make_scene("scene-a", 22)
        fast_scene_b = _make_scene("scene-b", 3, start_index=22)
        fast_frames = fast_scene_a + fast_scene_b
        slow_scenes = [
            ("scene-a", [(1000 + i, meta) for i, (_idx, meta) in enumerate(fast_scene_a)]),
            ("scene-b", [(2000 + i, meta) for i, (_idx, meta) in enumerate(fast_scene_b)]),
        ]
        slow_by_token = index_scene_frames_by_token(slow_scenes)

        schedule = build_dual_system_schedule(
            fast_frames,
            slow_by_token,
            slow_interval_sec=5.0,
        )

        slow_positions = [i for i, frame in enumerate(schedule) if frame.is_slow]
        self.assertEqual(slow_positions, [0, 10, 20, 22])
        self.assertEqual(schedule[10].slow_index, 1010)
        self.assertTrue(schedule[0].is_scene_start)
        self.assertTrue(schedule[22].is_scene_start)
        self.assertFalse(schedule[21].is_scene_start)

    def test_negative_interval_only_uses_scene_start(self):
        fast_frames = _make_scene("scene-a", 12)
        slow_by_token = {
            meta.sample_token: (100 + i, meta)
            for i, (_idx, meta) in enumerate(fast_frames)
        }
        schedule = build_dual_system_schedule(fast_frames, slow_by_token, -1.0)
        self.assertEqual([i for i, x in enumerate(schedule) if x.is_slow], [0])

    def test_fast_slow_timestamp_mismatch_fails(self):
        fast_frames = _make_scene("scene-a", 1)
        fast_meta = fast_frames[0][1]
        slow_meta = _Meta(
            sample_token=fast_meta.sample_token,
            scene_name=fast_meta.scene_name,
            timestamp_us=fast_meta.timestamp_us + 1,
        )
        with self.assertRaisesRegex(ValueError, "timestamp"):
            build_dual_system_schedule(
                fast_frames,
                {fast_meta.sample_token: (9, slow_meta)},
                5.0,
            )


class SystemSummaryTest(unittest.TestCase):
    def test_amortized_model_and_e2e_metrics(self):
        timings = []
        for i in range(10):
            is_slow = i == 0
            timings.append(
                FrameTiming(
                    frame_idx=i,
                    sample_token=f"token-{i}",
                    scene_name="scene-a",
                    is_slow=is_slow,
                    frame_wall_ms=50.0 if is_slow else 10.0,
                    input_wait_ms=1.0,
                    h2d_ms=0.5,
                    fast_ms=2.0,
                    slow_ms=10.0 if is_slow else 0.0,
                    align_ms=1.0,
                    post_ms=0.5,
                )
            )

        result = summarize_system_benchmark(
            name="test-system",
            timings=timings,
            total_wall_s=1.0,
            warmup=2,
            peak_memory_mb=123.0,
        )

        self.assertEqual(result["n_slow"], 1)
        self.assertEqual(result["n_regular"], 9)
        self.assertAlmostEqual(result["fps_e2e"], 10.0)
        self.assertAlmostEqual(result["model_latency_ms_mean"], 4.0)
        self.assertAlmostEqual(result["fps_model_amortized"], 250.0)
        self.assertAlmostEqual(result["regular_latency_ms"]["mean"], 10.0)
        self.assertAlmostEqual(result["slow_tick_latency_ms"]["mean"], 50.0)
        self.assertEqual(len(result["trace"]), 10)


if __name__ == "__main__":
    unittest.main()
