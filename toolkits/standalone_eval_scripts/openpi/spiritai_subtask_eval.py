# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Evaluate explicit SpiritAI subtask generation on recorded LeRobot v2.1 frames."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from rlinf.robotics.spiritai import CAMERA_KEYS, STATE_KEYS, pack_cartesian_fields
from rlinf.utils.logging import get_logger

logger = get_logger()


def _normalize_text(text: str) -> str:
    return " ".join(text.casefold().split()).rstrip(".")


def select_frames(
    labels: list[str], *, seed: int, radius: int, stride: int
) -> tuple[dict[int, list[str]], int | None]:
    """Select every segment midpoint and one reproducible boundary window."""
    if not labels or any(
        not isinstance(label, str) or not label.strip() for label in labels
    ):
        raise ValueError(
            "Expected nonempty subtask annotations for every episode frame."
        )
    if radius < stride or stride <= 0 or radius % stride:
        raise ValueError("Expected positive stride and radius divisible by stride.")

    starts = [0] + [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]]
    ends = starts[1:] + [len(labels)]
    selected: dict[int, list[str]] = {
        (start + end - 1) // 2: ["segment_center"]
        for start, end in zip(starts, ends, strict=True)
    }
    if len(starts) == 1:
        return dict(sorted(selected.items())), None

    index = int(np.random.default_rng(seed).integers(1, len(starts)))
    boundary = starts[index]
    before = max(starts[index - 1], boundary - 3)
    after = min(ends[index] - 1, boundary + 3)
    selected.setdefault(before, []).append("boundary_before")
    selected.setdefault(after, []).append("boundary_after")
    first, last = starts[index - 1], ends[index] - 1
    for offset in range(-radius, radius + 1, stride):
        if first <= boundary + offset <= last:
            selected.setdefault(boundary + offset, []).append("boundary_window")
    return dict(sorted(selected.items())), boundary


def _local_path(root: Path, template: str, **fields: Any) -> Path:
    path = (root / template.format(**fields)).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(f"Expected a dataset file under {root}, got {path}.")
    return path


def _read_episode(
    root: Path, info: dict, episode: int, length: int, args: argparse.Namespace
) -> tuple[Any, dict[int, list[str]], int | None]:
    import pyarrow.parquet as pq

    path = _local_path(
        root,
        info["data_path"],
        episode_chunk=episode // info["chunks_size"],
        episode_index=episode,
    )
    columns = [
        *STATE_KEYS,
        "episode_index",
        "frame_index",
        "timestamp",
        "task_index",
        "subtask",
    ]
    table = pq.read_table(path, columns=columns)
    if (
        len(table) != length
        or table["frame_index"].to_pylist() != list(range(length))
        or set(table["episode_index"].to_pylist()) != {episode}
    ):
        raise ValueError(
            f"Expected {length} ordered rows for episode {episode} in {path}."
        )
    selected, boundary = select_frames(
        table["subtask"].to_pylist(),
        seed=args.seed + episode,
        radius=args.boundary_radius,
        stride=args.boundary_stride,
    )
    return table, selected, boundary


def _camera_observations(
    root: Path, info: dict, episode: int, frames: list[int]
) -> list:
    from torchcodec.decoders import VideoDecoder

    views = []
    for name in CAMERA_KEYS:
        path = _local_path(
            root,
            info["video_path"],
            episode_chunk=episode // info["chunks_size"],
            episode_index=episode,
            video_key=name,
        )
        decoder = VideoDecoder(path, num_ffmpeg_threads=1, device="cpu")
        views.append(decoder.get_frames_at(indices=frames).data.permute(0, 2, 3, 1))
    return views


def _evaluate_episode(
    model: Any,
    root: Path,
    info: dict,
    episode: int,
    length: int,
    tasks: dict,
    phase_map: dict[str, str],
    args: argparse.Namespace,
):
    import torch

    table, selected, boundary = _read_episode(root, info, episode, length, args)
    frames = list(selected)
    observations = table.take(frames)
    images = _camera_observations(root, info, episode, frames)
    states = torch.from_numpy(
        pack_cartesian_fields(
            {key: observations[key].to_pylist() for key in STATE_KEYS},
            STATE_KEYS,
        )
    )
    instructions = [tasks[index] for index in observations["task_index"].to_pylist()]
    for start in range(0, len(frames), args.batch_size):
        stop = min(start + args.batch_size, len(frames))
        batch = slice(start, stop)
        env_obs = {
            "main_images": images[0][batch],
            "extra_view_images": torch.stack(
                [view[batch] for view in images[1:]], dim=1
            ),
            "states": states[batch],
            "task_descriptions": instructions[batch],
        }
        try:
            generated, result = model.predict_subtask_batch(env_obs)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Subtask generation failed for episode {episode}, frames {frames[batch]}."
            ) from exc
        for frame, prediction, eos in zip(
            frames[batch], generated, result["subtask_eos"], strict=True
        ):
            reference = table["subtask"][frame].as_py()
            ref_key, pred_key = _normalize_text(reference), _normalize_text(prediction)
            row = {
                "episode_index": episode,
                "frame_index": frame,
                "timestamp": table["timestamp"][frame].as_py(),
                "roles": selected[frame],
                "prediction": prediction,
                "reference": reference,
                "eos": eos,
                "nonempty": bool(prediction.strip()),
                "exact_match": ref_key == pred_key,
                "reference_phase": phase_map.get(ref_key),
                "prediction_phase": phase_map.get(pred_key),
                "planning_batch_seconds": result["planning_seconds"],
                "batch_size": stop - start,
            }
            if boundary is not None and "boundary_window" in selected[frame]:
                previous = _normalize_text(table["subtask"][boundary - 1].as_py())
                following = _normalize_text(table["subtask"][boundary].as_py())
                row["boundary_frame"] = boundary
                row["boundary_offset_frames"] = frame - boundary
                row["boundary_previous"] = table["subtask"][boundary - 1].as_py()
                row["boundary_next"] = table["subtask"][boundary].as_py()
                previous_key = phase_map.get(previous, previous)
                next_key = phase_map.get(following, following)
                prediction_key = phase_map.get(pred_key, pred_key)
                if previous_key == next_key:
                    row["boundary_prediction"] = "same_phase"
                elif prediction_key == previous_key:
                    row["boundary_prediction"] = "previous"
                elif prediction_key == next_key:
                    row["boundary_prediction"] = "next"
                else:
                    row["boundary_prediction"] = "other"
            yield row


def _summarize(rows: list[dict]) -> dict:
    groups = defaultdict(Counter)
    offsets = defaultdict(Counter)
    confusion = defaultdict(Counter)
    for row in rows:
        for key in ["all", *row["roles"]]:
            groups[key].update(
                samples=1,
                eos=int(row["eos"]),
                nonempty=int(row["nonempty"]),
                exact=int(row["exact_match"]),
                phase_scored=int(row["reference_phase"] is not None),
                phase_correct=int(
                    row["reference_phase"] is not None
                    and row["reference_phase"] == row["prediction_phase"]
                ),
            )
        if "boundary_offset_frames" in row:
            offsets[row["boundary_offset_frames"]].update([row["boundary_prediction"]])
        if row["reference_phase"] is not None:
            confusion[row["reference_phase"]][
                row["prediction_phase"] or "unmapped"
            ] += 1

    for counts in groups.values():
        counts["eos_rate"] = counts["eos"] / counts["samples"]
        counts["exact_accuracy"] = counts["exact"] / counts["samples"]
        if counts["phase_scored"]:
            counts["phase_accuracy"] = counts["phase_correct"] / counts["phase_scored"]
    return {
        "by_role": groups,
        "boundary_by_offset_frames": dict(sorted(offsets.items())),
        "phase_confusion": confusion,
    }


def main() -> None:
    """Load merged weights and score offline subtask predictions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Merged weights or export directory.",
    )
    parser.add_argument("--deployment-metadata", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New report directory."
    )
    parser.add_argument(
        "--phase-map",
        type=Path,
        help="Optional JSON mapping annotation text to phase names.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--boundary-radius", type=int, default=12, help="Frames on either side."
    )
    parser.add_argument(
        "--boundary-stride", type=int, default=3, help="Frames between samples."
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or (
        args.max_episodes is not None and args.max_episodes <= 0
    ):
        parser.error("--batch-size and --max-episodes must be positive.")
    if (
        args.seed < 0
        or args.boundary_stride <= 0
        or args.boundary_radius < args.boundary_stride
        or args.boundary_radius % args.boundary_stride
    ):
        parser.error(
            "Expected nonnegative seed and positive radius divisible by positive stride."
        )
    root = args.dataset_root.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.is_relative_to(root):
        parser.error("--output must be outside --dataset-root.")

    import torch
    from lerobot.datasets.utils import load_episodes, load_info, load_tasks
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi import get_model

    info = load_info(root)
    if info["codebase_version"] != "v2.1":
        raise ValueError(f"Expected LeRobot v2.1, got {info['codebase_version']!r}.")
    episodes = load_episodes(root)
    tasks, _ = load_tasks(root)
    episode_ids = sorted(episodes)[: args.max_episodes]
    if not episode_ids:
        raise ValueError(f"Expected at least one episode in {root}.")
    phase_map = {}
    if args.phase_map:
        raw_map = json.loads(args.phase_map.read_text())
        phase_map = {_normalize_text(label): phase for label, phase in raw_map.items()}
        if not phase_map or any(
            not isinstance(phase, str) or not phase.strip()
            for phase in phase_map.values()
        ):
            raise ValueError(
                "Expected phase map from annotation text to nonempty phase names."
            )

    manifest_path = args.deployment_metadata.expanduser().resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    config = OmegaConf.create(manifest["model"])
    config.model_path = str(args.checkpoint.expanduser().resolve(strict=True))
    config.precision = "bf16"
    config.is_lora = False
    config.openpi.task = "eval"
    config.openpi.policy_kind = "explicit_subtask"
    config.openpi.rtc_enabled = False
    config.openpi.validate_deployment = True
    config.openpi.deployment_metadata_path = str(manifest_path)
    config.openpi_data.norm_stats_path = str(
        args.norm_stats.expanduser().resolve(strict=True)
    )

    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(config, output / "model_config.yaml")
    if phase_map:
        (output / "phase_map.json").write_text(
            json.dumps(phase_map, ensure_ascii=False, indent=2) + "\n"
        )

    device = torch.device(args.device)
    logger.info(
        "Loading step %s on %s for %d episodes",
        manifest["step"],
        device,
        len(episode_ids),
    )
    model = get_model(config).to(device).eval().requires_grad_(False)
    started = time.perf_counter()
    rows = []
    with torch.inference_mode(), (output / "predictions.jsonl").open("w") as stream:
        for position, episode in enumerate(episode_ids, start=1):
            for row in _evaluate_episode(
                model,
                root,
                info,
                episode,
                episodes[episode]["length"],
                tasks,
                phase_map,
                args,
            ):
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if position == 1 or position % 10 == 0 or position == len(episode_ids):
                logger.info(
                    "Evaluated %d/%d episodes (%d observations)",
                    position,
                    len(episode_ids),
                    len(rows),
                )
    summary = {
        "dataset_root": str(root),
        "episode_ids": episode_ids,
        "fps": info["fps"],
        "checkpoint_step": manifest["step"],
        "weights_sha256": manifest["weights"]["sha256"],
        "device": str(device),
        "seed": args.seed,
        "boundary_radius_frames": args.boundary_radius,
        "boundary_stride_frames": args.boundary_stride,
        "elapsed_seconds": time.perf_counter() - started,
        **_summarize(rows),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    logger.info("Wrote %d predictions to %s", len(rows), output)
    if any(not row["eos"] for row in rows):
        raise RuntimeError(
            f"Expected EOS in every prediction. See {output / 'summary.json'}."
        )


if __name__ == "__main__":
    main()
