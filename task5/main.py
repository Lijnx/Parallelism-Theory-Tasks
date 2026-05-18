from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


MODEL_NAME = "yolov8s-pose.pt"
WINDOW_NAME = "YOLOv8 Pose"
QUEUE_SIZE = 32
CAMERA_INDEX = 0


@dataclass(frozen=True)
class FrameTask:
    frame_id: int
    frame: np.ndarray


@dataclass(frozen=True)
class FrameResult:
    frame_id: int
    frame: np.ndarray


class PipelineError(RuntimeError):
    pass


class FpsCounter:
    def __init__(self) -> None:
        self.last_time: float | None = None

    def tick(self) -> float:
        now = time.perf_counter()
        if self.last_time is None:
            self.last_time = now
            return 0.0

        elapsed = max(now - self.last_time, 1e-9)
        self.last_time = now
        return 1.0 / elapsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU pose estimation with YOLOv8 in single or multithreaded mode."
    )
    parser.add_argument(
        "--mode",
        choices=("video", "realtime"),
        required=True,
        help="input source: video file or realtime camera",
    )
    parser.add_argument(
        "--execution",
        choices=("single", "multi"),
        required=True,
        help="single-threaded or multithreaded processing",
    )
    parser.add_argument("--input", help="path to input video for video mode")
    parser.add_argument("--output", help="path to output video for video mode")
    parser.add_argument(
        "--workers",
        type=int,
        help="number of worker threads for multi mode",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.mode == "video":
        if not args.input:
            raise SystemExit("--input is required when --mode video")
        if not args.output:
            raise SystemExit("--output is required when --mode video")
    else:
        if args.input or args.output:
            print("Ignoring --input/--output in realtime mode.", file=sys.stderr)
    if args.execution == "multi":
        if args.workers is None:
            raise SystemExit("--workers is required when --execution multi")
        if args.workers <= 0:
            raise SystemExit("--workers must be a positive integer")


def load_yolo_class() -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise PipelineError(
            "ultralytics is not installed in .venv. Install project dependencies first."
        ) from exc
    return YOLO


def annotate_frame(model: Any, frame: np.ndarray) -> np.ndarray:
    results = model.predict(frame, device="cpu", verbose=False)
    return results[0].plot()


def configure_multi_runtime() -> None:
    cv2.setNumThreads(1)
    try:
        import torch
    except ImportError:
        return

    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass


def get_video_info(capture: cv2.VideoCapture) -> tuple[int, int, float]:
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0:
        raise PipelineError("Could not detect video resolution.")
    if fps <= 0:
        fps = 30.0
    return width, height, fps


def open_video_capture(source: str | int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise PipelineError(f"Could not open source: {source}")
    return capture


def open_video_writer(path: str, width: int, height: int, fps: float) -> cv2.VideoWriter:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise PipelineError(f"Could not open output video: {path}")
    return writer


def should_stop_display(frame: np.ndarray) -> bool:
    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(1) & 0xFF
    return key in (27, ord("q"))


def process_video_single(input_path: str, output_path: str) -> float:
    capture = open_video_capture(input_path)
    writer: cv2.VideoWriter | None = None
    started = time.perf_counter()

    try:
        width, height, fps = get_video_info(capture)
        writer = open_video_writer(output_path, width, height, fps)
        YOLO = load_yolo_class()
        model = YOLO(MODEL_NAME)

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(annotate_frame(model, frame))
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    return time.perf_counter() - started


def reader_thread(
    capture: cv2.VideoCapture,
    input_queue: queue.Queue[FrameTask | None],
    worker_count: int,
) -> None:
    frame_id = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            input_queue.put(FrameTask(frame_id=frame_id, frame=frame))
            frame_id += 1
    finally:
        for _ in range(worker_count):
            input_queue.put(None)


def worker_thread(
    input_queue: queue.Queue[FrameTask | None],
    output_queue: queue.Queue[FrameResult | BaseException | None],
) -> None:
    try:
        YOLO = load_yolo_class()
        model = YOLO(MODEL_NAME)
        while True:
            task = input_queue.get()
            if task is None:
                output_queue.put(None)
                return
            output_queue.put(
                FrameResult(frame_id=task.frame_id, frame=annotate_frame(model, task.frame))
            )
    except BaseException as exc:
        output_queue.put(exc)


def consume_ordered_results(
    output_queue: queue.Queue[FrameResult | BaseException | None],
    worker_count: int,
    handle_frame: Any,
) -> None:
    next_frame_id = 0
    buffered: dict[int, np.ndarray] = {}
    finished_workers = 0

    while finished_workers < worker_count:
        item = output_queue.get()
        if item is None:
            finished_workers += 1
            continue
        if isinstance(item, BaseException):
            raise item

        buffered[item.frame_id] = item.frame
        while next_frame_id in buffered:
            frame = buffered.pop(next_frame_id)
            if handle_frame(frame):
                return
            next_frame_id += 1


def run_video_multi_pass(input_path: str, output_path: str | None, worker_count: int) -> float:
    capture = open_video_capture(input_path)
    writer: cv2.VideoWriter | None = None
    input_queue: queue.Queue[FrameTask | None] = queue.Queue(maxsize=QUEUE_SIZE)
    output_queue: queue.Queue[FrameResult | BaseException | None] = queue.Queue(
        maxsize=QUEUE_SIZE
    )
    started = time.perf_counter()

    try:
        width, height, fps = get_video_info(capture)
        if output_path is not None:
            writer = open_video_writer(output_path, width, height, fps)

        reader = threading.Thread(
            target=reader_thread,
            args=(capture, input_queue, worker_count),
            daemon=True,
            name="reader",
        )
        workers = [
            threading.Thread(
                target=worker_thread,
                args=(input_queue, output_queue),
                daemon=True,
                name=f"worker-{index}",
            )
            for index in range(worker_count)
        ]

        reader.start()
        for worker in workers:
            worker.start()

        def handle_frame(frame: np.ndarray) -> bool:
            if writer is not None:
                writer.write(frame)
            return False

        consume_ordered_results(output_queue, worker_count, handle_frame)
        reader.join()
        for worker in workers:
            worker.join()
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    return time.perf_counter() - started


def process_video_multi(input_path: str, output_path: str, worker_count: int) -> float:
    configure_multi_runtime()
    return run_video_multi_pass(input_path, output_path, worker_count)


def process_realtime_single() -> None:
    capture = open_video_capture(CAMERA_INDEX)
    try:
        YOLO = load_yolo_class()
        model = YOLO(MODEL_NAME)
        fps_counter = FpsCounter()

        while True:
            ok, frame = capture.read()
            if not ok:
                raise PipelineError("Could not read a frame from the camera.")
            annotated = annotate_frame(model, frame)
            fps = fps_counter.tick()
            cv2.putText(
                annotated,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            if should_stop_display(annotated):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def process_realtime_multi(worker_count: int) -> None:
    configure_multi_runtime()
    capture = open_video_capture(CAMERA_INDEX)
    input_queue: queue.Queue[FrameTask | None] = queue.Queue(maxsize=QUEUE_SIZE)
    output_queue: queue.Queue[FrameResult | BaseException | None] = queue.Queue(
        maxsize=QUEUE_SIZE
    )
    stop_event = threading.Event()

    def realtime_reader() -> None:
        frame_id = 0
        try:
            while not stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    output_queue.put(PipelineError("Could not read a frame from the camera."))
                    return
                input_queue.put(FrameTask(frame_id=frame_id, frame=frame))
                frame_id += 1
        finally:
            for _ in range(worker_count):
                input_queue.put(None)

    reader = threading.Thread(target=realtime_reader, daemon=True, name="reader")
    workers = [
        threading.Thread(
            target=worker_thread,
            args=(input_queue, output_queue),
            daemon=True,
            name=f"worker-{index}",
        )
        for index in range(worker_count)
    ]

    fps_counter = FpsCounter()

    try:
        reader.start()
        for worker in workers:
            worker.start()

        def handle_frame(frame: np.ndarray) -> bool:
            fps = fps_counter.tick()
            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            stop = should_stop_display(frame)
            if stop:
                stop_event.set()
            return stop

        consume_ordered_results(output_queue, worker_count, handle_frame)
    finally:
        stop_event.set()
        capture.release()
        reader.join(timeout=2.0)
        for worker in workers:
            worker.join(timeout=2.0)
        cv2.destroyAllWindows()


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)

    try:
        if args.mode == "video" and args.execution == "single":
            elapsed = process_video_single(args.input, args.output)
            print(f"Processing time: {elapsed:.2f} s")
            return 0

        if args.mode == "video" and args.execution == "multi":
            elapsed = process_video_multi(args.input, args.output, args.workers)
            print(f"Processing time: {elapsed:.2f} s")
            return 0

        if args.mode == "realtime" and args.execution == "single":
            process_realtime_single()
            return 0

        process_realtime_multi(args.workers)
        return 0
    except PipelineError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
