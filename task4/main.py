from __future__ import annotations

import argparse
import logging
import os
import queue
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ui import UIError, WindowImage, compose_image, make_placeholder


DEFAULT_CAMERA = 0
DEFAULT_RESOLUTION = (640, 480)
DEFAULT_FPS = 15
LOG_DIR = Path(__file__).resolve().parent / "log"


class SensorError(RuntimeError):
    pass


class CameraOpenError(SensorError):
    pass


class CameraReadError(SensorError):
    pass


@dataclass(frozen=True)
class SensorValue:
    name: str
    value: int
    frequency_hz: float
    timestamp: float


def parse_camera(value: str) -> int | str:
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("camera must not be empty")
    try:
        camera_index = int(value)
    except ValueError:
        if value.startswith("-"):
            raise argparse.ArgumentTypeError("camera index must be non-negative")
        return value
    else:
        if camera_index < 0:
            raise argparse.ArgumentTypeError("camera index must be non-negative")
        return camera_index


def parse_resolution(value: str) -> tuple[int, int]:
    parts = value.strip().lower().split("x")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise argparse.ArgumentTypeError("resolution must have WIDTHxHEIGHT format")

    width, height = (int(parts[0]), int(parts[1]))
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("resolution values must be positive")
    return width, height


def parse_fps(value: str) -> int:
    if not value.isdigit():
        raise argparse.ArgumentTypeError("fps must be an integer")
    fps = int(value)
    if fps <= 0:
        raise argparse.ArgumentTypeError("fps must be positive")
    return fps


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_format = "%(asctime)s [%(levelname)s] %(threadName)s: %(message)s"

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8"),
        ],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Threaded camera and sensor visualization task."
    )
    parser.add_argument(
        "--camera",
        type=parse_camera,
        default=DEFAULT_CAMERA,
        help=f"camera index or device path, default: {DEFAULT_CAMERA}",
    )
    parser.add_argument(
        "--resolution",
        type=parse_resolution,
        default=DEFAULT_RESOLUTION,
        help="camera resolution in WIDTHxHEIGHT format, default: 640x480",
    )
    parser.add_argument(
        "--fps",
        type=parse_fps,
        default=DEFAULT_FPS,
        help=f"camera/display frames per second, default: {DEFAULT_FPS}",
    )
    return parser


def put_latest(output_queue: queue.Queue[Any], value: Any) -> None:
    while True:
        try:
            output_queue.get_nowait()
        except queue.Empty:
            break

    try:
        output_queue.put_nowait(value)
    except queue.Full:
        logging.debug("Dropped stale value because queue is full")


def drain_latest(input_queue: queue.Queue[Any], previous: Any = None) -> Any:
    value = previous
    while True:
        try:
            value = input_queue.get_nowait()
        except queue.Empty:
            return value


def check_camera_device_exists(camera: int | str) -> None:
    if os.name != "posix":
        return

    if isinstance(camera, int):
        device_path = Path(f"/dev/video{camera}")
    else:
        device_path = Path(camera)
        if not str(device_path).startswith("/dev/video"):
            return

    if not device_path.exists():
        raise CameraOpenError(f"Camera device does not exist: {device_path}")


class Sensor:
    def __init__(
        self,
        name: str,
        output_queue: queue.Queue[Any],
        stop_event: threading.Event,
        interval_s: float = 0.0,
    ) -> None:
        self.name = name
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.interval_s = interval_s
        self.error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run,
            name=name,
            daemon=True,
        )

    def start(self) -> None:
        logging.info("Starting %s", self.name)
        self.thread.start()

    def join(self, timeout: float | None = None) -> None:
        self.thread.join(timeout=timeout)

    def get(self) -> Any:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __del__(self) -> None:
        self.close()

    def _run(self) -> None:
        try:
            next_tick = time.perf_counter()
            while not self.stop_event.is_set():
                put_latest(self.output_queue, self.get())
                if self.interval_s > 0.0:
                    next_tick += self.interval_s
                    wait_s = max(0.0, next_tick - time.perf_counter())
                    self.stop_event.wait(wait_s)
        except SensorError as exc:
            self.error = exc
            logging.exception("%s failed", self.name)
            self.stop_event.set()
        except Exception as exc:
            self.error = exc
            logging.exception("Unexpected error in %s", self.name)
            self.stop_event.set()
        finally:
            self.close()
            logging.info("%s stopped", self.name)


class SensorCam(Sensor):
    def __init__(
        self,
        camera: int | str,
        resolution: tuple[int, int],
        fps: int,
        output_queue: queue.Queue[Any],
        stop_event: threading.Event,
    ) -> None:
        self.camera = camera
        self.resolution = resolution
        self.fps = fps
        self.cap = None
        self.closed = True

        check_camera_device_exists(camera)
        self.cap = cv2.VideoCapture(camera)
        self.closed = False

        width, height = resolution
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        if not self.cap.isOpened():
            self.close()
            raise CameraOpenError(f"Cannot open camera: {camera}")

        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.actual_width <= 0 or self.actual_height <= 0:
            self.actual_width, self.actual_height = resolution
        if self.actual_fps <= 0:
            self.actual_fps = float(fps)

        logging.info(
            "Camera opened: %s, requested=%dx%d@%d, actual=%dx%d@%.2f",
            camera,
            width,
            height,
            fps,
            self.actual_width,
            self.actual_height,
            self.actual_fps,
        )

        super().__init__("SensorCam", output_queue, stop_event)

    def get(self) -> np.ndarray:
        if self.cap is None:
            raise CameraReadError("Camera is not initialized")
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise CameraReadError(
                "Cannot read frame from camera. It may have been disconnected."
            )
        return frame

    def close(self) -> None:
        if not getattr(self, "closed", True):
            if self.cap is not None:
                self.cap.release()
            self.closed = True
            logging.info("Camera released")


class SensorX(Sensor):
    def __init__(
        self,
        name: str,
        frequency_hz: float,
        output_queue: queue.Queue[Any],
        stop_event: threading.Event,
    ) -> None:
        self.frequency_hz = frequency_hz
        self.value = 0
        super().__init__(
            name,
            output_queue,
            stop_event,
            interval_s=1.0 / frequency_hz,
        )

    def get(self) -> SensorValue:
        self.value += 1
        now = time.perf_counter()
        return SensorValue(
            name=self.name,
            value=self.value,
            frequency_hz=self.frequency_hz,
            timestamp=now,
        )


def install_signal_handlers(stop_event: threading.Event) -> None:
    def handle_signal(signum: int, _frame: Any) -> None:
        logging.info("Received signal %s, stopping", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


def main() -> int:
    args = build_parser().parse_args()
    setup_logging()
    logging.info(
        "Starting program: camera=%s, resolution=%dx%d, fps=%d",
        args.camera,
        args.resolution[0],
        args.resolution[1],
        args.fps,
    )

    stop_event = threading.Event()
    install_signal_handlers(stop_event)

    camera_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
    sensor_queues: dict[str, queue.Queue[SensorValue]] = {
        "SensorX-100Hz": queue.Queue(maxsize=1),
        "SensorX-10Hz": queue.Queue(maxsize=1),
        "SensorX-1Hz": queue.Queue(maxsize=1),
    }

    window: WindowImage | None = None
    sensors: list[Sensor] = []

    try:
        camera_sensor = SensorCam(
            args.camera,
            args.resolution,
            args.fps,
            camera_queue,
            stop_event,
        )
        sensors = [
            camera_sensor,
            SensorX(
                "SensorX-100Hz",
                100.0,
                sensor_queues["SensorX-100Hz"],
                stop_event,
            ),
            SensorX("SensorX-10Hz", 10.0, sensor_queues["SensorX-10Hz"], stop_event),
            SensorX("SensorX-1Hz", 1.0, sensor_queues["SensorX-1Hz"], stop_event),
        ]
        window = WindowImage(args.fps)

        for sensor in sensors:
            sensor.start()

        last_frame: np.ndarray | None = None
        last_sensor_values: dict[str, SensorValue] = {}
        loop_interval = 1.0 / args.fps
        actual_resolution = (camera_sensor.actual_width, camera_sensor.actual_height)
        actual_fps = camera_sensor.actual_fps

        while not stop_event.is_set():
            loop_started = time.perf_counter()
            last_frame = drain_latest(camera_queue, last_frame)

            for sensor_name, sensor_queue in sensor_queues.items():
                value = drain_latest(sensor_queue, last_sensor_values.get(sensor_name))
                if value is not None:
                    last_sensor_values[sensor_name] = value

            frame = (
                last_frame
                if last_frame is not None
                else make_placeholder(actual_resolution)
            )
            image = compose_image(
                frame,
                last_sensor_values,
                actual_resolution,
                actual_fps,
            )
            if not window.show(image):
                logging.info("Window requested program stop")
                stop_event.set()
                break

            elapsed = time.perf_counter() - loop_started
            stop_event.wait(max(0.0, loop_interval - elapsed))

        for sensor in sensors:
            sensor.join(timeout=0.5)

        failed_sensor = next((sensor for sensor in sensors if sensor.error), None)
        if failed_sensor is not None:
            raise SensorError(f"{failed_sensor.name} failed") from failed_sensor.error

    except CameraOpenError as exc:
        logging.error("Camera initialization failed: %s", exc)
        return 1
    except (SensorError, UIError):
        logging.exception("Program stopped because of an unrecoverable error")
        return 1
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        stop_event.set()
        for sensor in sensors:
            sensor.join(timeout=2.0)
            sensor.close()
        if window is not None:
            window.close()
        logging.info("Program finished")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
