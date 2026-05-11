from __future__ import annotations

import argparse
import logging
import math
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
    value: float
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
        self.started_at = time.perf_counter()
        super().__init__(
            name,
            output_queue,
            stop_event,
            interval_s=1.0 / frequency_hz,
        )

    def get(self) -> SensorValue:
        now = time.perf_counter()
        elapsed = now - self.started_at
        signal_value = math.sin(2.0 * math.pi * self.frequency_hz * elapsed)
        return SensorValue(
            name=self.name,
            value=signal_value,
            frequency_hz=self.frequency_hz,
            timestamp=now,
        )


class WindowImage:
    def __init__(self, fps: int, window_name: str = "Camera sensors") -> None:
        self.fps = fps
        self.window_name = window_name
        self.delay_ms = max(1, int(1000 / fps))
        self.closed = False
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            raise SensorError(f"Cannot create window: {exc}") from exc
        logging.info("Window created: %s", self.window_name)

    def show(self, img: np.ndarray) -> bool:
        try:
            cv2.imshow(self.window_name, img)
            key = cv2.waitKey(self.delay_ms) & 0xFF
            try:
                is_visible = cv2.getWindowProperty(
                    self.window_name,
                    cv2.WND_PROP_VISIBLE,
                )
            except cv2.error:
                logging.info("Window close button pressed")
                return False
            if is_visible < 1:
                logging.info("Window close button pressed")
                return False
            return key != ord("q")
        except cv2.error as exc:
            raise SensorError(f"Cannot show image: {exc}") from exc

    def close(self) -> None:
        if not self.closed:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                cv2.destroyAllWindows()
            self.closed = True
            logging.info("Window closed")

    def __del__(self) -> None:
        self.close()


def make_placeholder(resolution: tuple[int, int]) -> np.ndarray:
    width, height = resolution
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        img,
        "Waiting for camera frame...",
        (20, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    return img


def compose_image(
    frame: np.ndarray,
    sensor_values: dict[str, SensorValue],
    actual_resolution: tuple[int, int],
    actual_fps: float,
) -> np.ndarray:
    image = frame.copy()
    height, width = image.shape[:2]
    actual_width, actual_height = actual_resolution
    lines = [
        f"FPS: {actual_fps:.2f}",
        f"Resolution: {actual_width}x{actual_height}",
        "",
    ]

    for sensor_name in ("SensorX-100Hz", "SensorX-10Hz", "SensorX-1Hz"):
        value = sensor_values.get(sensor_name)
        if value is None:
            lines.append(f"{sensor_name}: no data")
        else:
            lines.append(
                f"{sensor_name}: {value.value:+.3f} ({value.frequency_hz:g} Hz)"
            )

    margin = 12
    padding_x = 14
    padding_y = 12
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness = 1
    line_gap = 8
    blank_gap = 12
    max_panel_width = width - 2 * margin

    visible_lines = [line for line in lines if line]
    while font_scale > 0.35:
        max_text_width = max(
            cv2.getTextSize(line, font, font_scale, thickness)[0][0]
            for line in visible_lines
        )
        if max_text_width + 2 * padding_x <= max_panel_width:
            break
        font_scale -= 0.04

    text_sizes = {
        line: cv2.getTextSize(line, font, font_scale, thickness)[0]
        for line in visible_lines
    }
    line_height = max(text_height for _, text_height in text_sizes.values())
    max_text_width = max(text_width for text_width, _ in text_sizes.values())

    panel_width = min(max_panel_width, max_text_width + 2 * padding_x)
    panel_height = 2 * padding_y
    for line in lines:
        panel_height += blank_gap if not line else line_height + line_gap
    panel_height -= line_gap
    panel_height = min(panel_height, height - 2 * margin)

    x1 = margin
    y1 = margin
    x2 = x1 + panel_width
    y2 = y1 + panel_height

    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.68, image, 0.32, 0, image)
    cv2.rectangle(image, (x1, y1), (x2, y2), (80, 80, 80), 1)

    x = x1 + padding_x
    y = y1 + padding_y + line_height
    for line in lines:
        if not line:
            y += blank_gap
            continue
        cv2.putText(
            image,
            line,
            (x, y),
            font,
            font_scale,
            (245, 245, 245),
            thickness,
            cv2.LINE_AA,
        )
        y += line_height + line_gap

    return image


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
    except SensorError:
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
        cv2.destroyAllWindows()
        logging.info("Program finished")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
