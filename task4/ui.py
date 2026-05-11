from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np


class UIError(RuntimeError):
    pass


class WindowImage:
    def __init__(self, fps: int, window_name: str = "Camera sensors") -> None:
        self.fps = fps
        self.window_name = window_name
        self.delay_ms = max(1, int(1000 / fps))
        self.closed = False
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            raise UIError(f"Cannot create window: {exc}") from exc
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
            raise UIError(f"Cannot show image: {exc}") from exc

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
    sensor_values: dict[str, Any],
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
                f"{sensor_name}: {value.value} ({value.frequency_hz:g} Hz)"
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
