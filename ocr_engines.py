from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class OCRResult:
    full_text: str
    boxes: List[Tuple[int, int, int, int]]


def _resolve_tesseract_exe() -> str | None:
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd and os.path.isfile(env_cmd):
        return env_cmd

    found = shutil.which("tesseract")
    if found:
        return found

    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for exe in candidates:
        if os.path.isfile(exe):
            return exe
    return None


def _as_rgb(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("Input image is None")
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def ocr_with_tesseract(orig_bgr: np.ndarray, preprocessed: np.ndarray | None = None) -> OCRResult:
    try:
        import pytesseract
    except Exception as ex:
        raise RuntimeError("pytesseract ist nicht installiert.") from ex

    tesseract_exe = _resolve_tesseract_exe()
    if tesseract_exe:
        pytesseract.pytesseract.tesseract_cmd = tesseract_exe

    img_for_ocr = preprocessed if preprocessed is not None else orig_bgr
    try:
        text = pytesseract.image_to_string(img_for_ocr, lang="deu+eng") or ""
        data = pytesseract.image_to_data(
            img_for_ocr,
            output_type=pytesseract.Output.DICT,
            lang="deu+eng",
        )
    except pytesseract.TesseractNotFoundError as ex:
        raise RuntimeError(
            "Tesseract wurde nicht gefunden. Bitte tesseract.exe in PATH setzen, "
            "oder TESSERACT_CMD auf die exe setzen "
            "(z.B. C:\\Program Files\\Tesseract-OCR\\tesseract.exe). "
            "Terminal/IDE danach neu starten."
        ) from ex

    boxes: List[Tuple[int, int, int, int]] = []
    n = len(data.get("text", []))
    for i in range(n):
        token = (data["text"][i] or "").strip()
        if not token:
            continue
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        if w > 0 and h > 0:
            boxes.append((x, y, x + w, y + h))

    return OCRResult(full_text=text.strip(), boxes=boxes)


_PADDLE_MODEL = None
_EASYOCR_READER = None


def _get_paddle_model():
    global _PADDLE_MODEL
    if _PADDLE_MODEL is not None:
        return _PADDLE_MODEL

    # Workaround for known Windows DLL/load-order conflicts:
    # import torch before paddleocr/albumentations stack.
    try:
        import torch  # noqa: F401
    except Exception as ex:
        raise RuntimeError(f"Torch Importfehler vor PaddleOCR: {ex}") from ex

    try:
        from paddleocr import PaddleOCR
    except Exception as ex:
        msg = str(ex)
        if "WinError 127" in msg or "shm.dll" in msg or "torch" in msg.lower():
            raise RuntimeError(
                "PaddleOCR konnte nicht importiert werden, weil Torch/Runtime in der venv fehlerhaft ist "
                f"(Details: {msg}). Das ist kein 'nicht installiert' Problem."
            ) from ex
        raise RuntimeError(f"PaddleOCR Importfehler: {msg}") from ex

    try:
        import paddle  # noqa: F401
    except Exception as ex:
        raise RuntimeError(
            "PaddleOCR ist installiert, aber 'paddlepaddle' fehlt. "
            "Installiere ein passendes Wheel fuer deine Python-Version/Plattform."
        ) from ex

    try:
        _PADDLE_MODEL = PaddleOCR(use_textline_orientation=True, lang="en")
    except Exception as ex_new:
        try:
            _PADDLE_MODEL = PaddleOCR(use_angle_cls=True, lang="en")
        except Exception as ex_old:
            raise RuntimeError(
                f"PaddleOCR konnte nicht initialisiert werden. "
                f"Neuer API-Call Fehler: {ex_new}. "
                f"Legacy API-Call Fehler: {ex_old}"
            ) from ex_old

    return _PADDLE_MODEL


def ocr_with_paddle(orig_bgr: np.ndarray, preprocessed: np.ndarray | None = None) -> OCRResult:
    model = _get_paddle_model()
    img_for_ocr = preprocessed if preprocessed is not None else orig_bgr

    rgb = _as_rgb(img_for_ocr)
    try:
        result = model.ocr(rgb, cls=True)
    except TypeError:
        result = model.ocr(rgb)
    except Exception as ex:
        msg = str(ex)
        if "ConvertPirAttribute2RuntimeAttribute" in msg:
            raise RuntimeError(
                "PaddleOCR/PaddlePaddle Backend-Fehler (PIR/OneDNN). "
                "Das ist kein unvollstaendiger Model-Download. "
                "Empfohlen: paddleocr<3 und paddlepaddle<3 (Python 3.10/3.11)."
            ) from ex
        raise RuntimeError(f"PaddleOCR Laufzeitfehler: {ex}") from ex

    lines: List[str] = []
    boxes: List[Tuple[int, int, int, int]] = []

    if not result:
        return OCRResult(full_text="", boxes=[])

    for entry in result:
        if not entry:
            continue
        for item in entry:
            if not item or len(item) < 2:
                continue

            quad = item[0]
            txt = item[1][0] if item[1] else ""
            if txt:
                lines.append(str(txt))

            try:
                xs = [int(p[0]) for p in quad]
                ys = [int(p[1]) for p in quad]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))
            except Exception:
                continue

    return OCRResult(full_text="\n".join(lines).strip(), boxes=boxes)


def _get_easyocr_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is not None:
        return _EASYOCR_READER

    try:
        import easyocr
    except Exception as ex:
        msg = str(ex)
        if "WinError 127" in msg or "shm.dll" in msg or "torch" in msg.lower():
            raise RuntimeError(
                "EasyOCR konnte nicht importiert werden, weil Torch/Runtime in der venv fehlerhaft ist "
                f"(Details: {msg}). Das ist kein 'nicht installiert' Problem."
            ) from ex
        raise RuntimeError(f"EasyOCR Importfehler: {msg}") from ex

    # CPU by default for broad compatibility.
    _EASYOCR_READER = easyocr.Reader(["de", "en"], gpu=False)
    return _EASYOCR_READER


def ocr_with_easyocr(orig_bgr: np.ndarray, preprocessed: np.ndarray | None = None) -> OCRResult:
    reader = _get_easyocr_reader()
    img_for_ocr = preprocessed if preprocessed is not None else orig_bgr
    rgb = _as_rgb(img_for_ocr)

    try:
        result = reader.readtext(rgb, detail=1, paragraph=False)
    except Exception as ex:
        raise RuntimeError(f"EasyOCR Laufzeitfehler: {ex}") from ex

    lines: List[str] = []
    boxes: List[Tuple[int, int, int, int]] = []
    for item in result:
        if not item or len(item) < 2:
            continue
        quad = item[0]
        txt = str(item[1]).strip()
        if txt:
            lines.append(txt)
        try:
            pts = np.array(quad, dtype=np.int32).reshape(-1, 2)
            x, y, w, h = cv2.boundingRect(pts)
            if w > 0 and h > 0:
                boxes.append((x, y, x + w, y + h))
        except Exception:
            continue

    return OCRResult(full_text="\n".join(lines).strip(), boxes=boxes)


def detect_text_regions_mser(
    bgr: np.ndarray,
    min_area: int = 80,
    max_area: int = 20000,
) -> List[Tuple[int, int, int, int]]:
    if bgr is None:
        return []

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)

    rects: List[Tuple[int, int, int, int]] = []
    for region in regions:
        x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
        area = w * h
        if area < min_area or area > max_area:
            continue
        aspect = w / max(h, 1)
        if aspect < 0.2 or aspect > 15.0:
            continue
        rects.append((x, y, x + w, y + h))

    return _nms_xyxy(rects, iou_threshold=0.3)


def _decode_east_predictions(scores: np.ndarray, geometry: np.ndarray, conf_threshold: float):
    rects = []
    confidences = []

    num_rows, num_cols = scores.shape[2:4]
    for y in range(num_rows):
        scores_data = scores[0, 0, y]
        x0 = geometry[0, 0, y]
        x1 = geometry[0, 1, y]
        x2 = geometry[0, 2, y]
        x3 = geometry[0, 3, y]
        angles = geometry[0, 4, y]

        for x in range(num_cols):
            score = float(scores_data[x])
            if score < conf_threshold:
                continue

            offset_x, offset_y = x * 4.0, y * 4.0
            angle = angles[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = x0[x] + x2[x]
            w = x1[x] + x3[x]

            end_x = int(offset_x + (cos * x1[x]) + (sin * x2[x]))
            end_y = int(offset_y - (sin * x1[x]) + (cos * x2[x]))
            start_x = int(end_x - w)
            start_y = int(end_y - h)

            rects.append((start_x, start_y, end_x, end_y))
            confidences.append(score)

    return rects, confidences


def detect_text_regions_east(
    bgr: np.ndarray,
    model_path: str | None = None,
    conf_threshold: float = 0.5,
    nms_threshold: float = 0.4,
    input_size: Tuple[int, int] = (320, 320),
) -> List[Tuple[int, int, int, int]]:
    if bgr is None:
        return []

    if model_path is None or not model_path.strip():
        model_path = os.path.join("models", "frozen_east_text_detection.pb")

    if not os.path.isfile(model_path):
        raise RuntimeError(
            f"EAST-Modell nicht gefunden: {model_path}. "
            "Bitte .pb Datei auswaehlen oder in models/ ablegen."
        )

    H, W = bgr.shape[:2]
    in_w, in_h = input_size
    r_w = W / float(in_w)
    r_h = H / float(in_h)

    resized = cv2.resize(bgr, (in_w, in_h))
    blob = cv2.dnn.blobFromImage(
        resized,
        1.0,
        (in_w, in_h),
        (123.68, 116.78, 103.94),
        swapRB=True,
        crop=False,
    )

    net = cv2.dnn.readNet(model_path)
    net.setInput(blob)
    scores, geometry = net.forward([
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3",
    ])

    rects, confidences = _decode_east_predictions(scores, geometry, conf_threshold)
    if not rects:
        return []

    boxes_xywh = []
    for (x1, y1, x2, y2) in rects:
        boxes_xywh.append([x1, y1, max(1, x2 - x1), max(1, y2 - y1)])

    indices = cv2.dnn.NMSBoxes(boxes_xywh, confidences, conf_threshold, nms_threshold)
    if len(indices) == 0:
        return []

    out_boxes: List[Tuple[int, int, int, int]] = []
    for idx in indices.flatten():
        x1, y1, x2, y2 = rects[int(idx)]
        sx1 = max(0, int(x1 * r_w))
        sy1 = max(0, int(y1 * r_h))
        sx2 = min(W, int(x2 * r_w))
        sy2 = min(H, int(y2 * r_h))
        if sx2 > sx1 and sy2 > sy1:
            out_boxes.append((sx1, sy1, sx2, sy2))

    return out_boxes


def _nms_xyxy(boxes: List[Tuple[int, int, int, int]], iou_threshold: float) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []

    b = np.array(boxes, dtype=np.float32)
    x1 = b[:, 0]
    y1 = b[:, 1]
    x2 = b[:, 2]
    y2 = b[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = areas.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return [boxes[i] for i in keep]
