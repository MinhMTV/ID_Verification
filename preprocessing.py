import cv2
import numpy as np


PREPROCESS_METHODS = ("preprocess", "adaptive", "otsu", "clahe", "none")


def _to_gray(bgr: np.ndarray) -> np.ndarray:
    if bgr is None:
        raise ValueError("Input image is None")
    if len(bgr.shape) == 2:
        return bgr.copy()
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _deskew_gray(gray: np.ndarray) -> np.ndarray:
    """
    Estimate skew angle from foreground pixels and rotate the grayscale image.
    Returns the rotated grayscale image (same size as input)."""

    if gray is None or gray.size == 0:
        return gray

    # Find foreground for angle estimation.
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(bw > 0))
    if coords.shape[0] < 50:
        return gray

    angle = cv2.minAreaRect(coords.astype(np.float32))[2]
    if angle < -45:
        angle = 90 + angle
    angle = -angle

    h, w = gray.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray,
        mat,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _adaptive_preprocess(gray: np.ndarray, h: int, block_size: int, c_value: int) -> np.ndarray:
    h = int(max(1, min(50, h)))
    block_size = int(max(3, min(101, block_size)))
    if block_size % 2 == 0:
        block_size += 1
    c_value = int(max(-100, min(100, c_value)))

    den = cv2.fastNlMeansDenoising(gray, h=h)
    return cv2.adaptiveThreshold(
        den,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_value,
    )


def preprocess_for_ocr(
    bgr: np.ndarray,
    method: str = "adaptive",
    adaptive_params: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """
    Multiple preprocessing strategies for OCR.
    Returns a single-channel uint8 image.
    """
    method = (method or "adaptive").strip().lower()
    gray = _to_gray(bgr)

    if method == "none":
        return gray

    if method == "preprocess":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # leichter denoise
        den = cv2.fastNlMeansDenoising(gray, h=3)

        # Kontrast leicht verbessern
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        thr = clahe.apply(den)

        return thr

    if method == "adaptive":
        if adaptive_params is None:
            return _adaptive_preprocess(gray, h=4, block_size=15, c_value=4)
        h, block_size, c_value = adaptive_params
        return _adaptive_preprocess(gray, h=h, block_size=block_size, c_value=c_value)
    
    if method == "otsu":
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thr

    if method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(gray)
        _, thr = cv2.threshold(cl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thr

    raise ValueError(f"Unbekannte Preprocessing-Methode: {method}. Verfuegbar: {PREPROCESS_METHODS}")
