import shutil
from pathlib import Path


def ocr_frames(frame_paths: list[dict]) -> list[dict]:
    if not frame_paths:
        return []

    tesseract_bin = shutil.which("tesseract")
    if not tesseract_bin:
        import warnings
        warnings.warn(
            "tesseract binary not found. Install Tesseract OCR to enable "
            "frame OCR. Skipping OCR."
        )
        return []

    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = tesseract_bin
    except ImportError:
        import warnings
        warnings.warn(
            "pytesseract is not installed. Install with: pip install pytesseract. "
            "Skipping OCR."
        )
        return []

    results = []
    for frame in frame_paths:
        path = frame.get("path", "") if isinstance(frame, dict) else str(frame)
        ts = frame.get("timestamp_sec", 0.0) if isinstance(frame, dict) else 0.0

        if not path or not Path(path).exists():
            continue

        try:
            data = pytesseract.image_to_data(
                str(path), output_type=pytesseract.Output.DICT
            )
            text_parts = []
            total_conf = 0.0
            conf_count = 0
            for i, text in enumerate(data.get("text", [])):
                txt = text.strip()
                if txt:
                    text_parts.append(txt)
                    conf = int(data.get("conf", [0])[i]) if i < len(data.get("conf", [])) else 0
                    if conf > 0:
                        total_conf += conf
                        conf_count += 1

            text = " ".join(text_parts)
            confidence = round(total_conf / conf_count, 2) if conf_count > 0 else 1.0

            if text:
                results.append({
                    "timestamp_sec": ts,
                    "text": text,
                    "confidence": confidence,
                    "source": "tesseract",
                })
        except Exception:
            continue

    return results
