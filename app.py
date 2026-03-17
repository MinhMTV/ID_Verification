from __future__ import annotations

import difflib
import csv
import datetime as dt
import os
import re
import threading
import unicodedata
from typing import List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from ocr_engines import (
    OCRResult,
    detect_text_regions_east,
    detect_text_regions_mser,
    ocr_with_easyocr,
    ocr_with_paddle,
    ocr_with_tesseract,
)
from preprocessing import PREPROCESS_METHODS, preprocess_for_ocr


try:
    from pdf2image import convert_from_path

    _PDF_AVAILABLE = True
except Exception:
    convert_from_path = None
    _PDF_AVAILABLE = False


DOC_CATEGORY_KEYWORDS = {
    "Verfahrenskarte": ["verfahrenskarte"],
    "Aufenthaltsberechtigungskarte": ["aufenthaltsberechtigungskarte"],
    "Aufenthaltstitel": ["aufenthaltstitel"],
    "Residence permit": ["residence permit"],
    "Reisedokument": ["reisedokument"],
    "Travel Document": ["travel document"],
    "Fremdenwesen Passport": ["fremdenwesen passport", "passport"],
    "Fremdenpass": ["fremdenpass"],
    "Asylberechtigt": ["asylberechtigt"],
    "Schutzberechtigt": ["schutzberechtigt"],
    "Identitaetskarte": ["identitaetskarte", "identitätskarte"],
    "Geduldete": ["geduldete"],
}


def _normalize_for_match(text: str) -> str:
    text = (text or "").lower().replace("*", " ")
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def classify_document_type(ocr_text: str) -> Tuple[str, float, str]:
    # 1) keyword matching
    norm_text = _normalize_for_match(ocr_text)
    if not norm_text:
        return "Unbekannt", 0.0, "kein OCR-Text"

    for category, keywords in DOC_CATEGORY_KEYWORDS.items():
        for kw in keywords:
            nkw = _normalize_for_match(kw)
            if nkw and nkw in norm_text:
                return category, 1.0, f"keyword: {kw}"

    # 2) fuzzy matching
    lines = [_normalize_for_match(x) for x in ocr_text.splitlines() if x.strip()]
    candidates = [c for c in lines if c]
    candidates.append(norm_text)

    best_cat = "Unbekannt"
    best_score = 0.0
    best_kw = ""
    for category, keywords in DOC_CATEGORY_KEYWORDS.items():
        for kw in keywords:
            nkw = _normalize_for_match(kw)
            if not nkw:
                continue
            score = 0.0
            for cand in candidates:
                score = max(score, difflib.SequenceMatcher(None, nkw, cand).ratio())
            if score > best_score:
                best_score = score
                best_cat = category
                best_kw = kw

    if best_score >= 0.62:
        return best_cat, best_score, f"fuzzy: {best_kw}"
    return "Unbekannt", best_score, "kein Match"


def _read_image_any(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        if not _PDF_AVAILABLE:
            raise RuntimeError("PDF geladen, aber pdf2image ist nicht installiert. Installiere pdf2image + poppler.")
        pages = convert_from_path(path, first_page=1, last_page=1)
        pil = pages[0].convert("RGB")
        rgb = np.array(pil)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    file_bytes = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Datei konnte nicht als Bild geladen werden (PNG/JPG/JPEG/JFIF/TIFF/BMP/PDF).")
    return bgr


class OCRApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ID OCR - Tesseract / Paddle / EasyOCR / MSER / EAST")
        self.geometry("1280x760")

        self.engine_var = tk.StringVar(value="tesseract")
        self.preprocess_var = tk.StringVar(value="adaptive")
        self.adaptive_h_var = tk.IntVar(value=3)
        self.adaptive_block_var = tk.IntVar(value=11)
        self.adaptive_c_var = tk.IntVar(value=25)
        self.train_filter_var = tk.StringVar(value="current")
        self.detect_var = tk.StringVar(value="ocr")
        self.use_detection_for_ocr_var = tk.BooleanVar(value=False)
        self.find_best_var = tk.BooleanVar(value=False)
        self.train_progress_var = tk.DoubleVar(value=0.0)
        self.east_model_var = tk.StringVar(value=os.path.join("models", "frozen_east_text_detection.pb"))
        self.train_image_paths: List[str] = []
        self.training_running = False
        self.training_thread: Optional[threading.Thread] = None

        self.file_path: Optional[str] = None
        self.orig_bgr: Optional[np.ndarray] = None
        self.pre_bgr: Optional[np.ndarray] = None
        self.display_bgr: Optional[np.ndarray] = None

        self.display_img_tk: Optional[ImageTk.PhotoImage] = None
        self.scale: float = 1.0
        self.current_boxes: List[Tuple[int, int, int, int]] = []

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Engine:").pack(side=tk.LEFT)
        ttk.Radiobutton(top, text="Tesseract", value="tesseract", variable=self.engine_var).pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(top, text="PaddleOCR", value="paddle", variable=self.engine_var).pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(top, text="EasyOCR", value="easyocr", variable=self.engine_var).pack(side=tk.LEFT, padx=6)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Label(top, text="Preprocessing:").pack(side=tk.LEFT)
        pre_cb = ttk.Combobox(top, textvariable=self.preprocess_var, state="readonly", width=10)
        pre_cb["values"] = PREPROCESS_METHODS
        pre_cb.pack(side=tk.LEFT, padx=6)
        ttk.Label(top, text="h").pack(side=tk.LEFT)
        ttk.Spinbox(top, from_=1, to=50, width=3, textvariable=self.adaptive_h_var).pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(top, text="block").pack(side=tk.LEFT)
        ttk.Spinbox(top, from_=3, to=99, width=3, textvariable=self.adaptive_block_var, increment=2).pack(side=tk.LEFT, padx=(2, 6))
        ttk.Label(top, text="C").pack(side=tk.LEFT)
        ttk.Spinbox(top, from_=1, to=100, width=3, textvariable=self.adaptive_c_var).pack(side=tk.LEFT, padx=(2, 10))

        ttk.Label(top, text="Detektion:").pack(side=tk.LEFT)
        det_cb = ttk.Combobox(top, textvariable=self.detect_var, state="readonly", width=18)
        det_cb["values"] = (
            "ocr",
            "mser",
            "east",
            "ocr+mser",
            "ocr+east",
            "ocr+mser+east",
        )
        det_cb.pack(side=tk.LEFT, padx=6)

        ttk.Checkbutton(
            top,
            text="Detection-ROIs fuer OCR nutzen",
            variable=self.use_detection_for_ocr_var,
        ).pack(side=tk.LEFT, padx=10)

        ttk.Button(top, text="Datei auswaehlen...", command=self.on_choose_file).pack(side=tk.LEFT, padx=10)
        ttk.Button(top, text="OCR starten", command=self.on_run_ocr).pack(side=tk.LEFT)
        self.btn_add_train = ttk.Button(top, text="Trainingsdaten hinzufuegen", command=self.on_choose_train_images)
        self.btn_add_train.pack(side=tk.LEFT, padx=8)
        self.btn_train = ttk.Button(top, text="Train Data", command=self.on_train_data)
        self.btn_train.pack(side=tk.LEFT, padx=4)
        ttk.Label(top, text="Train-Filter").pack(side=tk.LEFT, padx=(8, 2))
        train_cb = ttk.Combobox(top, textvariable=self.train_filter_var, state="readonly", width=10)
        train_cb["values"] = ("current", "adaptive", "preprocess", "otsu", "clahe", "none")
        train_cb.pack(side=tk.LEFT, padx=(2, 8))
        ttk.Checkbutton(top, text="Find best value", variable=self.find_best_var).pack(side=tk.LEFT, padx=8)

        self.status = ttk.Label(top, text="Bereit.")
        self.status.pack(side=tk.LEFT, padx=12)

        east = ttk.Frame(self, padding=(10, 0, 10, 8))
        east.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(east, text="EAST Modell (.pb):").pack(side=tk.LEFT)
        ttk.Entry(east, textvariable=self.east_model_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Button(east, text="Auswaehlen", command=self.on_choose_east_model).pack(side=tk.LEFT)

        train_status = ttk.Frame(self, padding=(10, 0, 10, 8))
        train_status.pack(side=tk.TOP, fill=tk.X)
        self.train_count_label = ttk.Label(train_status, text="Trainingsdaten: 0 Bilder")
        self.train_count_label.pack(side=tk.LEFT)
        self.train_state_label = ttk.Label(train_status, text="Training: idle")
        self.train_state_label.pack(side=tk.LEFT, padx=(14, 0))
        self.train_combo_label = ttk.Label(train_status, text="Kombination pro Bild: 0/0")
        self.train_combo_label.pack(side=tk.LEFT, padx=(14, 0))
        self.train_progress = ttk.Progressbar(
            train_status,
            orient=tk.HORIZONTAL,
            mode="determinate",
            maximum=100.0,
            variable=self.train_progress_var,
            length=280,
        )
        self.train_progress.pack(side=tk.LEFT, padx=(14, 0))

        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main, padding=10)
        main.add(left, weight=2)

        self.canvas = tk.Canvas(left, bg="#111111", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        right = ttk.Frame(main, padding=10)
        main.add(right, weight=1)

        ttk.Label(right, text="Erkannter Text:").pack(anchor="w")
        self.doc_type_label = ttk.Label(right, text="Dokumenttyp: -")
        self.doc_type_label.pack(anchor="w", pady=(4, 6))
        self.text = tk.Text(right, wrap="word", height=10)
        self.text.pack(fill=tk.BOTH, expand=True)

        hint = (
            "Tipps fuer Ausweis-Erkennung:\n"
            "- Preprocessing wechseln (adaptive/otsu/clahe).\n"
            "- Detektion: OCR-Boxen, MSER oder EAST.\n"
            "- Optional Detection-ROIs fuer OCR aktivieren."
        )
        ttk.Label(right, text=hint, justify="left").pack(anchor="w", pady=(10, 0))

        self.bind("<Configure>", lambda e: self._redraw())

    def set_status(self, msg: str):
        self.status.config(text=msg)
        self.update_idletasks()

    def _set_training_progress(self, pct: float, text: str):
        self.train_progress_var.set(max(0.0, min(100.0, pct)))
        self.train_state_label.config(text=f"Training: {text}")
        self.update_idletasks()

    def _set_combo_progress(self, combo_idx: int, total_combos: int):
        combo_idx = max(0, combo_idx)
        total_combos = max(0, total_combos)
        self.train_combo_label.config(text=f"Kombination pro Bild: {combo_idx}/{total_combos}")
        self.update_idletasks()

    def _set_training_controls(self, enabled: bool):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.btn_add_train.config(state=state)
        self.btn_train.config(state=state)

    def _post_status(self, msg: str):
        self.after(0, lambda m=msg: self.set_status(m))

    def _post_training_progress(self, pct: float, text: str):
        self.after(0, lambda p=pct, t=text: self._set_training_progress(p, t))

    def _post_combo_progress(self, combo_idx: int, total_combos: int):
        self.after(0, lambda i=combo_idx, t=total_combos: self._set_combo_progress(i, t))

    def _post_info(self, title: str, msg: str):
        self.after(0, lambda ttl=title, m=msg: messagebox.showinfo(ttl, m))

    def _post_warning(self, title: str, msg: str):
        self.after(0, lambda ttl=title, m=msg: messagebox.showwarning(ttl, m))

    def _on_training_finished(self):
        self.training_running = False
        self._set_training_controls(True)

    def on_choose_file(self):
        path = filedialog.askopenfilename(
            title="Bild/PDF auswaehlen",
            filetypes=[
                ("Images / PDF", "*.png *.jpg *.jpeg *.jfif *.tif *.tiff *.bmp *.pdf"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        self.file_path = path
        try:
            self.orig_bgr = _read_image_any(path)
            self.pre_bgr = None
            self.display_bgr = self.orig_bgr
            self.current_boxes = []
            self.set_status(f"Geladen: {os.path.basename(path)}")
            self.doc_type_label.config(text="Dokumenttyp: -")
            self.text.delete("1.0", tk.END)
            self._redraw()
        except Exception as ex:
            messagebox.showerror("Fehler", str(ex))

    def on_choose_east_model(self):
        path = filedialog.askopenfilename(
            title="EAST .pb Modell auswaehlen",
            filetypes=[("TensorFlow Graph", "*.pb"), ("All files", "*.*")],
        )
        if path:
            self.east_model_var.set(path)

    def on_choose_train_images(self):
        paths = filedialog.askopenfilenames(
            title="Trainingsbilder auswaehlen",
            filetypes=[
                ("Images / PDF", "*.png *.jpg *.jpeg *.jfif *.tif *.tiff *.bmp *.pdf"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return
        self.train_image_paths = list(paths)
        self.train_count_label.config(text=f"Trainingsdaten: {len(self.train_image_paths)} Bilder")
        self._set_training_progress(0.0, "bereit")
        self._set_combo_progress(0, 0)
        self.set_status(f"Trainingsdaten gesetzt: {len(self.train_image_paths)} Dateien")

    def _keyword_hit_count(self, text: str) -> int:
        norm = _normalize_for_match(text)
        hits = 0
        for keywords in DOC_CATEGORY_KEYWORDS.values():
            for kw in keywords:
                if _normalize_for_match(kw) in norm:
                    hits += 1
        return hits

    def _resize_for_training(self, bgr: np.ndarray, long_edge: int = 1400) -> np.ndarray:
        if bgr is None or bgr.size == 0:
            return bgr
        h, w = bgr.shape[:2]
        cur_long = max(h, w)
        if cur_long <= long_edge:
            return bgr
        scale = long_edge / float(cur_long)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _get_adaptive_params_from_ui(self) -> Tuple[int, int, int]:
        h = int(self.adaptive_h_var.get())
        block = int(self.adaptive_block_var.get())
        c_val = int(self.adaptive_c_var.get())
        h = max(1, min(50, h))
        block = max(3, min(99, block))
        if block % 2 == 0:
            block += 1
        c_val = max(1, min(100, c_val))
        return h, block, c_val

    def _get_train_method(self) -> str:
        mode = (self.train_filter_var.get() or "current").strip().lower()
        if mode == "current":
            return self.preprocess_var.get().strip().lower()
        return mode

    def _evaluate_adaptive_combo_single_image(
        self,
        image_path: str,
        h: int,
        block_size: int,
        c_val: int,
        train_method: str = "adaptive",
        train_engine: str = "tesseract",
        cached_bgr: np.ndarray | None = None,
    ) -> dict:
        processed = 0
        recognized_docs = 0
        keyword_hits = 0
        category = "Unbekannt"
        reason = ""

        try:
            bgr = cached_bgr if cached_bgr is not None else _read_image_any(image_path)
            if train_method == "adaptive":
                pre = preprocess_for_ocr(bgr, method="adaptive", adaptive_params=(h, block_size, c_val))
            else:
                pre = preprocess_for_ocr(bgr, method=train_method)
            if train_engine == "paddle":
                res = ocr_with_paddle(bgr, pre)
            elif train_engine == "easyocr":
                res = ocr_with_easyocr(bgr, pre)
            else:
                res = ocr_with_tesseract(bgr, pre)
            category, _, reason = classify_document_type(res.full_text)
            if category != "Unbekannt":
                recognized_docs = 1
            keyword_hits = self._keyword_hit_count(res.full_text)
            processed = 1
        except Exception:
            pass

        accuracy = float(recognized_docs)
        return {
            "h": h,
            "block_size": block_size,
            "c_value": c_val,
            "processed": processed,
            "recognized_docs": recognized_docs,
            "keyword_hits": keyword_hits,
            "accuracy": accuracy,
            "category": category,
            "reason": reason,
        }

    def _load_top_summary_seeds(self, summary_path: str, top_n: int = 3) -> List[Tuple[int, int, int]]:
        if not os.path.isfile(summary_path):
            return []
        rows = []
        try:
            with open(summary_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        h = int(row.get("h", "0"))
                        block_size = int(row.get("block_size", "0"))
                        c_val = int(row.get("c_value", "0"))
                        avg_acc = float(row.get("avg_accuracy", "0") or 0.0)
                        keyword_hits = int(float(row.get("keyword_hits", "0") or 0))
                        rows.append((avg_acc, keyword_hits, h, block_size, c_val))
                    except Exception:
                        continue
        except Exception:
            return []

        rows.sort(key=lambda x: (x[0], x[1]), reverse=True)
        out = []
        for _, __, h, b, c in rows[:top_n]:
            out.append((h, b, c))
        return out

    def _build_training_combos(
        self,
        max_candidates: int,
        base_h: int,
        base_block: int,
        base_c: int,
        summary_path: str,
    ) -> List[Tuple[int, int, int]]:
        combos: List[Tuple[int, int, int]] = []
        seen = set()

        def _add_combo(h: int, block_size: int, c_val: int):
            h = int(max(3, min(10, h)))
            block_size = int(max(3, min(99, block_size)))
            if block_size % 2 == 0:
                block_size += 1
            c_val = int(max(1, min(100, c_val)))
            key = (h, block_size, c_val)
            if key in seen or len(combos) >= max_candidates:
                return
            seen.add(key)
            combos.append(key)

        def _add_wave(h_vals, b_vals, c_vals):
            if len(combos) >= max_candidates:
                return
            for h in h_vals:
                for b in b_vals:
                    for c in c_vals:
                        _add_combo(h, b, c)
                        if len(combos) >= max_candidates:
                            return

        def _add_local_rings(anchor_h: int, anchor_b: int, anchor_c: int):
            # Large -> medium -> small neighborhoods around an anchor.
            for dh in [0, -2, 2, -1, 1]:
                for db in [0, -24, 24, -16, 16, -8, 8, -4, 4]:
                    for dc in [0, -30, 30, -20, 20, -10, 10, -5, 5, -2, 2]:
                        _add_combo(anchor_h + dh, anchor_b + db, anchor_c + dc)
                        if len(combos) >= max_candidates:
                            return

        # 1) Global coarse exploration with relatively large distances.
        _add_wave([3, 5, 7, 9], [11, 31, 51, 71, 91], [5, 25, 45, 65, 85])

        # 2) Warm-start seeds from previous summary + current UI defaults.
        seeds = [(base_h, base_block, base_c)]
        seeds.extend(self._load_top_summary_seeds(summary_path, top_n=3))
        unique_seeds = []
        for s in seeds:
            if s not in unique_seeds:
                unique_seeds.append(s)

        # 3) Multi-scale local search around each seed (not only one value).
        for sh, sb, sc in unique_seeds:
            _add_local_rings(sh, sb, sc)
            if len(combos) >= max_candidates:
                break

        # 4) Final filler exploration to keep diversity if budget remains.
        _add_wave(range(3, 11), range(13, 100, 14), range(7, 101, 14))
        return combos

    def on_train_data(self):
        if self.training_running:
            messagebox.showinfo("Info", "Training laeuft bereits.")
            return
        if not self.train_image_paths:
            messagebox.showinfo("Info", "Bitte zuerst Trainingsdaten hinzufuegen.")
            return

        self.training_running = True
        self._set_training_controls(False)
        find_best = bool(self.find_best_var.get())
        train_method = self._get_train_method()
        train_engine = self.engine_var.get().strip().lower()
        base_h, base_block, base_c = self._get_adaptive_params_from_ui()
        self._set_training_progress(0.0, "gestartet")
        self._set_combo_progress(0, 0)
        self.set_status(
            f"Training gestartet mit {len(self.train_image_paths)} Bildern (Filter={train_method}, Engine={train_engine})"
        )

        self.training_thread = threading.Thread(
            target=self._train_data_worker,
            args=(find_best, train_method, train_engine, list(self.train_image_paths), base_h, base_block, base_c),
            daemon=True,
        )
        self.training_thread.start()

    def _train_data_worker(
        self,
        find_best: bool,
        train_method: str,
        train_engine: str,
        image_paths: List[str],
        base_h: int,
        base_block: int,
        base_c: int,
    ):
        try:
            self._train_data_worker_impl(find_best, train_method, train_engine, image_paths, base_h, base_block, base_c)
        except Exception as ex:
            self._post_status("Training abgebrochen (Fehler).")
            self._post_warning("Train Data Fehler", str(ex))
        finally:
            self.after(0, self._on_training_finished)

    def _train_data_worker_impl(
        self,
        find_best: bool,
        train_method: str,
        train_engine: str,
        image_paths: List[str],
        base_h: int,
        base_block: int,
        base_c: int,
    ):
        out_path = os.path.join(os.getcwd(), "train_adaptive_grid_results.csv")
        summary_path = os.path.join(os.getcwd(), "train_adaptive_grid_summary.csv")
        run_id = dt.datetime.now().isoformat(timespec="seconds")
        self._post_training_progress(0.0, "gestartet")
        self._post_combo_progress(0, 0)
        self._post_status(f"Training gestartet mit {len(image_paths)} Bildern")

        # Fast mode defaults to keep runtime practical.
        max_candidates = 30
        early_stop_images = 5
        resize_long_edge = 1400
        if train_method == "adaptive":
            combos = self._build_training_combos(
                max_candidates=max_candidates,
                base_h=base_h,
                base_block=base_block,
                base_c=base_c,
                summary_path=summary_path,
            )
            seed_info = self._load_top_summary_seeds(summary_path, top_n=3)
        else:
            # Non-adaptive methods have no h/block/C sweep.
            combos = [(base_h, base_block, base_c)]
            seed_info = []

        # Preload and resize training images once (major speedup).
        loaded_images: List[Tuple[str, np.ndarray]] = []
        for path in image_paths:
            try:
                bgr = _read_image_any(path)
                loaded_images.append((path, self._resize_for_training(bgr, long_edge=resize_long_edge)))
            except Exception:
                continue

        total_images = len(loaded_images)
        total_combos = len(combos)
        total_steps = total_images * total_combos
        self._post_status(
            f"Sweep gestartet... {total_images} Bilder x {total_combos} Kombinationen "
            f"(Filter={train_method}, max {max_candidates if train_method == 'adaptive' else 1}), seeds={len(seed_info)+1 if train_method == 'adaptive' else 0}"
        )
        self._post_training_progress(0.0, f"bild 0/{total_images}")
        self._post_combo_progress(0, total_combos)

        rows = []
        combo_stats: dict[Tuple[int, int, int], dict] = {}
        step = 0

        disabled_combos: set[Tuple[int, int, int]] = set()

        # Requested order: all combinations for image1, then image2, ...
        for image_idx, (image_path, cached_bgr) in enumerate(loaded_images, start=1):
            image_name = os.path.basename(image_path)
            for combo_idx, (h, block_size, c_val) in enumerate(combos, start=1):
                key = (h, block_size, c_val)
                if key in disabled_combos:
                    row = {
                        "h": h,
                        "block_size": block_size,
                        "c_value": c_val,
                        "processed": 0,
                        "recognized_docs": 0,
                        "keyword_hits": 0,
                        "accuracy": 0.0,
                        "category": "Unbekannt",
                        "reason": "early_stop",
                    }
                else:
                    row = self._evaluate_adaptive_combo_single_image(
                        image_path,
                        h,
                        block_size,
                        c_val,
                        train_method=train_method,
                        train_engine=train_engine,
                        cached_bgr=cached_bgr,
                    )
                row["image_index"] = image_idx
                row["image_name"] = image_name
                rows.append(row)

                if key not in combo_stats:
                    combo_stats[key] = {
                        "h": h,
                        "block_size": block_size,
                        "c_value": c_val,
                        "processed": 0,
                        "recognized_docs": 0,
                        "keyword_hits": 0,
                    }
                combo_stats[key]["processed"] += row["processed"]
                combo_stats[key]["recognized_docs"] += row["recognized_docs"]
                combo_stats[key]["keyword_hits"] += row["keyword_hits"]

                # Early-stop combinations that show no signal in first N images.
                if image_idx >= early_stop_images and key not in disabled_combos:
                    stat = combo_stats[key]
                    if stat["recognized_docs"] == 0 and stat["keyword_hits"] == 0:
                        disabled_combos.add(key)

                step += 1
                if step % 25 == 0 or step == total_steps:
                    pct = (step / total_steps) * 100.0
                    self._post_training_progress(
                        pct,
                        f"bild {image_idx}/{total_images}, kombi {combo_idx}/{total_combos}",
                    )
                    self._post_combo_progress(combo_idx, total_combos)
                    self._post_status(
                        f"Training... Bild {image_idx}/{total_images}, Kombi {combo_idx}/{total_combos}"
                    )

        out_exists = os.path.isfile(out_path) and os.path.getsize(out_path) > 0
        with open(out_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not out_exists:
                writer.writerow(
                    [
                        "run_id",
                        "preprocess_method",
                        "ocr_engine",
                        "image_index",
                        "image_name",
                        "h",
                        "block_size",
                        "c_value",
                        "processed",
                        "recognized_docs",
                        "keyword_hits",
                        "accuracy",
                        "category",
                        "reason",
                    ]
                )
            for row in rows:
                writer.writerow(
                    [
                        run_id,
                        train_method,
                        train_engine,
                        row["image_index"],
                        row["image_name"],
                        row["h"],
                        row["block_size"],
                        row["c_value"],
                        row["processed"],
                        row["recognized_docs"],
                        row["keyword_hits"],
                        f"{row['accuracy']:.6f}",
                        row["category"],
                        row["reason"],
                    ]
                )

        if not combo_stats:
            self._post_status("Sweep fertig, aber keine gueltigen Ergebnisse.")
            self._post_warning("Train Data", "Keine gueltigen Ergebnisse erzeugt.")
            return

        summary_rows = []
        best = None
        best_key = (-1.0, -1, -1.0)  # avg_accuracy, total_keyword_hits, avg_keyword_hits
        for stat in combo_stats.values():
            processed = max(1, stat["processed"])
            avg_accuracy = stat["recognized_docs"] / processed
            avg_keyword_hits = stat["keyword_hits"] / processed
            srow = {
                "h": stat["h"],
                "block_size": stat["block_size"],
                "c_value": stat["c_value"],
                "processed": stat["processed"],
                "recognized_docs": stat["recognized_docs"],
                "keyword_hits": stat["keyword_hits"],
                "avg_accuracy": avg_accuracy,
                "avg_keyword_hits": avg_keyword_hits,
            }
            summary_rows.append(srow)
            key = (avg_accuracy, stat["keyword_hits"], avg_keyword_hits)
            if key > best_key:
                best_key = key
                best = srow

        summary_exists = os.path.isfile(summary_path) and os.path.getsize(summary_path) > 0
        with open(summary_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not summary_exists:
                writer.writerow(
                    [
                        "run_id",
                        "preprocess_method",
                        "ocr_engine",
                        "h",
                        "block_size",
                        "c_value",
                        "processed",
                        "recognized_docs",
                        "keyword_hits",
                        "avg_accuracy",
                        "avg_keyword_hits",
                    ]
                )
            for srow in summary_rows:
                writer.writerow(
                    [
                        run_id,
                        train_method,
                        train_engine,
                        srow["h"],
                        srow["block_size"],
                        srow["c_value"],
                        srow["processed"],
                        srow["recognized_docs"],
                        srow["keyword_hits"],
                        f"{srow['avg_accuracy']:.6f}",
                        f"{srow['avg_keyword_hits']:.6f}",
                    ]
                )

        msg = (
            f"Best Kombination:\n"
            f"Filter={train_method}\n"
            f"h={best['h']}, block_size={best['block_size']}, C={best['c_value']}\n"
            f"avg_accuracy={best['avg_accuracy']:.2%}, recognized_docs={best['recognized_docs']}, keyword_hits={best['keyword_hits']}\n\n"
            f"Getestete Kombinationen: {total_combos}\n\n"
            f"Details CSV:\n{out_path}\n"
            f"Summary CSV:\n{summary_path}"
        )
        self._post_status(
            f"Best: h={best['h']}, block={best['block_size']}, C={best['c_value']} | avg_acc={best['avg_accuracy']:.2%}"
        )
        self._post_training_progress(100.0, "fertig")
        self._post_combo_progress(total_combos, total_combos)
        self._post_info("Train Data fertig", msg)

    def _run_ocr_engine(self, orig: np.ndarray, pre: np.ndarray | None) -> OCRResult:
        engine = self.engine_var.get().strip().lower()
        if engine == "tesseract":
            return ocr_with_tesseract(orig, pre)
        if engine == "easyocr":
            return ocr_with_easyocr(orig, pre)
        return ocr_with_paddle(orig, pre)

    def _detect_regions(self, method: str) -> List[Tuple[int, int, int, int]]:
        if self.orig_bgr is None:
            return []

        method = method.strip().lower()
        if method == "mser":
            return detect_text_regions_mser(self.orig_bgr)
        if method == "east":
            return detect_text_regions_east(self.orig_bgr, model_path=self.east_model_var.get().strip())
        return []

    def _parse_detection_mode(self) -> Tuple[bool, bool, bool]:
        mode = self.detect_var.get().strip().lower()
        use_ocr_boxes = "ocr" in mode
        use_mser = "mser" in mode
        use_east = "east" in mode
        return use_ocr_boxes, use_mser, use_east

    def _merge_boxes(self, *box_groups: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        merged: List[Tuple[int, int, int, int]] = []
        for group in box_groups:
            if group:
                merged.extend(group)
        if not merged:
            return []

        xywh = []
        scores = []
        for x1, y1, x2, y2 in merged:
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            xywh.append([x1, y1, w, h])
            scores.append(1.0)

        indices = cv2.dnn.NMSBoxes(xywh, scores, score_threshold=0.0, nms_threshold=0.35)
        if indices is None or len(indices) == 0:
            return merged
        return [merged[int(i)] for i in np.array(indices).flatten()]

    def _ocr_on_boxes(self, boxes: List[Tuple[int, int, int, int]], pre_full: np.ndarray) -> OCRResult:
        if self.orig_bgr is None:
            return OCRResult(full_text="", boxes=[])

        H, W = self.orig_bgr.shape[:2]
        merged_text: List[str] = []
        merged_boxes: List[Tuple[int, int, int, int]] = []

        for (x1, y1, x2, y2) in sorted(boxes, key=lambda b: (b[1], b[0])):
            x1 = max(0, min(W - 1, x1))
            y1 = max(0, min(H - 1, y1))
            x2 = max(1, min(W, x2))
            y2 = max(1, min(H, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            if (x2 - x1) * (y2 - y1) < 120:
                continue

            roi_orig = self.orig_bgr[y1:y2, x1:x2]
            roi_pre = pre_full[y1:y2, x1:x2] if pre_full is not None else None

            try:
                part = self._run_ocr_engine(roi_orig, roi_pre)
            except Exception:
                continue

            if part.full_text:
                merged_text.append(part.full_text.strip())

            if part.boxes:
                for bx1, by1, bx2, by2 in part.boxes:
                    merged_boxes.append((x1 + bx1, y1 + by1, x1 + bx2, y1 + by2))
            else:
                merged_boxes.append((x1, y1, x2, y2))

        return OCRResult(full_text="\n".join(t for t in merged_text if t).strip(), boxes=merged_boxes)

    def on_run_ocr(self):
        if not self.file_path or self.orig_bgr is None:
            messagebox.showinfo("Info", "Bitte zuerst eine Datei auswaehlen.")
            return

        try:
            self.set_status("Preprocessing...")
            pre_method = self.preprocess_var.get().strip().lower()
            if pre_method == "adaptive":
                pre = preprocess_for_ocr(self.orig_bgr, method=pre_method, adaptive_params=self._get_adaptive_params_from_ui())
            else:
                pre = preprocess_for_ocr(self.orig_bgr, method=pre_method)
            self.pre_bgr = cv2.cvtColor(pre, cv2.COLOR_GRAY2BGR) if len(pre.shape) == 2 else pre.copy()
            self.display_bgr = self.pre_bgr
            self._redraw()

            self.set_status("OCR laeuft...")
            use_ocr_boxes, use_mser, use_east = self._parse_detection_mode()

            mser_boxes = self._detect_regions("mser") if use_mser else []
            east_boxes = self._detect_regions("east") if use_east else []
            detected_boxes = self._merge_boxes(mser_boxes, east_boxes)
            use_roi_ocr = bool(self.use_detection_for_ocr_var.get()) and len(detected_boxes) > 0

            if use_roi_ocr:
                res = self._ocr_on_boxes(detected_boxes, pre)
            else:
                res = self._run_ocr_engine(self.orig_bgr, pre)

            ocr_boxes = res.boxes if use_ocr_boxes else []
            boxes_to_draw = self._merge_boxes(ocr_boxes, detected_boxes)

            self._show_text(res.full_text)
            self._show_document_type(res.full_text)
            self._draw_boxes(boxes_to_draw)
            self.set_status(f"Fertig. Boxen: {len(boxes_to_draw)}")
        except Exception as ex:
            messagebox.showerror("Fehler", str(ex))
            self.set_status("Fehler.")

    def _show_text(self, txt: str):
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, txt if txt else "(kein Text erkannt)")

    def _show_document_type(self, txt: str):
        category, score, reason = classify_document_type(txt)
        if category == "Unbekannt":
            self.doc_type_label.config(text=f"Dokumenttyp: Unbekannt ({score:.0%})")
        else:
            self.doc_type_label.config(text=f"Dokumenttyp: {category} ({score:.0%}, {reason})")

    def _redraw(self):
        img = self.display_bgr if self.display_bgr is not None else self.orig_bgr
        if img is None:
            self.canvas.delete("all")
            return

        self.canvas.delete("all")
        ch = max(1, self.canvas.winfo_height())
        cw = max(1, self.canvas.winfo_width())

        h, w = img.shape[:2]
        scale = min(cw / w, ch / h)
        self.scale = scale

        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((new_w, new_h), Image.BILINEAR)

        self.display_img_tk = ImageTk.PhotoImage(pil)
        self.canvas.create_image(0, 0, anchor="nw", image=self.display_img_tk)
        self._draw_boxes_on_canvas(self.current_boxes)

    def _draw_boxes(self, boxes: List[Tuple[int, int, int, int]]):
        self.current_boxes = list(boxes) if boxes else []
        self._redraw()

    def _draw_boxes_on_canvas(self, boxes: List[Tuple[int, int, int, int]]):
        if not boxes:
            return
        for (x1, y1, x2, y2) in boxes:
            sx1, sy1 = int(x1 * self.scale), int(y1 * self.scale)
            sx2, sy2 = int(x2 * self.scale), int(y2 * self.scale)
            self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline="#00FF7F", width=2)


def main():
    app = OCRApp()
    app.mainloop()


if __name__ == "__main__":
    main()
