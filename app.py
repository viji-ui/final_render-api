"""
Flask API + Dashboard server for the Gas Cylinder Inspection & Tracking demo.

Render-ready version of app.py — same routes and logic, with:
- CORS enabled (so the React Native app / any frontend can call it cross-origin)
- PORT read from the environment (Render assigns this dynamically)
- debug mode off (never run debug=True in production)
- served via gunicorn, not the Flask dev server (see Procfile)

Pipeline:
  React Native app (camera scan OR file upload) --> POSTs the QR image
  --> this API decodes the QR itself (OpenCV) --> extracts cylinder JSON
  --> computes status --> stores state --> returns result to the UI
  --> Dashboard (web) polls the API and renders live
"""

import os
import json
import threading
from datetime import datetime

import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow requests from any origin (mobile app, browser, etc.)

lock = threading.Lock()
state = {
    "latest": None,
    "history": [],  # most recent first
    "stats": {"total": 0, "passed": 0, "failed": 0, "due": 0},
}
MAX_HISTORY = 50

qr_detector = cv2.QRCodeDetector()


def compute_status(cyl: dict) -> str:
    """SAFE / DUE / EXPIRED based on next_inspection date."""
    next_insp = cyl.get("next_inspection")
    if not next_insp:
        return "due"
    try:
        next_date = datetime.strptime(next_insp, "%Y-%m-%d")
    except ValueError:
        return "due"
    days_left = (next_date - datetime.now()).days
    if days_left < 0:
        return "expired"
    if days_left <= 60:
        return "due"
    return "safe"


def decode_qr_from_bytes(image_bytes: bytes):
    """Decode a QR code from raw image bytes using OpenCV. Returns decoded text or None."""
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return None

    # Try direct detection first
    data, points, _ = qr_detector.detectAndDecode(img)
    if data:
        return data

    # Fallback: upscale + grayscale + threshold, helps with small/low-contrast QR crops
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    data, points, _ = qr_detector.detectAndDecode(thresh)
    return data or None


def record_scan(data: dict) -> dict:
    """Shared logic: compute status, store in state, return the entry."""
    status = compute_status(data)
    entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "cylinder": data,
        "status": status,
    }
    with lock:
        state["latest"] = entry
        state["history"].insert(0, entry)
        state["history"] = state["history"][:MAX_HISTORY]
        state["stats"]["total"] += 1
        if status == "safe":
            state["stats"]["passed"] += 1
        elif status == "expired":
            state["stats"]["failed"] += 1
        else:
            state["stats"]["due"] += 1
    print(f"[scan] {data.get('cylinder_id')} -> {status}")
    return entry


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/scan", methods=["POST"])
def receive_scan():
    """Called when the caller already has decoded cylinder JSON (e.g. scanner.py locally)."""
    data = request.get_json(force=True, silent=True)
    if not data or "cylinder_id" not in data:
        return jsonify({"error": "invalid payload, expected cylinder JSON"}), 400
    entry = record_scan(data)
    return jsonify({"ok": True, "status": entry["status"], "cylinder": entry["cylinder"]})


@app.route("/api/scan-qr", methods=["POST"])
def scan_qr_image():
    """
    Called directly from the UI: upload or camera-captured QR image goes here.
    Expects multipart/form-data with field name 'image'.
    The API decodes the QR itself and returns the extracted cylinder info + status.
    """
    if "image" not in request.files:
        return jsonify({"error": "no image uploaded, expected multipart field 'image'"}), 400

    file = request.files["image"]
    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "uploaded image is empty"}), 400

    qr_text = decode_qr_from_bytes(image_bytes)
    if not qr_text:
        return jsonify({"error": "no QR code detected in image"}), 422

    try:
        data = json.loads(qr_text)
    except json.JSONDecodeError:
        return jsonify({"error": "QR code did not contain valid JSON", "raw": qr_text}), 422

    if not isinstance(data, dict) or "cylinder_id" not in data:
        return jsonify({"error": "invalid cylinder payload in QR", "raw": data}), 400

    entry = record_scan(data)
    return jsonify({"ok": True, "status": entry["status"], "cylinder": entry["cylinder"], "time": entry["time"]})


@app.route("/api/latest")
def get_latest():
    with lock:
        return jsonify(state["latest"])


@app.route("/api/history")
def get_history():
    with lock:
        return jsonify(state["history"])


@app.route("/api/stats")
def get_stats():
    with lock:
        return jsonify(state["stats"])


@app.route("/api/reset", methods=["POST"])
def reset():
    with lock:
        state["latest"] = None
        state["history"] = []
        state["stats"] = {"total": 0, "passed": 0, "failed": 0, "due": 0}
    return jsonify({"ok": True})


@app.route("/healthz")
def healthz():
    # Render (and most hosts) can hit this to confirm the service is alive.
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    # Local dev only. On Render, gunicorn imports `app` directly (see Procfile)
    # and this block never runs.
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)
