"""Check yoga photos with the same model used in the live demo.

    python check_image.py                      # opens a web page: drag & drop / choose / paste photos
    python check_image.py photo1.jpg photo2.png --pose tree     # command line, prints a report
    python check_image.py --folder my_photos --pose warrior2    # whole folder + summary + CSV

Results (annotated image + text report) are also saved to outputs/image_checks/.
The page is served only on your own computer (127.0.0.1).
"""
from __future__ import annotations

import argparse
import base64
import csv
import glob
import json
import os
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from yogacoach.analyze import analyze_image, load_image_bytes  # noqa: E402
from yogacoach.config import TARGET_POSES  # noqa: E402

MAX_BYTES = 40 * 1024 * 1024
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".tif", ".tiff")


def safe_name(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name))[0]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)[:60] or "image"


def save_result(save_dir: str, name: str, rep) -> str | None:
    try:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.join(save_dir, safe_name(name) + "_checked")
        cv2.imwrite(base + ".jpg", rep.annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(base + ".txt", "w", encoding="utf-8") as f:
            f.write(rep.text + "\n")
        return base + ".jpg"
    except Exception:  # noqa: BLE001 - saving is best-effort
        return None


def make_handler(backend, model_path, save_dir):
    lock = threading.Lock()            # the pose model is not thread-safe
    html = open(os.path.join(ROOT, "yogacoach", "web", "index.html"), "rb").read()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep the console quiet
            pass

        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            p = urlparse(self.path).path
            if p in ("/", "/index.html"):
                self._send(200, html, "text/html; charset=utf-8")
            elif p == "/favicon.ico":
                self._send(204, b"")
            else:
                self._send(404, b'{"error":"not found"}')

        def do_POST(self):
            u = urlparse(self.path)
            if u.path != "/analyze":
                return self._send(404, b'{"error":"not found"}')
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BYTES:
                return self._send(413, json.dumps({"error": "image missing or too large"}).encode())
            data = self.rfile.read(n)
            pose = q.get("pose", "auto")
            leniency = q.get("leniency", "lenient")
            mode = q.get("mode", "hybrid")
            if pose not in ["auto"] + TARGET_POSES or leniency not in ("strict", "normal", "lenient") \
                    or mode not in ("hybrid", "dl", "rules"):
                return self._send(400, json.dumps({"error": "bad parameter"}).encode())
            if mode != "rules" and not os.path.isfile(model_path or ""):
                mode = "rules"             # no trained model file: fall back to geometry only
            try:
                img = load_image_bytes(data)
            except Exception as e:  # noqa: BLE001
                body = dict(status="no_person", text=f"Could not read this file as an image ({e}).", warnings=["Unreadable image."],
                            rows=[], feedback=[], probs={}, image=None, pose=None, pose_title="", score=None,
                            leniency=leniency, mode=mode)
                return self._send(200, json.dumps(body).encode())
            try:
                with lock:
                    rep = analyze_image(img, backend, model_path, pose, mode, leniency)
            except Exception as e:  # noqa: BLE001
                return self._send(500, json.dumps({"error": f"analysis failed: {e}"}).encode())
            save_result(save_dir, q.get("name", "upload"), rep)
            ok, buf = cv2.imencode(".jpg", rep.annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
            url = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
            self._send(200, json.dumps(rep.to_json(url)).encode())

    return Handler


def run_server(backend, model_path, args):
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(backend, model_path, args.save_dir))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Yoga image checker running at {url}   (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


def run_cli(backend, model_path, files, args):
    rows = []
    for f in files:
        try:
            img = load_image_bytes(open(f, "rb").read())
        except Exception as e:  # noqa: BLE001
            print(f"\n=== {f}\nCould not read image: {e}")
            continue
        rep = analyze_image(img, backend, model_path, args.pose, args.mode, args.leniency)
        out = save_result(args.save_dir, f, rep)
        print(f"\n=== {f}\n{rep.text}\n(saved: {out})")
        rows.append([f, rep.status, rep.pose or "", f"{rep.score:.0f}" if rep.score is not None else "",
                     "; ".join(rep.feedback) if rep.status == "incorrect" else ""])
    if len(rows) > 1 or args.folder:
        print("\n---- summary ----")
        for st in ("correct", "incorrect", "wrong_pose", "no_person"):
            print(f"{st:11s}: {sum(r[1] == st for r in rows)}")
        csv_path = os.path.join(args.save_dir, "summary.csv")
        os.makedirs(args.save_dir, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "status", "pose", "score", "feedback"])
            w.writerows(rows)
        print("summary table:", csv_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*", help="image files (omit to open the web page)")
    ap.add_argument("--folder", help="check every image in this folder")
    ap.add_argument("--pose", default="auto", choices=["auto"] + TARGET_POSES)
    ap.add_argument("--leniency", default="lenient", choices=["strict", "normal", "lenient"])
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "dl", "rules"])
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "yogacoach.pt"))
    ap.add_argument("--pose-variant", default="full", choices=["lite", "full", "heavy"])
    ap.add_argument("--pose-model", default=None)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--save-dir", default=os.path.join(ROOT, "outputs", "image_checks"))
    args = ap.parse_args()

    if args.mode != "rules" and not os.path.isfile(args.model):
        print(f"[warn] model file {args.model} not found - falling back to rules-only mode")
        args.mode = "rules"

    from yogacoach.pose_backend import PoseBackend
    backend = PoseBackend(args.pose_variant, args.pose_model, video=False)
    model_path = args.model if args.mode != "rules" else None

    files = list(args.images)
    if args.folder:
        files += sorted(f for f in glob.glob(os.path.join(args.folder, "*")) if f.lower().endswith(IMG_EXT))
    if files:
        run_cli(backend, model_path, files, args)
    else:
        run_server(backend, args.model, args)


if __name__ == "__main__":
    main()
