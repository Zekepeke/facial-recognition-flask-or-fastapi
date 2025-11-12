from flask import Flask, request, jsonify
import cv2, numpy as np, json, pathlib, datetime
import mediapipe as mp
from keras_facenet import FaceNet

app = Flask(__name__)

DB_PATH = pathlib.Path(__file__).with_name("face_db.json")

# ----- Face detection (MediaPipe) -----
mp_face = mp.solutions.face_detection

# ----- Face embedding (FaceNet) -----
embedder = FaceNet()  # 512-D embeddings

def load_db():
    if DB_PATH.exists():
        with open(DB_PATH, "r") as f:
            return json.load(f)
    return {}  # minimal: { "person_id": [ [emb1...], [emb2...] ], ... }

def save_db(db):
    with open(DB_PATH, "w") as f:
        json.dump(db, f)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

# face preprocessing function using MediaPipe
def preprocess_face_from_bytes(img_bytes):
    """Decode -> detect face -> crop -> resize to 160x160 RGB -> return np.uint8 image."""
    nparr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # detect with MediaPipe
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
        result = fd.process(rgb)
    if not result.detections:
        raise ValueError("No face detected")

    # take the highest score detection
    det = max(result.detections, key=lambda d: d.score[0])
    h, w, _ = rgb.shape
    bbox = det.location_data.relative_bounding_box
    x1 = max(int(bbox.xmin * w), 0)
    y1 = max(int(bbox.ymin * h), 0)
    x2 = min(int((bbox.xmin + bbox.width) * w), w)
    y2 = min(int((bbox.ymin + bbox.height) * h), h)

    # a little padding around the box
    pad = int(0.15 * max(x2 - x1, y2 - y1))
    x1 = max(x1 - pad, 0); y1 = max(y1 - pad, 0)
    x2 = min(x2 + pad, w); y2 = min(y2 + pad, h)

    face = rgb[y1:y2, x1:x2]
    if face.size == 0:
        raise ValueError("Bad crop")

    face = cv2.resize(face, (160, 160), interpolation=cv2.INTER_AREA)
    return face  # RGB uint8

# the facenet embedding function
def embed_face(face_rgb_160):
    """Return a 512-D FaceNet embedding as np.ndarray (float32)."""
    # keras-facenet handles prewhiten internally
    # pass list of faces
    emb = embedder.embeddings([face_rgb_160])[0].astype("float32")
    return emb

@app.route("/enroll/<person_id>", methods=["POST"])
def enroll(person_id):
    """POST multipart/form-data with key 'file' (image). Adds embedding under person_id."""
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "missing file"}), 400
    file = request.files["file"].read()
    try:
        face = preprocess_face_from_bytes(file)
        emb = embed_face(face)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    db = load_db()
    db.setdefault(person_id, [])
    db[person_id].append(emb.tolist())
    save_db(db)
    return jsonify({"ok": True, "person_id": person_id, "num_samples": len(db[person_id])})

@app.route("/identify", methods=["POST"])
def identify():
    """
    POST multipart/form-data with 'file' (image).
    Returns best match if similarity >= threshold.
    """
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "missing file"}), 400
    # cosine similarity
    threshold = float(request.form.get("threshold", 0.65)) # default 0.65
    img_bytes = request.files["file"].read()

    try:
        face = preprocess_face_from_bytes(img_bytes)
        probe = embed_face(face)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    db = load_db()
    if not db:
        return jsonify({"ok": True, "match": None, "reason": "empty gallery"})

    best_id, best_sim = None, -1.0
    for person_id, emb_list in db.items():
        # compare against all of that person's samples 
        # take MAX similarity
        sims = [cosine_sim(probe, np.array(e, dtype="float32")) for e in emb_list]
        person_sim = max(sims) if sims else -1.0
        if person_sim > best_sim:
            best_sim, best_id = person_sim, person_id

    match = best_id if best_sim >= threshold else None
    return jsonify({
        "ok": True,
        "match": match,
        "similarity": round(best_sim, 4),
        "threshold": threshold
    })

@app.route("/whoami", methods=["GET"])
def whoami():
    # simple health/meta
    return jsonify({
        "ok": True,
        "model": "keras-facenet (FaceNet) + MediaPipe FaceDetection",
        "db_path": str(DB_PATH),
        "last_modified": datetime.datetime.fromtimestamp(DB_PATH.stat().st_mtime).isoformat() if DB_PATH.exists() else None
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)