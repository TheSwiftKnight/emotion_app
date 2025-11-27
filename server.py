import os
import cv2
import numpy as np
import tensorflow as tf
import base64
import eventlet
import time
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from pythonosc.udp_client import SimpleUDPClient

import socket
import struct
import threading

# ======================
# 全域 BPM（Max → Python）
# ======================
current_bpm = 120.0  # 初始值

def bpm_listener():
    """從 UDP 9001 接收 Max 傳來的 BPM（文字或 binary float）"""
    global current_bpm
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 9001))   # 要跟 Max 一樣的 port
    print("Listening raw BPM on UDP 9001 ...")
    while True:
        data, addr = sock.recvfrom(1024)
        try:
            # 1) 試試看純文字 float
            try:
                text = data.decode("utf-8").strip()
                val = float(text)
                current_bpm = val
                print(f"[PY] BPM updated (text): {current_bpm}")
                continue
            except Exception:
                pass  # 不是純文字，就往下試

            # 2) Max binary float 格式：b'float....<4 bytes>'
            if data.startswith(b'float') and len(data) >= 4:
                bpm_bytes = data[-4:]
                val = struct.unpack('>f', bpm_bytes)[0]  # big-endian float
                current_bpm = float(val)
                print(f"[PY] BPM updated (binary): {current_bpm}")
            else:
                print("[PY] Unknown BPM packet:", data)

        except Exception as e:
            print("[PY] BPM parse error (unexpected):", data, e)

# 只讓 BPM listener 開一次
bpm_thread = None
def start_bpm_listener_once():
    global bpm_thread
    if bpm_thread is None or not bpm_thread.is_alive():
        bpm_thread = threading.Thread(target=bpm_listener, daemon=True)
        bpm_thread.start()
        print("[PY] BPM listener started.")
    else:
        print("[PY] BPM listener already running.")


# ================
# Flask & SocketIO
# ================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# ============
# 路徑 & 模型
# ============
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'share/emotion_detection_model_100epochs.h5'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

OSC_IP = "127.0.0.1"
OSC_PORT = 8000

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
model = None

# Phase1 預先計算結果的全域暫存
precomputed = {
    "video_path": None,
    "fps": None,
    "total_frames": None,
    "all_probs": None,    # numpy array (N, 7)
    "face_boxes": None    # list of (x, y, w, h) 或 None
}

# OSC 客戶端
osc_client = None
try:
    osc_client = SimpleUDPClient(OSC_IP, OSC_PORT)
    print(f"OSC client initialized: {OSC_IP}:{OSC_PORT}")
except Exception as e:
    print(f"Warning: Could not initialize OSC client: {e}")
    print("OSC messages will not be sent to Max/MSP")


# =====================
# 建構 / 載入模型的函數
# =====================
def create_model_architecture(input_shape=(48, 48, 1), num_classes=7):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', name="conv_1")(inputs)
    x = Conv2D(64, kernel_size=(3, 3), activation='relu', name="conv_2")(x)
    x = MaxPooling2D(pool_size=(2, 2), name="pool_1")(x)
    x = Dropout(0.1, name="dropout_1")(x)
    x = Conv2D(128, kernel_size=(3, 3), activation='relu', name="conv_3")(x)
    x = MaxPooling2D(pool_size=(2, 2), name="pool_2")(x)
    x = Dropout(0.1, name="dropout_2")(x)
    x = Conv2D(256, kernel_size=(3, 3), activation='relu', name="conv_4")(x)
    x = MaxPooling2D(pool_size=(2, 2), name="pool_3")(x)
    x = Dropout(0.1, name="dropout_3")(x)
    x = Flatten(name="flatten")(x)
    x = Dense(512, activation='relu', name="dense_1")(x)
    x = Dropout(0.2, name="dropout_4")(x)
    outputs = Dense(num_classes, activation='softmax', name="output")(x)
    return Model(inputs=inputs, outputs=outputs, name="emotion_model")


def load_emotion_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully via load_model.")
        except Exception as e:
            print(f"Direct load failed, trying to build architecture first: {e}")
            model = create_model_architecture()
            model.load_weights(MODEL_PATH)
            print("Model weights loaded successfully.")
    else:
        print(f"警告：找不到模型檔案 {MODEL_PATH}。請確保檔案存在。")


def predict_emotion_with_probabilities(face_bgr, model):
    """給一張臉（BGR），回傳 7 維機率"""
    face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized_face = cv2.resize(face_gray, (48, 48)) / 255.0
    face_img = np.expand_dims(resized_face, axis=(0, -1))
    predictions = model.predict(face_img, verbose=0)
    return predictions[0]


# ============
# Flask Routes
# ============
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """上傳影片 → 存檔 → 背景啟動 Phase1 預先計算"""
    global precomputed

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = 'uploaded_video.mp4'
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # 每次上傳新的影片，先重設 precomputed
    precomputed = {
        "video_path": filepath,
        "fps": None,
        "total_frames": None,
        "all_probs": None,
        "face_boxes": None
    }

    # 在背景執行 Phase1，不阻塞 HTTP 回應
    socketio.start_background_task(run_phase1, filepath)

    return jsonify({
        'message': 'File uploaded successfully, Phase1 started.',
        'filepath': filepath
    })


# =====================
# Phase1：預先掃完整部影片
# =====================
def run_phase1(video_path):
    """先把整部影片掃完，算每一幀的情緒機率 & 臉框，結果存到 precomputed 裡。"""
    global precomputed, model

    if model is None:
        load_emotion_model()
        if model is None:
            print("[PY] Model not found in Phase1!")
            return

    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    print("Starting video processing (Phase1: precompute probs)...")

    if not cap.isOpened():
        print("[PY] Cannot open video file in Phase1.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("FPS:", fps, "Total frames:", total_frames)

    all_probs = []
    face_boxes = []

    last_prob = np.ones(len(emotion_labels)) / len(emotion_labels)
    last_box = None

    SAMPLE_STEP = 5
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % SAMPLE_STEP == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5
            )

            if len(faces) > 0:
                x, y, w, h = faces[0]
                face = frame[y:y + h, x:x + w]
                prob = predict_emotion_with_probabilities(face, model)
                last_prob = prob
                last_box = (int(x), int(y), int(w), int(h))
            else:
                prob = last_prob
                last_box = None
        else:
            prob = last_prob

        all_probs.append(prob)
        face_boxes.append(last_box)

        if frame_idx % 50 == 0:
            print(f"Phase1: Processed frame {frame_idx}/{total_frames}")

        frame_idx += 1

    cap.release()

    all_probs = np.stack(all_probs)
    N = all_probs.shape[0]
    print("Collected probs for frames:", N)

    # 存到全域 precomputed
    precomputed["video_path"] = video_path
    precomputed["fps"] = fps
    precomputed["total_frames"] = total_frames
    precomputed["all_probs"] = all_probs
    precomputed["face_boxes"] = face_boxes

    print("[PY] Phase1 precompute finished.")

    # 告訴前端 Phase1 完成，可以啟用「開始分析」按鈕
    socketio.emit('phase1_complete', {
        "msg": "Phase1 precompute finished.",
        "frames": int(N),
        "fps": float(fps)
    })


# ============================
# SocketIO：Phase2 開始分析
# ============================
@socketio.on('start_processing')
def handle_process_video(data=None):
    """按下「開始分析」後，使用 Phase1 的結果播放影片＋丟 OSC 到 Max。"""
    global precomputed, osc_client

    # 確認 Phase1 跑完
    if precomputed["all_probs"] is None or precomputed["video_path"] is None:
        emit('error', {'msg': 'Phase1 not finished yet. Please upload and wait.'})
        return

    video_path = precomputed["video_path"]
    all_probs = precomputed["all_probs"]
    face_boxes = precomputed["face_boxes"]
    fps = precomputed["fps"]
    N = all_probs.shape[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        emit('error', {'msg': 'Cannot open video file in Phase2.'})
        return

    if fps is None or fps <= 0:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

    frame_idx = 0
    last_osc_time = -1e9

    print(f"[PY] Phase2: playback with 4-beat forecast, FPS={fps}, N={N}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= N:
                print("[PY] Video ended in Phase2.")
                break

            current_time_sec = frame_idx / fps

            # --- NOW ---
            probs_now = all_probs[frame_idx]
            idx_now = int(np.argmax(probs_now))
            emotion_now = emotion_labels[idx_now]
            prob_now = float(probs_now[idx_now])

            # --- FUTURE：未來四拍平均 ---
            safe_bpm = max(current_bpm, 1e-3)
            beat_seconds = 60.0 / safe_bpm
            four_beat_seconds = beat_seconds * 4.0

            window_frames = int(fps * four_beat_seconds)
            if window_frames < 1:
                window_frames = 1

            start = frame_idx + 1
            end = min(N, frame_idx + 1 + window_frames)

            if start < end:
                window = all_probs[start:end]
                probs_future = window.mean(axis=0)
            else:
                probs_future = probs_now.copy()

            idx_future = int(np.argmax(probs_future))
            emotion_future = emotion_labels[idx_future]
            prob_future = float(probs_future[idx_future])

            # --- 每一拍送 OSC ---
            if current_time_sec - last_osc_time >= beat_seconds:
                if osc_client is not None:
                    try:
                        # 1. 未來四拍的主情緒 + 機率 + 四拍秒數
                        osc_client.send_message(
                            "/emotion_label_future",
                            [emotion_future, prob_future, float(four_beat_seconds)]
                        )
                        # 2. FUTURE 七維
                        for label, p in zip(emotion_labels, probs_future):
                            osc_client.send_message(
                                "/emotion_prob_future", [label, float(p)]
                            )
                        # 3. NOW 主情緒
                        osc_client.send_message(
                            "/emotion_label", [emotion_now, prob_now]
                        )
                        # 4. NOW 七維
                        for label, p in zip(emotion_labels, probs_now):
                            osc_client.send_message(
                                "/emotion_prob", [label, float(p)]
                            )

                        last_osc_time = current_time_sec

                    except Exception as e:
                        print(f"OSC send error: {e}")

            # --- 畫臉框 ---
            box = face_boxes[frame_idx] if frame_idx < len(face_boxes) else None
            if box is not None:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # --- 畫文字（NOW & FUTURE） ---
            cv2.putText(
                frame,
                f"NOW: {emotion_now} {prob_now*100:.1f}%",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            cv2.putText(
                frame,
                f"FUT[4 beats ~ {four_beat_seconds:.2f}s]: "
                f"{emotion_future} {prob_future*100:.1f}%",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            # --- 底部：NOW 七個情緒 ---
            h, w_, _ = frame.shape
            line_height = 20
            x_left = 20
            for i, (label, p) in enumerate(zip(emotion_labels, probs_now)):
                text = f"{label}: {p*100:.1f}%"
                y = h - 20 - i * line_height
                cv2.putText(
                    frame,
                    text,
                    (x_left, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

            # --- 丟給前端畫面 ---
            frame = cv2.resize(frame, None, fx=0.6, fy=0.6)
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('video_frame', {'image': frame_base64})

            # --- 同時送 NOW 的 7 維比例到前端圖表 ---
            emotion_data = {
                label: float(p) * 100
                for label, p in zip(emotion_labels, probs_now)
            }
            socketio.emit('emotion_update', emotion_data)

            socketio.sleep(1.0 / fps)
            frame_idx += 1

    finally:
        cap.release()

    emit('processing_complete', {'msg': '影片分析完成 (Phase2)'})
    print("Video processing finished and released (Phase2).")


if __name__ == '__main__':
    # 啟動 BPM listener（從 Max 收 BPM）
    start_bpm_listener_once()

    # 第一次啟動時載入模型
    load_emotion_model()

    print("Server starting on http://127.0.0.1:5000")
    socketio.run(app, debug=True, port=5000)
