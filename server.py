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

# 全域 BPM（給所有地方用）
current_bpm = 120.0

def bpm_listener():
    global current_bpm
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 9001))   # 要跟 Max 一樣的 port
    print("Listening raw BPM on UDP 9001 ...")
    while True:
        data, addr = sock.recvfrom(1024)
        try:
            # 試試看純文字格式
            try:
                text = data.decode("utf-8").strip()
                val = float(text)
                current_bpm = val
                print(f"[PY] BPM updated (text): {current_bpm}")
                continue
            except Exception:
                pass  # 不是文字就往下

            # Max binary float 格式：b'float....<4 bytes>'
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


# 初始化 Flask 和 SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# 設定路徑
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'share/emotion_detection_model_100epochs.h5'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# OSC 設定（用於 Max/MSP 通訊）
OSC_IP = "127.0.0.1"
OSC_PORT = 8000
SEND_INTERVAL = 0.5  # 每 0.5 秒發送一次 OSC 訊息

# 全域變數儲存模型
model = None
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
processing_sessions = {}  # 紀錄每個使用者目前的處理狀態
emotion_colors = {
    'Angry': (0, 0, 255),
    'Disgust': (0, 255, 0),
    'Fear': (128, 0, 128),
    'Happy': (0, 255, 255),
    'Neutral': (128, 128, 128),
    'Sad': (255, 0, 0),
    'Surprise': (180, 105, 255)
}
osc = SimpleUDPClient("127.0.0.1", 8000)

# 初始化 OSC 客戶端
osc_client = None
try:
    osc_client = SimpleUDPClient(OSC_IP, OSC_PORT)
    print(f"OSC client initialized: {OSC_IP}:{OSC_PORT}")
except Exception as e:
    print(f"Warning: Could not initialize OSC client: {e}")
    print("OSC messages will not be sent to Max/MSP")

# --- 這是你原本 ipynb 中的模型建構邏輯 (為了確保模型載入正確) ---
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
            # 嘗試直接載入
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully via load_model.")
        except Exception as e:
            print(f"Direct load failed, trying to build architecture first: {e}")
            # 如果直接載入失敗，先建構架構再載入權重 (根據你的 ipynb 邏輯)
            model = create_model_architecture()
            model.load_weights(MODEL_PATH)
            print("Model weights loaded successfully.")
    else:
        print(f"警告：找不到模型檔案 {MODEL_PATH}。請確保檔案存在。")

# 預測函數
def predict_emotion(face_img):
    resized_face = cv2.resize(face_img, (48, 48)) / 255.0
    face_input = np.expand_dims(resized_face, axis=(0, -1))
    predictions = model.predict(face_input, verbose=0)
    return predictions[0]

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = 'uploaded_video.mp4'
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    return jsonify({'message': 'File uploaded successfully', 'filepath': filepath})

# --- SocketIO Events ---

def _stop_session(sid, message=None):
    """Helper to停止指定 session 的處理流程。"""
    session = processing_sessions.get(sid)
    if session:
        session['stop'] = True
    if message:
        socketio.emit('error', {'msg': message}, to=sid)


def process_video_stream(sid, mode):
    """背景任務：依照 mode 進行影片或即時攝影的情緒分析。"""
    global model, osc_client
    session = processing_sessions.get(sid)
    if session is None:
        return

    if model is None:
        load_emotion_model()
        if model is None:
            _stop_session(sid, 'Model not found!')
            processing_sessions.pop(sid, None)
            return

    if mode not in ('upload', 'camera'):
        _stop_session(sid, '未知的來源模式')
        processing_sessions.pop(sid, None)
        return

    cap = None
    if mode == 'upload':
        video_path = os.path.join(UPLOAD_FOLDER, 'uploaded_video.mp4')
        if not os.path.exists(video_path):
            _stop_session(sid, 'Video file not found. Please upload first.')
            processing_sessions.pop(sid, None)
            return
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)

    if not cap or not cap.isOpened():
        _stop_session(sid, '無法開啟影片或攝影機來源')
        processing_sessions.pop(sid, None)
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    source_desc = '即時攝影' if mode == 'camera' else '影片'
    print(f"Starting {source_desc} processing for session {sid} ...")

    last_send_time = 0
    last_probabilities = np.zeros(len(emotion_labels))
    last_emotion = 'Neutral'
    max_idx = 0

    while True:
        if session.get('stop'):
            print(f"Session {sid} requested stop.")
            break

        ret, frame = cap.read()
        if not ret:
            if mode == 'camera':
                socketio.sleep(0.05)
                continue
            print(f"{source_desc} processing completed for session {sid}.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        current_emotions = {}
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            try:
                probabilities = predict_emotion(face)
                last_probabilities = probabilities
                max_idx = int(np.argmax(probabilities))
                last_emotion = emotion_labels[max_idx]

                emotion_data = {}
                for i, label in enumerate(emotion_labels):
                    emotion_data[label] = float(probabilities[i]) * 100
                current_emotions = emotion_data

                label_text = f"{emotion_labels[max_idx]}: {probabilities[max_idx]*100:.1f}%"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                break  # 一次只處理第一張臉，維持與舊邏輯一致
            except Exception as e:
                print(f"Prediction error: {e}")

        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('video_frame', {'image': frame_base64}, to=sid)

        if current_emotions:
            socketio.emit('emotion_update', current_emotions, to=sid)

        if osc_client is not None:
            now = time.time()
            if now - last_send_time >= SEND_INTERVAL:
                try:
                    for label, prob in zip(emotion_labels, last_probabilities):
                        osc_client.send_message("/emotion_prob", [label, float(prob)])

                    top_prob = float(last_probabilities[max_idx])
                    osc_client.send_message("/emotion_label", [last_emotion, top_prob])

                    for label, prob in zip(emotion_labels, last_probabilities):
                        osc_client.send_message("/emotion", [label, float(prob)])

                    last_send_time = now
                except Exception as e:
                    print(f"OSC send error: {e}")

        socketio.sleep(0.03)

    cap.release()
    processing_sessions.pop(sid, None)
    socketio.emit(
        'processing_complete',
        {'msg': f'{source_desc}分析結束', 'mode': mode},
        to=sid
    )
    print(f"{source_desc} processing finished and released for session {sid}.")


@socketio.on('start_processing')
def handle_process_video():
    global model, osc_client
    if model is None:
        load_emotion_model()
        if model is None:
            emit('error', {'msg': 'Model not found!'})
            return

    video_path = os.path.join(UPLOAD_FOLDER, 'uploaded_video.mp4')
    if not os.path.exists(video_path):
        emit('error', {'msg': 'Video file not found. Please upload first.'})
        return

    # ---------- Phase1：先跑完整部影片，收集 all_probs + face_boxes ----------
    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                         'haarcascade_frontalface_default.xml')

    print("Starting video processing (Phase1: precompute probs)...")

    if not cap.isOpened():
        raise RuntimeError("Cannot open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("FPS:", fps, "Total frames:", total_frames)

    all_probs = []        # 每幀的情緒機率 (N, 7)
    face_boxes = []       # 每幀的臉框 (x, y, w, h) 或 None

    last_prob = np.ones(len(emotion_labels)) / len(emotion_labels)
    last_box = None

    SAMPLE_STEP = 5
    frame_idx = 0

    def predict_emotion_with_probabilities(face_bgr, model):
        face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(face_gray, (48, 48)) / 255.0
        face_img = np.expand_dims(resized_face, axis=(0, -1))
        predictions = model.predict(face_img, verbose=0)
        return predictions[0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % SAMPLE_STEP == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

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

    all_probs = np.stack(all_probs)  # shape = (N, 7)
    N = all_probs.shape[0]
    print("Collected probs for frames:", N)

    # ---------- Phase2：重新播放影片，用 all_probs 做 NOW + 未來四拍，畫好再丟給前端 ----------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    N = all_probs.shape[0]
    frame_idx = 0
    last_osc_time = -1e9

    print(f"[PY] Video FPS = {fps}, frames in probs = {N}")
    print("Starting Phase2: playback with 4-beat forecast...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= N:
                print("[PY] Video ended.")
                break

            current_time_sec = frame_idx / fps

            # ===== NOW：這幀的 7 維情緒 =====
            probs_now = all_probs[frame_idx]
            idx_now = int(np.argmax(probs_now))
            emotion_now = emotion_labels[idx_now]
            prob_now = float(probs_now[idx_now])

            # ===== FUTURE：未來四拍平均 =====
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

            # ===== 每一拍送一次 OSC =====
            if current_time_sec - last_osc_time >= beat_seconds:
                if osc_client is not None:
                    # 1. 未來四拍的主情緒 + 機率 + 這四拍大概幾秒
                    osc_client.send_message("/emotion_label_future",[emotion_future, prob_future, float(four_beat_seconds)])

                    

                    # 3. FUTURE：未來四拍平均的七個情緒機率（你要的部分）
                    for label, p in zip(emotion_labels, probs_future):
                        osc_client.send_message("/emotion_prob_future", [label, float(p)])
                        
                    # 4. NOW 主情緒
                    osc_client.send_message("/emotion_label", [emotion_now, prob_now])
                        
                    # 2. NOW：當前這一幀的七個情緒機率
                    for label, p in zip(emotion_labels, probs_now):
                        osc_client.send_message("/emotion_prob", [label, float(p)])

                    

                last_osc_time = current_time_sec

            # ===== 畫臉框 =====
            box = face_boxes[frame_idx] if frame_idx < len(face_boxes) else None
            if box is not None:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

            # ===== 畫文字（左上、左下） =====
            cv2.putText(frame, f"NOW: {emotion_now} {prob_now*100:.1f}%",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame,
                        f"FUT[4 beats ~ {four_beat_seconds:.2f}s]: {emotion_future} {prob_future*100:.1f}%",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            h, w, _ = frame.shape
            line_height = 20
            x_left = 20
            for i, (label, p) in enumerate(zip(emotion_labels, probs_now)):
                text = f"{label}: {p*100:.1f}%"
                y = h - 20 - i * line_height
                cv2.putText(frame, text, (x_left, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            # ===== 丟給前端框框顯示 =====
            frame = cv2.resize(frame, None, fx=0.6, fy=0.6)
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('video_frame', {'image': frame_base64})

            # 同時送 NOW 的 7 維比例到前端圖表
            emotion_data = {label: float(p)*100 for label, p in zip(emotion_labels, probs_now)}
            socketio.emit('emotion_update', emotion_data)

            socketio.sleep(1.0 / fps)
            frame_idx += 1

    finally:
        cap.release()

    emit('processing_complete', {'msg': '影片分析完成'})
    print("Video processing finished and released.")


@socketio.on('stop_processing')
def handle_stop_processing():
    sid = request.sid
    session = processing_sessions.get(sid)
    if session:
        session['stop'] = True
        emit('status', {'msg': '停止指令已送出，請稍候...'})


@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    session = processing_sessions.get(sid)
    if session:
        session['stop'] = True

if __name__ == '__main__':
    # 第一次啟動時嘗試載入模型
    load_emotion_model()
    print("Server starting on http://127.0.0.1:5000")
    socketio.run(app, debug=True, port=5000)