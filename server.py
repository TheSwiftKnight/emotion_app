import os
import cv2
import numpy as np
import tensorflow as tf
import base64
import time
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from pythonosc.udp_client import SimpleUDPClient

import socket
import struct
import threading

import mediapipe as mp
import math

# --- 手勢追蹤全域變數 ---
hand_tracking_active = False
hand_control_enabled = True
hand_thread = None
# 用來傳給前端做畫面互動
current_interaction = {'scale': 1.0, 'pan_x': 0.0, 'pan_y': 0.0}

# ======================
# 全域 BPM（給所有地方用）
# ======================
current_bpm = 120.0

# ======================
# Phase1 結果快取（給 PLAY / 未來 camera 共用）
# ======================
cached_probs = None       # numpy array, shape (N, 7)
cached_boxes = None       # list of face boxes per frame
cached_fps = None         # float
cached_video_path = None  # str

# 停止旗標（Phase1 / Phase2 共用）
stop_requested = False


def bpm_listener():
    """從 UDP 9001 接收 Max 傳來的 BPM（文字或 binary float）"""
    global current_bpm
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 9001))   # 要跟 Max 一樣的 port
    print("Listening raw BPM on UDP 9001 ...")
    while True:
        data, addr = sock.recvfrom(1024)
        try:
            # 1) 試文字 float
            try:
                text = data.decode("utf-8").strip()
                val = float(text)
                current_bpm = val
                print(f"[PY] BPM updated (text): {current_bpm}")
                continue
            except Exception:
                pass  # 不是文字就往下

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


# 初始化 Flask 和 SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# 路徑 / 模型 / OSC 設定
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'share/emotion_detection_model_100epochs.h5'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

OSC_IP = "127.0.0.1"
OSC_PORT = 8000

model = None
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emotion_colors = {
    'Angry': (0, 0, 255),
    'Disgust': (0, 255, 0),
    'Fear': (128, 0, 128),
    'Happy': (0, 255, 255),
    'Neutral': (128, 128, 128),
    'Sad': (255, 0, 0),
    'Surprise': (180, 105, 255)
}

osc_client = None
try:
    osc_client = SimpleUDPClient(OSC_IP, OSC_PORT)
    print(f"OSC client initialized: {OSC_IP}:{OSC_PORT}")
except Exception as e:
    print(f"Warning: Could not initialize OSC client: {e}")
    print("OSC messages will not be sent to Max/MSP")


# --- 模型建構 ---
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


# ======================
# 共用 Phase2 播放函式（給 PHASE2 按鈕用）
# ======================
def run_phase2(video_path, all_probs, face_boxes, fps, mode_label='phase2'):
    """
    Phase2 播放邏輯：
    - video_path: 影片路徑
    - all_probs: 每幀 7 維情緒機率 (N,7)
    - face_boxes: 每幀臉框 (或 None)
    - fps: 影片 FPS
    - mode_label: 'phase2' / 未來 camera 也可以用別的字
    """
    global osc_client, current_bpm, stop_requested

    # --- 新增：啟動手勢追蹤 ---
    global hand_tracking_active, hand_thread
    if not hand_tracking_active:
        hand_tracking_active = True
        hand_thread = threading.Thread(target=hand_tracking_loop, daemon=True)
        hand_thread.start()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        emit('error', {'msg': 'Cannot reopen video file for Phase2.'})
        return

    N = all_probs.shape[0]
    frame_idx = 0
    last_osc_time = -1e9

    print(f"[PY] Phase2 start, mode={mode_label}, FPS={fps}, frames={N}")

    try:
        while True:
            if stop_requested:
                print("[PY] Phase2 stop requested by client.")
                break

            ret, frame = cap.read()
            if not ret or frame_idx >= N:
                print("[PY] Phase2 video ended.")
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
                    # 未來四拍
                    osc_client.send_message(
                        "/emotion_label_future",
                        [emotion_future, prob_future, float(four_beat_seconds)]
                    )
                    for label, p in zip(emotion_labels, probs_future):
                        osc_client.send_message("/emotion_prob_future", [label, float(p)])

                    # 當前
                    osc_client.send_message("/emotion_label", [emotion_now, prob_now])
                    for label, p in zip(emotion_labels, probs_now):
                        osc_client.send_message("/emotion_prob", [label, float(p)])

                last_osc_time = current_time_sec

            # ===== 畫臉框 =====
            # ===== 畫臉框（依情緒改顏色）=====
            box = face_boxes[frame_idx] if frame_idx < len(face_boxes) else None
            if box is not None:
                x, y, w, h = box

                # 使用 NOW 的情緒顏色
                color = emotion_colors.get(emotion_now, (0, 255, 0))

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # 在臉框旁顯示文字
                cv2.putText(
                    frame,
                    f"{emotion_now} {prob_now*100:.1f}%",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            # ===== 畫文字 =====
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
                f"FUT[4 beats ~ {four_beat_seconds:.2f}s]: {emotion_future} {prob_future*100:.1f}%",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            # ===== 底部每個情緒的條狀文字 =====
            h, w, _ = frame.shape
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

            # ===== 丟給前端 =====
            frame = cv2.resize(frame, None, fx=0.6, fy=0.6)
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('video_frame', {'image': frame_base64})

            emotion_data = {
                label: float(p) * 100 for label, p in zip(emotion_labels, probs_now)
            }
            socketio.emit('emotion_update', emotion_data)

            socketio.sleep(1.0 / fps)
            frame_idx += 1

    finally:
        cap.release()
        # --- 關閉手勢追蹤 ---
        hand_tracking_active = False 
        if hand_thread and hand_thread.is_alive():
            hand_thread.join(timeout=1.0)

    if stop_requested:
        msg = 'Phase2 播放已停止'
    else:
        msg = 'Phase2 播放完成'
    emit('processing_complete', {'msg': msg, 'mode': mode_label})
    print(f"[PY] Phase2 finished. mode={mode_label}, stop={stop_requested}")


# ======================
# Flask Routes
# ======================
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


# ======================
# SocketIO Events
# ======================

@socketio.on('start_phase1')
def handle_start_phase1(data):
    """
    PHASE1：只做預先分析 + 快取，不播放。
    """
    global model, cached_probs, cached_boxes, cached_fps, cached_video_path, stop_requested

    mode = data.get('mode', 'upload')
    print('[PY] start_phase1, mode =', mode)

    if mode != 'upload':
        emit('error', {'msg': f'目前只支援上傳影片模式的 PHASE1，mode={mode}'})
        return

    if model is None:
        load_emotion_model()
        if model is None:
            emit('error', {'msg': 'Model not found!'})
            return

    video_path = os.path.join(UPLOAD_FOLDER, 'uploaded_video.mp4')
    if not os.path.exists(video_path):
        emit('error', {'msg': 'Video file not found. Please upload first.'})
        return

    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    print("Starting Phase1: precompute probs (no playback)...")

    if not cap.isOpened():
        emit('error', {'msg': 'Cannot open video file.'})
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1  # 避免除以 0
    print("FPS:", fps, "Total frames:", total_frames)

    # 一開始先丟 0%
    socketio.emit('phase1_progress', {'progress': 0.0})


    all_probs = []        # 每幀情緒機率 (N, 7)
    face_boxes = []       # 每幀臉框 (x, y, w, h) 或 None

    last_prob = np.ones(len(emotion_labels)) / len(emotion_labels)
    last_box = None

    SAMPLE_STEP = 5
    frame_idx = 0

    stop_requested = False  # 清掉舊的 stop

    def predict_emotion_with_probabilities(face_bgr, model_local):
        face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(face_gray, (48, 48)) / 255.0
        face_img = np.expand_dims(resized_face, axis=(0, -1))
        predictions = model_local.predict(face_img, verbose=0)
        return predictions[0]

    while True:
        if stop_requested:
            print("[PY] Phase1 stop requested by client.")
            break

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

        # 更新進度條（每一幀都更新，如果覺得太頻繁可以改成每 N 幀一次）
        progress = (frame_idx + 1) / total_frames * 100.0
        socketio.emit('phase1_progress', {'progress': float(progress)})

        frame_idx += 1

        socketio.sleep(0)


    cap.release()

    if len(all_probs) == 0:
        emit('error', {'msg': 'Phase1 沒有讀到任何 frame'})
        return

    all_probs = np.stack(all_probs)
    N = all_probs.shape[0]
    print("Phase1 collected probs for frames:", N)

    cached_probs = all_probs
    cached_boxes = face_boxes
    cached_fps = fps
    cached_video_path = video_path
    print("[PY] Phase1 results cached.")

    # 保險：Phase1 結束時設定 100%
    socketio.emit('phase1_progress', {'progress': 100.0})

    if stop_requested:
        msg = "Phase1 已停止，暫不播放 Phase2。"
    else:
        msg = "Phase1 分析完成，現在可以按 PLAY（PHASE2）播放。"

    emit('processing_complete', {'msg': msg, 'mode': 'phase1'})



@socketio.on('start_phase2')
def handle_start_phase2(data):
    """
    PHASE2：只用快取結果播放，不再跑模型。
    """
    global cached_probs, cached_boxes, cached_fps, cached_video_path, stop_requested

    print("[PY] start_phase2 called")

    if cached_probs is None or cached_video_path is None or cached_fps is None:
        emit('error', {'msg': '目前沒有快取結果，請先按 PHASE1 進行分析。'})
        return

    stop_requested = False
    run_phase2(
        cached_video_path,
        cached_probs,
        cached_boxes,
        cached_fps,
        mode_label='phase2'
    )


@socketio.on('stop_processing')
def handle_stop_processing():
    """
    前端 STOP 按鈕：把 stop_requested 設成 True，
    Phase1 / Phase2 的迴圈都會在下一次迭代時停下。
    """
    global stop_requested
    stop_requested = True
    print("[PY] stop_processing received, stop_requested set to True")

@socketio.on('start_camera')
def handle_start_camera():
    global camera_running, camera_thread

    print("[PY] start_camera called")

    if camera_running:
        emit('error', {'msg': 'Camera 已經在運作'})
        return

    if model is None:
        load_emotion_model()

    camera_running = True
    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    camera_thread.start()

    # ⭐ 這裡「不要」 emit processing_complete
    # Camera 結束時在 camera_loop 裡 emit 就好


@socketio.on('stop_camera')
def handle_stop_camera():
    global camera_running
    print("[PY] stop_camera received")
    camera_running = False


@socketio.on('toggle_hand_control')
def handle_toggle_hand_control(data):
    global hand_control_enabled
    # data['enabled'] 應該是 true/false
    hand_control_enabled = data.get('enabled', True)
    state_str = "ON" if hand_control_enabled else "OFF"
    print(f"[PY] Hand Control toggled: {state_str}")

# ==============================
# Camera Mode：即時偵測
# ==============================
camera_running = False
camera_thread = None

def camera_loop():
    global camera_running, model, osc_client, current_bpm, hand_control_enabled

    # 1. 先停止手勢執行緒，並給予緩衝時間釋放攝影機
    global hand_tracking_active
    hand_tracking_active = False
    time.sleep(0.5)  # 等待資源釋放
    
    # 2. 初始化
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    mp_hands = mp.solutions.hands
    
    if not cap.isOpened():
        print("[PY] 無法開啟攝影機 (Camera busy or not found)")
        socketio.emit('error', {'msg': '無法開啟攝影機，請稍後再試'})
        camera_running = False
        return

    print("[PY] Camera mode started.")
    fps = 30
    last_osc_time = time.time()

    # 使用 with 確保手勢模型資源會被釋放
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        while camera_running:
            try:
                ret, frame = cap.read()
                if not ret: 
                    print("[PY] Camera read failed")
                    break
                
                # 翻轉畫面
                frame = cv2.flip(frame, 1)

                # =================================================
                # A. 手勢控制邏輯 (獨立區塊)
                # =================================================
                if hand_control_enabled:
                    try:
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = hands.process(img_rgb)

                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                # 計算 Pan & Zoom
                                wrist_x = hand_landmarks.landmark[0].x 
                                pan_val = max(0.0, min(1.0, wrist_x))

                                thumb_tip = hand_landmarks.landmark[4]
                                index_tip = hand_landmarks.landmark[8]
                                dist = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
                                
                                # 映射距離到 Zoom 係數
                                zoom_factor = 0.7 + (dist * 2.0)

                                # 發送控制訊號
                                if osc_client:
                                    osc_client.send_message("/pan", float(pan_val))
                                    osc_client.send_message("/zoom", float(zoom_factor))

                                rotation_y = (pan_val - 0.5) * 60 
                                socketio.emit('interaction_update', {
                                    'scale': float(zoom_factor),
                                    'rotate': float(rotation_y)
                                })
                                break # 只抓一隻手
                    except Exception as e:
                        print(f"[Hand Error] {e}")

                # =================================================
                # B. 情緒辨識邏輯 (獨立區塊，不受手勢影響)
                # =================================================
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                # 預設值 (如果沒抓到臉，顯示 Neutral 或平均值)
                emotion_now = "Neutral"
                prob_now = 0.0
                probs_now = np.ones(7) / 7 

                if len(faces) > 0:
                    try:
                        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                        x, y, w, h = faces[0]
                        
                        # 畫框
                        color = emotion_colors.get(emotion_now, (0, 255, 0))
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                        # [修正] 預測邏輯維度處理
                        face_roi = gray[y:y+h, x:x+w]
                        resized = cv2.resize(face_roi, (48, 48)) / 255.0
                        
                        # 修正這裡：分兩步擴充維度，確保形狀為 (1, 48, 48, 1)
                        face_input = np.expand_dims(resized, axis=-1) # (48, 48, 1)
                        face_input = np.expand_dims(face_input, axis=0) # (1, 48, 48, 1)
                        
                        preds = model.predict(face_input, verbose=0)[0]
                        probs_now = preds
                        idx_now = int(np.argmax(preds))
                        emotion_now = emotion_labels[idx_now]
                        prob_now = float(preds[idx_now])

                        # 更新框框文字
                        color = emotion_colors.get(emotion_now, (0, 255, 0))
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, f"{emotion_now} {prob_now*100:.0f}%", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    except Exception as e:
                        print(f"[Emotion Predict Error] {e}")
                        # 發生錯誤時保持預設值，不崩潰

                # =================================================
                # C. 發送數據 (關鍵：必須在所有 if 之外)
                # =================================================
                
                # 1. 發送影像 (給前端顯示)
                frame_resized = cv2.resize(frame, None, fx=0.6, fy=0.6)
                _, buffer = cv2.imencode('.jpg', frame_resized)
                frame_b64 = base64.b64encode(buffer).decode('utf-8')
                socketio.emit('video_frame', {'image': frame_b64})

                # 2. 為了 PIP 邏輯正常，也發送 camera_frame (前端會隱藏)
                pip_frame = cv2.resize(frame, (320, 240))
                _, pip_buffer = cv2.imencode('.jpg', pip_frame)
                pip_b64 = base64.b64encode(pip_buffer).decode('utf-8')
                socketio.emit('camera_frame', {'image': pip_b64})

                # 3. 發送情緒數據 (這行一定要執行，前端比例條才會動)
                emotion_dict = {
                    label: float(p)*100 for label, p in zip(emotion_labels, probs_now)
                }
                socketio.emit('emotion_update', emotion_dict)

                # 4. 發送 OSC
                beat_sec = 60.0 / max(current_bpm, 1e-3)
                now_t = time.time()
                if now_t - last_osc_time >= beat_sec:
                    if osc_client:
                        osc_client.send_message("/emotion_label_future", [emotion_now, prob_now, beat_sec * 4])
                        for label, p in zip(emotion_labels, probs_now):
                            osc_client.send_message("/emotion_prob_future", [label, float(p)])
                        osc_client.send_message("/emotion_label", [emotion_now, prob_now])
                        for label, p in zip(emotion_labels, probs_now):
                            osc_client.send_message("/emotion_prob", [label, float(p)])
                    last_osc_time = now_t

                socketio.sleep(1.0 / fps)

            except Exception as e:
                print(f"[Camera Loop Fatal Error] {e}")
                break

    cap.release()
    print("[PY] Camera mode stopped.")
    socketio.emit('processing_complete', {'msg': 'Camera 模式停止', 'mode': 'camera'})
    
def hand_tracking_loop():
    global hand_tracking_active, hand_control_enabled, osc_client

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        cap = cv2.VideoCapture(0)
        print("[PY] Hand Tracking & PIP Camera Started")
        
        while hand_tracking_active:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # [新增] 1. 水平翻轉畫面 (鏡像效果)
            frame = cv2.flip(frame, 1)

            # --- A. 處理 PIP 畫面回傳 ---
            pip_frame = cv2.resize(frame, (320, 240)) 
            _, buffer = cv2.imencode('.jpg', pip_frame)
            pip_base64 = base64.b64encode(buffer).decode('utf-8')
            socketio.emit('camera_frame', {'image': pip_base64})

            # --- B. 手勢辨識與控制 ---
            if hand_control_enabled:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                zoom_factor = 1.0 
                pan_val = 0.5     

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # [修改] 2. 計算 Pan (左右位置)
                        # 因為畫面已經翻轉 (鏡像)，手在畫面右邊 x 就是 1，聲音也要往右，所以直接用 x
                        wrist_x = hand_landmarks.landmark[0].x 
                        pan_val = max(0.0, min(1.0, wrist_x))

                        # [保留] 3. 計算 Zoom (使用優化過的距離公式)
                        thumb_tip = hand_landmarks.landmark[4]
                        index_tip = hand_landmarks.landmark[8]
                        dist = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
                        
                        # 0.7(基礎遠距) + 2.5(倍率)
                        zoom_factor = 0.7 + (dist * 2.5)
                        break 

                    # 發送 OSC 給 Max
                    if osc_client:
                        osc_client.send_message("/pan", float(pan_val))
                        osc_client.send_message("/zoom", float(zoom_factor))

                    # 發送 Socket 給前端改變 CSS
                    rotation_y = (pan_val - 0.5) * 60 
                    socketio.emit('interaction_update', {
                        'scale': float(zoom_factor),
                        'rotate': float(rotation_y)
                    })
            
            time.sleep(0.033)

        cap.release()
        print("[PY] Hand Tracking Stopped")
        
if __name__ == '__main__':
    load_emotion_model()
    start_bpm_listener_once()
    print("Server starting on http://127.0.0.1:5000")
    socketio.run(app, debug=False, port=5000, use_reloader=False)

