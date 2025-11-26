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
def handle_process_video(data=None):
    sid = request.sid
    mode = (data or {}).get('mode', 'upload')

    if processing_sessions.get(sid):
        emit('error', {'msg': '已有分析進行中，請先停止或等待完成。'})
        return

    processing_sessions[sid] = {'mode': mode, 'stop': False}
    socketio.start_background_task(process_video_stream, sid, mode)


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