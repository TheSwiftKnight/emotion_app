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

    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("Starting video processing...")
    
    # OSC 時間控制
    last_send_time = 0
    last_probabilities = np.zeros(len(emotion_labels))
    last_emotion = 'Neutral'
    max_idx = 0  # 初始化 max_idx
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # 影片結束，發送最後一幀（如果有）然後停止
            print("Video processing completed.")
            break

        # 為了效能，每 3 幀處理一次，或根據需要調整
        # 这里为了演示流畅性，每一帧都处理，但你可以加计数器跳帧

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        current_emotions = {}
        
        # 為了視覺化，我們把處理過的畫面傳回前端
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w] # 模型需要灰階輸入
            
            try:
                probabilities = predict_emotion(face)
                
                # 更新最後的情緒數據（用於 OSC）
                last_probabilities = probabilities
                max_idx = np.argmax(probabilities)
                last_emotion = emotion_labels[max_idx]
                
                # 準備數據傳給前端
                emotion_data = {}
                for i, label in enumerate(emotion_labels):
                    emotion_data[label] = float(probabilities[i]) * 100
                
                current_emotions = emotion_data # 存下來發送
                
                # 在畫面上畫框 (Optional: 如果你想在後端畫好再傳回去)
                label_text = f"{emotion_labels[max_idx]}: {probabilities[max_idx]*100:.1f}%"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
            except Exception as e:
                print(f"Prediction error: {e}")

        # 將 Frame 轉為 Base64 圖片傳回前端顯示
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # 發送數據到前端
        socketio.emit('video_frame', {'image': frame_base64})
        
        if current_emotions:
            socketio.emit('emotion_update', current_emotions)
        
        # 發送 OSC 訊息到 Max/MSP
        if osc_client is not None:
            now = time.time()
            if now - last_send_time >= SEND_INTERVAL:
                try:
                    # 發送每個情緒的機率到 /emotion_prob
                    for label, prob in zip(emotion_labels, last_probabilities):
                        osc_client.send_message("/emotion_prob", [label, float(prob)])
                    
                    # 發送最大機率的情緒到 /emotion_label
                    top_prob = float(last_probabilities[max_idx])
                    osc_client.send_message("/emotion_label", [last_emotion, top_prob])
                    
                    # 發送每個情緒到 /emotion（與 notebook 一致）
                    for label, prob in zip(emotion_labels, last_probabilities):
                        osc_client.send_message("/emotion", [label, float(prob)])
                    
                    last_send_time = now
                except Exception as e:
                    print(f"OSC send error: {e}")

        # 控制速度，模擬真實播放 (如果不加這個，處理速度快的話影片會像快轉)
        socketio.sleep(0.03) 

    cap.release()
    # 發送處理完成信號，讓前端知道影片已結束
    emit('processing_complete', {'msg': '影片分析完成'})
    print("Video processing finished and released.")

if __name__ == '__main__':
    # 第一次啟動時嘗試載入模型
    load_emotion_model()
    print("Server starting on http://127.0.0.1:5000")
    socketio.run(app, debug=True, port=5000)