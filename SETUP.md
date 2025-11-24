# 環境設置指南 (Environment Setup Guide)

本專案需要 Python 3.9+ 和相關依賴套件。以下是兩種設置方式：

## 方法 1: 使用 Conda (推薦)

### 步驟 1: 創建 Conda 環境

```bash
# 創建新的 conda 環境（Python 3.9）
conda create -n emotion_app python=3.9 -y

# 激活環境
conda activate emotion_app
```

### 步驟 2: 安裝依賴

```bash
# 方法 A: 使用 conda 安裝大部分套件，然後用 pip 安裝特定版本
conda install -c conda-forge flask=2.3.3 flask-socketio=5.3.6 tensorflow numpy eventlet -y
pip install opencv-python-headless

# 方法 B: 全部使用 pip 安裝（在 conda 環境中）
pip install -r requirements.txt
```

### 步驟 3: 驗證安裝

```bash
python -c "import cv2, flask, flask_socketio, tensorflow, numpy, eventlet; print('所有套件安裝成功！')"
```

---

## 方法 2: 使用 Python venv (虛擬環境)

### 步驟 1: 創建虛擬環境

```bash
# 在專案目錄下創建虛擬環境
python3 -m venv venv

# 激活虛擬環境
# macOS/Linux:
source venv/bin/activate

# Windows:
# venv\Scripts\activate
```

### 步驟 2: 升級 pip

```bash
pip install --upgrade pip
```

### 步驟 3: 安裝依賴

```bash
pip install -r requirements.txt
```

### 步驟 4: 驗證安裝

```bash
python -c "import cv2, flask, flask_socketio, tensorflow, numpy, eventlet; print('所有套件安裝成功！')"
```

---

## 運行應用程式

### 使用 Conda 環境

```bash
# 激活環境
conda activate emotion_app

# 確保使用 conda 環境中的 Python
cd /Users/kaitsao/Desktop/emotion_app
python server.py
# 或使用完整路徑
/opt/anaconda3/envs/emotion_app/bin/python server.py
```

### 使用 venv

```bash
# 激活虛擬環境
source venv/bin/activate  # macOS/Linux
# 或 venv\Scripts\activate  # Windows

# 運行服務器
python server.py
```

---

## 常見問題排除

### 問題 1: ModuleNotFoundError: No module named 'cv2'

**解決方案：**
```bash
# 在 conda 環境中
conda activate emotion_app
pip install opencv-python-headless

# 或在 venv 中
source venv/bin/activate
pip install opencv-python-headless
```

### 問題 2: 使用錯誤的 Python 解釋器

**解決方案：**
```bash
# 檢查當前使用的 Python
which python
python --version

# 確保使用正確的環境
conda activate emotion_app  # 或 source venv/bin/activate
```

### 問題 3: TensorFlow 安裝失敗

**解決方案：**
```bash
# 對於 macOS (Apple Silicon)
pip install tensorflow-macos tensorflow-metal

# 或使用 conda
conda install -c conda-forge tensorflow -y
```

---

## 依賴套件清單

- **flask==2.3.3** - Web 框架
- **flask-socketio==5.3.6** - WebSocket 支持
- **eventlet==0.36.1** - 異步網絡庫
- **opencv-python-headless** - 計算機視覺庫（無 GUI 版本）
- **tensorflow>=2.10.0** - 機器學習框架
- **numpy>=1.21.0** - 數值計算庫

---

## 注意事項

1. **Python 版本**: 建議使用 Python 3.9 或 3.10
2. **opencv-python-headless**: 使用 headless 版本以避免 GUI 依賴問題
3. **TensorFlow**: 根據您的系統（CPU/GPU）選擇合適的版本
4. **Conda vs pip**: 建議在 conda 環境中使用 conda 安裝 tensorflow，然後用 pip 安裝其他套件

