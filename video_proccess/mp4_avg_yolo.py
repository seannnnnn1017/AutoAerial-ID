import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO  # 確保已安裝 ultralytics 模組
import matplotlib.pyplot as plt

def extract_frames(video_path, interval_seconds=1):
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Extracting frames from video: {video_path}")
    for i in tqdm(range(total_frames), desc="Reading video frames"):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1
    
    cap.release()
    print(f"Frames extracted: {len(frames)}")
    return frames

def load_yolo_model():
    # 加載 YOLO 模型（使用預訓練權重）
    model = YOLO('yolov10x.pt')  # 使用較小的 YOLO 模型版本
    return model

def detect_objects(model, frame):
    # 使用 YOLO 檢測物體
    results = model(frame, conf=0.1)  # 設定置信度閾值
    bboxes = []
    for result in results:  # 遍歷每個檢測結果
        for box in result.boxes:  # 獲取檢測框
            if box.cls == 2 or box.cls == 7:  # 只選擇類別為 "person" 的目標（類別ID為0）
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # 獲取框的座標
                bboxes.append((x1, y1, x2, y2))
    return bboxes


def generate_weight_mask(frame, bboxes):
    # 根據檢測框生成權重遮罩
    mask = np.ones_like(frame, dtype=np.float64)  # 預設所有區域權重為1
    for x1, y1, x2, y2 in bboxes:
        mask[y1:y2, x1:x2] = 0.01  # 將人物區域的權重設置為較低值
    return mask

def estimate_background_with_yolo(frames, yolo_model):
    print("Estimating background with YOLO...")
    acc_frame = np.zeros_like(frames[0], dtype=np.float64)
    acc_weight = np.zeros_like(frames[0], dtype=np.float64)

    for frame in tqdm(frames, desc="Analyzing frames"):
        bboxes = detect_objects(yolo_model, frame)  # 檢測人物
        clean_mask = generate_weight_mask(frame, bboxes)  # 生成權重遮罩
        
        acc_frame += frame.astype(np.float64) * clean_mask
        acc_weight += clean_mask

    acc_weight[acc_weight == 0] = 1
    background = (acc_frame / acc_weight).astype(np.uint8)
    return background

# 測試影片提取
video_path = "images\highway_test.mp4"
frames = extract_frames(video_path, interval_seconds=1)

# 加載 YOLO 模型
yolo_model = load_yolo_model()

# 估算背景
background = estimate_background_with_yolo(frames, yolo_model)

# 儲存背景影像
cv2.imwrite('./estimated_background_yolo.jpg', background)
print("Background estimation completed and saved.")

# 顯示背景影像
plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
plt.title("Estimated Background")
plt.axis('off')
plt.show()
