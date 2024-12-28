import gradio as gr
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

def extract_frames(video_path, interval_seconds=1):
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in tqdm(range(total_frames), desc="Reading video frames"):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames

def load_yolo_model():
    model = YOLO('yolov10x.pt')
    return model

def detect_objects(model, frame):
    results = model(frame, conf=0.1)
    bboxes = []
    for result in results:
        for box in result.boxes:
            if box.cls == 2 or box.cls == 7:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                bboxes.append((x1, y1, x2, y2))
    return bboxes

def generate_weight_mask(frame, bboxes):
    mask = np.ones_like(frame, dtype=np.float64)
    for x1, y1, x2, y2 in bboxes:
        mask[y1:y2, x1:x2] = 0.01
    return mask

def estimate_background_with_yolo(frames, yolo_model):
    acc_frame = np.zeros_like(frames[0], dtype=np.float64)
    acc_weight = np.zeros_like(frames[0], dtype=np.float64)

    for frame in tqdm(frames, desc="Analyzing frames"):
        bboxes = detect_objects(yolo_model, frame)
        clean_mask = generate_weight_mask(frame, bboxes)

        acc_frame += frame.astype(np.float64) * clean_mask
        acc_weight += clean_mask

    acc_weight[acc_weight == 0] = 1
    background = (acc_frame / acc_weight).astype(np.uint8)
    return background

def process_video(video_file):
    frames = extract_frames(video_file, interval_seconds=1)
    yolo_model = load_yolo_model()
    background = estimate_background_with_yolo(frames, yolo_model)
    return cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

def demo(video_file):
    # 修复：直接使用视频文件路径
    background = process_video(video_file)
    return background

demo_interface = gr.Interface(
    fn=demo,
    inputs=gr.Video(label="Upload Video"),
    outputs=gr.Image(label="Estimated Background"),
    title="YOLO Background Estimation",
    description="Upload a video to estimate the background using YOLO."
)

demo_interface.launch()
