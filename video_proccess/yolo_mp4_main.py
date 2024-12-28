from ultralytics import YOLO
import cv2

view = True
model = YOLO("yolov10x.pt")  # 載入 YOLO 模型
input_video_path = "images/752750716.004314.mp4"  # 影片路徑
output_image_path = "output/result_no_person.jpg"  # 輸出影像路徑
cap = cv2.VideoCapture(input_video_path)

main_frame_replace = []
replace_done = False  # 追蹤是否完成替換

# 找一張有人的影格，並儲存人的位置
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用 YOLO 偵測物件
    results = model(frame)
    person_count = 0
    height, width, _ = frame.shape  # 取得影像尺寸

    for result in results:
        for box in result.boxes:
            if box.cls == 2:  # 偵測到 "person"
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 增加邊界框的範圍
                margin = 20  # 可調整，單位為像素
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(width, x2 + margin)
                y2 = min(height, y2 + margin)
                
                main_frame_replace.append([x1, y1, x2, y2])
                main_frame = frame.copy()  # 儲存包含人的初始影像

    if person_count > 0:  # 偵測到至少一個人後退出
        break

print(main_frame_replace)  # 確認人物位置座標

# 嘗試用其他影格替換人的位置
while cap.isOpened() and not replace_done:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    overlap_count = 0

    for idx, i in enumerate(main_frame_replace):
        x2_min, y2_min, x2_max, y2_max = map(int, i)
        overlap = False

        # 檢查每個人的框是否仍然有人
        for result in results:
            for box in result.boxes:
                if box.cls == 2:
                    x1_min, y1_min, x1_max, y1_max = map(int, box.xyxy[0])
                    if x1_min < x2_max and x1_max > x2_min and y1_min < y2_max and y1_max > y2_min:
                        overlap = True
                        break

        # 如果沒有重疊，進行替換
        if not overlap:
            main_frame[y2_min:y2_max, x2_min:x2_max] = frame[y2_min:y2_max, x2_min:x2_max]
        else:
            overlap_count += 1

    # 如果沒有重疊的區域，表示替換完成
    if overlap_count == 0:
        replace_done = True

    # 可選：視覺化處理過程
    if view:
        cv2.imshow("Replacing Frame", main_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 儲存最終結果並顯示
cv2.imwrite(output_image_path, main_frame)
print(f"最終影像已儲存到 {output_image_path}")
cv2.imshow("Final Result", main_frame)
cv2.waitKey()
cap.release()
cv2.destroyAllWindows()
