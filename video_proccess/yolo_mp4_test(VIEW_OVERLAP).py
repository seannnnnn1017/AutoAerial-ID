from ultralytics import YOLO
import cv2
view=True
model = YOLO("yolov10x.pt")  
# 載入影片
input_video_path = "images/752750716.004314.mp4"  # 影片路徑
cap = cv2.VideoCapture(input_video_path)

main_frame_replace=[]
# 找一張有人的照片
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 使用 YOLO 模型進行物件偵測
    results = model(frame)
    
    # 複製影格以進行標註
    annotated_frame = frame.copy()
    person_count = 0
    # 只保留標註為 "person" 的偵測結果
    for result in results:
        for box in result.boxes:  # 提取每個偵測框
            if box.cls == 0:  # 在 COCO 資料集中，"person" 的類別索引通常為 0
                person_count += 1
                # 取得邊框座標
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                main_frame_replace.append([x1, y1, x2, y2])
                main_frame=frame.copy()
    if person_count > 0:
        break
                # 繪製邊框與標註
                #cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #cv2.putText(annotated_frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # 顯示處理過的影格
    #cv2.imshow("YOLO Person Detection", annotated_frame)

    # 按 'q' 鍵退出顯示
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


print(main_frame_replace) #把人物位置存起來
for i in main_frame_replace:
    cv2.rectangle(main_frame, (i[0], i[1]), (i[2], i[3]), (0, 255, 0), 2)
    cv2.putText(main_frame, "Person", (i[0], i[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.imshow("YOLO Person Detection", main_frame)
cv2.waitKey()

replace_count=0
while cap.isOpened(): #找影片中其他時間段的圖片來取代人物
    ret, frame = cap.read()
    if not ret:
        break
    
    # 使用 YOLO 模型進行物件偵測
    results = model(frame)
    
    # 複製影格以進行標註
    annotated_frame = frame.copy()
    # 初始化所有 main_frame_replace 範圍為藍色框
    for i in main_frame_replace:
        if view:
            cv2.rectangle(annotated_frame, (i[0], i[1]), (i[2], i[3]), (255, 0, 0), 2)
            cv2.putText(annotated_frame, "Person", (i[0], i[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 只保留標註為 "person" 的偵測結果
    for result in results:
        for box in result.boxes:  # 提取每個偵測框
            if box.cls == 0:  # 偵測到 "person"
                person_count += 1
                # 取得邊框座標
                x1_min, y1_min, x1_max, y1_max = map(int, box.xyxy[0])
                
                # 檢查是否與 main_frame_replace 中的任一框重疊
                overlap = False
                for idx, i in enumerate(main_frame_replace):
                    x2_min, y2_min, x2_max, y2_max = map(int, i)
                    
                    # 判斷重疊並改變顏色
                    if x1_min < x2_max and x1_max > x2_min and y1_min < y2_max and y1_max > y2_min:
                        overlap = True
                        if view:
                            # 改變被碰到的矩形顏色為紅色
                            cv2.rectangle(annotated_frame, (x2_min, y2_min), (x2_max, y2_max), (0, 0, 255), 2)
                            cv2.putText(annotated_frame, "Person", (x2_min, y2_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                # 畫框：綠色代表不重疊，藍色代表重疊
                color = (0, 255, 0) if not overlap else (0, 0, 255)
                if view:
                    cv2.rectangle(annotated_frame, (x1_min, y1_min), (x1_max, y1_max), color, 2)


    # 顯示影像
    if view:
        cv2.imshow('Annotated Frame', annotated_frame)
    # 按下 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if replace_count == len(main_frame_replace):
        break
                # 繪製邊框與標註
                #cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #cv2.putText(annotated_frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # 顯示處理過的影格
    cv2.imshow("YOLO Person Detection", annotated_frame)

    # 按 'q' 鍵退出顯示
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
