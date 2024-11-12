import cv2

# 定義前處理函數
def image_process(img):

    
    # 計算 x 軸方向的梯度
    dstx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    
    # 計算 y 軸方向的梯度
    dsty = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    
    # 將負值轉換為正值
    dstx = cv2.convertScaleAbs(dstx)
    dsty = cv2.convertScaleAbs(dsty)
    
    # 加權融合 x 和 y 的梯度圖像
    dst = cv2.addWeighted(dstx, 0.5, dsty, 0.5, 0)
    return dst


input_video_path = "images/752750716.004314.mp4"  # 影片路徑
cap = cv2.VideoCapture(input_video_path)

#初始化前一影格
ret, prev_frame = cap.read()
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_frame_edge = image_process(prev_frame_gray)  # 對前一影格進行前處理

# 設定閾值
threshold_value = 50
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 將當前影格轉換為灰度
    current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 對當前影格進行前處理
    current_frame_edge = image_process(current_frame_gray)

    # 計算當前影格與前一影格的差異
    frame_diff = cv2.absdiff(prev_frame_edge, current_frame_edge)

    # 應用閾值來高亮顯示移動物體
    _, thresh = cv2.threshold(frame_diff, threshold_value, 255, cv2.THRESH_BINARY)

    # 使用形態學運算來去除雜訊並加強移動區域
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 找到移動物體的輪廓並在原影格上標示
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # 過濾掉小區域
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(current_frame_edge, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 顯示結果
    cv2.imshow("Moving Objects with Edge Detection", current_frame_edge)

    # 更新前一影格
    prev_frame_edge = current_frame_edge

    # 按 'q' 鍵退出顯示
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
