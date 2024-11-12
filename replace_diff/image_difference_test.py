import cv2

# 讀取兩張圖片
image1 = cv2.imread('images/canny_person.jpg')
image2 = cv2.imread('images/canny_person(1).jpg')

# 調整圖片大小
width = 640
height1 = int((width / image1.shape[1]) * image1.shape[0])
height2 = int((width / image2.shape[1]) * image2.shape[0])
image1 = cv2.resize(image1, (width, height1))
image2 = cv2.resize(image2, (width, height2))

# 將圖片轉為灰度
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# 計算兩張圖片的差分
diff = cv2.absdiff(gray1, gray2)
cv2.imshow(f'Difference with threshold', diff)
cv2.waitKey(0)
# 設定不同的靈敏度閾值
thresholds = [60, 70, 80]

# 顯示每個閥值的差異
for threshold in thresholds:
    # 將差分圖像進行二值化處理
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # 尋找輪廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 在原圖上標記差異區域
    marked_image = image1.copy()
    cv2.drawContours(marked_image, contours, -1, (0, 255, 0), 2)
    
    # 顯示結果
    cv2.imshow(f'Difference with threshold {threshold}', marked_image)

cv2.waitKey(0)
cv2.destroyAllWindows()