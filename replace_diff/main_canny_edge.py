import cv2
import numpy as np
import canny_edge
# 讀取兩張圖片
image1 = cv2.imread('images\people.jpg')
image2 = cv2.imread('images\people(2).jpg')
# 調整圖片大小
width = 640
height1 = int((width / image1.shape[1]) * image1.shape[0])
height2 = int((width / image2.shape[1]) * image2.shape[0])
image1 = cv2.resize(image1, (width, height1))
image2 = cv2.resize(image2, (width, height2))
# 邊緣檢測
edge1=canny_edge.canny_edge(image1)
edge2=canny_edge.canny_edge(image2)
# 計算兩張圖片的差分
diff = cv2.absdiff(edge1, edge2)
diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) 
cv2.imshow('A', edge1)
cv2.imshow('B', edge2)
cv2.imshow('Differences', diff)
cv2.waitKey(0)
# 設定二值化的閾值
threshold = 50
_, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

# 尋找二值圖像中的輪廓
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 初始化用來包含所有綠框的最小和最大邊界
x_min, y_min = np.inf, np.inf
x_max, y_max = -np.inf, -np.inf

# 在原始圖像上繪製綠色矩形框來標記差異區域，並更新邊界
for contour in contours:
    if cv2.contourArea(contour) > 25:  # 可以調整面積閾值來過濾較小的噪聲
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #image2[y:y+h, x:x+w] = image1[y:y+h, x:x+w]
        # 更新包含所有綠框的邊界
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
        

# 繪製紅色矩形框，包圍所有綠框的範圍
if x_min != np.inf and y_min != np.inf and x_max != -np.inf and y_max != -np.inf:
    # 繪製紅框
    cv2.rectangle(image2, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

    x_min=max(x_min-20,0)
    y_min=max(y_min-20,0)
    x_max=min(x_max+20,image2.shape[1])
    y_max=min(y_max+20,image2.shape[0])
    # 替換紅色框內的區域為 image1 對應區域

    image2[y_min:y_max, x_min:x_max] = image1[y_min:y_max, x_min:x_max]

# 顯示結果
cv2.imshow('Differences', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()
