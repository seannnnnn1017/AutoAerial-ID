import cv2
import numpy as np

# 設置畫布大小
canvas = np.ones((400, 400, 3), dtype="uint8") * 255

# 第一個矩形的座標
x1_min, y1_min, x1_max, y1_max = 100, 100, 200, 200

# 第二個矩形的座標（你可以改變這些數值來測試不重疊情況）
x2_min, y2_min, x2_max, y2_max = 250, 250, 350, 350

# 畫出第一個矩形，顏色為藍色
cv2.rectangle(canvas, (x1_min, y1_min), (x1_max, y1_max), (255, 0, 0), 2)
cv2.putText(canvas, "Rect 1", (x1_min, y1_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# 畫出第二個矩形，顏色為綠色
cv2.rectangle(canvas, (x2_min, y2_min), (x2_max, y2_max), (0, 255, 0), 2)
cv2.putText(canvas, "Rect 2", (x2_min, y2_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# 判斷不重疊的條件
# 不重疊的情況：一個矩形完全在另一個矩形的左邊、右邊、上方或下方
if x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max:
    cv2.putText(canvas, "No Overlap", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
else:
    # 若重疊，則計算重疊區域，並畫出重疊區域（紅色）
    overlap_x_min = max(x1_min, x2_min)
    overlap_y_min = max(y1_min, y2_min)
    overlap_x_max = min(x1_max, x2_max)
    overlap_y_max = min(y1_max, y2_max)
    cv2.rectangle(canvas, (overlap_x_min, overlap_y_min), (overlap_x_max, overlap_y_max), (0, 0, 255), -1)
    cv2.putText(canvas, "Overlap", (overlap_x_min, overlap_y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# 顯示圖像
cv2.imshow("Non-Overlap Visualization", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
