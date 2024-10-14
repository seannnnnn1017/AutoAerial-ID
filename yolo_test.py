from ultralytics import YOLO
import cv2
import os

# 載入 YOLOv9 模型
model = YOLO('yolov9c.pt')  # 確保您有正確的模型檔案

# 讀取圖片
image = cv2.imread('C:/Users/fishd/Desktop/Github/空拍影像/images/yolo_test.jpg')  # 替換為您的圖片路徑

# 獲取圖片的高度和寬度
height, width, _ = image.shape

# 計算每個區塊的高和寬
block_height = height // 6
block_width = width // 6

# 創建切割圖片的列表
images = []

# 將圖片切割為 3x3 的 9 份
for i in range(6):
    for j in range(6):
        # 根據行列切割圖片
        block = image[i * block_height:(i + 1) * block_height, j * block_width:(j + 1) * block_width]
        images.append(block)

# 指定保存的目錄
save_dir = 'C:/Users/fishd/Desktop/Github/空拍影像/results'

# 檢查目錄是否存在，不存在則創建
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

annotated_images = []

# 對每一份圖片進行物體檢測並保存結果
for i, img in enumerate(images):
    results = model.track(img, show=True, conf=0.2)  # 設定信心閾值為 0.2
    # 繪製標記到圖片
    annotated_img = results[0].plot()  # 獲取標記過的圖片
    annotated_images.append(annotated_img)
    # 保存檢測後的圖片
    cv2.imwrite(os.path.join(save_dir, f'annotated_image_{i}.jpg'), annotated_img)  # 保存完整的圖片

# 將切割的圖片合成成一張完整的圖
combined_image = cv2.vconcat([cv2.hconcat(annotated_images[i * 6:(i + 1) * 6]) for i in range(6)])  # 修改為 6x6 的合成


# 保存合成的圖片
cv2.imwrite(os.path.join(save_dir, 'combined_image.jpg'), combined_image)  # 保存合成的圖片
