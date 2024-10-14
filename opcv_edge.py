import cv2

def image_process(img):
    width = 640
    height1 = int((width / img.shape[1]) * img.shape[0])
    img = cv2.resize(img, (width, height1))
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


if __name__=='__main__':
    # 測試程式 (請將 'input.jpg' 替換為你的圖像路徑)
    img = cv2.imread('images\people(2).jpg', cv2.IMREAD_GRAYSCALE)
    
    cv2.imshow('edge',image_process(img))
    cv2.waitKey(0)
