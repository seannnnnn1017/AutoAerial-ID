import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

def canny_edge(img ,low_threshold = 50 ,high_threshold = 150):
    gray = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    width = 640
    height1 = int((width / img.shape[1]) * img.shape[0])
    height2 = int((width / gray.shape[1]) * gray.shape[0])
    img = cv2.resize(img, (width, height1))
    gray = cv2.resize(gray, (width, height2))

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)
        #定義Canny的參數並應用
    low_threshold = 50
    high_threshold = 150
    masked_edges = cv2.Canny(blur_gray,low_threshold,high_threshold)
        #定義Hough轉換參數
    rho = 1
    theta = np.pi/180
    threshold = 1
    min_line_length = 40
    max_line_gap = 1
        #製作與我們的圖像大小相同的空白
    line_image = np.copy(img)*0 
        #在邊緣檢測到的圖像上運行霍夫轉換

    lines = cv2.HoughLinesP(masked_edges,rho,theta,threshold,np.array([]),min_line_length,max_line_gap)
        #遞廻輸出“lines”並在空白處繪製線條

    for i in lines:
        for x1,y1,x2,y2 in i:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
        #創建彩色之二進制圖像與線圖像組合
    color_edges = np.dstack((masked_edges,masked_edges,masked_edges))
        #在邊緣圖像上繪製線條
    combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 



    return combo

if __name__=="__main__":
    
    img = cv2.imread("images\white_wall(1).jpg")
        
    cv2.imshow('edge',canny_edge(img))
    cv2.waitKey(0)