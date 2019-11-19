import cv2

def myICA():
    img1 = cv2.imread("/Users/zhangchen/Documents/资料/学校材料/课程/神经网络及应用/NN作业用到的材料/风景图/1.bmp")
    img2 = cv2.imread("/Users/zhangchen/Documents/资料/学校材料/课程/神经网络及应用/NN作业用到的材料/风景图/2.bmp")
    img3 = cv2.imread("/Users/zhangchen/Documents/资料/学校材料/课程/神经网络及应用/NN作业用到的材料/风景图/3.bmp")
    img4 = cv2.imread("/Users/zhangchen/Documents/资料/学校材料/课程/神经网络及应用/NN作业用到的材料/风景图/4.bmp")
    Image = cv2.addWeighted(img1, 0.8, img2, 0.2, 0)
    cv2.imshow('Image', Image)

    cv2.waitKey(0)
    cv2.destroyAllWindow()

myICA()