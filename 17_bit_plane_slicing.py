import matplotlib.pyplot as plt 
import cv2 
import numpy as np 

def main():
    img_3D = cv2.imread("/home/mohon/4_1/code/dip_code/images/rgb_flower1.png")

    img_gray = cv2.cvtColor(img_3D, cv2.COLOR_BGR2GRAY)
    planes = []
    planes.append(img_gray)

    planes += img_plane(img_gray)
    planes +=  [planes[1] + planes[2] + planes[3] + planes[4] + planes[5] + planes[6] + planes[7] + planes[8]]
    display(planes)  



def img_plane(img):
    planes = []

    for i in range(8):
        tmp = img & (1<<i)
        planes.append(tmp)
    
    return planes
    

def display(img_set):
    
    for i in range(len(img_set)):
        plt.subplot(4,3, i+1)
        plt.imshow(img_set[i], cmap = 'gray')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
