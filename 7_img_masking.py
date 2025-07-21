import matplotlib.pyplot as plt
import cv2


def main():
    img_3D = cv2.imread("/home/mohon/4_1/cse4161/images/rgb_flower.png", cv2.IMREAD_COLOR)

    masking_area = 1000
    img_3D[:masking_area, :masking_area] = 0    

    img_3D = cv2.cvtColor(img_3D, cv2.COLOR_BGR2RGB)
    plt.imshow(img_3D)
    plt.title('Modified Image')
    plt.show()




if __name__ == "__main__":
    main()