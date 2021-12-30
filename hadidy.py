
import numpy as np
import cv2 
import  matplotlib.pyplot as plt




def plt_image(img):
    plt.axis("off")
    plt.imshow(img,cmap="gray")
    plt.show()




def subtract_background(image, radius=50, light_bg=False):
        from skimage.morphology import white_tophat, black_tophat, disk
        str_el = disk(radius) #you can also use 'ball' here to get a slightly smoother result at the cost of increased computing time
        if light_bg:
            return  black_tophat(image, str_el)
        else:
            return white_tophat(image, str_el)

img=cv2.imread("highway1.jpg",0)
img=subtract_background(img)
plt_image(img)

