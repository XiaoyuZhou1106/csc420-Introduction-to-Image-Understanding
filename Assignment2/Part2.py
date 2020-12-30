
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

#A function to contract a matrix, where every element represents the min sum of gradient of that pixel.
def FindMinGradient(image, gradient):
    M = np.zeros(image.shape) #construct a matrix to store every min gradient for every pixels
    #for the last row, its min sum gradient is itself
    for i in range(image.shape[1]):
        M[image.shape[0] - 1] = gradient[image.shape[0] - 1][i]
    for m in range(image.shape[0] - 1):
        for n in range(image.shape[1]):
            r = image.shape[0] - 2 - m #temperary row, from bottom to top, except the last row
            c = image.shape[1] - 1 - n #temperary column, from right to left
            sub_gradient = M[r+1][c]
            if c - 1 > 0: # test whether it is out of range
                if M[r+1][c-1] < sub_gradient:
                    sub_gradient = M[r+1][c-1]
            if c + 1 < image.shape[1]: # test whether it is out of range
                if M[r+1][c+1] < sub_gradient:
                    sub_gradient = M[r+1][c+1]
            M[r][c] = sub_gradient + gradient[r][c]
    return M

#Find the path of the min sum of gradient in M
def FindMinPath(M, image):
    path = []
    min_gradient = min(M[0])
    min_col = 0
    for i in range(image.shape[1]):
        if M[0][i] == min_gradient:
            min_col = i
    path.append(min_col)
    for j in range(image.shape[0] - 1):
        temp_col = min_col
        temp_min = M[j+1][temp_col]
        if min_col - 1 > 0: # test whether it is out of range
            if M[j+1][min_col-1] < temp_min:
                temp_min = M[j+1][min_col-1]
                temp_col = min_col-1
        if min_col + 1 < image.shape[1]: # test whether it is out of range
            if M[j+1][min_col+1] < temp_min:
                temp_min = M[j+1][min_col+1]
                temp_col = min_col+1
        path.append(temp_col)
        min_col = temp_col
    return path


#calculate the gradient.
def CalculateGradient(image):
    gx = np.gradient(image)[0]
    gy = np.gradient(image)[1]
    gradient = np.zeros(gy.shape)
    for x in range(gx.shape[0]):
        for y in range(gx.shape[1]):
            gradient[x][y] = (gx[x][y] ** 2 + gy[x][y] ** 2) ** 0.5
    return gradient

#Remove the pixels in the path with min sum gradient from the image
def RemoveColumn(image):
    gradient = CalculateGradient(image)
    M = FindMinGradient(image, gradient)
    path = FindMinPath(M, image)

    #copy the orignal image except the pixels in the path line by line.
    new_image = np.zeros((image.shape[0], image.shape[1]-1))
    for i in range(image.shape[0]):
        new_col = 0
        for j in range(image.shape[1]):
            if j != path[i]:
                new_image[i][new_col] = image[i][j]
                new_col += 1
    return new_image

#main function for seam carving
def ImageResizing(image, size):
    #first, do column delation:
    new_image = image
    while new_image.shape[1] > size[1]:
        new_image = RemoveColumn(new_image)

    #do row delation:
    row_image = new_image.transpose() #initialize for row reduction
    while row_image.shape[1] > size[0]:
        row_image = RemoveColumn(row_image)
    return row_image.transpose()


if __name__ == '__main__':
    #seam carving for ex1.
    img = cv2.imread('./ex1.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = gray*1.1
    blur = cv2.GaussianBlur(gray,(5,5),7)
    image = ImageResizing(blur, (968, 957))
    plt.imshow(image, cmap='gray')
    #
    # #cropped ex1:
    # crop_img = blur[:968, :957]
    # plt.imshow(crop_img, cmap='gray')
    #
    # #scaled ex1:
    # scaled_img = cv2.resize(blur, (957, 968))
    # plt.imshow(scaled_img, cmap='gray')
    #
    # #seam carving for ex2.
    # img = cv2.imread('./ex2.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # gray = gray*1.1
    # blur = cv2.GaussianBlur(gray,(5,5),7)
    # #for size 961 x 1200
    # image = ImageResizing(blur, (961, 1200))
    # plt.imshow(image, cmap='gray')
    # #for size 861 x 1200
    # image = ImageResizing(blur, (861, 1200))
    # plt.imshow(image, cmap='gray')
    #
    # #cropped ex2:
    # crop_img = blur[:861, :1200]
    # plt.imshow(crop_img, cmap='gray')
    #
    # #scaled ex2:
    # scaled_img = cv2.resize(blur, (1200, 861))
    # plt.imshow(scaled_img, cmap='gray')
    #
    # #seam carving for ex3.
    # img = cv2.imread('./ex3.jpg')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # gray = gray*1.1
    # blur = cv2.GaussianBlur(gray,(5,5),7)
    # #for size 870 x 1440
    # image = ImageResizing(blur, (870, 1440))
    # plt.imshow(image, cmap='gray')
    # #for size 870 x 1200
    # image = ImageResizing(blur, (870, 1200))
    # plt.imshow(image, cmap='gray')
    #
    # #cropped ex3:
    # crop_img = blur[:870, :1200]
    # plt.imshow(crop_img, cmap='gray')
    #
    # #scaled ex3:
    # scaled_img = cv2.resize(blur, (1200, 870))
    # plt.imshow(scaled_img, cmap='gray')

