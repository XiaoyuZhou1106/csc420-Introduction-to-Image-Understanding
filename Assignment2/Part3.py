import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import scipy.linalg as la

def SecondMomentMatrix(image, window):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = gray*1.1

    blur = cv2.GaussianBlur(gray,(5,5),7)
    Ix = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)

    IxIy = np.multiply(Ix, Iy)
    Ix2 = np.multiply(Ix, Ix)
    Iy2 = np.multiply(Iy, Iy)

    Ix2_blur = cv2.GaussianBlur(Ix2,(7,7),10)
    Iy2_blur = cv2.GaussianBlur(Iy2,(7,7),10)
    IxIy_blur = cv2.GaussianBlur(IxIy,(7,7),10)

    lambda0 = np.zeros(gray.shape)
    lambda1 = np.zeros(gray.shape)
    for x in range(image.shape[0] - window.shape[0]):
        for y in range(image.shape[1] - window.shape[1]):
            #calculate every M matrix
            M_matrix = [[0, 0], [0, 0]]
            for m in range(window.shape[0]):
                for n in range(window.shape[1]):
                    temp_matrix = np.array([[Ix2_blur[x+m][y+n], IxIy_blur[x+m][y+n]], [IxIy_blur[x+m][y+n], Iy2_blur[x+m][y+n]]])
                    M_matrix += window[m][n] * temp_matrix
            #find the eigenvalues:
            eigvals, eigvecs = la.eig(M_matrix)
            eigvals = eigvals.real
            lambda0[x][y] = eigvals[0]
            lambda1[x][y] = eigvals[1]
    return lambda0, lambda1

#find a Gaussian kernel with size 3
def GuassianBlurring(sigma):
    k = sigma
    matrix_size = 3
    keneral = np.zeros((matrix_size, matrix_size))
    for i in range(matrix_size):
        for j in range(matrix_size):
            temp_x = i - k
            temp_y = j - k
            keneral[i][j] = (1 / 2 * np.pi * (sigma ** 2)) * np.exp(-(temp_x ** 2 + temp_y ** 2)/ (2 *sigma **2))

    return keneral

if __name__ == '__main__':
    #set the window
    window = np.array([[0.25, 0.25], [0.25, 0.25]])
    #find eigenvalues for image1:
    img1 = cv2.imread('./image1.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    lam0_img1, lam1_img1 = SecondMomentMatrix(img1, window)
    # plt.plot(lam0_img1, lam1_img1, '.', color='black') #comment this line, or it will influence next image

    #find the corner for image1
    img1_temp = img1.copy()
    img1_temp[np.logical_and(lam0_img1 > 0.05 * np.amax(lam0_img1), lam1_img1 > 0.05 * np.amax(lam1_img1))] = [255, 255, 0]
    plt.imshow(img1_temp)
    #
    # #find eigenvalues for image2:
    # img2 = cv2.imread('./image2.jpg')
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # lam0_img2, lam1_img2 = SecondMomentMatrix(img2, window)
    # # plt.plot(lam0_img2, lam1_img2, '.', color='black') #comment this line, or it will influence next image
    #
    # #find the corner for image2
    # img2_temp = img2.copy()
    # img2_temp[np.logical_and(lam0_img2 > 0.05 * np.amax(lam0_img2), lam1_img2 > 0.05 * np.amax(lam1_img2))] = [255, 255, 0]
    # plt.imshow(img2_temp)
    #
    #
    # #Generate a Gaussian kernel with sigma = 1
    # kernel1 = GuassianBlurring(1)
    # #for image1:
    # lam0_img1_k1, lam1_img1_k1 = SecondMomentMatrix(img1, kernel1)
    # plt.plot(lam0_img1_k1, lam1_img1_k1, '.', color='black')
    # #find the corners
    # img1_temp_k1 = img1.copy()
    # img1_temp_k1[np.logical_and(lam0_img1_k1 > 0.05 * np.amax(lam0_img1_k1), lam1_img1_k1 > 0.05 * np.amax(lam1_img1_k1))] = [255, 255, 0]
    # plt.imshow(img1_temp_k1)
    #
    # #for image2:
    # lam0_img2_k1, lam1_img2_k1 = SecondMomentMatrix(img2, kernel1)
    # plt.plot(lam0_img2_k1, lam1_img2_k1, '.', color='black')
    # #find the corners
    # img2_temp_k1 = img2.copy()
    # img2_temp_k1[np.logical_and(lam0_img2_k1 > 0.05 * np.amax(lam0_img2_k1), lam1_img2_k1 > 0.05 * np.amax(lam1_img2_k1))] = [255, 255, 0]
    # plt.imshow(img2_temp_k1)
    #
    # #Generate a Gaussian kernel with sigma = 100000
    # kernel2 = GuassianBlurring(100000)
    # #for image1:
    # lam0_img1_k2, lam1_img1_k2 = SecondMomentMatrix(img1, kernel2)
    # plt.plot(lam0_img1_k2, lam1_img1_k2, '.', color='black')
    # #find the corners
    # img1_temp_k2 = img1.copy()
    # img1_temp_k2[np.logical_and(lam0_img1_k2 > 0.05 * np.amax(lam0_img1_k2), lam1_img1_k2 > 0.05 * np.amax(lam1_img1_k2))] = [255, 255, 0]
    # plt.imshow(img1_temp_k2)
    #
    # #for image2:
    # lam0_img2_k2, lam1_img2_k2 = SecondMomentMatrix(img2, kernel2)
    # plt.plot(lam0_img2_k2, lam1_img2_k2, '.', color='black')
    # #find the corners
    # img2_temp_k2 = img2.copy()
    # img2_temp_k2[np.logical_and(lam0_img2_k2 > 0.05 * np.amax(lam0_img2_k2), lam1_img2_k2 > 0.05 * np.amax(lam1_img2_k2))] = [255, 255, 0]
    # plt.imshow(img2_temp_k2)

