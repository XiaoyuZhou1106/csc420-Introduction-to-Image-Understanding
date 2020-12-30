
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#Step I:The function that returns a 2D Gaussian matrix
def GuassianBlurring(sigma):
    k = int(3 * sigma)
    matrix_size = 2 * k + 1
    keneral = np.zeros((matrix_size, matrix_size))
    for i in range(matrix_size):
        for j in range(matrix_size):
            temp_x = i - k
            temp_y = j - k
            keneral[i][j] = (1 / 2 * np.pi * (sigma ** 2)) * np.exp(-(temp_x ** 2 + temp_y ** 2)/ (2 *sigma **2))

    return keneral

#Step2: The function that return the gradient magnitude of the image
def GradientMagnitude(image):
    sobelX = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
    sobelY = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))

    gradientX = np.zeros((image.shape)) #the gradients of image along X-axis
    gradientY = np.zeros((image.shape)) #the gradients of image along y-axis

    #find the gradient for every G(x, y)
    for x in range(image.shape[0] - 2):
        for y in range(image.shape[1] - 2):
            tempMatrix = image[x:x+3, y:y+3]
            gradientX[x][y] = np.sum(np.multiply(sobelX, tempMatrix))
            gradientY[x][y] = np.sum(np.multiply(sobelY, tempMatrix))

    gradientFinal = np.sqrt(gradientY ** 2 + gradientX ** 2)
    return gradientFinal

#Step3: Implement threshold algorithm to find the edge.
def ThresholdAlgorithm(gradientImage):
    #1: initialized tau
    tauOrignal = np.sum(gradientImage) / (gradientImage.shape[0] * gradientImage.shape[1])
    epsilon = 0

    tauTemp = UpdateTau(gradientImage, tauOrignal) #find tau_i then compare.
    while abs(tauTemp - tauOrignal) > epsilon: #iteration for 2-4
        tauOrignal = tauTemp
        tauTemp = UpdateTau(gradientImage, tauOrignal)

    #5, assign value to the edge-mapped image
    edgeImage = np.zeros((gradientImage.shape))
    for x in range(gradientImage.shape[0]):
        for y in range(gradientImage.shape[1]):
            if gradientImage[x][y] >= tauOrignal:
                edgeImage[x][y] = 255
            else:
                edgeImage[x][y] = 0
    return edgeImage

#a helper function to take an image and a gaussian matrix to flur the image
def FlurImage(image, keneral):
    flurImage = np.zeros((image.shape[0] - keneral.shape[0] + 1, image.shape[1]-keneral.shape[0] + 1))

    for x in range(image.shape[0] - keneral.shape[0] + 1):
        for y in range(image.shape[1] - keneral.shape[0] + 1):
            tempMatrix = image[x:x+keneral.shape[0], y:y+keneral.shape[0]]
            flurImage[x][y] = np.sum(np.multiply(keneral, tempMatrix))
    return flurImage

#a helper function to update the latest tau in the ThresholdAlgorithm
def UpdateTau(gradientImage, tauOrignal):
    lower = []
    upper = []
    for x in range(gradientImage.shape[0]):
        for y in range(gradientImage.shape[1]):
            if gradientImage[x][y] < tauOrignal:
                lower.append(gradientImage[x][y])
            else:
                upper.append(gradientImage[x][y])
    mL = sum(lower) / len(lower)
    mH = sum(upper) / len(upper)
    tauTemp = (mL + mH) / 2
    return tauTemp


if __name__ == '__main__':
    #visualize for step 1:
    #for sigma = 1:
    Blur1 = GuassianBlurring(1)
    plt.imshow(Blur1)
    #for sigma = 3:
    Blur2 = GuassianBlurring(3)
    plt.imshow(Blur2)

    #Step4: test
    #test for image 1
    image1 = np.asarray(Image.open('./Q4_image_1.jpg').convert('RGB'))
    #convert the image into grayscle
    gray_image1 = 0.2125 * image1[:, :, 0] + 0.7154 * image1[:, :, 1] + 0.0721 * image1[:, :, 2]
    plt.imshow(gray_image1, cmap='gray')
    plt.show()
    #find the blurred image
    blurImage1 = FlurImage(gray_image1, GuassianBlurring(3))
    plt.imshow(blurImage1, cmap='gray')
    plt.show()
    #find the Gradient image
    gradientImage1 = GradientMagnitude(blurImage1)
    plt.imshow(gradientImage1, cmap='gray')
    plt.show()
    #find the edge image
    edge_image1 = ThresholdAlgorithm(gradientImage1)
    plt.imshow(edge_image1, cmap='gray')
    plt.show()

    #test for image 2
    image2 = np.asarray(Image.open('./Q4_image_2.jpg').convert('RGB'))
    #convert the image into grayscle
    grayImage2 = 0.2125 * image2[:, :, 0] + 0.7154 * image2[:, :, 1] + 0.0721 * image2[:, :, 2]
    plt.imshow(grayImage2, cmap='gray')
    plt.show()
    #find the blurred image
    blurImage2 = FlurImage(grayImage2, GuassianBlurring(3))
    plt.imshow(blurImage2, cmap='gray')
    plt.show()
    #find the Gradient image
    gradientImage2 = GradientMagnitude(blurImage2)
    plt.imshow(gradientImage2, cmap='gray')
    plt.show()
    #find the edge image
    edge_image2 = ThresholdAlgorithm(gradientImage2)
    plt.imshow(edge_image2, cmap='gray')
    plt.show()

    #test for my own image
    image3 = np.asarray(Image.open('./Q4_image_3.jpg').convert('RGB'))
    #convert the image into grayscle
    grayImage3 = 0.2125 * image3[:, :, 0] + 0.7154 * image3[:, :, 1] + 0.0721 * image3[:, :, 2]
    plt.imshow(grayImage3, cmap='gray')
    plt.show()
    #find the blurred image
    blurImage3 = FlurImage(grayImage3, GuassianBlurring(3))
    plt.imshow(blurImage3, cmap='gray')
    plt.show()
    #find the Gradient image
    gradientImage3 = GradientMagnitude(blurImage3)
    plt.imshow(gradientImage3, cmap='gray')
    plt.show()
    #find the edge image
    edge_image3 = ThresholdAlgorithm(gradientImage3)
    plt.imshow(edge_image3, cmap='gray')
    plt.show()

