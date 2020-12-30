import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def ConnectedComponentLabeling(image):
    label = np.zeros((image.shape))
    currentLabel = 1
    queue = []

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            if image[x][y] >= 10 and label[x][y] == 0: #if it is a foreground pixel and has not labbelled
                # (the reason for pixel > 10 is because not all black pixel in the image are 0. So I
                # assummed the value less than 10 is black
                label[x][y] = currentLabel
                queue.append((x, y))
                while len(queue) > 0:
                    tempPixel = queue[0]
                    queue.remove(tempPixel)
                    pixelX, pixelY = tempPixel
                    neighbors = [(pixelX - 1, pixelY - 1), (pixelX, pixelY - 1), (pixelX + 1, pixelY - 1),
                                 (pixelX - 1, pixelY), (pixelX + 1, pixelY),
                                 (pixelX - 1, pixelY + 1), (pixelX, pixelY + 1), (pixelX + 1, pixelY + 1)]
                    for item in neighbors:
                        if 0 <= item[0] < image.shape[0] and 0 <= item[1] < image.shape[1]:
                            if image[item[0]][item[1]] >= 10 and label[item[0]][item[1]] == 0:
                                label[item[0]][item[1]] = currentLabel
                                queue.append(item)
                currentLabel += 1
    return label

if __name__ == '__main__':
    image = np.asarray(Image.open('./Q6.png').convert('RGB'))
    label = ConnectedComponentLabeling(image[:,:,0])
    print(label.max()) #this is the number of cells that the algorithm return
    plt.imshow(label, cmap='gray')

