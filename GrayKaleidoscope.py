# first you import the necessary libraries
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# then open the image and convert it into an array with the grayscale pixel values
theimage = Image.open("ferb.jpeg").convert('L')
image_array = np.array(theimage)

# you create variables height and width and assign them
height, width = image_array.shape

# then create a array for first quadrant and assign those pixel values to it
firstquad = image_array[:height//2, :width//2]

# then assign the values for the second third and fourth quadrants respectively by using the flip method in numpy
image_array[height//2:, :width//2] = np.flipud(firstquad)
image_array[:height//2, width//2:] = np.fliplr(firstquad)
image_array[height//2: , width//2:] = np.fliplr(image_array[height//2:, :width//2])

# then you reconstruct the image from the given pixel values
result_image = Image.fromarray(image_array)

# and finally plot it using the matplotlib library
plt.imshow(result_image, cmap = 'gray')
plt.axis('off')
plt.show()
