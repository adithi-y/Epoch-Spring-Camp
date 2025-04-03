import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


theimage = Image.open("ferb.jpeg").convert('L')
image_array = np.array(theimage)

height, width = image_array.shape
firstquad = image_array[:height//2, :width//2]

image_array[height//2:, :width//2] = np.flipud(firstquad)
image_array[:height//2, width//2:] = np.fliplr(firstquad)
image_array[height//2: , width//2:] = np.fliplr(image_array[height//2:, :width//2])

result_image = Image.fromarray(image_array)
result_image.save("mirrored_image.jpg")

plt.imshow(result_image, cmap = 'gray')
plt.axis('off')
plt.show()


# first you import the necessary libraries
# then open the image and convert it into an array with the grayscale pixel values
# you create variables height and width and assign them
# then create a array for first quadrant and assign those pixel values to it
# then assign the values for the second third and fourth quadrants respectively by using the flip method in numpy
# then you reconstruct the image from the given pixel values and save it
# and finally plot it using the matplotlib library