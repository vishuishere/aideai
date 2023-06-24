# from PIL import Image
# #Read the two images
# image1 = Image.open('media/brain/output/output_31_1.png')
# image1.show()
# image2 = Image.open('media/brain/output/output_31_2.png')
# image2.show()
# #resize, first image
# image1 = image1.resize((426, 240))
# image1_size = image1.size
# image2_size = image2.size
# new_image = Image.new('RGB',(2*image1_size[0], image1_size[1]), (250,250,250))
# new_image.paste(image1,(0,0))
# new_image.paste(image2,(0,0))
# new_image.save("merged_image.jpg","JPEG")
# new_image.show()
# result = Image.blend(image1, image2, alpha=0.5)
# result.save("blended_merged_image.jpg","JPEG")
# result.show()

import cv2
import numpy as np
# C:\My Projects\AIDE\AIDE BE\aide_ai\aideai\media\brain\output
# Load the two images
image1 = cv2.imread('media/brain/output/output_31_1.png')
image2 = cv2.imread('media/brain/output/output_31_2.png')
image3 = cv2.imread('media/brain/output/output_31_3.png')

image1 = cv2.resize(image1, (500, 500))
image2 = cv2.resize(image2, (500, 500))
image3 = cv2.resize(image3, (500, 500))

import matplotlib.pyplot as plt
import numpy as np


# Create a figure with subplots
fig, axes = plt.subplots(1, 3)

# Plot the images in the subplots
axes[0].imshow(image1, cmap='gray')
axes[0].set_title('Image')

axes[1].imshow(image2, cmap='gray')
axes[1].set_title('Label')

axes[2].imshow(image3, cmap='gray')
axes[2].set_title('Output')

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.savefig('ttt.png')
