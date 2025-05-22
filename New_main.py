import numpy as np
import matplotlib.pyplot as plt

watermark = np.load('w.npy')

# They have the watermark locked and loaded!

print(watermark.shape)
# The watermark is a 2D array

# Plot the watermark
plt.imshow(watermark, cmap='gray')
plt.title('Watermark')
plt.axis('off')
plt.show()