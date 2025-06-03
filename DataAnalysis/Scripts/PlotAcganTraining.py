'''
This script plots three images achieved during training process of ACGAN with given
row names (number of epoch for each image) and column names (image class)
'''

import matplotlib.pyplot as plt
from PIL import Image

# Path to selected files
png_files = [
    '..\\DataConfig\\train_imgs\\eff\\0_0.png',
    '..\\DataConfig\\train_imgs\\nof\\0_0.png',
    '..\\DataConfig\\train_imgs\\eff\\epoch50_0.png',
    '..\\DataConfig\\train_imgs\\nof\\epoch50_0.png',
    '..\\DataConfig\\train_imgs\\eff\\epoch150_0.png',
    '..\\DataConfig\\train_imgs\\nof\\epoch150_0.png'
]

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))

columns = ['Effusion', 'No Finding']
for ax, col in zip(axes[0], columns):
    ax.set_title(col, fontsize=18, pad=10)

rows = ['Epoch 0', 'Epoch 50', 'Epoch 150']
for i, row in enumerate(rows):
    axes[i, 0].text(-0.1, 0.5, row, fontsize=18, 
                    ha='right', va='center', transform=axes[i, 0].transAxes)

for ax, img_path in zip(axes.ravel(), png_files):
    img = Image.open(img_path)
    ax.imshow(img)
    ax.axis('off')

plt.tight_layout()
plt.show()