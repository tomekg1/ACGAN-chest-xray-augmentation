'''
This script plots eight images of chest x-ray with bouding box, which help to locate
each of eight pathologies specified in BBox_List_2017.csv file
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import random

# Paths
data_dir = '..\\Data'
bbox_csv_path = '..\\DataConfig\BBox_List_2017.csv'

bbox_df = pd.read_csv(bbox_csv_path)

print(bbox_df.columns)
print(bbox_df.head())


unique_labels = bbox_df['Finding Label'].unique()

rows = 2
cols = 7
num_subplots = rows * cols


if len(unique_labels) > num_subplots:
    selected_labels = random.sample(list(unique_labels), num_subplots)
else:
    selected_labels = unique_labels
    num_subplots = len(selected_labels)
    rows = min(2, num_subplots)
    cols = (num_subplots + rows - 1) // rows

fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
fig.tight_layout(pad=3.0)

if num_subplots == 1:
    axes = [axes]
else:
    axes = axes.flatten()


for i, label in enumerate(selected_labels):
    if i >= num_subplots:
        break
        
    label_records = bbox_df[bbox_df['Finding Label'] == label]
    
    if len(label_records) == 0:
        continue
        
    record = label_records.sample(1).iloc[0]
    image_name = record['Image Index']
    
    # find path to image
    image_path = None
    for dir_num in range(1, 13):
        search_dir = os.path.join(data_dir, f'images_{str(dir_num).zfill(3)}', 'images')
        potential_path = os.path.join(search_dir, image_name)
        if os.path.exists(potential_path):
            image_path = potential_path
            break
    
    if image_path is None:
        print(f"Image {image_name} not found in any directory")
        continue
    
    # load image
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        continue
    
    # bounding box
    x = record['Bbox [x']
    y = record['y']
    w = record['w']
    h = record['h]']
    
    ax = axes[i]
    ax.imshow(img, cmap='gray')
    ax.set_title(f"{label}", fontsize=18)
    ax.axis('off')
    
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.show()
