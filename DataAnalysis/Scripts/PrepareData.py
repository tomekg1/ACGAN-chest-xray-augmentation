'''
This script provides data preparation for NIH chest X-rays dataset.
It transforms 1024x1024 png files to 128x128 jpg files and puts them
in folders accordingly to their description in given csv file
'''

import os
from PIL import Image
import pandas as pd


df = pd.read_csv('DataConfig\Data_Entry_2017.csv')

# Filter out rows where Finding Labels contains multiple labels (contains '|')
df_single_label = df[~df['Finding Labels'].str.contains('|', regex=False)]

base_path = 'Data'
output_base_path = 'ProcessedData'

# Output dimension
DIMS = (128, 128)

# Total number of "No Finding"
no_finding_count = 0
max_no_finding = 10000

# Directory name indexing
path_idx = 1

for _, row in df_single_label.iterrows():
    finding_label = row['Finding Labels']
    image_file = row['Image Index']
    
    # Path to image file
    image_path = os.path.join(base_path, f"images_{str(path_idx).zfill(3)}", 'images', image_file)
    
    # If image path does not exist change it to the next directory
    if not os.path.exists(image_path):
        path_idx += 1
        image_path = os.path.join(base_path, f"images_{str(path_idx).zfill(3)}", 'images', image_file)
        print(f"New path: {image_path}")
    
    if os.path.exists(image_path):
        img = Image.open(image_path)
        img = img.resize(DIMS)
        
        # Change extension to jpg
        img = img.convert("RGB")
        
        # Creation of output folder based on finding label name
        label_path = os.path.join(output_base_path, finding_label)
        if not os.path.exists(label_path):
            os.makedirs(label_path)
        
        # No Finding label limits
        if finding_label == "No Finding":
            if no_finding_count >= max_no_finding:
                continue
            no_finding_count += 1
        
        # Save path
        output_image_path = os.path.join(label_path, image_file.replace('.png', '.jpg'))
        
        # Save image in jpg format
        img.save(output_image_path, 'JPEG')

print("Processing finished!")
