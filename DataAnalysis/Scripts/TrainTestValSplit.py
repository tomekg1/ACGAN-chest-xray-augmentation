'''
This script provides data preparation for NIH chest X-rays dataset.
It splits the processed data (PrepareData.py) to train and validation
datasets with selected ratio and chosen classes
'''

import os
import random
import shutil

input_data_dir = 'ProcessedData' 
output_data_dir = 'SplitData_Eff_NoF'

# Data split
split_ratios = {'train': 0.5, 'val': 0.5}

# Create directories
for split in split_ratios.keys():
    split_path = os.path.join(output_data_dir, split)
    if not os.path.exists(split_path):
        os.makedirs(split_path)

choosen_classes = [
        #'Atelectasis',
    #'Cardiomegaly',
    #'Consolidation',
    #'Edema',
    'Effusion',
    #'Emphysema',
    #'Fibrosis',
    #'Hernia',
    #'Infiltration',
    #'Mass',
    'No Finding'
    #'Nodule',
    #'Pleural_Thickening',
    #'Pneumonia',
    #'Pneumothorax'
]

# Process each Finding Label class
for class_name in choosen_classes:
    class_path = os.path.join(input_data_dir, class_name)
    
    if os.path.isdir(class_path):
        images = os.listdir(class_path)
        # random.shuffle(images)
        
        num_images = len(images)
        train_split = int(num_images * split_ratios['train'])
        val_split = train_split + int(num_images * split_ratios['val'])
        
        for i, image in enumerate(images):
            src_path = os.path.join(class_path, image)
            
            if i < train_split:
                dst_folder = os.path.join(output_data_dir, 'train', class_name)
            elif i < val_split:
                dst_folder = os.path.join(output_data_dir, 'val', class_name)

            
            os.makedirs(dst_folder, exist_ok=True)
            shutil.copy(src_path, os.path.join(dst_folder, image))

print("Split finished!")