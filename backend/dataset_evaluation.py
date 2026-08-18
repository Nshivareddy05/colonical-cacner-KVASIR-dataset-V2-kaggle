import os
import random

DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))

def get_random_sample(dataset_name):
    # dataset_name can be 'cancer-0' or 'cancer-1'
    dataset_path = os.path.join(DATASET_ROOT, dataset_name)
    if not os.path.exists(dataset_path):
        return None
    
    valid_extensions = ('.png', '.jpg', '.jpeg')
    all_files = []
    
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.lower().endswith(valid_extensions):
                all_files.append(os.path.join(root, f))
                
    if not all_files:
        return None
        
    return random.choice(all_files)

def get_histopathology_samples(count=8):
    dataset_path = os.path.join(DATASET_ROOT, 'Histopathology')
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(DATASET_ROOT, 'histopathology')
        if not os.path.exists(dataset_path):
            return []
            
    valid_extensions = ('.png', '.jpg', '.jpeg')
    all_files = []
    
    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.lower().endswith(valid_extensions):
                rel_path = os.path.relpath(os.path.join(root, f), DATASET_ROOT)
                rel_path = rel_path.replace('\\', '/')
                all_files.append(rel_path)
                
    if not all_files:
        return []
        
    if len(all_files) > count:
        return random.sample(all_files, count)
    return all_files
