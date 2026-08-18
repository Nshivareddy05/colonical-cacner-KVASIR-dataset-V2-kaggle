import os, hashlib, time
start = time.time()
dataset_dir = 'dataset'
hash_map = {}
count = 0
for root, _, files in os.walk(dataset_dir):
    for f in files:
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(root, f)
            with open(path, 'rb') as file:
                file_hash = hashlib.md5(file.read()).hexdigest()
            if 'cancer-1' in path:
                hash_map[file_hash] = 'cancer-1'
            elif 'cancer-0' in path:
                hash_map[file_hash] = 'cancer-0'
            count += 1
print(f"Hashed {count} files in {time.time() - start:.2f} seconds")
