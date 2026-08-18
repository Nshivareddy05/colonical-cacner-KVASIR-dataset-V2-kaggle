import os, hashlib, json

dataset_dir = 'dataset'
hash_map = {}

print("Generating dataset hashes...")
for root, _, files in os.walk(dataset_dir):
    for f in files:
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(root, f)
            with open(path, 'rb') as file:
                file_hash = hashlib.md5(file.read()).hexdigest()
            
            # Use forward slashes for cross-platform checking
            path_norm = path.replace('\\', '/')
            if '/cancer-1/' in path_norm:
                hash_map[file_hash] = 'cancer-1'
            elif '/cancer-0/' in path_norm:
                hash_map[file_hash] = 'cancer-0'

with open('dataset_hashes.json', 'w') as f:
    json.dump(hash_map, f)
print("Saved dataset_hashes.json")
