import os
import time
import torch
from PIL import Image
import torchvision.transforms as T

# Import engine components
from reader import ZeroCopyPatchLakeReader

class COCOJpegBaseline:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]
        
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # ViT Linear Projection simulation
        self.proj = torch.nn.Conv2d(3, 768, kernel_size=16, stride=16)
        self.proj.eval()

    def run_epoch(self, batch_size=100):
        total_images = len(self.image_paths)
        start_time = time.time()
        
        for i in range(0, total_images, batch_size):
            batch_paths = self.image_paths[i:i + batch_size]
            tensors = []
            
            for path in batch_paths:
                img = Image.open(path).convert('RGB')
                tensors.append(self.transform(img))
                
            batch_tensor = torch.stack(tensors)
            
            with torch.no_grad():
                _ = self.proj(batch_tensor).flatten(2).transpose(1, 2)
                
        return time.time() - start_time

if __name__ == "__main__":
    COCO_DIR = "bench_data/coco/val2017"
    PLAKE_FILE = "bench_data/coco/coco_val2017.plake"
    NUM_IMAGES = 5000

    print(f"\nExecuting Large-Scale COCO Benchmark ({NUM_IMAGES} Images)...")
    print("=" * 60)

    # 1. Baseline Run
    print("Running Standard JPEG-to-CPU Pipeline...")
    baseline = COCOJpegBaseline(COCO_DIR)
    baseline_time = baseline.run_epoch(batch_size=100)
    baseline_fps = NUM_IMAGES / baseline_time

    # 2. PatchLake Run
    print("Running PatchLake Zero-Copy Pipeline...")
    reader = ZeroCopyPatchLakeReader(PLAKE_FILE)
    start_time = time.time()
    
    num_batches = reader.reader.num_record_batches
    for batch_idx in range(num_batches):
        _ = reader.stream_batch(batch_idx)
        
    plake_time = time.time() - start_time
    plake_fps = NUM_IMAGES / plake_time

    # 3. Output
    print("\nRESULTS: COCO Dataset Throughput")
    print("-" * 60)
    print(f"Standard JPEG Pipeline: {baseline_time:.2f} sec | {baseline_fps:.2f} img/sec")
    print(f"PatchLake Zero-Copy:    {plake_time:.2f} sec | {plake_fps:.2f} img/sec")
    print("-" * 60)
    
    speedup = baseline_time / plake_time
    print(f"★ AT SCALE SPEEDUP: PatchLake is {speedup:.2f}x FASTER")
    print("=" * 60)
