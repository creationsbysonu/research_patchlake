import os
import time
import shutil
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import pyarrow as pa
import pyarrow.ipc as ipc

# Import engine components
from format import create_patchlake_schema
from ingest import PatchExtractor, write_plake_stripe
from reader import ZeroCopyPatchLakeReader

# -------------------------------------------------------------------
# BASELINE DATASET & DATALOADER (Standard JPEG CPU Pipeline)
# -------------------------------------------------------------------
class JPEGBaselineDataset:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_paths = [
            os.path.join(image_folder, f) 
            for f in os.listdir(image_folder) 
            if f.endswith('.jpg')
        ]
        # Standard ViT preprocessing pipeline executed on CPU per batch
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Simulation of ViT Linear Projection layer
        self.proj = torch.nn.Conv2d(3, 768, kernel_size=16, stride=16)
        self.proj.eval()

    def load_and_patchify_batch(self, batch_paths):
        tensors = []
        for path in batch_paths:
            # 1. CPU Disk I/O & JPEG Decoding
            img = Image.open(path).convert('RGB')
            # 2. CPU Resize, Normalization, Tensorization
            tensor_img = self.transform(img)
            tensors.append(tensor_img)
        
        # 3. Stack into batch: [B, 3, 224, 224]
        batch_tensor = torch.stack(tensors)
        
        # 4. On-the-fly CPU Patchify: [B, 3, 224, 224] -> [B, 196, 768]
        with torch.no_grad():
            patches = self.proj(batch_tensor).flatten(2).transpose(1, 2)
        
        return patches

    def benchmark_stream(self, batch_size=10):
        total_images = len(self.image_paths)
        start_time = time.time()
        
        for i in range(0, total_images, batch_size):
            batch_paths = self.image_paths[i:i + batch_size]
            _ = self.load_and_patchify_batch(batch_paths)
            
        end_time = time.time()
        return end_time - start_time

# -------------------------------------------------------------------
# SYNTHETIC DATASET GENERATOR
# -------------------------------------------------------------------
def generate_benchmark_data(num_images=100, output_dir="bench_data"):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print(f"\n[1/3] Generating {num_images} synthetic images on disk...")
    jpeg_folder = os.path.join(output_dir, "jpegs")
    os.makedirs(jpeg_folder)
    
    extractor = PatchExtractor()
    plake_records = []

    for i in range(num_images):
        # Generate 256x256 random synthetic RGB image
        img_np = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(img_np)
        
        # Save as JPEG for Baseline
        img.save(os.path.join(jpeg_folder, f"img_{i:04d}.jpg"), quality=85)
        
        # Extractor for PatchLake .plake format
        cls_vec, patch_bytes, q_min, q_scale = extractor.process_and_quantize(img)
        plake_records.append({
            "image_id": i,
            "timestamp": int(time.time() * 1000000),
            "camera_id": f"CAM_{i % 4}",
            "class_label": "benchmark_class",
            "cls_vector": cls_vec,
            "patch_tensors": patch_bytes.tolist(),
            "quant_min": q_min,
            "quant_scale": q_scale
        })

    print("[2/3] Writing PatchLake (.plake) format file...")
    plake_path = os.path.join(output_dir, "dataset.plake")
    write_plake_stripe(plake_records, plake_path)
    
    return jpeg_folder, plake_path

# -------------------------------------------------------------------
# MAIN BENCHMARK SUITE
# -------------------------------------------------------------------
if __name__ == "__main__":
    NUM_TEST_IMAGES = 100  # Scalable parameter for stress testing
    BATCH_SIZE = 10

    jpeg_dir, plake_file = generate_benchmark_data(num_images=NUM_TEST_IMAGES)
    
    print("\n[3/3] Executing Comparative Benchmarking Suite...")
    print("=" * 60)

    # 1. Run Baseline Test
    baseline = JPEGBaselineDataset(jpeg_dir)
    baseline_time = baseline.benchmark_stream(batch_size=BATCH_SIZE)
    baseline_fps = NUM_TEST_IMAGES / baseline_time
    baseline_patches_per_sec = (NUM_TEST_IMAGES * 196) / baseline_time

    # 2. Run PatchLake Test
    reader = ZeroCopyPatchLakeReader(plake_file)
    start_time = time.time()
    
    num_batches = reader.reader.num_record_batches
    for batch_idx in range(num_batches):
        _ = reader.stream_batch(batch_idx)
        
    plake_time = time.time() - start_time
    plake_fps = NUM_TEST_IMAGES / plake_time
    plake_patches_per_sec = (NUM_TEST_IMAGES * 196) / plake_time

    # Calculate Storage Footprint
    jpeg_storage_bytes = sum(
        os.path.getsize(os.path.join(jpeg_dir, f)) 
        for f in os.listdir(jpeg_dir)
    )
    plake_storage_bytes = os.path.getsize(plake_file)

    # Output Results
    print(f"\nRESULTS: Ingestion Throughput Benchmarks ({NUM_TEST_IMAGES} Images)")
    print("-" * 60)
    print(f"Standard JPEG Pipeline Time: {baseline_time:.4f} sec | {baseline_fps:.2f} img/sec | {baseline_patches_per_sec:.2f} patches/sec")
    print(f"PatchLake Zero-Copy Time:    {plake_time:.4f} sec | {plake_fps:.2f} img/sec | {plake_patches_per_sec:.2f} patches/sec")
    print("-" * 60)
    
    speedup = baseline_time / plake_time if plake_time > 0 else 0
    print(f"★ THROUGHPUT SPEEDUP: PatchLake is {speedup:.2f}x FASTER than Standard Baseline")
    
    print("\nSTORAGE FOOTPRINT ANALYSIS")
    print("-" * 60)
    print(f"Raw JPEG Directory Size:   {jpeg_storage_bytes / (1024*1024):.2f} MB")
    print(f"PatchLake (.plake) Size:   {plake_storage_bytes / (1024*1024):.2f} MB")
    print(f"Uncompressed Tensor Size:  {(NUM_TEST_IMAGES * 196 * 768 * 4) / (1024*1024):.2f} MB (Reference Raw FP32)")
    print("=" * 60)
