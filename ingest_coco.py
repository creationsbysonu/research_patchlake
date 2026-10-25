import os
import time
import zipfile
import urllib.request
from PIL import Image
import pyarrow as pa
import pyarrow.ipc as ipc

# Import your existing engine components
from format import create_patchlake_schema
from ingest import PatchExtractor, write_plake_stripe

COCO_VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
DATA_DIR = "bench_data/coco"
ZIP_PATH = os.path.join(DATA_DIR, "val2017.zip")
IMG_DIR = os.path.join(DATA_DIR, "val2017")
PLAKE_OUT = os.path.join(DATA_DIR, "coco_val2017.plake")

def download_and_extract_coco():
    """Downloads and unzips the COCO 2017 Validation dataset if missing."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if not os.path.exists(IMG_DIR):
        if not os.path.exists(ZIP_PATH):
            print(f"[1/4] Downloading COCO val2017 (~778 MB)...")
            urllib.request.urlretrieve(COCO_VAL_URL, ZIP_PATH)
        
        print("[2/4] Extracting images...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
    else:
        print("[1/4] COCO dataset already exists locally.")

def ingest_coco_to_patchlake():
    """Reads real JPEGs, patchifies, quantizes, and writes to .plake."""
    extractor = PatchExtractor()
    plake_records = []
    
    # Get list of all JPEGs
    image_files = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
    total_images = len(image_files)
    
    print(f"[3/4] Ingesting {total_images} real-world images into PatchLake...")
    start_time = time.time()
    
    for idx, filename in enumerate(image_files):
        img_path = os.path.join(IMG_DIR, filename)
        
        try:
            # 1. Load real image (handles varying resolutions and aspect ratios)
            img = Image.open(img_path).convert('RGB')
            
            # 2. Extract and Quantize
            cls_vec, patch_bytes, q_min, q_scale = extractor.process_and_quantize(img)
            
            # 3. Build Lakehouse Record
            plake_records.append({
                "image_id": int(filename.split('.')[0]),  # COCO filenames are IDs
                "timestamp": int(time.time() * 1000000),
                "camera_id": "COCO_VAL",
                "class_label": "general_eval",
                "cls_vector": cls_vec,
                "patch_tensors": patch_bytes.tolist(),
                "quant_min": q_min,
                "quant_scale": q_scale
            })
            
            if (idx + 1) % 500 == 0:
                print(f"  -> Processed {idx + 1} / {total_images} images")
                
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    ingest_time = time.time() - start_time
    print(f"Extraction complete in {ingest_time:.2f} seconds.")

    print(f"[4/4] Writing massive zero-copy storage container...")
    write_plake_stripe(plake_records, PLAKE_OUT)
    
    final_size_mb = os.path.getsize(PLAKE_OUT) / (1024 * 1024)
    print("=" * 60)
    print(f"SUCCESS: COCO Validation set converted to PatchLake!")
    print(f"Final File Size: {final_size_mb:.2f} MB")
    print(f"Output Location: {PLAKE_OUT}")
    print("=" * 60)

if __name__ == "__main__":
    download_and_extract_coco()
    ingest_coco_to_patchlake()
