import torch
import torchvision.transforms as T
from PIL import Image
import pyarrow as pa
import pyarrow.ipc as ipc
import numpy as np
import time

# Import the schema from our previous script
from format import create_patchlake_schema

class PatchExtractor:
    def __init__(self, patch_size=16, embed_dim=768):
        """
        Simulates the Patch Embedding layer of a Vision Transformer (ViT-Base).
        """
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # In a real pipeline, this would be the frozen weights of CLIP or LLaVA.
        # We use a randomized Conv2d layer to simulate the exact mathematical operation.
        self.proj = torch.nn.Conv2d(
            in_channels=3, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        self.proj.eval() # Disable gradient tracking

        # Standard ViT preprocessing pipeline
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process_and_quantize(self, pil_image: Image.Image):
        """
        Transforms an image into quantized patch tensors and a global CLS vector.
        """
        # 1. Preprocessing (CPU bound)
        x = self.transform(pil_image).unsqueeze(0) # Shape: [1, 3, 224, 224]

        with torch.no_grad():
            # 2. Patchification
            # Conv2d output: [1, 768, 14, 14]
            # Flatten(2) -> [1, 768, 196]
            # Transpose -> [1, 196, 768]
            patches = self.proj(x).flatten(2).transpose(1, 2).squeeze(0)
            
            # Simulate a global CLS token by averaging the patches
            cls_token = patches.mean(dim=0) # Shape: [768]

            # 3. Asymmetric SQ8 Quantization
            # Find min and max to scale the float32 range into uint8 (0-255)
            p_min = patches.min().item()
            p_max = patches.max().item()
            scale = (p_max - p_min) / 255.0 if p_max != p_min else 1.0
            
            # Apply transformation: quantized = round((value - min) / scale)
            quantized = torch.clamp(torch.round((patches - p_min) / scale), 0, 255).to(torch.uint8)

        # Flatten into 1D arrays for Arrow serialization
        return (
            cls_token.numpy().tolist(),         # 768 floats
            quantized.numpy().flatten(),        # 150,528 uint8 integers
            p_min,
            scale
        )

def write_plake_stripe(records, output_path):
    """
    Serializes a list of dictionaries into a .plake Arrow IPC file.
    """
    schema = create_patchlake_schema()
    
    # Convert Python dictionaries to Arrow columnar arrays based on the strict schema
    table = pa.Table.from_pylist(records, schema=schema)
    
    # Write to disk using Arrow IPC format (zero-copy compatible)
    with pa.OSFile(output_path, 'wb') as sink:
        with ipc.new_file(sink, schema) as writer:
            writer.write_table(table)
    
    return table.num_rows

if __name__ == "__main__":
    print("Initializing PatchLake Ingestion Pipeline...")
    extractor = PatchExtractor()
    
    records = []
    num_test_images = 10
    
    print(f"Processing {num_test_images} synthetic images...")
    start_time = time.time()
    
    for i in range(num_test_images):
        # Generate a synthetic random image
        dummy_img = Image.fromarray(np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8))
        
        # Run extraction and quantization
        cls_vec, patch_bytes, q_min, q_scale = extractor.process_and_quantize(dummy_img)
        
        # Package with relational metadata
        records.append({
            "image_id": i + 1000,
            "timestamp": int(time.time() * 1000000), # microseconds
            "camera_id": f"CAM_{i % 3}",
            "class_label": "synthetic_test",
            "cls_vector": cls_vec,
            "patch_tensors": patch_bytes.tolist(),
            "quant_min": q_min,
            "quant_scale": q_scale
        })

    # Write to disk
    output_file = "test_batch.plake"
    rows_written = write_plake_stripe(records, output_file)
    
    end_time = time.time()
    
    print("-" * 40)
    print(f"Successfully wrote {rows_written} records to {output_file}")
    print(f"Time elapsed: {end_time - start_time:.4f} seconds")
    print(f"Total patches processed: {rows_written * 196}")
