import pyarrow as pa

def create_patchlake_schema(patch_count=196, dim=768):
    """
    Defines the zero-copy Arrow IPC schema for PatchLake.
    Combines relational metadata with flattened, fixed-size tensors for ViT ingestion.
    """
    
    # 1. Tensor Types
    # The global CLS token is unquantized (float32) for high-precision ANN search
    cls_type = pa.list_(pa.float32(), dim) 
    
    # The patch tensors are heavily quantized to 8-bit unsigned integers (uint8)
    # The list size is strictly defined (196 * 768) to force a FixedSizeList memory layout
    tensor_element_type = pa.uint8()
    patch_tensor_type = pa.list_(tensor_element_type, patch_count * dim)

    # 2. Relational & Physical Layout
    schema = pa.schema([
        # Standard Relational Metadata (for SQL predicate pushdown)
        pa.field("image_id", pa.int64(), nullable=False),
        pa.field("timestamp", pa.timestamp("us"), nullable=False), # Microsecond precision
        pa.field("camera_id", pa.string()),
        pa.field("class_label", pa.string()),
        
        # Hybrid Vector Indexing Zone
        pa.field("cls_vector", cls_type, nullable=False),
        
        # Tensor Matrix Data Zone
        pa.field("patch_tensors", patch_tensor_type, nullable=False),
        
        # Residual Quantization (RQC) Reconstruction Parameters
        # These allow PyTorch to un-quantize the uint8 back to float32 upon read
        pa.field("quant_min", pa.float32(), nullable=False),
        pa.field("quant_scale", pa.float32(), nullable=False)
    ])
    
    # 3. Inject file-level metadata to identify the custom format
    metadata = {
        b"format": b"patchlake",
        b"version": b"1.0.0",
        b"patch_count": str(patch_count).encode(),
        b"embed_dim": str(dim).encode()
    }
    
    return schema.with_metadata(metadata)

if __name__ == "__main__":
    # Test the schema generation
    schema = create_patchlake_schema()
    print("PatchLake Schema Generated Successfully:")
    print("-" * 40)
    print(schema.to_string())
    print("\nFile Metadata:", schema.metadata)
