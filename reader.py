import pyarrow as pa
import pyarrow.ipc as ipc
import numpy as np
import torch
import time

class ZeroCopyPatchLakeReader:
    def __init__(self, plake_file_path):
        # 1. mmap the file (OS maps disk to virtual memory, bypassing RAM allocation)
        self.source = pa.memory_map(plake_file_path, 'r')
        self.reader = ipc.RecordBatchFileReader(self.source)

    def stream_batch(self, batch_idx=0):
        batch = self.reader.get_record_batch(batch_idx)
        
        # --- THE ZERO COPY HANDOFF ---
        
         # 2. Get the FixedSizeList array of patch tensors
        patch_array = batch.column("patch_tensors")
        
        # 3. Access the raw underlying C-buffer
        # Because it's a FixedSizeList, we must access the child array (.values) 
        # which holds the actual flat contiguous uint8 memory block.
        flat_values = patch_array.values
        
        # In a primitive array, buffer[0] is the null bitmap, buffer[1] is the raw data
        raw_buffer = flat_values.buffers()[1]

        
        # 4. Wrap the raw buffer in NumPy (Zero Copy)
        np_data = np.frombuffer(raw_buffer, dtype=np.uint8)
        
        # 5. Wrap the NumPy array in PyTorch (Zero Copy)
        # Reshape to [Batch_Size, 196 patches, 768 dimensions]
        num_rows = batch.num_rows
        torch_tensors = torch.from_numpy(np_data).view(num_rows, 196, 768)
        
        # --- PROOF OF ZERO COPY ---
        print("\n--- Memory Address Verification ---")
        print(f"Arrow Buffer Pointer:   {hex(raw_buffer.address)}")
        print(f"PyTorch Tensor Pointer: {hex(torch_tensors.data_ptr())}")
        
        if hex(raw_buffer.address) == hex(torch_tensors.data_ptr()):
            print("SUCCESS: Perfect Zero-Copy Handoff Achieved! No RAM duplicated.")
        else:
            print("FAILED: Memory was copied.")

        # --- DEQUANTIZATION (RQC) ---
        # To use this in an actual ViT, we convert the uint8 back to float32
        # using the metadata columns we saved during ingestion.
        q_min = torch.tensor(batch.column("quant_min").to_numpy()).view(-1, 1, 1)
        q_scale = torch.tensor(batch.column("quant_scale").to_numpy()).view(-1, 1, 1)
        
        # Math: X_float = (X_quantized * scale) + min
        ready_tensors = (torch_tensors.to(torch.float32) * q_scale) + q_min
        
        return ready_tensors

if __name__ == "__main__":
    print("Initializing Zero-Copy Reader...")
    reader = ZeroCopyPatchLakeReader("test_batch.plake")
    
    start_time = time.time()
    vit_ready_tensors = reader.stream_batch(0)
    end_time = time.time()
    
    print("\n--- Tensor Stats ---")
    print(f"Final Tensor Shape: {vit_ready_tensors.shape}")
    print(f"Final Tensor Dtype: {vit_ready_tensors.dtype}")
    print(f"End-to-End Read Latency: {(end_time - start_time) * 1000:.2f} ms")
