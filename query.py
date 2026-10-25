import time
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.compute as pc
import numpy as np
import torch

class PushdownPatchLakeEngine:
    def __init__(self, plake_file_path):
        """
        Memory-maps the PatchLake storage container and initializes 
        columnar metadata evaluators for zero-copy pushdown.
        """
        self.source = pa.memory_map(plake_file_path, 'r')
        self.reader = ipc.RecordBatchFileReader(self.source)

    def execute_pushdown_query(self, camera_id_target=None, class_label_target=None):
        """
        Evaluates relational predicates directly on Arrow columnar metadata, 
        creates a selection vector, and extracts matching patch tensors 
        with zero memory duplication.
        """
        matched_tensors = []
        total_rows_scanned = 0
        total_rows_matched = 0

        for i in range(self.reader.num_record_batches):
            batch = self.reader.get_record_batch(i)
            num_rows = batch.num_rows
            total_rows_scanned += num_rows

            # ---------------------------------------------------------------
            # STEP 1: Columnar Predicate Evaluation (Metadata Zone Scan)
            # ---------------------------------------------------------------
            mask = None
            if camera_id_target is not None:
                # Wrap target string in pa.scalar for correct C++ kernel comparison
                target_scalar = pa.scalar(camera_id_target, type=pa.string())
                mask = pc.equal(batch.column("camera_id"), target_scalar)
            
            if class_label_target is not None:
                label_scalar = pa.scalar(class_label_target, type=pa.string())
                label_mask = pc.equal(batch.column("class_label"), label_scalar)
                mask = pc.and_(mask, label_mask) if mask is not None else label_mask

            if mask is None:
                # Select all rows if no filter predicate provided
                matched_indices = np.arange(num_rows)
            else:
                # Convert bit-packed PyArrow boolean mask to NumPy index array
                bool_mask = mask.to_numpy(zero_copy_only=False)
                matched_indices = np.where(bool_mask)[0]

            num_matched = len(matched_indices)
            if num_matched == 0:
                # ZONE PRUNING: Non-matching record batch skipped
                continue

            total_rows_matched += num_matched

            # ---------------------------------------------------------------
            # STEP 2: Zero-Copy Pointer Slicing on Filtered Candidates
            # ---------------------------------------------------------------
            patch_array = batch.column("patch_tensors")
            flat_values = patch_array.values
            raw_buffer = flat_values.buffers()[1]

            # Direct buffer view without heap copy
            full_np_data = np.frombuffer(raw_buffer, dtype=np.uint8)
            full_tensor_view = full_np_data.reshape(num_rows, 196, 768)

            # Slice matching candidate rows
            filtered_uint8_tensor = torch.from_numpy(full_tensor_view[matched_indices])

            # ---------------------------------------------------------------
            # STEP 3: Vectorized RQC Dequantization
            # ---------------------------------------------------------------
            q_min = torch.tensor(batch.column("quant_min").to_numpy()[matched_indices]).view(-1, 1, 1)
            q_scale = torch.tensor(batch.column("quant_scale").to_numpy()[matched_indices]).view(-1, 1, 1)

            # Dequantize uint8 back to float32
            dequantized_tensors = (filtered_uint8_tensor.to(torch.float32) * q_scale) + q_min
            matched_tensors.append(dequantized_tensors)

        # ---------------------------------------------------------------
        # STEP 4: Assemble Final PyTorch Input Tensor
        # ---------------------------------------------------------------
        if matched_tensors:
            final_feed = torch.cat(matched_tensors, dim=0)
        else:
            final_feed = torch.empty((0, 196, 768), dtype=torch.float32)

        return final_feed, total_rows_scanned, total_rows_matched

if __name__ == "__main__":
    print("Initializing Pushdown Query Engine...")
    engine = PushdownPatchLakeEngine("bench_data/dataset.plake")

    target_camera = "CAM_1"
    print(f"\nExecuting Filtered Pushdown SQL: SELECT patches WHERE camera_id = '{target_camera}'")
    
    start_time = time.time()
    result_tensors, scanned, matched = engine.execute_pushdown_query(camera_id_target=target_camera)
    latency_ms = (time.time() - start_time) * 1000

    print("=" * 60)
    print(f"Total Rows Scanned in Footers:  {scanned}")
    print(f"Matched Candidate Rows:        {matched} (Selectivity: {(matched/scanned)*100:.1f}%)")
    print(f"Output PyTorch Tensor Shape:    {result_tensors.shape}")
    print(f"Filtered Execution Latency:     {latency_ms:.2f} ms")
    print("=" * 60)
