import time
import pyarrow as pa
import pyarrow.ipc as ipc
import duckdb
import torch
import numpy as np

class DuckDBPatchLakeEngine:
    def __init__(self, plake_file_path):
        """
        Loads the .plake Arrow IPC container and registers it 
        as an in-memory zero-copy virtual table inside DuckDB.
        """
        self.plake_file_path = plake_file_path
        
        # 1. Memory-map the Arrow IPC file
        self.source = pa.memory_map(plake_file_path, 'r')
        self.reader = ipc.RecordBatchFileReader(self.source)
        
        # 2. Read all record batches into an Arrow Table
        self.arrow_table = self.reader.read_all()
        
        # 3. Initialize embedded DuckDB connection
        self.con = duckdb.connect(database=':memory:')
        
        # 4. Register PyArrow table as a virtual SQL table named 'patchlake'
        self.con.register('patchlake', self.arrow_table)

    def execute_sql(self, sql_query):
        """
        Executes SQL queries over the registered .plake table 
        and fetches the result as a PyArrow Table.
        """
        return self.con.execute(sql_query).to_arrow_table()


    def query_and_stream_to_pytorch(self, sql_query):
        """
        Executes a SQL query, filters the Arrow table using DuckDB, 
        and performs a zero-copy handoff of filtered patch tensors directly to PyTorch.
        """
        start_time = time.time()
        
        # Execute SQL and fetch filtered result as PyArrow Table
        result_arrow_table = self.execute_sql(sql_query)
        query_latency = (time.time() - start_time) * 1000
        
        num_rows = result_arrow_table.num_rows
        if num_rows == 0:
            return torch.empty((0, 196, 768), dtype=torch.float32), query_latency
        
        # Extract patch tensors array from filtered query output
        patch_column = result_arrow_table.column("patch_tensors")
        
        # Flatten chunked arrays if DuckDB splits the batches
        if isinstance(patch_column, pa.ChunkedArray):
            flat_values = patch_column.chunks[0].values
        else:
            flat_values = patch_column.values
            
        raw_buffer = flat_values.buffers()[1]
        
        # Zero-copy view into PyTorch
        np_data = np.frombuffer(raw_buffer, dtype=np.uint8)
        uint8_tensors = torch.from_numpy(np_data).view(num_rows, 196, 768)
        
        # Vectorized RQC Dequantization
        q_min = torch.tensor(result_arrow_table.column("quant_min").to_numpy()).view(-1, 1, 1)
        q_scale = torch.tensor(result_arrow_table.column("quant_scale").to_numpy()).view(-1, 1, 1)
        
        ready_tensors = (uint8_tensors.to(torch.float32) * q_scale) + q_min
        
        return ready_tensors, query_latency


if __name__ == "__main__":
    print("Initializing DuckDB Virtual Table Engine for PatchLake...")
    engine = DuckDBPatchLakeEngine("bench_data/dataset.plake")
    
    print("\n" + "=" * 60)
    print("DEMO 1: Relational Metadata Aggregation")
    print("=" * 60)
    
    # Query 1: Standard SQL aggregation over camera_id metadata
    agg_sql = """
        SELECT 
            camera_id, 
            COUNT(*) as image_count,
            AVG(quant_scale) as avg_quant_scale
        FROM patchlake 
        GROUP BY camera_id
        ORDER BY camera_id
    """
    rel_result = engine.execute_sql(agg_sql)
    print("SQL Query Result (Native Arrow Dict):")
    print(rel_result.to_pydict())
    
    print("\n" + "=" * 60)
    print("DEMO 2: Filtered SQL Pushdown directly to PyTorch")
    print("=" * 60)
    
    # Query 2: Filtered SQL query retrieving patch tensors for CAM_2
    filter_sql = """
        SELECT patch_tensors, quant_min, quant_scale 
        FROM patchlake 
        WHERE camera_id = 'CAM_2'
    """
    tensors, latency = engine.query_and_stream_to_pytorch(filter_sql)
    
    print(f"Executed SQL: {filter_sql.strip()}")
    print(f"Query Latency: {latency:.2f} ms")
    print(f"Output PyTorch Tensor Shape: {tensors.shape}")
    print(f"Output PyTorch Tensor Type:  {tensors.dtype}")
    print("=" * 60)
