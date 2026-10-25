# PatchLake

**A columnar "lakehouse" file format for pre-patchified Vision Transformer embeddings.**

PatchLake stores the *output* of a ViT patch-embedding layer (`196 × 768` uint8 tensors + a
float32 CLS vector) next to the relational metadata that describes the image, in a single
Apache Arrow IPC file (`.plake`). Because the tensors live in one contiguous, fixed-size Arrow
buffer, a query engine can filter on metadata and hand the surviving rows straight to PyTorch
with **zero copies** (`Arrow buffer address == torch tensor data_ptr`).

The project answers one question: *if the patchification step is materialised at ingest time
instead of being recomputed on every training epoch, how much of the CPU dataloader disappears?*

```
JPEG (any resolution)
  │   ViT preprocessing: Resize 224×224 → ToTensor → Normalize(ImageNet)
  ▼
[1, 3, 224, 224] float32
  │   Conv2d(3→768, kernel=16, stride=16)   ← stands in for the ViT patch-embedding layer
  ▼
[1, 768, 14, 14] → flatten(2) → transpose(1,2) → [1, 196, 768]
  │                    └── mean over the 196 patches ──► CLS vector [768] float32
  │   SQ8 quantisation:  scale = (max-min)/255,  q = clamp(round((x-min)/scale), 0, 255)
  ▼
196×768 uint8  +  quant_min  +  quant_scale
  │   Arrow IPC file (.plake)
  ▼
memory_map → column predicate on metadata → buffer view → torch.from_numpy → dequantised batch
```

---

## Headline results

Measured on this machine, CPU only, single run, no warm-up (details in
[Benchmarks](#benchmarks) and [Reality checks](#reality-checks--what-these-numbers-do-not-prove)):

| Pipeline (100 images) | Wall time | Images/sec | Patches/sec |
|---|---|---|---|
| Standard JPEG pipeline (decode → resize → normalize → patchify) | 0.2081 s | 480.51 | 94,179.09 |
| PatchLake zero-copy read (mmap → buffer view → dequant) | 0.0304 s | 3,285.48 | 643,953.24 |
| **Speed-up** | | **6.84×** | **6.84×** |

| Storage (100 images) | Size |
|---|---|
| Raw JPEG directory (`quality=85`, 256×256) | 4.78 MB |
| `.plake` file | 14.66 MB |
| Equivalent raw float32 tensors (reference) | 57.42 MB |

`.plake` is **3.92× smaller than raw fp32** and **≈3.1× larger than the JPEG originals** for
this low-resolution synthetic set (on real COCO photographs it comes out slightly *smaller* than
the JPEGs — see [Real-world scale](#real-world-scale-ingest_cocopy--coco_val2017plake)). It
trades disk for CPU; it does not compress pixels.

Zero-copy is verified at runtime by comparing pointers, not asserted in prose:

```
--- Memory Address Verification ---
Arrow Buffer Pointer:   0x15804c6b8
PyTorch Tensor Pointer: 0x15804c6b8
SUCCESS: Perfect Zero-Copy Handoff Achieved! No RAM duplicated.
```

---

## Repository layout

```
engine/                          ← git repository root
├── format.py                    Schema definition (the actual file-format spec)
├── ingest.py                    JPEG → patches → SQ8 → .plake writer (+ 10-image demo)
├── ingest_coco.py               Download COCO val2017 and ingest all 5,000 images
├── reader.py                    Zero-copy reader, .plake → ViT-ready float32 tensor
├── query.py                     Pushdown engine: metadata filter + zero-copy tensor slice
├── duckdb_engine.py             Same file exposed as a SQL table through embedded DuckDB
├── benchmark.py                 JPEG-baseline vs PatchLake throughput benchmark
├── test_batch.plake             Generated demo artifact (10 rows, 1.5 MB)
├── bench_data/
│   ├── jpegs/                   Generated synthetic JPEGs (100 files, 5.1 MB)
│   ├── dataset.plake            Generated benchmark artifact (100 rows, 14.7 MB)
│   └── coco/
│       ├── val2017.zip          Downloaded COCO archive (778 MB)
│       ├── val2017/             5,000 real JPEGs (787 MB)
│       └── coco_val2017.plake   Real-world artifact (5,000 rows, 732.7 MB)
└── __pycache__/                 Interpreter cache (should not be committed)
```

Every module is runnable on its own (`python format.py`, `python reader.py`, …) and every
module's `__main__` prints the evidence for the claim it makes. **Run all commands from the
`engine/` directory** — the scripts use relative paths such as `bench_data/dataset.plake`.

### Script-by-script

| Script | Claim it demonstrates | Needs GPU? | Runtime |
|---|---|---|---|
| `format.py` | The schema is a real Arrow schema with a fixed-size tensor layout | no | instant |
| `ingest.py` | Images become quantised patches + metadata rows | no | 0.11 s / 10 imgs |
| `reader.py` | The buffer is handed to PyTorch without a copy | no | ~24 ms |
| `query.py` | Metadata predicates prune rows before tensors are touched | no | ~29 ms |
| `duckdb_engine.py` | The same file is queryable with plain SQL | no | ~17 ms |
| `ingest_coco.py` | It scales to 5,000 real images (732.7 MB file) | no | minutes + 778 MB download |
| `benchmark.py` | The end-to-end throughput gain over the JPEG pipeline | no | ~2 s |

---

## File format specification

A `.plake` file is an **uncompressed Apache Arrow IPC *file*** (`ipc.new_file`, footer + magic
bytes), produced from a table with this schema. Its identity is stamped into the schema
metadata, so a reader can detect a foreign file before touching any data.

| Field | Arrow type | Nullable | Bytes/row | Purpose |
|---|---|---|---|---|
| `image_id` | `int64` | no | 8 | Source image identifier (COCO filename stem for COCO runs) |
| `timestamp` | `timestamp[us]` | no | 8 | Ingestion time, microsecond precision |
| `camera_id` | `string` | yes | var | Sensor identifier — the primary pushdown predicate |
| `class_label` | `string` | yes | var | Scenario label — the secondary predicate |
| `cls_vector` | `fixed_size_list<float32>[768]` | no | 3,072 | Global embedding, unquantised, for future ANN search |
| `patch_tensors` | `fixed_size_list<uint8>[150528]` | no | 150,528 | The `196 × 768` patch matrix, SQ8 quantised, flattened row-major |
| `quant_min` | `float32` | no | 4 | Per-row dequantisation offset |
| `quant_scale` | `float32` | no | 4 | Per-row dequantisation step |

Schema metadata (written by `format.py`):

```
format:      'patchlake'
version:     '1.0.0'
patch_count: '196'
embed_dim:   '768'
```

### Why `FixedSizeList` and not a nested list or a blob

`fixed_size_list<uint8>[150528]` makes the payload **one contiguous, offset-free buffer per
record batch**. There is no per-row offsets array and no per-element overhead, so the reader can
reinterpret the raw C buffer directly:

```python
flat_values  = batch.column("patch_tensors").values     # child array, contiguous
raw_buffer   = flat_values.buffers()[1]                 # buffer[0] is the null bitmap
np_data      = np.frombuffer(raw_buffer, dtype=np.uint8)  # zero copy
tensors      = torch.from_numpy(np_data).view(rows, 196, 768)  # still zero copy
```

Dequantisation is then a single vectorised expression, with `quant_min`/`quant_scale` broadcast
as `[-1, 1, 1]`:

```
X_float = X_uint8 * quant_scale + quant_min
```

### Measured file geometry

| File | Rows | Bytes | Bytes/row | Record batches |
|---|---|---|---|---|
| `test_batch.plake` | 10 | 1,538,786 | 153,878 | 1 |
| `bench_data/dataset.plake` | 100 | 15,367,474 | 153,674 | 1 |
| `bench_data/coco/coco_val2017.plake` | 5,000 | 768,262,266 | 153,652 | **1** |

The arithmetic matches the format exactly: `150,528` (uint8 patches) + `3,072` (f32 CLS) +
`8 + 8 + 4 + 4` (scalars) `+` variable-length strings `+` Arrow IPC framing ≈ 153.6 KB/row.

SQ8 gives a **4× reduction** on the patch payload versus float32 (`150,528` vs `602,112` bytes
per image), which is what makes the format 3.92× smaller than raw tensors overall.

---

## Quick start

### 1. Environment

The project was developed and measured in a Conda environment named `patchlake`:

```bash
conda create -n patchlake python=3.11 -y
conda activate patchlake
pip install pyarrow torch torchvision duckdb numpy pillow
```

Exact versions this README's numbers were produced with:

| Component | Version |
|---|---|
| Python | 3.11.17 |
| pyarrow | 25.0.0 |
| torch | 2.5.1 |
| torchvision | 0.20.1 |
| duckdb | 2.0.0 |
| numpy | 2.4.6 |
| pillow | 12.3.0 |

Torch is CPU-only here; no CUDA/MPS-specific code path is used anywhere.
On some machines `import torchvision` prints a `libjpeg` warning — it is harmless, the pipeline
decodes JPEGs with Pillow, not `torchvision.io`.

There is no `requirements.txt`, no test suite and no CI in the repository at the time of writing.

### 2. Reproduce every claim in ~30 seconds

```bash
cd engine
python format.py        # prints the Arrow schema and format metadata
python ingest.py        # writes test_batch.plake from 10 synthetic images
python reader.py        # prints the pointer-equality zero-copy proof
python query.py         # filtered pushdown over the 100-row dataset
python duckdb_engine.py # SQL aggregation + filtered SQL → PyTorch

# ⚠️ benchmark.py DELETES bench_data/ before regenerating it.
# Move the COCO data out of the way first if you have already downloaded it:
mv bench_data/coco /tmp/coco_backup
python benchmark.py
mv /tmp/coco_backup bench_data/coco
```

To rebuild the real-world dataset (downloads 778 MB, ingests 5,000 images):

```bash
python ingest_coco.py
```

---

## Captured terminal output

Verbatim output from the runs on this machine (macOS, Apple M4, 16 GB RAM, CPU only). Every
number quoted elsewhere in this README comes from one of these runs.

### `python format.py`

```
PatchLake Schema Generated Successfully:
----------------------------------------
image_id: int64 not null
timestamp: timestamp[us] not null
camera_id: string
class_label: string
cls_vector: fixed_size_list<item: float>[768] not null
  child 0, item: float
patch_tensors: fixed_size_list<item: uint8>[150528] not null
  child 0, item: uint8
quant_min: float not null
quant_scale: float not null
-- schema metadata --
format: 'patchlake'
version: '1.0.0'
patch_count: '196'
embed_dim: '768'
```

### `python ingest.py`

```
Initializing PatchLake Ingestion Pipeline...
Processing 10 synthetic images...
----------------------------------------
Successfully wrote 10 records to test_batch.plake
Time elapsed: 0.1121 seconds
Total patches processed: 1960
```

### `python reader.py`

```
Initializing Zero-Copy Reader...

--- Memory Address Verification ---
Arrow Buffer Pointer:   0x16e107f38
PyTorch Tensor Pointer: 0x16e107f38
SUCCESS: Perfect Zero-Copy Handoff Achieved! No RAM duplicated.

--- Tensor Stats ---
Final Tensor Shape: torch.Size([10, 196, 768])
Final Tensor Dtype: torch.float32
End-to-End Read Latency: 23.87 ms
```

### `python query.py`

```
Initializing Pushdown Query Engine...

Executing Filtered Pushdown SQL: SELECT patches WHERE camera_id = 'CAM_1'
============================================================
Total Rows Scanned in Footers:  100
Matched Candidate Rows:        25 (Selectivity: 25.0%)
Output PyTorch Tensor Shape:    torch.Size([25, 196, 768])
Filtered Execution Latency:     29.20 ms
============================================================
```

### `python duckdb_engine.py`

```
Initializing DuckDB Virtual Table Engine for PatchLake...

============================================================
DEMO 1: Relational Metadata Aggregation
============================================================
SQL Query Result (Native Arrow Dict):
{'camera_id': ['CAM_0', 'CAM_1', 'CAM_2', 'CAM_3'], 'image_count': [25, 25, 25, 25], 'avg_quant_scale': [0.01589482441544533, 0.015805992633104324, 0.016251770444214345, 0.016140993908047675]}

============================================================
DEMO 2: Filtered SQL Pushdown directly to PyTorch
============================================================
Executed SQL: SELECT patch_tensors, quant_min, quant_scale
        FROM patchlake
        WHERE camera_id = 'CAM_2'
Query Latency: 16.50 ms
Output PyTorch Tensor Shape: torch.Size([25, 196, 768])
Output PyTorch Tensor Type:  torch.float32
============================================================
```

### `python benchmark.py`

```
[1/3] Generating 100 synthetic images on disk...
[2/3] Writing PatchLake (.plake) format file...

[3/3] Executing Comparative Benchmarking Suite...
============================================================

--- Memory Address Verification ---
Arrow Buffer Pointer:   0x15804c6b8
PyTorch Tensor Pointer: 0x15804c6b8
SUCCESS: Perfect Zero-Copy Handoff Achieved! No RAM duplicated.

RESULTS: Ingestion Throughput Benchmarks (100 Images)
------------------------------------------------------------
Standard JPEG Pipeline Time: 0.2081 sec | 480.51 img/sec | 94179.09 patches/sec
PatchLake Zero-Copy Time:    0.0304 sec | 3285.48 img/sec | 643953.24 patches/sec
------------------------------------------------------------
★ THROUGHPUT SPEEDUP: PatchLake is 6.84x FASTER than Standard Baseline

STORAGE FOOTPRINT ANALYSIS
------------------------------------------------------------
Raw JPEG Directory Size:   4.78 MB
PatchLake (.plake) Size:   14.66 MB
Uncompressed Tensor Size:  57.42 MB (Reference Raw FP32)
============================================================
```

---

## Benchmarks

### Methodology (`benchmark.py`)

Both arms consume the *same* 100 images:

1. **Baseline** — `JPEGBaselineDataset` globs the JPEG directory and, per batch of 10, runs the
   full CPU dataloader path: `Image.open()` → `convert('RGB')` → resize/normalise → `torch.stack`
   → `Conv2d` patchify → `[B, 196, 768]` float32. This is what a normal ViT `DataLoader` does on
   every epoch.
2. **PatchLake** — `ZeroCopyPatchLakeReader` mmaps `dataset.plake`, and for each record batch
   does the buffer handoff plus the dequantisation expression.

Storage is then compared as the sum of JPEG file sizes against the `.plake` file size, with the
raw float32 tensor footprint printed as a reference.

### Results (100 images, batch size 10)

| | Time | Images/sec | Patches/sec | vs. baseline |
|---|---|---|---|---|
| JPEG baseline | 0.2081 s | 480.51 | 94,179 | 1.00× |
| PatchLake | 0.0304 s | 3,285.48 | 643,953 | **6.84×** |

The speed-up is concentrated exactly where the design intends: JPEG entropy decoding, resize,
normalisation and the 768×16×16 convolution all move to ingest time and never run again.

### Real-world scale (`ingest_coco.py` → `coco_val2017.plake`)

COCO 2017 `val2017`: 5,000 real photographs of varying resolution and aspect ratio, ingested
through the identical code path (`Resize(224,224)` handles the shape variance).

| Metric | Value |
|---|---|
| Images ingested | 5,000 |
| Output file | 732.67 MB (`bench_data/coco/coco_val2017.plake`) |
| Bytes per row | 153,652 |
| Record batches | 1 |
| Source JPEGs for comparison | 786.9 MB (161.1 KB/image average) |
| DuckDB open + register (`read_all` + `create` + `register`) | 617.7 ms |
| `COUNT(*)`/`SUM(quant_scale)` over all 5,000 rows (metadata only) | 2.1 ms |
| `GROUP BY` aggregation over all 5,000 rows | 17.0 ms |
| Filtered pull of 237 rows → `(237, 196, 768)` `float32` | 149.7 ms |

Two things this shows: metadata-only SQL over the whole corpus costs ~2 ms because only the
string/scalar columns are ever read, and pulling 237 images' worth of dequantised tensors costs
~150 ms — ~0.63 ms per image, versus the ~2 ms/image the JPEG path needs just to produce the
patches.

---

## How the query path works

`query.py::PushdownPatchLakeEngine.execute_pushdown_query()` — four explicit stages:

```
.plake ──memory_map──► RecordBatchFileReader
                            │
    ┌───────────────────────┴──────────────────────────────────────────┐
    │ STEP 1  Columnar predicate  (metadata zone scan)                 │
    │         pc.equal(camera_id, 'CAM_1')  →  bit-packed boolean mask  │
    │         np.where(mask)                →  selection vector         │
    │         no match in batch?            →  ZONE PRUNING, skip       │
    ├──────────────────────────────────────────────────────────────────┤
    │ STEP 2  Zero-copy pointer slicing on the survivors                │
    │         patch_array.values.buffers()[1]  →  np.frombuffer         │
    │         view(rows, 196, 768)[selection]  →  torch.from_numpy      │
    ├──────────────────────────────────────────────────────────────────┤
    │ STEP 3  Vectorised RQC dequantisation                            │
    │         uint8 × quant_scale + quant_min  →  float32               │
    ├──────────────────────────────────────────────────────────────────┤
    │ STEP 4  torch.cat over batches  →  one contiguous ViT input batch │
    └──────────────────────────────────────────────────────────────────┘
```

`duckdb_engine.py` exposes the same container as a virtual table so the *engine* does the
filtering instead:

```python
engine = DuckDBPatchLakeEngine("bench_data/dataset.plake")
tensors, latency = engine.query_and_stream_to_pytorch(
    "SELECT patch_tensors, quant_min, quant_scale FROM patchlake WHERE camera_id = 'CAM_2'"
)
# tensors: (25, 196, 768) float32
```

DuckDB never copies the tensor payload either — it hands back an Arrow table whose buffer is
reinterpreted by `np.frombuffer`, and DuckDB's own vectorised executor handles the predicate and
the aggregation. That is what makes "SQL for the metadata, tensors for the GPU, no copies in
between" practical without writing a storage engine.

---

## Reality checks — what these numbers do **not** prove

The honest reading of the results above:

- **It is a pipeline comparison, not a format comparison.** The baseline does work (decode,
  resize, patchify) that PatchLake already did at ingest. The 6.84× measures "materialising the
  patch step removes that work from the training loop", which is the actual hypothesis. It does
  not mean reading a `.plake` file is 6.84× cheaper than reading an equally-prepared file.
- **The images are synthetic, and the storage verdict depends on resolution.** 100
  uniform-random 256×256 frames compress far worse than photographs under JPEG, which is why the
  synthetic run shows `.plake` at 3.1× the JPEG size. On the real COCO set the same comparison
  inverts: 786.9 MB of JPEGs became a 732.7 MB `.plake` file (~7% smaller), simply because every
  image is downsampled to 224×224. So the format is not compressing anything — at low resolution
  it costs more disk, at high resolution it can cost less. Only these two points were measured.
- **No warm-up, single run, CPU only.** Variance was not quantified; expect a few percent run to
  run. Nothing here is a GPU dataloader measurement.
- **`fixed_size_list<float32>` CLS vectors are written but never read by any query path**, and
  there is **no ANN index** despite the schema comment about "high-precision ANN search".
- **Zone pruning is effectively untested.** Every file produced by the writers holds the whole
  table in **one** record batch, so "skip the batch" degenerates to "scan everything or
  everything matched". Row-group sizing per batch is a genuine open problem (see
  [Known limitations](#known-limitations--gotchas)).
- **The dequantisation is not validated numerically.** No test asserts the round-trip error bound
  (`≤ scale/2`).

---

## Known limitations & gotchas

Correctness / design debt a reviewer should know about:

1. **`benchmark.py` deletes `bench_data/`.** `generate_benchmark_data()` calls
   `shutil.rmtree(output_dir)` unconditionally, which also removes `bench_data/coco/` — 2.3 GB
   of downloaded data and the 733 MB artifact. This is why the benchmark above was executed via a
   throw-away runner that passed a different `output_dir`; the code path is otherwise identical.
2. **One record batch per file.** `write_plake_stripe()` builds a single `pa.Table` and writes one
   batch, so 5,000 COCO rows land in one 733 MB batch. Batch-level pruning cannot work, and
   reading is all-or-nothing at the batch granularity.
3. **Ingest is Python-object bound.** Rows are accumulated as Python `dict`s with
   `patch_tensors` produced by `.tolist()` — 150,528 Python ints per row. That dominates ingest
   CPU and memory (the COCO run holds all 5,000 rows in RAM before writing). Building
   `pa.array` straight from the NumPy buffer would be the obvious fix.
4. **The patch-embedding weights are random.** `PatchExtractor.proj` is an unseeded
   `Conv2d(3, 768, 16, 16)`; it simulates the *shape and cost* of a ViT patch embedding, not its
   semantics. The CLS vector is the mean of the patches, not a trained CLS token.
5. **The DuckDB path only reads `chunks[0]`.** `query_and_stream_to_pytorch()` uses
   `patch_column.chunks[0]` and ignores further chunks, so a result DuckDB splits across chunks
   would be truncated/misaligned. The single-batch artifacts hide this.
6. **`196` and `768` are hardcoded** in `reader.py`, `query.py` and `duckdb_engine.py` rather than
   read from the schema metadata, so a file with a different patch count would mis-resolve.
7. **DuckDB registration loads the whole file into RAM** (`reader.read_all()`), measured at 617.7 ms
   for 733 MB. The `memory_map` path in `query.py` does not.
8. **No tests, no CI, no lint, no license file, no `requirements.txt`.** Verification currently
   means reading the printed pointer addresses.
9. **Relative paths.** All entry points assume `cwd == engine/`.
10. **Not a replacement for an image store.** `.plake` holds derived tensors; the JPEGs remain the
    source of truth for anything the patch grid cannot reconstruct.

---

## Artifacts and what is committed

`.plake` files are *build outputs* — fully reproducible from `ingest.py`, `benchmark.py` and
`ingest_coco.py`. The repository is therefore source-only: **10 files, ~1.5 MB**, so cloning is
instant.

```bash
git clone https://github.com/creationsbysonu/research_patchlake.git
cd research_patchlake
```

| Path | Size | In the repo? |
|---|---|---|
| `*.py`, `README.md`, `.gitignore` | ~28 KB | **yes** |
| `test_batch.plake` | 1.5 MB | **yes** — 10 synthetic images, so `python reader.py` works with no setup |
| `bench_data/coco/coco_val2017.plake` | 733 MB | no — rebuild with `ingest_coco.py` |
| `bench_data/coco/val2017.zip` + 5,000 JPEGs | 778 MB + 787 MB | no — downloaded on demand by `ingest_coco.py` |
| `bench_data/dataset.plake`, `bench_data/jpegs/` | 14.7 MB + 5.1 MB | no — rebuild with `benchmark.py` |
| `__pycache__/` | — | no |

GitHub enforces a hard 100 MB per-file limit and a 2 GB push payload limit; `coco_val2017.plake`
(733 MB) and `val2017.zip` (778 MB) exceed both, so `.gitignore` excludes `bench_data/`,
`__pycache__/`, `*.pyc` and `*.zip`. Every figure in this README was produced by actually running
the scripts above, so a reviewer can regenerate the datasets and reproduce the tables end to end.

---

## Roadmap

- Replace the random `Conv2d` with frozen CLIP / DINOv2 / ViT-Base weights (and store model id +
  revision in the schema metadata).
- Row-group a.k.a. record-batch sizing at write time (e.g. 64–256 rows per batch) so zone
  pruning and batch-level prefetch become real; add a min/max statistics block per batch.
- Vectorise ingestion: build `pa.array(np.ndarray)` directly instead of `.tolist()`.
- Fix the multi-chunk bug in the DuckDB path and drop the in-memory `read_all()` in favour of
  `DuckDB`'s Arrow IPC scan.
- ANN index over `cls_vector` (the format already pays 3 KB/row for it).
- Optional LZ4/ZSTD IPC compression for cold tiers; measure the read-time cost on dequant.
- Tests: dequant round-trip bound (`≤ scale/2`), pointer-equality invariant, predicate/selectivity
  correctness against a NumPy oracle, and a multi-batch pruning test.
- Move dequantisation to the GPU and benchmark a real training loop (dataLoader-to-step latency),
  which is the number that actually matters.

---

## Reference

- Format: [Apache Arrow IPC file format](https://arrow.apache.org/docs/format/Columnar.html)
  (`fixed_size_list` layout, buffer/offset semantics).
- DuckDB zero-copy Arrow integration: <https://duckdb.org/docs/guides/python/sql_on_arrow>.
- Quantisation scheme: per-row asymmetric scalar quantisation of 8 bits (SQ8), with the
  reconstruction pair `(quant_min, quant_scale)` stored next to the tensor.
- Real-world data: COCO 2017 `val2017` (<http://images.cocodataset.org/zips/val2017.zip>),
  downloaded on demand and not part of this repository.
