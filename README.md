# Helical challenge

> [!WARNING]  
> TBD:
> Fix quantization using mtq

## Usage

```python
time python3 main.py
```

## Configuration 

Controlled by `conf/config.yaml`. See below for an example.

```yaml
# ============================================================================
# GLOBAL SETTINGS
# ============================================================================
seed: 42

# ============================================================================
# MODEL CONFIGS
# ============================================================================
model:
  name: "gf-12L-38M-i4096"  # Model identifier
#   'gf-12L-38M-i4096', 'gf-12L-38M-i4096-CLcancer', 'gf-20L-151M-i4096', 'gf-12L-40M-i2048', 'gf-6L-10M-i2048', 'gf-12L-104M-i4096', 'gf-12L-104M-i4096-CLcancer', 'gf-18L-316M-i4096', 'gf-12L-40M-i2048-CZI-CellxGene'
  batch_size: 128              # Base batch size for inference
  max_length: 2048            # Maximum sequence length
  device: 'cuda:0'
  # Alternative models (uncomment to use):
  # name: "geneformer-6L-30M"
  # name: "geneformer-12L-95M"

# ============================================================================
# DATA CONFIGS
# ============================================================================
data:
  pertpy_data_identifier: "norman_2019"
  condition_filter: "control"
  
  # Preprocessing options
  gene_names_column: "index"
  use_raw_counts: false
  preprocess: true
  normalize: true
  filter_genes: true
  min_genes: 200      # Minimum genes per cell
  min_cells: 3        # Minimum cells per gene
  
  # Subset data for testing
  max_cells: 1000   # Limit number of cells for faster testing
  max_genes: 100   # Limit number of genes

# ============================================================================
# PERTURBATION CONFIGS
# ============================================================================
perturbation:
  # Option 1: Explicit gene list (set num_genes to null if using this)
  gene_list: null
  # gene_list: ["BRCA1", "TP53", "MYC", "EGFR", "KRAS"]
  
  # Option 2: Number of top variable genes (set gene_list to null if using this)
  num_genes: 10
  
  # Perturbation settings
  perturbation_type: "knockout"  # Options: knockout, overexpression
  
  # Gene selection method when using num_genes
  selection_method: "top_expressed"  # Options: highly_variable, top_expressed, random
  
  # Additional perturbation options
  perturbation_strength: 2.0  # For overexpression experiments

# ============================================================================
# OPTIMIZATION CONFIGS
# ============================================================================
optimization:
  # Methods to run: batching, quantization, onnx, distributed, all
  methods:
    - all
  
  # Batching optimization
  batching:
    enabled: true
    batch_size_multiplier: 2  # Multiply base batch_size by this factor

  # Quantization optimization
  quantization:
    enabled: true
    dtype: "qint8"           # Options: qint8, float16
    quantize_layers:
      - "torch.nn.Linear"
    dynamic: true            # Use dynamic quantization
  
  # ONNX optimization
  onnx:
    enabled: true
    opset_version: 14
    optimize: true
    use_gpu: true           # Use ONNX GPU runtime if available
    precision: "FP16"
  
  # Distributed optimization
  distributed:
    enabled: true
    backend: "nccl"         # Options: nccl, gloo
    world_size: null        # Auto-detect number of GPUs
    use_data_parallel: true

# ============================================================================
# OUTPUT CONFIGS
# ============================================================================
output:
  dir: "outputs"                    # Base output directory
  save_outputs: true                # Save model outputs (embeddings and perturbations)
  
  # Output format options
  compression: true                 # Options: null, "gzip", "lzf"
  precision: "float32"              # Options: float32, float16
  
  # What to save
  save_embeddings: true
  save_perturbations: true
  save_metadata: true
  save_cell_ids: true
  save_gene_names: true
  
  # Comparison outputs
  save_detailed_comparisons: true   # Save per-gene comparison CSVs
  
  # Logging
  log_level: "INFO"                 # Options: DEBUG, INFO, WARNING, ERROR
  log_to_file: true
  log_file: "pipeline.log"

# ============================================================================
# HARDWARE CONFIGS
# ============================================================================
hardware:
  device: "auto"                    # Options: auto, cuda, cpu, cuda:0, cuda:1, etc.
  mixed_precision: false            # Use automatic mixed precision (AMP)
  cudnn_benchmark: true             # Enable cuDNN benchmarking
  
  # Memory management
  empty_cache_between_methods: true # Clear GPU cache between optimization methods

# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================
experiment:
  name: "geneformer_perturbation"
  tags: []
  notes: ""

# ============================================================================
# REPRODUCIBILITY
# ============================================================================
reproducibility:
  deterministic: true               # Set seed

# ============================================================================
# ADVANCED OPTIONS
# ============================================================================
advanced:
  # Compilation (PyTorch 2.0+)
  compile_model: false
  compile_backend: "inductor"       # Options: inductor, aot_eager, cudagraphs
  
  # Error handling
  continue_on_error: true           # Continue if one optimization method fails

# ============================================================================
# VALIDATION
# ============================================================================
validation:
  # Thresholds for output comparison
  min_correlation: 0.95             # Minimum correlation with baseline
  max_mse: 0.01                     # Maximum MSE with baseline
  warn_on_threshold: true           # Warn if thresholds are exceeded
  
  # Sanity checks
  check_nan: true                   # Check for NaN values in outputs
  check_inf: true                   # Check for Inf values in outputs

```

## Outputs

See `outputs/`. Compares post-perturbation embeddings generated by each optimization method using Pearson's correlation and MSE. Compares performance based on CPU & GPU utilization.

> [!WARNING]  
> ONNX inference is set up using torch-ort but is not currently tested. 
