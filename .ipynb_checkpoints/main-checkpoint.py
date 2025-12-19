"""
Hydra-parameterized script for in-silico perturbation using the Helical API and a Geneformer model.
"""
import os
import time
import psutil
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import warnings
from tqdm.auto import trange

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np
import torch
import anndata as ad
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine
import pertpy

# Set datasetdir to avoid re-downloading data
from scanpy import settings
settings.datasetdir = "/data"

# Helical imports
try:
    from helical.models.geneformer.model import Geneformer
    from helical.models.geneformer.geneformer_tokenizer import TranscriptomeTokenizer
    from helical.models.geneformer.geneformer_config import GeneformerConfig
    from helical.models.geneformer.geneformer_utils import pad_tensor_list, gen_attention_mask, get_model_input_size
    from helical.models.scgpt.dataset import Dataset
    HELICAL_AVAILABLE = True
except ImportError:
    HELICAL_AVAILABLE = False
    warnings.warn("Helical library not found. Please install: pip install helical")

# ONNX inference using torch_ort (not working)
try:
    from torch_ort import ORTInferenceModule, OpenVINOProviderOptions
except ImportError:
    pass

# Distributed inference using torch.distributed
try:
    from torch.distributed import init_process_group, destroy_process_group
    import torch.multiprocessing as mp
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False

# Quantization using nvidia-modelopt
import modelopt.torch.quantization as mtq

# Ignore warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ModelOutputs:
    """Container for model outputs."""
    method: str
    embeddings: np.ndarray
    perturbation_results: Dict[str, np.ndarray]
    cell_ids: Optional[List[str]] = None
    gene_names: Optional[List[str]] = None
    
    def save(self, output_dir: Path, compression: Optional[str] = None, precision: str = "float32"):
        """Save outputs to disk."""
        method_dir = output_dir / self.method
        method_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert precision if needed
        embeddings = self.embeddings.astype(precision)
        
        # Save embeddings
        if compression:
            np.savez_compressed(method_dir / "embeddings.npz", embeddings=embeddings)
        else:
            np.save(method_dir / "embeddings.npy", embeddings)
        
        # Save perturbation results
        for gene, perturbed_emb in self.perturbation_results.items():
            perturbed_emb = perturbed_emb.astype(precision)
            if compression:
                np.savez_compressed(
                    method_dir / f"perturbation_{gene}.npz", 
                    perturbation=perturbed_emb
                )
            else:
                np.save(method_dir / f"perturbation_{gene}.npy", perturbed_emb)
        
        # Save metadata
        metadata = {
            'method': self.method,
            'embedding_shape': list(self.embeddings.shape),
            'num_perturbations': len(self.perturbation_results),
            'genes_perturbed': list(self.perturbation_results.keys()),
            'precision': precision,
            'compression': compression,
        }
        
        if self.cell_ids:
            metadata['num_cells'] = len(self.cell_ids)
            pd.Series(self.cell_ids, name='cell_id').to_csv(
                method_dir / "cell_ids.csv", index=False
            )
        
        if self.gene_names:
            pd.Series(self.gene_names, name='gene').to_csv(
                method_dir / "gene_names.csv", index=False
            )
        
        # Save metadata as JSON
        import json
        with open(method_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved outputs for {self.method} to {method_dir}")


@dataclass
class ComparisonMetrics:
    """Container for comparison metrics between methods."""
    method_a: str
    method_b: str
    
    # Embedding comparison
    embedding_pearson: float
    embedding_spearman: float
    embedding_cosine_similarity: float
    embedding_mse: float
    embedding_max_abs_diff: float
    
    # Per-gene perturbation comparison
    perturbation_correlations: Dict[str, float]
    perturbation_mse: Dict[str, float]
    
    # Summary statistics
    mean_perturbation_correlation: float
    min_perturbation_correlation: float
    max_perturbation_correlation: float
    
    timestamp: str


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    method: str
    runtime_seconds: float
    cpu_percent: float
    memory_mb: float
    gpu_memory_mb: Optional[float]
    gpu_utilization: Optional[float]
    throughput_cells_per_sec: float
    num_cells: int
    num_genes_perturbed: int
    timestamp: str
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


class PerformanceMonitor:
    """Monitor system performance during execution."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.gpu_available = torch.cuda.is_available()
        
    def get_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        metrics = {
            'cpu_percent': self.process.cpu_percent(),
            'memory_mb': self.process.memory_info().rss / 1024 / 1024,
        }
        
        if self.gpu_available:
            try:
                metrics['gpu_memory_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
                # Note: torch.cuda.utilization() may not be available on all systems
                try:
                    metrics['gpu_utilization'] = torch.cuda.utilization()
                except:
                    metrics['gpu_utilization'] = None
            except:
                metrics['gpu_memory_mb'] = None
                metrics['gpu_utilization'] = None
        else:
            metrics['gpu_memory_mb'] = None
            metrics['gpu_utilization'] = None
            
        return metrics


class OutputComparator:
    """Compare outputs from different optimization methods."""
    
    @staticmethod
    def compare_embeddings(emb_a: np.ndarray, emb_b: np.ndarray) -> Dict[str, float]:
        """Compare two embedding matrices."""
        # Flatten for correlation
        flat_a = emb_a.flatten()
        flat_b = emb_b.flatten()
        
        # Pearson correlation
        pearson_r, _ = pearsonr(flat_a, flat_b)
        
        # Spearman correlation
        spearman_r, _ = spearmanr(flat_a, flat_b)
        
        # Cosine similarity (average across cells)
        cosine_sims = []
        for i in range(emb_a.shape[0]):
            cos_sim = 1 - cosine(emb_a[i], emb_b[i])
            cosine_sims.append(cos_sim)
        avg_cosine_sim = np.mean(cosine_sims)
        
        # MSE
        mse = np.mean((emb_a - emb_b) ** 2)
        
        # Max absolute difference
        max_abs_diff = np.max(np.abs(emb_a - emb_b))
        
        return {
            'pearson': pearson_r,
            'spearman': spearman_r,
            'cosine_similarity': avg_cosine_sim,
            'mse': mse,
            'max_abs_diff': max_abs_diff,
        }
    
    @staticmethod
    def compare_perturbations(
        pert_a: Dict[str, np.ndarray], 
        pert_b: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compare perturbation results."""
        correlations = {}
        mse_values = {}
        
        common_genes = set(pert_a.keys()) & set(pert_b.keys())
        
        for gene in common_genes:
            emb_a = pert_a[gene]
            emb_b = pert_b[gene]
            
            # Correlation
            flat_a = emb_a.flatten()
            flat_b = emb_b.flatten()
            corr, _ = pearsonr(flat_a, flat_b)
            correlations[gene] = corr
            
            # MSE
            mse = np.mean((emb_a - emb_b) ** 2)
            mse_values[gene] = mse
        
        return correlations, mse_values
    
    @classmethod
    def compare_outputs(
        cls, 
        output_a: ModelOutputs, 
        output_b: ModelOutputs
    ) -> ComparisonMetrics:
        """Compare two ModelOutputs objects."""
        logger.info(f"Comparing {output_a.method} vs {output_b.method}")
        
        # Compare embeddings
        emb_metrics = cls.compare_embeddings(output_a.embeddings, output_b.embeddings)
        
        # Compare perturbations
        pert_corr, pert_mse = cls.compare_perturbations(
            output_a.perturbation_results,
            output_b.perturbation_results
        )
        
        # Summary statistics
        corr_values = list(pert_corr.values())
        mean_corr = np.mean(corr_values) if corr_values else 0.0
        min_corr = np.min(corr_values) if corr_values else 0.0
        max_corr = np.max(corr_values) if corr_values else 0.0
        
        return ComparisonMetrics(
            method_a=output_a.method,
            method_b=output_b.method,
            embedding_pearson=emb_metrics['pearson'],
            embedding_spearman=emb_metrics['spearman'],
            embedding_cosine_similarity=emb_metrics['cosine_similarity'],
            embedding_mse=emb_metrics['mse'],
            embedding_max_abs_diff=emb_metrics['max_abs_diff'],
            perturbation_correlations=pert_corr,
            perturbation_mse=pert_mse,
            mean_perturbation_correlation=mean_corr,
            min_perturbation_correlation=min_corr,
            max_perturbation_correlation=max_corr,
            timestamp=datetime.now().isoformat()
        )


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class GeneformerPerturbationPipeline:
    """Main pipeline for Geneformer perturbation analysis."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.monitor = PerformanceMonitor()
        
        # Check if Helical is available
        if not HELICAL_AVAILABLE:
            raise ImportError(
                "Helical library is required. Install with: pip install helical"
            )
        
        # Setup device
        if cfg.hardware.device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(cfg.hardware.device)
        
        logger.info(f"Using device: {self.device}")
        
        # Setup reproducibility
        if cfg.reproducibility.deterministic:
            torch.manual_seed(cfg.seed)
            np.random.seed(cfg.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        
        if cfg.hardware.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
        
        # Setup output directory
        self.output_dir = Path(cfg.output.dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging to file
        if cfg.output.log_to_file:
            log_file = self.output_dir / cfg.output.log_file
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
            logger.addHandler(file_handler)
            logger.setLevel(getattr(logging, cfg.output.log_level))
    
    # ------------------------------------------------------------------------
    # Data loading, pre-processing
    # ------------------------------------------------------------------------
    
    def load_model(self) -> Geneformer:
        """Load Geneformer model."""
        logger.info("Loading Geneformer model...")
        
        config = GeneformerConfig(
            model_name=self.cfg.model.name,
            batch_size=self.cfg.model.batch_size,
            device=self.cfg.model.device,
        )
        
        model = Geneformer(configurer=config)
        
        # Compile model if enabled (PyTorch 2.0+)
        if self.cfg.advanced.compile_model:
            try:
                logger.info(f"Compiling model with backend: {self.cfg.advanced.compile_backend}")
                compiled_model = torch.compile(model.model, backend=self.cfg.advanced.compile_backend)
                model.model = compiled_model
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        
        logger.info(f"Model loaded: {self.cfg.model.name}")
        return model
    
    def load_and_tokenize_data(self, model: Geneformer) -> ad.AnnData:
        """Load and tokenize AnnData."""
        logger.info(f"Loading data: {self.cfg.data.pertpy_data_identifier}")
        
        # adata = ad.read_h5ad(self.cfg.data.input_path)
        adata = pertpy.data.norman_2019()
        if self.cfg.data.condition_filter is not None:
            adata=adata[adata.obs.perturbation_name == self.cfg.data.condition_filter]
            
        logger.info(f"Loaded {adata.n_obs} cells, {adata.n_vars} genes")
        
        # Apply data preprocessing if configured
        if self.cfg.data.preprocess:
            logger.info("Preprocessing data...")
            
            if self.cfg.data.filter_genes:
                import scanpy as sc
                sc.pp.filter_cells(adata, min_genes=self.cfg.data.min_genes)
                sc.pp.filter_genes(adata, min_cells=self.cfg.data.min_cells)
                logger.info(f"After filtering: {adata.n_obs} cells, {adata.n_vars} genes")
            
            if self.cfg.data.normalize:
                import scanpy as sc
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
                logger.info("Data normalized")
        
        # Subset data if configured (for testing)
        if hasattr(self.cfg.data, 'max_cells') and self.cfg.data.max_cells:
            n_cells = min(self.cfg.data.max_cells, adata.n_obs)
            adata = adata[:n_cells, :]
            logger.info(f"Subsetted to {n_cells} cells for testing")
        
        if hasattr(self.cfg.data, 'max_genes') and self.cfg.data.max_genes:
            n_genes = min(self.cfg.data.max_genes, adata.n_vars)
            adata = adata[:, :n_genes]
            logger.info(f"Subsetted to {n_genes} genes for testing")
        
        # Tokenize data
        logger.info("Tokenizing data...")
        tokenized_data = model.process_data(
                                            adata, 
                                            gene_names = self.cfg.data.gene_names_column, 
                                            use_raw_counts = bool(self.cfg.data.use_raw_counts)
                                           )
        
        logger.info("Tokenization complete")
        return adata, tokenized_data
    
    def get_genes_to_perturb(self, adata: ad.AnnData) -> List[str]:
        """Get list of genes to perturb based on config."""
        if self.cfg.perturbation.gene_list:
            genes = self.cfg.perturbation.gene_list
            logger.info(f"Using explicit gene list: {genes}")
        elif self.cfg.perturbation.num_genes:
            num_genes = self.cfg.perturbation.num_genes
            selection_method = self.cfg.perturbation.selection_method
            
            if selection_method == "highly_variable":
                if 'highly_variable' in adata.var.columns:
                    genes = adata.var_names[adata.var['highly_variable']][:num_genes].tolist()
                else:
                    logger.warning("No highly_variable column found, using top expressed genes")
                    # Select genes with highest mean expression
                    mean_expr = np.array(adata.X.mean(axis=0)).flatten()
                    top_indices = np.argsort(mean_expr)[-num_genes:]
                    genes = adata.var_names[top_indices].tolist()
            elif selection_method == "top_expressed":
                # Select genes with highest mean expression
                mean_expr = np.array(adata.X.mean(axis=0)).flatten()
                top_indices = np.argsort(mean_expr)[-num_genes:]
                genes = adata.var_names[top_indices].tolist()
            elif selection_method == "random":
                np.random.seed(self.cfg.seed)
                genes = np.random.choice(adata.var_names, num_genes, replace=False).tolist()
            else:
                genes = adata.var_names[:num_genes].tolist()
            
            logger.info(f"Selected {num_genes} genes using method: {selection_method}")
        else:
            raise ValueError("Must specify either gene_list or num_genes in perturbation config")
        
        return genes
    
    # ------------------------------------------------------------------------
    # Inference functions
    # ------------------------------------------------------------------------
    
    def extract_embeddings(self, model: Geneformer, data: Dataset) -> np.ndarray:
        """Extract embeddings from the model."""
        logger.info("Extracting embeddings...")
  
        with torch.no_grad():
            # Use mixed precision if enabled
            if self.cfg.hardware.mixed_precision:
                with torch.cuda.amp.autocast():
                    embeddings = model.get_embeddings(data)
            else:
                embeddings = model.get_embeddings(data)
        logger.info(f"Extracted embeddings shape: {embeddings.shape}")
        return embeddings
    
    def perform_perturbations(
        self, 
        model: Geneformer, 
        adata: ad.AnnData, 
        genes: List[str]
    ) -> Dict:
        """Perform in silico gene perturbation."""
        logger.info(f"Performing perturbation on {len(genes)} genes...")
        
        results = {}
        perturbation_type = self.cfg.perturbation.perturbation_type
        perturbed_data_dict = {}
        for gene in genes:
            logger.info(f"Perturbing gene: {gene} (type: {perturbation_type})")

            perturbed_data = adata.copy()

            if gene in perturbed_data.var_names:
                gene_idx = perturbed_data.var_names.get_loc(gene)

                if perturbation_type == "knockout":
                    # Zero out gene expression
                    perturbed_data.X[:, gene_idx] = 0
                elif perturbation_type == "overexpression":
                    # Increase gene expression (e.g., multiply by factor)
                    strength = getattr(self.cfg.perturbation, 'perturbation_strength', 2.0)
                    perturbed_data.X[:, gene_idx] *= strength
                   
                perturbed_data_dict[gene] = model.process_data(
                                                               perturbed_data, 
                                                               use_raw_counts = bool(self.cfg.data.use_raw_counts)
                                                              )

            else:
                logger.warning(f"Gene {gene} not found in data, skipping")
        
        logger.info("Perturbations complete.")
        return perturbed_data_dict
    
    def get_perturbation_results(
        self, 
        model: Geneformer, 
        perturbed_data_dict: Dict, 
        genes: List[str]
    ) -> Dict[str, np.ndarray]:
        """Extract post-perturbation embeddings"""
        logger.info(f"Extracting embeddings for perturbations on {len(genes)} genes...")
        
        results = {}
        perturbation_type = self.cfg.perturbation.perturbation_type
        
        for gene in genes:
            if gene in perturbed_data_dict.keys():
                logger.info(f"Extraxcting embeddings for perturbed: {gene} (type: {perturbation_type})")

                perturbed_emb = self.extract_embeddings(model, perturbed_data_dict[gene])
                results[gene] = perturbed_emb
            else:
                logger.warning(f"Gene {gene} not found in perturbed data, skipping")
        
        logger.info("Finished getting perturbation results.")
        return results
    
    # ------------------------------------------------------------------------
    # Optimization Methods
    # ------------------------------------------------------------------------
    
    def run_baseline(
        self, 
        model: Geneformer, 
        data: Dataset,
        adata: ad.AnnData, 
        genes: List[str],
        perturbed_data_dict: Dict
    ) -> Tuple[PerformanceMetrics, ModelOutputs]:
        """Run baseline inference without optimization."""
        logger.info("Running BASELINE (no optimization)...")
        
        start_time = time.time()
        start_metrics = self.monitor.get_metrics()
        
        # Extract embeddings
        embeddings = self.extract_embeddings(model, data)
        
        # Perform perturbations
        # DEPRECATED
        # perturbation_results = self.perform_perturbation(model, adata, genes)
        perturbation_results = self.get_perturbation_results(model, perturbed_data_dict, genes)
        
        end_time = time.time()
        end_metrics = self.monitor.get_metrics()
        
        runtime = end_time - start_time
        
        perf_metrics = PerformanceMetrics(
            method='baseline',
            runtime_seconds=runtime,
            cpu_percent=end_metrics['cpu_percent'],
            memory_mb=end_metrics['memory_mb'],
            gpu_memory_mb=end_metrics['gpu_memory_mb'],
            gpu_utilization=end_metrics['gpu_utilization'],
            throughput_cells_per_sec=adata.n_obs / runtime if runtime > 0 else 0,
            num_cells=adata.n_obs,
            num_genes_perturbed=len(genes),
            timestamp=datetime.now().isoformat(),
            additional_metrics={'optimization': 'none'}
        )
        
        outputs = ModelOutputs(
            method='baseline',
            embeddings=embeddings,
            perturbation_results=perturbation_results,
            cell_ids=adata.obs_names.tolist() if hasattr(data, 'obs_names') else None,
            gene_names=genes
        )
        
        return perf_metrics, outputs
    
    def run_with_batching(
        self, 
        model: Geneformer, 
        data: Dataset,
        adata: ad.AnnData, 
        genes: List[str],
        perturbed_data_dict: Dict
    ) -> Tuple[PerformanceMetrics, ModelOutputs]:
        """Run inference with optimized batching."""
        logger.info("Running with BATCHING optimization...")
        
        if not self.cfg.optimization.batching.enabled:
            logger.warning("Batching is disabled in config")
            return None, None
        
        # Get optimized batch size
        original_batch_size = self.cfg.model.batch_size
        multiplier = self.cfg.optimization.batching.batch_size_multiplier
        optimized_batch_size = int(original_batch_size * multiplier)
        
        logger.info(f"Using batch size: {optimized_batch_size} (original: {original_batch_size})")
        
        start_time = time.time()
        start_metrics = self.monitor.get_metrics()
        
        # Temporarily modify batch size
        old_batch_size = self.cfg.model.batch_size
        model.forward_batch_size = optimized_batch_size
        
        # Extract embeddings
        embeddings = self.extract_embeddings(model, data)
        
        # Perform perturbations
        # DEPRECATED
        # perturbation_results = self.perform_perturbation(model, adata, genes)
        # Get perturbation results
        perturbation_results = self.get_perturbation_results(model, perturbed_data_dict, genes)
                                                             
        # Restore batch size
        model.forward_batch_size = old_batch_size
        
        end_time = time.time()
        end_metrics = self.monitor.get_metrics()
        
        runtime = end_time - start_time
        
        perf_metrics = PerformanceMetrics(
            method='batching',
            runtime_seconds=runtime,
            cpu_percent=end_metrics['cpu_percent'],
            memory_mb=end_metrics['memory_mb'],
            gpu_memory_mb=end_metrics['gpu_memory_mb'],
            gpu_utilization=end_metrics['gpu_utilization'],
            throughput_cells_per_sec=adata.n_obs / runtime if runtime > 0 else 0,
            num_cells=adata.n_obs,
            num_genes_perturbed=len(genes),
            timestamp=datetime.now().isoformat(),
            additional_metrics={
                'batch_size': optimized_batch_size,
                'original_batch_size': original_batch_size,
                'multiplier': multiplier
            }
        )
        
        outputs = ModelOutputs(
            method='batching',
            embeddings=embeddings,
            perturbation_results=perturbation_results,
            cell_ids=adata.obs_names.tolist() if hasattr(data, 'obs_names') else None,
            gene_names=genes
        )
        
        return perf_metrics, outputs
    
    def run_with_quantization(
        self, 
        model: Geneformer, 
        data: Dataset,
        adata: ad.AnnData, 
        genes: List[str],
        perturbed_data_dict: Dict
    ) -> Tuple[PerformanceMetrics, ModelOutputs]:
        """Run inference with model quantization."""
        logger.info("Running with QUANTIZATION optimization...")
        
        if not self.cfg.optimization.quantization.enabled:
            logger.warning("Quantization is disabled in config")
            return None, None
        
        # Get quantization settings
        dtype_str = self.cfg.optimization.quantization.dtype
        
        if dtype_str == "qint8" or dtype_str == "fp8":
            
            qconf = {"qint8": mtq.INT8_SMOOTHQUANT_CFG,
                     "fp8": mtq.FP8_DEFAULT_CFG}
            
            logger.info("Applying QINT8 quantization using nvidia-modelopt...")
            
            MODEL_INPUT_SIZE = get_model_input_size(model.model)
            PAD_TOKEN_ID = model.tk.gene_token_dict["<pad>"]
            
            # Calibration for [quantize()]
            def calib_loop(model):
                silent = False
                device = self.cfg.model.device
                subset = data.select(range(self.cfg.optimization.quantization.calibration_data_size))
                total_batch_length = len(subset)
                forward_batch_size = self.cfg.model.batch_size
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
                    for i in trange(0, total_batch_length, forward_batch_size, leave=(not silent)):
                        max_range = min(i + forward_batch_size, total_batch_length)

                        minibatch = subset.select([i for i in range(i, max_range)])

                        minibatch.set_format(type="torch", device=device)
                        lengths = minibatch["length"]
                        max_len = int(max(lengths))

                        input_data_minibatch = minibatch["input_ids"]

                        input_data_minibatch = pad_tensor_list(
                            input_data_minibatch, max_len, PAD_TOKEN_ID, MODEL_INPUT_SIZE
                        ).to(device)

                        model(
                            input_ids=input_data_minibatch,
                            attention_mask=gen_attention_mask(minibatch),
                            output_attentions=False,
                        )
       
            quantized_model = mtq.quantize(model.model, qconf[dtype_str], calib_loop)
            model.model = quantized_model
        elif dtype_str == "float16":
            # Apply half precision
            logger.info("Applying float16 precision...")
            quantized_model = model.model.half()
            model.model = quantized_model
        else:
            logger.warning(f"Unknown dtype: {dtype_str}, using original model")

        start_time = time.time()
        start_metrics = self.monitor.get_metrics()
        
        # Extract embeddings
        embeddings = self.extract_embeddings(model, data)
        
        # Perform perturbations
        # DEPRECATED
        # perturbation_results = self.perform_perturbation(model, adata, genes)
        perturbation_results = self.get_perturbation_results(model, perturbed_data_dict, genes)
        
        end_time = time.time()
        end_metrics = self.monitor.get_metrics()
        
        runtime = end_time - start_time
        
        perf_metrics = PerformanceMetrics(
            method='quantization',
            runtime_seconds=runtime,
            cpu_percent=end_metrics['cpu_percent'],
            memory_mb=end_metrics['memory_mb'],
            gpu_memory_mb=end_metrics['gpu_memory_mb'],
            gpu_utilization=end_metrics['gpu_utilization'],
            throughput_cells_per_sec=adata.n_obs / runtime if runtime > 0 else 0,
            num_cells=adata.n_obs,
            num_genes_perturbed=len(genes),
            timestamp=datetime.now().isoformat(),
            additional_metrics={'quantization_dtype': dtype_str}
        )
        
        outputs = ModelOutputs(
            method='quantization',
            embeddings=embeddings,
            perturbation_results=perturbation_results,
            cell_ids=adata.obs_names.tolist() if hasattr(data, 'obs_names') else None,
            gene_names=genes
        )
        
        return perf_metrics, outputs
    
    def run_with_onnx(
        self, 
        model: Geneformer, 
        data: Dataset,
        adata: ad.AnnData,  
        genes: List[str],
        perturbed_data_dict: Dict
    ) -> Tuple[PerformanceMetrics, ModelOutputs]:
        """Run inference with ONNX export."""
        if not ONNX_AVAILABLE:
            logger.error("ONNX not available. Install with: pip install onnx onnxruntime")
            return None, None
        
        if not self.cfg.optimization.onnx.enabled:
            logger.warning("ONNX is disabled in config")
            return None, None
        
        logger.info("Running with ONNX optimization...")
        
        try:
            provider_options = OpenVINOProviderOptions(backend = "GPU" if bool(self.cfg.optimization.onnx.use_gpu) else "CPU", 
                                                       precision = self.cfg.optimization.onnx.precision)
            onnx_model = model.copy()
            onnx_model.model = ORTInferenceModule(onnx_model.model, 
                                                  provider_options = provider_options)
            model = onnx_model

        except Exception as e:
            logger.warning(f"ONNX export failed: {e}. Using original model.")
        
        start_time = time.time()
        start_metrics = self.monitor.get_metrics()
        
        embeddings = self.extract_embeddings(model, data)
        
        # Perform perturbation 
        # DEPRECATED
        # perturbation_results = self.perform_perturbation(model, adata, genes)
        
        perturbation_results = self.get_perturbation_results(model, perturbed_data_dict, genes)
        
        end_time = time.time()
        end_metrics = self.monitor.get_metrics()
        
        runtime = end_time - start_time
        
        perf_metrics = PerformanceMetrics(
            method='onnx',
            runtime_seconds=runtime,
            cpu_percent=end_metrics['cpu_percent'],
            memory_mb=end_metrics['memory_mb'],
            gpu_memory_mb=end_metrics['gpu_memory_mb'],
            gpu_utilization=end_metrics['gpu_utilization'],
            throughput_cells_per_sec=adata.n_obs / runtime if runtime > 0 else 0,
            num_cells=adata.n_obs,
            num_genes_perturbed=len(genes),
            timestamp=datetime.now().isoformat(),
            additional_metrics={
                'onnx_opset': self.cfg.optimization.onnx.opset_version,
                'note': 'Using PyTorch model (full ONNX runtime not implemented)'
            }
        )
        
        outputs = ModelOutputs(
            method='onnx',
            embeddings=embeddings,
            perturbation_results=perturbation_results,
            cell_ids=adata.obs_names.tolist() if hasattr(data, 'obs_names') else None,
            gene_names=genes
        )
        
        return perf_metrics, outputs
    
    def run_with_distributed(
        self, 
        model: Geneformer, 
        data: Dataset,
        adata: ad.AnnData, 
        genes: List[str],
        perturbed_data_dict: Dict
    ) -> Tuple[PerformanceMetrics, ModelOutputs]:
        """Run inference with distributed processing."""
        if not DISTRIBUTED_AVAILABLE:
            logger.error("Distributed processing not available")
            return None, None
        
        if not self.cfg.optimization.distributed.enabled:
            logger.warning("Distributed is disabled in config")
            return None, None
        
        logger.info("Running with DISTRIBUTED optimization...")
        
        num_gpus = torch.cuda.device_count()
        
        if (num_gpus > 1) & bool(self.cfg.optimization.distributed.use_data_parallel):
            
            logger.info(f"Using {num_gpus} GPUs with DataParallel")
            model_to_parallelize = model.model.cpu()
            parallelized_model = torch.nn.DataParallel(model_to_parallelize)
            parallelized_model = parallelized_model.cuda()
            model.model = parallelized_model
            
        else:
            logger.warning("Only 1 GPU available, distributed mode may not provide benefits")
        
        start_time = time.time()
        start_metrics = self.monitor.get_metrics()
        
        # Extract embeddings
        embeddings = self.extract_embeddings(model, data)
        
        # Perform perturbations
        # DEPRECATED
        # perturbation_results = self.perform_perturbation(model, adata, genes)
        
        perturbation_results = self.get_perturbation_results(model, perturbed_data_dict, genes)
        
        end_time = time.time()
        end_metrics = self.monitor.get_metrics()
        
        runtime = end_time - start_time
        
        perf_metrics = PerformanceMetrics(
            method='distributed',
            runtime_seconds=runtime,
            cpu_percent=end_metrics['cpu_percent'],
            memory_mb=end_metrics['memory_mb'],
            gpu_memory_mb=end_metrics['gpu_memory_mb'],
            gpu_utilization=end_metrics['gpu_utilization'],
            throughput_cells_per_sec=adata.n_obs / runtime if runtime > 0 else 0,
            num_cells=adata.n_obs,
            num_genes_perturbed=len(genes),
            timestamp=datetime.now().isoformat(),
            additional_metrics={
                'num_gpus': num_gpus,
                'backend': self.cfg.optimization.distributed.backend
            }
        )
        
        outputs = ModelOutputs(
            method='distributed',
            embeddings=embeddings,
            perturbation_results=perturbation_results,
            cell_ids=adata.obs_names.tolist() if hasattr(data, 'obs_names') else None,
            gene_names=genes
        )
        
        return perf_metrics, outputs
    
    # ------------------------------------------------------------------------
    # Validation and Output
    # ------------------------------------------------------------------------
    
    def validate_outputs(self, outputs: ModelOutputs) -> bool:
        """Validate outputs against configured thresholds."""
        if not self.cfg.validation.check_nan and not self.cfg.validation.check_inf:
            return True
        
        valid = True
        
        # Check embeddings
        if self.cfg.validation.check_nan:
            if np.isnan(outputs.embeddings).any():
                logger.error(f"NaN values found in {outputs.method} embeddings")
                valid = False
        
        if self.cfg.validation.check_inf:
            if np.isinf(outputs.embeddings).any():
                logger.error(f"Inf values found in {outputs.method} embeddings")
                valid = False
        
        # Check perturbations
        for gene, perturbed_emb in outputs.perturbation_results.items():
            if self.cfg.validation.check_nan and np.isnan(perturbed_emb).any():
                logger.error(f"NaN values found in {outputs.method} perturbation for {gene}")
                valid = False
            
            if self.cfg.validation.check_inf and np.isinf(perturbed_emb).any():
                logger.error(f"Inf values found in {outputs.method} perturbation for {gene}")
                valid = False
        
        return valid
    
    def save_performance_metrics(self, metrics_list: List[PerformanceMetrics]) -> pd.DataFrame:
        """Save performance metrics to CSV."""
        # Convert to DataFrame
        records = []
        for metrics in metrics_list:
            record = {
                'method': metrics.method,
                'runtime_seconds': metrics.runtime_seconds,
                'cpu_percent': metrics.cpu_percent,
                'memory_mb': metrics.memory_mb,
                'gpu_memory_mb': metrics.gpu_memory_mb,
                'gpu_utilization': metrics.gpu_utilization,
                'throughput_cells_per_sec': metrics.throughput_cells_per_sec,
                'num_cells': metrics.num_cells,
                'num_genes_perturbed': metrics.num_genes_perturbed,
                'timestamp': metrics.timestamp,
            }
            # Add additional metrics
            for key, value in metrics.additional_metrics.items():
                record[f'extra_{key}'] = value
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Save to CSV
        output_path = self.output_dir / f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Performance metrics saved to: {output_path}")
        
        return df
    
    def save_comparison_metrics(self, comparison_list: List[ComparisonMetrics]) -> pd.DataFrame:
        """Save comparison metrics to CSV."""
        # Convert to DataFrame
        records = []
        for comp in comparison_list:
            record = {
                'method_a': comp.method_a,
                'method_b': comp.method_b,
                'embedding_pearson': comp.embedding_pearson,
                'embedding_spearman': comp.embedding_spearman,
                'embedding_cosine_similarity': comp.embedding_cosine_similarity,
                'embedding_mse': comp.embedding_mse,
                'embedding_max_abs_diff': comp.embedding_max_abs_diff,
                'mean_perturbation_correlation': comp.mean_perturbation_correlation,
                'min_perturbation_correlation': comp.min_perturbation_correlation,
                'max_perturbation_correlation': comp.max_perturbation_correlation,
                'timestamp': comp.timestamp,
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Save to CSV
        output_path = self.output_dir / f"comparison_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Comparison metrics saved to: {output_path}")
        
        # Save detailed per-gene comparisons if enabled
        if self.cfg.output.save_detailed_comparisons:
            for comp in comparison_list:
                gene_records = []
                for gene in comp.perturbation_correlations.keys():
                    gene_records.append({
                        'gene': gene,
                        'correlation': comp.perturbation_correlations[gene],
                        'mse': comp.perturbation_mse[gene],
                    })
                
                if gene_records:
                    gene_df = pd.DataFrame(gene_records)
                    gene_output_path = self.output_dir / f"gene_comparison_{comp.method_a}_vs_{comp.method_b}.csv"
                    gene_df.to_csv(gene_output_path, index=False)
                    logger.info(f"Per-gene comparison saved to: {gene_output_path}")
        
        return df
    
    def print_summary(self, perf_df: pd.DataFrame, comp_df: pd.DataFrame):
        """Print comprehensive summary."""
        logger.info("\n" + "="*100)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("="*100)
        logger.info(f"\n{perf_df.to_string()}\n")
        
        # Calculate speedup relative to baseline
        if 'baseline' in perf_df['method'].values:
            baseline_time = perf_df[perf_df['method'] == 'baseline']['runtime_seconds'].values[0]
            logger.info("\nSPEEDUP RELATIVE TO BASELINE:")
            logger.info("-" * 50)
            for _, row in perf_df.iterrows():
                if row['method'] != 'baseline':
                    speedup = baseline_time / row['runtime_seconds'] if row['runtime_seconds'] > 0 else 0
                    logger.info(f"{row['method']:20s}: {speedup:.2f}x")
        
        if not comp_df.empty:
            logger.info("\n" + "="*100)
            logger.info("OUTPUT COMPARISON SUMMARY")
            logger.info("="*100)
            logger.info(f"\n{comp_df.to_string()}\n")
            
            # Highlight any concerning differences
            logger.info("\nOUTPUT QUALITY ASSESSMENT:")
            logger.info("-" * 50)
            for _, row in comp_df.iterrows():
                if row['method_b'] != 'baseline':
                    continue
                
                quality = "EXCELLENT"
                if row['embedding_pearson'] < 0.99:
                    quality = "GOOD"
                if row['embedding_pearson'] < 0.95:
                    quality = "FAIR"
                if row['embedding_pearson'] < 0.90:
                    quality = "POOR"
                
                logger.info(f"{row['method_a']:20s}: {quality} (r={row['embedding_pearson']:.4f})")
        
        logger.info("="*100)
    
    # ------------------------------------------------------------------------
    # Main Execution
    # ------------------------------------------------------------------------
    
    def run(self):
        """Main execution method."""
        logger.info("="*100)
        logger.info("GENEFORMER PERTURBATION PIPELINE")
        logger.info("="*100)
        logger.info(f"Experiment: {self.cfg.experiment.name}")
        if self.cfg.experiment.tags:
            logger.info(f"Tags: {', '.join(self.cfg.experiment.tags)}")
        if self.cfg.experiment.notes:
            logger.info(f"Notes: {self.cfg.experiment.notes}")
        
        logger.info(f"\nConfiguration:\n{OmegaConf.to_yaml(self.cfg)}")
        
        # Load model
        model = self.load_model()
        
        # Load and tokenize data
        adata, data = self.load_and_tokenize_data(model)
        
        # Get genes to perturb
        genes = self.get_genes_to_perturb(adata)
        
        # Get perturbed and tokenized data (once!)
        perturbed_data_dict = self.perform_perturbations(model, adata, genes)
        
        # Storage for results
        perf_metrics_list = []
        outputs_dict = {}
        
        # Always run baseline first
        logger.info("\n" + "="*100)
        try:
            perf_metrics, outputs = self.run_baseline(model, data, 
                                                      adata, genes,
                                                      perturbed_data_dict)
            
            # Validate outputs
            if self.validate_outputs(outputs):
                perf_metrics_list.append(perf_metrics)
                outputs_dict['baseline'] = outputs
                
                # Save baseline outputs
                if self.cfg.output.save_outputs:
                    outputs.save(
                        self.output_dir / "outputs",
                        compression=self.cfg.output.compression,
                        precision=self.cfg.output.precision
                    )
            else:
                logger.error("Baseline validation failed")
                if not self.cfg.advanced.continue_on_error:
                    return
        except Exception as e:
            logger.error(f"Baseline failed: {e}", exc_info=True)
            if not self.cfg.advanced.continue_on_error:
                return
        
        # Clear cache if configured
        if self.cfg.hardware.empty_cache_between_methods and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Run optimization methods
        methods = self.cfg.optimization.methods
        
        method_runners = {
            'batching': self.run_with_batching,
            'quantization': self.run_with_quantization,
            # 'onnx': self.run_with_onnx, # Commented out for now to avoid extra compute
            'distributed': self.run_with_distributed,
        }
        
        # Expand 'all' to all available methods
        if 'all' in methods:
            methods = list(method_runners.keys())
        
        for method in methods:
            if method not in method_runners:
                logger.warning(f"Unknown method: {method}")
                continue
            
            logger.info("\n" + "="*100)
            
            try:
                runner = method_runners[method]
                perf_metrics, outputs = runner(model, data, 
                                               adata, genes,
                                               perturbed_data_dict)
                
                if perf_metrics is None or outputs is None:
                    logger.warning(f"{method} returned no results (may be disabled)")
                    continue
                
                # Validate outputs
                if self.validate_outputs(outputs):
                    perf_metrics_list.append(perf_metrics)
                    outputs_dict[method] = outputs
                    
                    if self.cfg.output.save_outputs:
                        outputs.save(
                            self.output_dir / "outputs",
                            compression=self.cfg.output.compression,
                            precision=self.cfg.output.precision
                        )
                else:
                    logger.error(f"{method} validation failed")
                    if not self.cfg.advanced.continue_on_error:
                        break
                
                # Clear cache if configured
                if self.cfg.hardware.empty_cache_between_methods and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"{method} failed: {e}", exc_info=True)
                if not self.cfg.advanced.continue_on_error:
                    break
        
        # Compare all outputs to baseline
        logger.info("\n" + "="*100)
        logger.info("COMPARING OUTPUTS TO BASELINE")
        logger.info("="*100)
        
        comparison_list = []
        baseline_outputs = outputs_dict.get('baseline')
        
        if baseline_outputs:
            for method, outputs in outputs_dict.items():
                if method != 'baseline':
                    comp_metrics = OutputComparator.compare_outputs(baseline_outputs, outputs)
                    comparison_list.append(comp_metrics)
                    
                    # Check thresholds
                    if self.cfg.validation.warn_on_threshold:
                        if comp_metrics.embedding_pearson < self.cfg.validation.min_correlation:
                            logger.warning(
                                f"{method}: Correlation {comp_metrics.embedding_pearson:.4f} "
                                f"below threshold {self.cfg.validation.min_correlation}"
                            )
                        if comp_metrics.embedding_mse > self.cfg.validation.max_mse:
                            logger.warning(
                                f"{method}: MSE {comp_metrics.embedding_mse:.6f} "
                                f"above threshold {self.cfg.validation.max_mse}"
                            )
        
        # Save all metrics
        if perf_metrics_list:
            perf_df = self.save_performance_metrics(perf_metrics_list)
        else:
            logger.error("No methods completed successfully")
            return
        
        if comparison_list:
            comp_df = self.save_comparison_metrics(comparison_list)
        else:
            comp_df = pd.DataFrame()
        
        # Print summary
        self.print_summary(perf_df, comp_df)
        
        logger.info(f"\nAll outputs saved to: {self.output_dir}")
        logger.info("\n" + "="*100)
        logger.info("PIPELINE COMPLETE!")
        logger.info("="*100)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Main entry point."""
    try:
        pipeline = GeneformerPerturbationPipeline(cfg)
        pipeline.run()
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
