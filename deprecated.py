import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from typing import List, Dict, Tuple
import time
from datetime import datetime
import logging
import os
import tempfile

# Helical imports
from helical.models.geneformer.model import Geneformer
from helical.models.scgpt.dataset import Dataset

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------
# Distributed inference wrappers
# ------------------------------------------------------------------------
def setup_ddp(rank: int, world_size: int, backend: str = 'nccl'):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Reduce memory usage
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Clear cache
    torch.cuda.empty_cache()

def cleanup_ddp():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()
    torch.cuda.empty_cache()

def get_perturbation_results_ddp(
    rank: int,
    model: Geneformer, 
    perturbed_data_dict: Dict, 
    genes: List[str]
) -> Dict[str, np.ndarray]:
    """Extract post-perturbation embeddings (DDP)"""
    logger.info(f"Extracting embeddings for perturbations on {len(genes)} genes...")

    # Move model to GPU [rank]
    model.model = model.model.to(rank)
    world_size = dist.get_world_size()

    # Distribute genes across GPUs
    genes_per_gpu = len(genes) // world_size
    start_idx = rank * genes_per_gpu
    end_idx = start_idx + genes_per_gpu if rank < world_size - 1 else len(genes)
    local_genes = genes[start_idx:end_idx]

    local_results = {}

    perturbation_type = self.cfg.perturbation.perturbation_type

    for gene in local_genes:
        if gene in perturbed_data_dict.keys():
            logger.info(f"Extraxcting embeddings for perturbed: {gene} (type: {perturbation_type})")

            perturbed_emb = self.extract_embeddings(model, perturbed_data_dict[gene], rank)
            local_results[gene] = perturbed_emb
        else:
            logger.warning(f"Gene {gene} not found in perturbed data, skipping")

    logger.info("Finished getting perturbation results.")
    return local_results

def run_inference_worker(
    self,
    rank: int,
    world_size: int,
    model: Geneformer,
    data: Dataset,
    perturbed_data_dict: Dict,
    genes: List[str],
    backend: str,
    temp_dir: str
):
    """Worker function for each GPU process."""
    try:
        setup_ddp(rank, world_size, backend)

        logger.info(f"Rank {rank}: Starting inference")

        # Move model to GPU [rank]
        model.model = model.model.to(rank)

        # Extract embeddings
        local_embeddings = self.extract_embeddings(model, data, rank)

        # Perform perturbations
        local_perturbation_results = self.get_perturbation_results_ddp(
            rank, model, perturbed_data_dict, genes
        )

        # Save results to disk instead of shared memory
        result_path = os.path.join(temp_dir, f'rank_{rank}_results.pt')
        torch.save({
            'embeddings': local_embeddings,
            'perturbation_results': local_perturbation_results
        }, result_path)

        logger.info(f"Rank {rank}: Saved results to {result_path}")

        cleanup_ddp()

    except Exception as e:
        logger.error(f"Error in rank {rank}: {str(e)}")
        import traceback
        traceback.print_exc()

        # Save error to file
        error_path = os.path.join(temp_dir, f'rank_{rank}_error.txt')
        with open(error_path, 'w') as f:
            f.write(str(e))
            f.write('\n')
            f.write(traceback.format_exc())