import os
import shutil
import tempfile
import pytest
import torch
import numpy as np

from dsa.optimizer import DiskSparseRiemannianAdam


def test_dsa_euclidean_step():
    """Проверка оптимизатора в евклидовом пространстве (k=0.0)"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        num_entities = 1000
        embedding_dim = 16
        
        initial_embeddings = torch.randn(num_entities, embedding_dim) * 0.05
        
        optimizer = DiskSparseRiemannianAdam(
            params={"entity_emb": initial_embeddings},
            lr=0.05,
            k=0.0, 
            disk_dir=temp_dir,
            max_queue_size=100
        )
        
        initial_w0 = optimizer.state_files["entity_emb"]["w"][0].copy()
        
        batch_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        idx_np = batch_indices.numpy()
        
        for _ in range(5):
            weights_np = optimizer.state_files["entity_emb"]["w"][idx_np].copy()
            current_weights = torch.from_numpy(weights_np).requires_grad_(True)
            
            loss = torch.mean(current_weights ** 2)
            loss.backward()
            
            optimizer.step(updates={"entity_emb": (batch_indices, current_weights.grad.cpu())})
        
        optimizer.shutdown(timeout=2.0)
        
        updated_w0 = optimizer.state_files["entity_emb"]["w"][0].copy()
        assert not np.array_equal(initial_w0, updated_w0), "Веса на диске должны были обновиться!"
        
    finally:

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def test_dsa_hyperbolic_step():
    """Проверка работы в гиперболическом пространстве Пуанкаре (k=1.0)"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        num_entities = 500
        embedding_dim = 8
        
        initial_embeddings = torch.randn(num_entities, embedding_dim) * 0.01
        
        optimizer = DiskSparseRiemannianAdam(
            params={"entity_emb": initial_embeddings},
            lr=0.01,
            k=1.0,
            disk_dir=temp_dir,
            max_queue_size=50
        )
        
        batch_indices = torch.tensor([5, 10, 15], dtype=torch.long)
        idx_np = batch_indices.numpy()
        
        weights_np = optimizer.state_files["entity_emb"]["w"][idx_np].copy()
        current_weights = torch.from_numpy(weights_np).requires_grad_(True)
        
        loss = torch.mean(current_weights ** 2)
        loss.backward()
        
        optimizer.step(updates={"entity_emb": (batch_indices, current_weights.grad.cpu())})
        optimizer.shutdown(timeout=2.0)
        
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
