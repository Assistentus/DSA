import os
import math
import torch
import numpy as np
from torch.optim.optimizer import Optimizer
from typing import Dict, Tuple, Optional, Union
from .async_writer import AsyncDiskWriter

class DiskSparseRiemannianAdam(Optimizer):
    """
    Disk-based Sparse Riemannian Adam (DSA).
    
    Специализированный оптимизатор для обучения сверхбольших графовых эмбеддингов.
    - Хранит состояния (m, v) и веса (w) на NVMe через np.memmap.
    - Масштабирует использование памяти от размера батча O(B), а не графа O(N).
    - Реализует Riemannian Gradient Scaling и проекцию (Retraction) для шара Пуанкаре.
    - Использует фоновую асинхронную запись для устранения простоев GPU.
    """
    
    def __init__(self, params: Union[list, dict], lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                 weight_decay=0, k=1.0, disk_dir="./disk_cache", max_queue_size=150):
        
        if isinstance(params, dict):
            self.param_names = list(params.keys())
            params_list = list(params.values())
        else:
            params_list = list(params)
            self.param_names = [f"param_{i}" for i in range(len(params_list))]
        
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                        weight_decay=weight_decay, k=k)
        
        super().__init__(params_list, defaults)
        
        self.disk_dir = os.path.abspath(disk_dir)
        os.makedirs(self.disk_dir, exist_ok=True)
        
        self.state_files = {}
        param_shapes = {}
        
        # Инициализация хранилища на диске
        for name, param in zip(self.param_names, self.param_groups[0]['params']):
            shape = param.shape
            param_shapes[name] = shape
            
            # w, m, v файлы
            paths = {
                'w': os.path.join(self.disk_dir, f"{name}_w.dat"),
                'm': os.path.join(self.disk_dir, f"{name}_m.dat"),
                'v': os.path.join(self.disk_dir, f"{name}_v.dat")
            }
            
            for suffix, path in paths.items():
                if not os.path.exists(path):
                    mm = np.memmap(path, dtype='float32', mode='w+', shape=shape)
                    mm[:] = 0.0
                    # Если это веса и они уже есть в тензоре, сохраняем начальное состояние
                    if suffix == 'w':
                        mm[:] = param.detach().cpu().numpy()
                    mm.flush()
                    del mm
            
            self.state_files[name] = {
                'w': np.memmap(paths['w'], dtype='float32', mode='r+', shape=shape),
                'm': np.memmap(paths['m'], dtype='float32', mode='r+', shape=shape),
                'v': np.memmap(paths['v'], dtype='float32', mode='r+', shape=shape)
            }
            
        # Запуск фонового писателя
        self.writer = AsyncDiskWriter(
            memmaps={f"{n}_{s}": self.state_files[n][s] for n in self.param_names for s in ['w', 'm', 'v']},
            max_queue_size=max_queue_size
        )

    @torch.no_grad()
    def step(self, updates: Dict[str, Tuple[torch.Tensor, torch.Tensor]], 
             current_k: Optional[Union[float, torch.Tensor]] = None):
        """
        updates: словарь {'param_name': (indices_tensor, grads_tensor)}
        """
        # Глобальный счетчик шагов хранится в первом параметре для совместимости
        param_base = self.param_groups[0]['params'][0]
        state = self.state[param_base]
        if len(state) == 0:
            state['step'] = 0
        state['step'] += 1
        
        group = self.param_groups[0]
        lr = group['lr']
        beta1, beta2 = group['beta1'], group['beta2']
        eps = group['eps']
        weight_decay = group['weight_decay']
        
        bc1 = 1.0 - beta1 ** state['step']
        bc2 = 1.0 - beta2 ** state['step']
        
        k = current_k.item() if isinstance(current_k, torch.Tensor) else (current_k if current_k is not None else group['k'])

        for name, (indices, grads) in updates.items():
            if name not in self.state_files or indices.numel() == 0:
                continue
            
            # 1. Агрегация градиентов для дублирующихся индексов в батче
            if indices.numel() != torch.unique(indices).numel():
                unique_idx, inv = torch.unique(indices, return_inverse=True)
                agg_grads = torch.zeros((len(unique_idx), grads.shape[1]), device=grads.device, dtype=grads.dtype)
                agg_grads.index_add_(0, inv, grads)
                active_indices, active_grads = unique_idx, agg_grads
            else:
                active_indices, active_grads = indices, grads

            idx_np = active_indices.cpu().numpy()
            states = self.state_files[name]
            
            # 2. Чтение из mmap в оперативную память (только нужные строки)
            w = torch.from_numpy(states['w'][idx_np].copy())
            m = torch.from_numpy(states['m'][idx_np].copy())
            v = torch.from_numpy(states['v'][idx_np].copy())
            g_euc = active_grads.cpu().float()

            # 3. Riemannian Gradient Scaling (Poincaré Ball)
            if k > 1e-9:
                w_sq_norm = torch.sum(w * w, dim=-1, keepdim=True)
                max_radius_sq = (1.0 - 1e-5) / k
                w_sq_norm = torch.clamp(w_sq_norm, max=max_radius_sq)
                rescale_factor = ((1.0 - k * w_sq_norm) ** 2) / 4.0
                g_riem = g_euc * rescale_factor
            else:
                g_riem = g_euc

            # 4. Adam Step
            if weight_decay > 0:
                w.mul_(1.0 - lr * weight_decay)

            m.mul_(beta1).add_(g_riem, alpha=1.0 - beta1)
            v.mul_(beta2).addcmul_(g_riem, g_riem, value=1.0 - beta2)

            m_hat = m / bc1
            v_hat = v / bc2
            step_size = lr * m_hat / (torch.sqrt(v_hat) + eps)
            w.sub_(step_size)

            # 5. Retraction (удержание на многообразии)
            if k > 1e-9:
                w_norm = torch.norm(w, p=2, dim=-1, keepdim=True)
                max_radius = math.sqrt((1.0 - 1e-5) / k)
                w = torch.where(w_norm > max_radius, w / w_norm * max_radius, w)

            # 6. Передача на асинхронную фоновую запись
            self.writer.put_update(name, active_indices, w, m, v)

    def shutdown(self):
        """Вызывать в конце обучения для финализации записи на диск."""
        if hasattr(self, 'writer'):
            self.writer.shutdown()
            
    def zero_grad(self, set_to_none: bool = False):
        """Заглушка для совместимости, так как мы работаем с явными градиентами."""
        pass
