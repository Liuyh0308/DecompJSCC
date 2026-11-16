# -*- coding: utf-8 -*-
"""
分布式训练工具模块

支持三种训练模式：
1. single: 单 GPU 训练
2. dp: DataParallel 多 GPU 训练（单进程）
3. ddp: DistributedDataParallel 多 GPU 训练（多进程，推荐）

使用方式：
- 单卡/DP: python train_xxx.py --dist-mode single/dp --gpu-ids 0,1
- DDP:     torchrun --nproc_per_node=2 train_xxx.py --dist-mode ddp

工程规范：
- 所有分布式相关逻辑集中在本模块
- 训练脚本只需调用本模块的高层接口
- 支持平滑切换不同分布式模式
"""

import os
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from typing import Optional, List, Tuple, Union
from enum import Enum


class DistMode(Enum):
    """分布式训练模式"""
    SINGLE = "single"  # 单 GPU
    DP = "dp"          # DataParallel
    DDP = "ddp"        # DistributedDataParallel


class DistributedContext:
    """
    分布式训练上下文管理器
    
    封装所有分布式训练相关的状态和操作，提供统一接口。
    
    Attributes:
        mode: 分布式模式 (single/dp/ddp)
        device: 当前进程使用的主设备
        local_rank: 本地进程排名（DDP 模式下有效）
        world_size: 总进程数
        is_main: 是否是主进程（只有主进程负责打印、保存等）
        device_ids: DP 模式下使用的 GPU 列表
    """
    
    def __init__(
        self,
        mode: str = "single",
        gpu_ids: Optional[str] = None,
        backend: str = "nccl",
        init_method: str = "env://",
        timeout_minutes: int = 360,
    ):
        """
        初始化分布式上下文
        
        Args:
            mode: 分布式模式，可选 "single", "dp", "ddp"
            gpu_ids: 逗号分隔的 GPU 编号（相对当前可见 GPU），如 "0,1"
            backend: DDP 通信后端，通常用 "nccl"（GPU）或 "gloo"（CPU）
            init_method: DDP 初始化方法，通常用 "env://"
            timeout_minutes: DDP collective watchdog 超时时间，HR 验证较慢时需高于默认 10 分钟
        """
        self.mode = DistMode(mode.lower())
        self.backend = backend
        self.init_method = init_method
        if timeout_minutes <= 0:
            raise ValueError("timeout_minutes must be positive")
        self.timeout_minutes = timeout_minutes
        
        # 解析 GPU IDs
        self._gpu_ids_list = self._parse_gpu_ids(gpu_ids)
        
        # 初始化分布式环境
        self._setup()
    
    def _parse_gpu_ids(self, gpu_ids_str: Optional[str]) -> Optional[List[int]]:
        """解析 GPU ID 字符串"""
        if not gpu_ids_str or not gpu_ids_str.strip():
            return None
        ids = [int(x.strip()) for x in gpu_ids_str.strip().split(",") if x.strip()]
        return ids if ids else None
    
    def _setup(self):
        """根据模式初始化分布式环境"""
        n_gpu_available = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        if self.mode == DistMode.SINGLE:
            self._setup_single(n_gpu_available)
        elif self.mode == DistMode.DP:
            self._setup_dp(n_gpu_available)
        elif self.mode == DistMode.DDP:
            self._setup_ddp(n_gpu_available)
        else:
            raise ValueError(f"未知的分布式模式: {self.mode}")
    
    def _setup_single(self, n_gpu_available: int):
        """单 GPU 模式初始化"""
        self.local_rank = 0
        self.world_size = 1
        self.is_main = True
        
        if n_gpu_available == 0:
            self.device = torch.device("cpu")
            self.device_ids = None
            self.n_gpu = 0
        else:
            # 如果指定了 GPU，用第一个；否则用 cuda:0
            if self._gpu_ids_list:
                gpu_id = self._gpu_ids_list[0]
                self._validate_gpu_id(gpu_id, n_gpu_available)
                self.device = torch.device(f"cuda:{gpu_id}")
                self.device_ids = [gpu_id]
            else:
                self.device = torch.device("cuda:0")
                self.device_ids = [0]
            self.n_gpu = 1
    
    def _setup_dp(self, n_gpu_available: int):
        """DataParallel 模式初始化"""
        self.local_rank = 0
        self.world_size = 1
        self.is_main = True
        
        if n_gpu_available == 0:
            raise RuntimeError("DP 模式需要至少一块 GPU")
        
        if self._gpu_ids_list:
            for gpu_id in self._gpu_ids_list:
                self._validate_gpu_id(gpu_id, n_gpu_available)
            self.device = torch.device(f"cuda:{self._gpu_ids_list[0]}")
            self.device_ids = self._gpu_ids_list
            self.n_gpu = len(self._gpu_ids_list)
        else:
            # 使用所有可见 GPU
            self.device = torch.device("cuda:0")
            self.device_ids = list(range(n_gpu_available))
            self.n_gpu = n_gpu_available
    
    def _setup_ddp(self, n_gpu_available: int):
        """DistributedDataParallel 模式初始化"""
        if n_gpu_available == 0:
            raise RuntimeError("DDP 模式需要至少一块 GPU")
        
        # 从环境变量获取分布式信息（由 torchrun 设置）
        if "LOCAL_RANK" not in os.environ:
            raise RuntimeError(
                "DDP 模式需要通过 torchrun 启动。\n"
                "用法: torchrun --nproc_per_node=N train_xxx.py --dist-mode ddp"
            )
        
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        self.is_main = (rank == 0)
        
        # 如果指定了 GPU IDs，使用映射；否则每个进程用对应的 GPU
        if self._gpu_ids_list:
            if self.local_rank >= len(self._gpu_ids_list):
                raise RuntimeError(
                    f"进程数 ({self.world_size}) 超过指定的 GPU 数 ({len(self._gpu_ids_list)})"
                )
            gpu_id = self._gpu_ids_list[self.local_rank]
            self._validate_gpu_id(gpu_id, n_gpu_available)
        else:
            gpu_id = self.local_rank
            if gpu_id >= n_gpu_available:
                raise RuntimeError(
                    f"进程 {self.local_rank} 尝试使用 GPU {gpu_id}，"
                    f"但只有 {n_gpu_available} 块 GPU 可见"
                )
        
        self.device = torch.device(f"cuda:{gpu_id}")
        self.device_ids = [gpu_id]
        self.n_gpu = 1  # DDP 每个进程只用一块 GPU
        
        # 设置当前进程使用的 GPU
        torch.cuda.set_device(self.device)
        
        # 初始化进程组
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.backend,
                init_method=self.init_method,
                timeout=timedelta(minutes=self.timeout_minutes),
            )
    
    def _validate_gpu_id(self, gpu_id: int, n_gpu_available: int):
        """验证 GPU ID 是否有效"""
        if gpu_id < 0 or gpu_id >= n_gpu_available:
            raise ValueError(
                f"GPU ID {gpu_id} 无效。当前可见 GPU 共 {n_gpu_available} 块，"
                f"有效编号为 0 到 {n_gpu_available - 1}"
            )
    
    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        根据分布式模式包装模型
        
        Args:
            model: 原始模型（应已在正确设备上）
            find_unused_parameters: MoE 等稀疏模型需设为 True，否则 DDP 可能死锁
        
        Returns:
            包装后的模型
        """
        if self.mode == DistMode.SINGLE:
            return model
        
        elif self.mode == DistMode.DP:
            if self.n_gpu > 1:
                model = DataParallel(model, device_ids=self.device_ids)
                self.print_main(f"  [DP] 使用 {self.n_gpu} 块 GPU: {self.device_ids}")
            return model
        
        elif self.mode == DistMode.DDP:
            model = DistributedDataParallel(
                model,
                device_ids=self.device_ids,
                output_device=self.device_ids[0],
                find_unused_parameters=False,
            )
            self.print_main(f"  [DDP] 使用 {self.world_size} 个进程")
            return model
        
        return model
    
    def get_dataloader(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        collate_fn=None,
        pin_memory: bool = True,
        drop_last: bool = False,
        is_train: bool = True,
    ) -> DataLoader:
        """
        创建 DataLoader，DDP 模式下训练集使用 DistributedSampler
        
        Args:
            dataset: 数据集
            batch_size: 批大小（DDP 模式下是每个进程的批大小）
            shuffle: 是否打乱
            num_workers: 数据加载线程数
            collate_fn: 数据整理函数
            pin_memory: 是否锁页内存
            drop_last: 是否丢弃最后不完整的 batch
            is_train: 是否是训练集
                      - True: DDP 模式下使用 DistributedSampler（数据分割到各进程）
                      - False: 验证/测试集，每个进程使用完整数据集，便于正确计算指标
        
        Returns:
            DataLoader
        
        注意：
            验证集设置 is_train=False 很重要！否则每个进程只看到部分数据，
            导致验证指标不准确或不变。
        """
        sampler = None
        
        if self.mode == DistMode.DDP and is_train:
            # 只有训练集使用 DistributedSampler
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=dist.get_rank(),
                shuffle=shuffle,
            )
            shuffle = False  # 使用 sampler 时不能 shuffle
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=drop_last,
        )
    
    def set_epoch(self, dataloader: DataLoader, epoch: int):
        """
        设置 DataLoader 的 epoch（DDP 模式下需要，确保每个 epoch shuffle 不同）
        """
        if self.mode == DistMode.DDP and hasattr(dataloader, 'sampler'):
            sampler = dataloader.sampler
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)
    
    def get_state_dict(self, model: torch.nn.Module) -> dict:
        """
        获取模型的 state_dict，自动处理 DP/DDP 包装
        
        Args:
            model: 模型（可能被 DP/DDP 包装）
        
        Returns:
            不带 "module." 前缀的 state_dict
        """
        if hasattr(model, "module"):
            return model.module.state_dict()
        return model.state_dict()
    
    def load_state_dict(
        self,
        model: torch.nn.Module,
        state_dict: dict,
        strict: bool = True,
    ):
        """
        加载 state_dict，兼容有/无 "module." 前缀
        
        Args:
            model: 模型
            state_dict: 要加载的权重
            strict: 是否严格匹配
        """
        # 获取真正的模型（去掉 DP/DDP 包装）
        target = model.module if hasattr(model, "module") else model
        
        # 检查 state_dict 是否有 "module." 前缀
        has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
        
        if has_module_prefix:
            # 去掉前缀
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        
        target.load_state_dict(state_dict, strict=strict)
    
    def save_checkpoint(
        self,
        state: dict,
        filepath: str,
        only_main: bool = True,
    ):
        """
        保存检查点
        
        Args:
            state: 要保存的状态字典
            filepath: 保存路径
            only_main: 是否只在主进程保存（推荐 True，避免重复保存）
        """
        if only_main and not self.is_main:
            return
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        torch.save(state, filepath)
    
    def barrier(self):
        """同步所有进程（DDP 模式下）"""
        if self.mode == DistMode.DDP and dist.is_initialized():
            dist.barrier()
    
    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
    ) -> torch.Tensor:
        """
        对张量执行 all-reduce 操作（DDP 模式下）
        
        Args:
            tensor: 要归约的张量
            op: 归约操作（默认求和）
        
        Returns:
            归约后的张量
        """
        if self.mode == DistMode.DDP and dist.is_initialized():
            dist.all_reduce(tensor, op=op)
        return tensor
    
    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        """对张量求平均（DDP 模式下跨进程平均）"""
        if self.mode == DistMode.DDP and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor = tensor / self.world_size
        return tensor
    
    def print_main(self, *args, **kwargs):
        """只在主进程打印"""
        if self.is_main:
            print(*args, **kwargs)
    
    def cleanup(self):
        """清理分布式环境"""
        if self.mode == DistMode.DDP and dist.is_initialized():
            dist.destroy_process_group()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
    
    def __repr__(self):
        return (
            f"DistributedContext(\n"
            f"  mode={self.mode.value},\n"
            f"  device={self.device},\n"
            f"  n_gpu={self.n_gpu},\n"
            f"  device_ids={self.device_ids},\n"
            f"  world_size={self.world_size},\n"
            f"  local_rank={self.local_rank},\n"
            f"  is_main={self.is_main}\n"
            f")"
        )


def add_distributed_args(parser):
    """
    为 ArgumentParser 添加分布式训练相关参数
    
    Args:
        parser: argparse.ArgumentParser 实例
    
    Returns:
        修改后的 parser
    """
    group = parser.add_argument_group("分布式训练参数")
    
    group.add_argument(
        "--dist-mode",
        type=str,
        default="single",
        choices=["single", "dp", "ddp"],
        help="分布式模式: single(单卡), dp(DataParallel), ddp(DistributedDataParallel)"
    )
    
    group.add_argument(
        "--gpu-ids",
        type=str,
        default="",
        help="逗号分隔的 GPU 编号（相对当前可见 GPU），如 '0,1'。留空则使用所有可见 GPU"
    )
    
    group.add_argument(
        "--dist-backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo"],
        help="DDP 通信后端（GPU 用 nccl，CPU 用 gloo）"
    )

    group.add_argument(
        "--dist-timeout-minutes",
        type=int,
        default=360,
        help="DDP collective timeout minutes; HR validation can exceed PyTorch's 10-minute default"
    )
    
    return parser


def create_dist_context(args) -> DistributedContext:
    """
    从命令行参数创建分布式上下文
    
    Args:
        args: 解析后的命令行参数（需包含 dist_mode, gpu_ids, dist_backend）
    
    Returns:
        DistributedContext 实例
    """
    return DistributedContext(
        mode=getattr(args, "dist_mode", "single"),
        gpu_ids=getattr(args, "gpu_ids", ""),
        backend=getattr(args, "dist_backend", "nccl"),
        timeout_minutes=getattr(args, "dist_timeout_minutes", 360),
    )


def setup_seed(seed: int, dist_ctx: Optional[DistributedContext] = None):
    """
    设置随机种子，确保可复现性
    
    DDP 模式下每个进程使用不同的种子（seed + rank），避免数据重复
    
    Args:
        seed: 基础随机种子
        dist_ctx: 分布式上下文（可选）
    """
    import random
    import numpy as np
    
    if dist_ctx and dist_ctx.mode == DistMode.DDP:
        # DDP 模式：每个进程用不同种子
        seed = seed + dist_ctx.local_rank
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# 兼容旧代码的函数（逐步废弃）
# =============================================================================

def parse_gpu_ids(gpu_ids_str):
    """
    [兼容函数] 解析 GPU ID 字符串
    建议使用 DistributedContext 替代
    """
    if not gpu_ids_str or not str(gpu_ids_str).strip():
        return None
    ids = [int(x.strip()) for x in str(gpu_ids_str).strip().split(",") if x.strip()]
    return ids if ids else None


def get_device(use_multi_gpu=None, gpu_ids=None):
    """
    [兼容函数] 获取设备信息
    建议使用 DistributedContext 替代
    """
    n_gpu_total = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpu_total == 0:
        return torch.device("cpu"), 0, None
    
    if gpu_ids is not None and len(gpu_ids) > 0:
        for i in gpu_ids:
            if i < 0 or i >= n_gpu_total:
                raise ValueError(
                    f"gpu_ids 包含无效编号 {i}。当前可见 GPU 共 {n_gpu_total} 块，"
                    f"有效编号为 0 到 {n_gpu_total - 1}"
                )
        dev = torch.device(f"cuda:{gpu_ids[0]}")
        n_gpu = len(gpu_ids)
        return dev, n_gpu, gpu_ids
    
    dev = torch.device("cuda:0")
    n_gpu = n_gpu_total
    return dev, n_gpu, None


def wrap_model_multi_gpu(model, device, use_multi_gpu=True, device_ids=None):
    """
    [兼容函数] 使用 DataParallel 包装模型
    建议使用 DistributedContext.wrap_model() 替代
    """
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if device_ids is not None and len(device_ids) > 0:
        n_use = len(device_ids)
    else:
        n_use = n_gpu
    
    if n_use > 1 and use_multi_gpu:
        if device_ids is not None and len(device_ids) > 0:
            model = DataParallel(model, device_ids=device_ids)
            print(f"  [DP] 使用 {len(device_ids)} 块 GPU: {device_ids}")
        else:
            model = DataParallel(model)
            print(f"  [DP] 使用 {n_gpu} 块 GPU")
    return model


def get_state_dict_to_save(model):
    """
    [兼容函数] 获取用于保存的 state_dict
    建议使用 DistributedContext.get_state_dict() 替代
    """
    if hasattr(model, "module"):
        return model.module.state_dict()
    return model.state_dict()


def load_state_dict_compat(model, state_dict, strict=True):
    """
    [兼容函数] 兼容加载 state_dict
    建议使用 DistributedContext.load_state_dict() 替代
    """
    target = model.module if hasattr(model, "module") else model
    if not any(k.startswith("module.") for k in state_dict.keys()):
        target.load_state_dict(state_dict, strict=strict)
        return
    new_sd = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    target.load_state_dict(new_sd, strict=strict)
