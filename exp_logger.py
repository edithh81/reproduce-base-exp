"""
Experiment logger for baseline reproducibility.

Logs per-epoch: train time, eval time, recall, ndcg, GPU memory peak, config.
Writes both a structured JSON lines file and a human-readable summary.

Usage:
    from exp_logger import ExpLogger
    logger = ExpLogger(model_name='kucnet', dataset='last-fm', config=vars(args),
                       log_dir='../logs/kucnet')
    ...
    logger.log_epoch(epoch, train_time, eval_time, recall, ndcg)
    ...
    logger.finish(best_epoch, best_recall, best_ndcg)
"""

import os
import json
import time
import torch
from datetime import datetime


class ExpLogger:
    def __init__(self, model_name: str, dataset: str, config: dict, log_dir: str):
        self.model_name = model_name
        self.dataset = dataset
        self.config = config
        self.log_dir = log_dir
        self.start_time = time.time()

        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_id = f'{dataset}_{timestamp}'
        self.jsonl_path = os.path.join(log_dir, f'{self.run_id}.jsonl')
        self.summary_path = os.path.join(log_dir, f'{self.run_id}_summary.txt')

        # Write header / config
        self._write_jsonl({
            'type': 'config',
            'model': model_name,
            'dataset': dataset,
            'timestamp': timestamp,
            'config': _serialize_config(config),
            'gpu': _gpu_info(),
        })

        with open(self.summary_path, 'w') as f:
            f.write(f'{"="*70}\n')
            f.write(f' {model_name.upper()} on {dataset}\n')
            f.write(f' Started: {timestamp}\n')
            f.write(f'{"="*70}\n\n')
            f.write(f'Config:\n')
            for k, v in _serialize_config(config).items():
                f.write(f'  {k}: {v}\n')
            f.write(f'\nGPU: {_gpu_info()}\n')
            f.write(f'\n{"Epoch":<8}{"Train(s)":<12}{"Eval(s)":<12}{"Recall@20":<12}{"NDCG@20":<12}{"GPU Peak(MB)":<14}{"Elapsed(s)":<12}\n')
            f.write(f'{"-"*70}\n')

        self.epoch_records = []

    def log_epoch(self, epoch: int, train_time: float, eval_time: float,
                  recall: float, ndcg: float):
        gpu_peak_mb = 0.0
        if torch.cuda.is_available():
            gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            torch.cuda.reset_peak_memory_stats()

        elapsed = time.time() - self.start_time

        record = {
            'type': 'epoch',
            'epoch': epoch,
            'train_time_s': round(train_time, 2),
            'eval_time_s': round(eval_time, 2),
            'recall_20': round(recall, 6),
            'ndcg_20': round(ndcg, 6),
            'gpu_peak_mb': round(gpu_peak_mb, 1),
            'elapsed_s': round(elapsed, 2),
        }
        self._write_jsonl(record)
        self.epoch_records.append(record)

        with open(self.summary_path, 'a') as f:
            f.write(f'{epoch:<8}{train_time:<12.2f}{eval_time:<12.2f}{recall:<12.6f}{ndcg:<12.6f}{gpu_peak_mb:<14.1f}{elapsed:<12.2f}\n')

    def finish(self, best_epoch: int, best_recall: float, best_ndcg: float):
        total_time = time.time() - self.start_time
        total_train = sum(r['train_time_s'] for r in self.epoch_records)
        total_eval = sum(r['eval_time_s'] for r in self.epoch_records)
        max_gpu = max((r['gpu_peak_mb'] for r in self.epoch_records), default=0)

        summary = {
            'type': 'summary',
            'best_epoch': best_epoch,
            'best_recall_20': round(best_recall, 6),
            'best_ndcg_20': round(best_ndcg, 6),
            'total_time_s': round(total_time, 2),
            'total_train_time_s': round(total_train, 2),
            'total_eval_time_s': round(total_eval, 2),
            'gpu_peak_mb': round(max_gpu, 1),
            'n_epochs': len(self.epoch_records),
        }
        self._write_jsonl(summary)

        with open(self.summary_path, 'a') as f:
            f.write(f'\n{"="*70}\n')
            f.write(f' BEST: epoch {best_epoch}  Recall@20={best_recall:.6f}  NDCG@20={best_ndcg:.6f}\n')
            f.write(f' Total time:  {total_time:.2f}s  (train: {total_train:.2f}s  eval: {total_eval:.2f}s)\n')
            f.write(f' GPU peak:    {max_gpu:.1f} MB\n')
            f.write(f' Epochs:      {len(self.epoch_records)}\n')
            f.write(f'{"="*70}\n')

        print(f'\n[ExpLogger] Logs saved to:')
        print(f'  JSONL:   {self.jsonl_path}')
        print(f'  Summary: {self.summary_path}')

    def _write_jsonl(self, record: dict):
        with open(self.jsonl_path, 'a') as f:
            f.write(json.dumps(record) + '\n')


def _gpu_info() -> str:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(torch.cuda.current_device())
        total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024**2)
        return f'{name} ({total:.0f} MB)'
    return 'CPU only'


def _serialize_config(config: dict) -> dict:
    """Make config JSON-serializable."""
    out = {}
    for k, v in config.items():
        try:
            json.dumps(v)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(v)
    return out
