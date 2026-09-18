import time
import torch

class EfficiencyTracker:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.total_tokens = 0
        
    def start(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.start_time = time.perf_counter()
        
    def stop(self, tokens_generated: int):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end_time = time.perf_counter()
        self.total_tokens = tokens_generated
        
    def get_metrics(self):
        latency = self.end_time - self.start_time
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        tokens_per_sec = self.total_tokens / latency if latency > 0 else 0
        
        return {
            "latency_sec": latency,
            "tokens_per_sec": tokens_per_sec,
            "total_tokens": self.total_tokens,
            "peak_vram_mb": peak_vram_mb
        }
