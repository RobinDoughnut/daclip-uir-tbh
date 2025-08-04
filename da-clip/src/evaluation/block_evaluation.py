import torch
import torch.nn as nn
import time
import psutil
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

@dataclass
class BlockMetrics:
    name: str
    accuracy: float
    inference_time: float
    memory_usage: float
    parameter_count: int
    flops: float
    attention_pattern: str

class BlockEvaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.metrics = []
        
    def evaluate_block(self, block: nn.Module, test_data: torch.Tensor, 
                      block_name: str) -> BlockMetrics:
        """Evaluate a single attention block"""
        block = block.to(self.device)
        block.eval()
        
        # Measure parameters
        param_count = sum(p.numel() for p in block.parameters())
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):  # Warm up
                _ = block(test_data)
            
            torch.cuda.synchronize() if self.device == 'cuda' else None
            start_time = time.time()
            for _ in range(100):  # Measure
                output = block(test_data)
            torch.cuda.synchronize() if self.device == 'cuda' else None
            end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        
        # Measure memory usage
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            _ = block(test_data)
            memory_usage = torch.cuda.max_memory_allocated() / 1024**2  # MB
        else:
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024**2
            _ = block(test_data)
            memory_after = process.memory_info().rss / 1024**2
            memory_usage = memory_after - memory_before
        
        # Calculate FLOPs (simplified)
        input_size = test_data.numel()
        # Rough estimate: attention is O(n²), MLP is O(n)
        flops = input_size * input_size + input_size * 4  # Simplified
        
        # Determine attention pattern
        if hasattr(block, 'sparse_attention'):
            attention_pattern = "Sparse"
        elif hasattr(block, 'linear_attention'):
            attention_pattern = "Linear"
        elif hasattr(block, 'alpha'):  # DeepNorm
            attention_pattern = "DeepNorm"
        elif hasattr(block, 'attn') and isinstance(block.attn, nn.MultiheadAttention):
            attention_pattern = "Standard"
        else:
            attention_pattern = "Custom"
        
        return BlockMetrics(
            name=block_name,
            accuracy=0.0,  # Would need task-specific evaluation
            inference_time=avg_time * 1000,  # Convert to ms
            memory_usage=memory_usage,
            parameter_count=param_count,
            flops=flops,
            attention_pattern=attention_pattern
        )
    
    def compare_blocks(self, blocks: Dict[str, nn.Module], 
                      test_data: torch.Tensor) -> List[BlockMetrics]:
        """Compare multiple attention blocks"""
        results = []
        
        for name, block in blocks.items():
            print(f"Evaluating {name}...")
            metrics = self.evaluate_block(block, test_data, name)
            results.append(metrics)
            self.metrics.append(metrics)
        
        return results
    
    def generate_report(self, save_path: str = "block_evaluation_report.html"):
        """Generate comprehensive evaluation report"""
        if not self.metrics:
            print("No metrics to report!")
            return
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Inference time comparison
        names = [m.name for m in self.metrics]
        times = [m.inference_time for m in self.metrics]
        axes[0, 0].bar(names, times)
        axes[0, 0].set_title('Inference Time (ms)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Memory usage comparison
        memory = [m.memory_usage for m in self.metrics]
        axes[0, 1].bar(names, memory)
        axes[0, 1].set_title('Memory Usage (MB)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Parameter count comparison
        params = [m.parameter_count for m in self.metrics]
        axes[1, 0].bar(names, params)
        axes[1, 0].set_title('Parameter Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # FLOPs comparison
        flops = [m.flops for m in self.metrics]
        axes[1, 1].bar(names, flops)
        axes[1, 1].set_title('FLOPs (estimated)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('block_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed results
        print("\n" + "="*80)
        print("RESIDUAL ATTENTION BLOCK EVALUATION REPORT")
        print("="*80)
        
        for metric in self.metrics:
            print(f"\n{metric.name}:")
            print(f"  Attention Pattern: {metric.attention_pattern}")
            print(f"  Inference Time: {metric.inference_time:.2f} ms")
            print(f"  Memory Usage: {metric.memory_usage:.2f} MB")
            print(f"  Parameters: {metric.parameter_count:,}")
            print(f"  FLOPs: {metric.flops:,.0f}")
        
        # Find best performing block
        best_time = min(self.metrics, key=lambda x: x.inference_time)
        best_memory = min(self.metrics, key=lambda x: x.memory_usage)
        
        print(f"\n🏆 Best Performance:")
        print(f"  Fastest: {best_time.name} ({best_time.inference_time:.2f} ms)")
        print(f"  Most Memory Efficient: {best_memory.name} ({best_memory.memory_usage:.2f} MB)")

def run_block_evaluation():
    """Main evaluation function"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create test data
    batch_size, seq_len, d_model = 8, 196, 768  # Typical for ViT
    test_data = torch.randn(batch_size, seq_len, d_model).to(device)
    
    # Create simple test blocks for demonstration
    # In a real scenario, you would import actual transformer blocks
    
    class DemoAttentionBlock(nn.Module):
        def __init__(self, d_model, num_heads, name="demo"):
            super().__init__()
            self.name = name
            self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model)
            )
        
        def forward(self, x):
            # Standard transformer block: attention + residual + norm + MLP + residual + norm
            attn_out, _ = self.attn(x, x, x)
            x = self.norm1(x + attn_out)
            mlp_out = self.mlp(x)
            x = self.norm2(x + mlp_out)
            return x
    
    # Create different variations for comparison
    blocks = {
        "Standard Block": DemoAttentionBlock(768, 12, "standard"),
        "8-Head Block": DemoAttentionBlock(768, 8, "8head"),
        "16-Head Block": DemoAttentionBlock(768, 16, "16head"),
    }
    
    # Run evaluation
    evaluator = BlockEvaluator(device)
    results = evaluator.compare_blocks(blocks, test_data)
    
    # Generate report
    evaluator.generate_report()
    
    return results

if __name__ == "__main__":
    run_block_evaluation()