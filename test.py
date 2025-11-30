import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=10000, d_model=512, nhead=8, num_layers=6, max_seq_len=512):
        super(SimpleTransformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # 嵌入层 + 位置编码
        x = self.embedding(x) * np.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # 创建注意力mask
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        
        # Transformer编码器
        x = self.transformer(x, mask=mask)
        
        # 输出层
        x = self.output_layer(x)
        return x

def check_matmul_device():
    """检查torch.matmul算子使用的设备并比较性能"""
    print("检查torch.matmul算子设备...")
    
    # 创建测试张量 - 使用更大的矩阵以更好地体现性能差异
    size = 1024
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    
    # CPU上的matmul - 运行100次
    print("CPU matmul测试...")
    start_time = time.time()
    for i in range(100):
        result_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - start_time
    print(f"CPU matmul 100次总时间: {cpu_time:.4f}秒")
    print(f"CPU matmul平均每次时间: {cpu_time/100:.6f}秒")
    print(f"CPU matmul结果设备: {result_cpu.device}")
    
    # 检查MPS是否可用
    if torch.backends.mps.is_available():
        a_mps = a_cpu.to('mps')
        b_mps = b_cpu.to('mps')
        
        # MPS上的matmul - 运行100次
        print("\nMPS matmul测试...")
        
        # 预热运行几次以避免第一次运行的初始化开销
        for _ in range(5):
            _ = torch.matmul(a_mps, b_mps)
        
        # 正式测试100次
        start_time = time.time()
        for i in range(100):
            result_mps = torch.matmul(a_mps, b_mps)
        mps_time = time.time() - start_time
        print(f"MPS matmul 100次总时间: {mps_time:.4f}秒")
        print(f"MPS matmul平均每次时间: {mps_time/100:.6f}秒")
        print(f"MPS matmul结果设备: {result_mps.device}")
        
        # 性能比较
        speedup = cpu_time / mps_time
        print(f"\n性能比较:")
        print(f"CPU总时间: {cpu_time:.4f}秒")
        print(f"MPS总时间: {mps_time:.4f}秒")
        print(f"加速比 (CPU/MPS): {speedup:.2f}x")
        
        if speedup > 1:
            print(f"🎉 MPS比CPU快 {speedup:.2f} 倍")
        else:
            print(f"⚠️ CPU比MPS快 {1/speedup:.2f} 倍")
        
        # 检查结果是否相同（允许小的数值差异）
        result_mps_cpu = result_mps.cpu()
        diff = torch.abs(result_cpu - result_mps_cpu).max()
        print(f"CPU和MPS结果最大差异: {diff.item():.8f}")
        
        # 额外测试：不同矩阵大小的性能
        print("\n" + "="*50)
        print("不同矩阵大小的性能测试")
        print("="*50)
        
        sizes = [256, 512, 1024, 2048]
        for test_size in sizes:
            print(f"\n测试矩阵大小: {test_size}x{test_size}")
            test_a_cpu = torch.randn(test_size, test_size)
            test_b_cpu = torch.randn(test_size, test_size)
            
            # CPU测试
            start = time.time()
            for _ in range(20):  # 减少次数以加快测试
                _ = torch.matmul(test_a_cpu, test_b_cpu)
            test_cpu_time = time.time() - start
            
            # MPS测试
            test_a_mps = test_a_cpu.to('mps')
            test_b_mps = test_b_cpu.to('mps')
            
            # 预热
            for _ in range(3):
                _ = torch.matmul(test_a_mps, test_b_mps)
            
            start = time.time()
            for _ in range(20):
                _ = torch.matmul(test_a_mps, test_b_mps)
            test_mps_time = time.time() - start
            
            test_speedup = test_cpu_time / test_mps_time
            print(f"CPU: {test_cpu_time/20:.6f}s/次, MPS: {test_mps_time/20:.6f}s/次, 加速: {test_speedup:.2f}x")
            
    else:
        print("MPS不可用")

# 在main函数中调用这个函数
if __name__ == "__main__":
    print("PyTorch版本:", torch.__version__)
    print("MPS可用:", torch.backends.mps.is_available())
    
    if torch.backends.mps.is_available():
        print("MPS设备:", )
    
    check_matmul_device()

def benchmark_model(device='cpu', batch_size=4, seq_len=128):
    """在指定设备上测试模型性能"""
    print(f"\n在 {device.upper()} 上进行性能测试...")
    
    # 创建模型和测试数据
    model = SimpleTransformer(
        vocab_size=10000,
        d_model=256,  # 减小模型大小以便更快测试
        nhead=8,
        num_layers=4,
        max_seq_len=seq_len
    ).to(device)
    
    # 创建测试数据
    input_ids = torch.randint(0, 10000, (batch_size, seq_len)).to(device)
    
    # 预热（避免第一次运行的初始化开销）
    for _ in range(3):
        _ = model(input_ids)
    
    # 性能测试
    start_time = time.time()
    num_iterations = 50
    
    for i in range(num_iterations):
        output = model(input_ids)
        
        # 模拟训练步骤
        if i % 10 == 0:  # 每10次进行一次反向传播
            loss = output.mean()
            loss.backward()
    
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_iteration = total_time / num_iterations
    
    print(f"设备: {device}")
    print(f"总时间: {total_time:.4f}秒")
    print(f"每次迭代平均时间: {avg_time_per_iteration:.4f}秒")
    print(f"吞吐量: {num_iterations/total_time:.2f} 迭代/秒")
    
    return total_time, avg_time_per_iteration

def compare_performance():
    """比较CPU和MPS性能"""
    print("=" * 50)
    print("CPU vs MPS 性能比较")
    print("=" * 50)
    
    # 检查设备可用性
    mps_available = torch.backends.mps.is_available()
    print(f"MPS可用: {mps_available}")
    
    if mps_available:
        print("MPS设备: ")
    
    # 检查matmul设备
    check_matmul_device()
    
    # 测试不同批大小和序列长度
    test_configs = [
        (4, 128),
        (8, 128),
        (4, 256),
    ]
    
    cpu_times = []
    mps_times = []
    
    for batch_size, seq_len in test_configs:
        print(f"\n测试配置: batch_size={batch_size}, seq_len={seq_len}")
        print("-" * 40)
        
        # CPU测试
        cpu_total, cpu_avg = benchmark_model('cpu', batch_size, seq_len)
        cpu_times.append((cpu_total, cpu_avg))
        
        # MPS测试（如果可用）
        if mps_available:
            mps_total, mps_avg = benchmark_model('mps', batch_size, seq_len)
            mps_times.append((mps_total, mps_avg))
            
            # 计算加速比
            speedup = cpu_avg / mps_avg
            print(f"\n加速比 (CPU/MPS): {speedup:.2f}x")
            
            if speedup > 1:
                print(f"MPS比CPU快 {speedup:.2f} 倍")
            else:
                print(f"CPU比MPS快 {1/speedup:.2f} 倍")
    
    # 总结
    if mps_available and mps_times:
        print("\n" + "=" * 50)
        print("性能比较总结")
        print("=" * 50)
        
        avg_cpu_time = np.mean([t[1] for t in cpu_times])
        avg_mps_time = np.mean([t[1] for t in mps_times])
        overall_speedup = avg_cpu_time / avg_mps_time
        
        print(f"平均CPU时间: {avg_cpu_time:.4f}秒/迭代")
        print(f"平均MPS时间: {avg_mps_time:.4f}秒/迭代")
        print(f"总体加速比: {overall_speedup:.2f}x")
        
        if overall_speedup > 1:
            print(f"🎉 MPS总体表现更好，加速 {overall_speedup:.2f} 倍")
        else:
            print(f"⚠️ CPU总体表现更好，MPS慢 {1/overall_speedup:.2f} 倍")

def memory_usage_comparison():
    """比较内存使用情况"""
    print("\n" + "=" * 50)
    print("内存使用比较")
    print("=" * 50)
    
    if torch.backends.mps.is_available():
        # 在MPS上测试内存使用
        model_mps = SimpleTransformer(d_model=256, num_layers=4).to('mps')
        input_mps = torch.randint(0, 10000, (4, 128)).to('mps')
        
        # MPS内存统计
        if hasattr(torch.mps, 'memory_allocated'):
            torch.mps.empty_cache()
            initial_mem = torch.mps.memory_allocated() / 1024**2  # MB
            
            output_mps = model_mps(input_mps)
            peak_mem = torch.mps.memory_allocated() / 1024**2
            
            print(f"MPS内存使用:")
            print(f"  初始: {initial_mem:.2f} MB")
            print(f"  峰值: {peak_mem:.2f} MB")
            print(f"  增加: {peak_mem - initial_mem:.2f} MB")
    
    # CPU内存统计
    import psutil
    process = psutil.Process()
    initial_cpu_mem = process.memory_info().rss / 1024**2
    
    model_cpu = SimpleTransformer(d_model=256, num_layers=4)
    input_cpu = torch.randint(0, 10000, (4, 128))
    output_cpu = model_cpu(input_cpu)
    
    final_cpu_mem = process.memory_info().rss / 1024**2
    
    print(f"CPU内存使用:")
    print(f"  初始: {initial_cpu_mem:.2f} MB")
    print(f"  峰值: {final_cpu_mem:.2f} MB")
    print(f"  增加: {final_cpu_mem - initial_cpu_mem:.2f} MB")

if __name__ == "__main__":
    print("PyTorch版本:", torch.__version__)
    print("MPS可用:", torch.backends.mps.is_available())
    
    if torch.backends.mps.is_available():
        print("MPS设备:", )
    
    # 运行性能比较
    compare_performance()
    
    # 运行内存使用比较
    memory_usage_comparison()
    
    print("\n测试完成！")
