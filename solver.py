import random
import math
import time
import numpy as np
from scipy.stats import kendalltau

# ===================================================================
# == Part 1: Flexible Functions for Flow Shop Scheduling Problem ==
# ===================================================================

def create_random_instance(num_jobs, num_machines, max_proc_time=20):
    """動態生成指定數量的工作和機器，並為每個工作隨機分配處理時間。"""
    jobs = []
    for i in range(num_jobs):
        proc_times = [random.randint(1, max_proc_time) for _ in range(num_machines)]
        jobs.append({'id': i, 'proc': proc_times})
    return jobs

def calculate_makespan(perm, jobs, num_machines):
    """根據給定的工作順序 (perm)，計算完成所有工作的總時間 (makespan)。"""
    machine_finish_times = [0] * num_machines
    for job_id in perm:
        job_proc_times = jobs[job_id]['proc']
        machine_finish_times[0] += job_proc_times[0]
        for m in range(1, num_machines):
            start_time = max(machine_finish_times[m], machine_finish_times[m-1])
            machine_finish_times[m] = start_time + job_proc_times[m]
    return machine_finish_times[-1]

def neighbor_swap(perm):
    """從一個排序中隨機交換兩個元素，產生一個新的「鄰近」排序。"""
    a, b = sorted(random.sample(range(len(perm)), 2))
    new = perm.copy()
    new[a], new[b] = new[b], new[a]
    return new

# ===================================================================
# == Part 2: Machine Learning Sampler (Placeholder)              ==
# ===================================================================

class MLInverseSampler:
    """一個機器學習採樣器的佔位符，使用「菁英庫」策略來模擬其行為。"""
    def __init__(self, num_jobs):
        self.num_jobs = num_jobs
        self.elite_archive = []

    def train(self, examples):
        """佔位符訓練函式：將好的範例加入我們的菁英庫。"""
        self.elite_archive.extend(examples)
        # 簡單去重並限制菁英庫大小，防止無限增長
        unique_perms = list(set(tuple(p) for p in self.elite_archive))
        self.elite_archive = [list(p) for p in unique_perms[:50]] # 最多保留50個菁英解
        
    def sample(self, n):
        """佔位符採樣函式：從菁英庫中生成新的候選解。"""
        samples = []
        if not self.elite_archive:
            for _ in range(n):
                perm = list(range(self.num_jobs))
                random.shuffle(perm)
                samples.append(perm)
        else:
            for _ in range(n):
                base_perm = random.choice(self.elite_archive)
                new_perm = neighbor_swap(base_perm)
                samples.append(new_perm)
        return samples

# ===================================================================
# == Part 3: Simulated Annealing Algorithm                         ==
# ===================================================================

def simulated_annealing(jobs, num_machines, start_perm=None, T0=100.0, alpha=0.995, steps=5000):
    """使用模擬退火演算法尋找最佳的工作排序。"""
    n = len(jobs)
    current_perm = start_perm[:] if start_perm else random.sample(range(n), n)
    current_makespan = calculate_makespan(current_perm, jobs, num_machines)
    best_perm, best_makespan = current_perm[:], current_makespan
    T = T0
    for _ in range(steps):
        candidate_perm = neighbor_swap(current_perm)
        candidate_makespan = calculate_makespan(candidate_perm, jobs, num_machines)
        delta = candidate_makespan - current_makespan
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-12)):
            current_perm, current_makespan = candidate_perm, candidate_makespan
            if current_makespan < best_makespan:
                best_perm, best_makespan = current_perm[:], current_makespan
        T *= alpha
    return best_perm, best_makespan

# ===================================================================
# == Part 4: Hybrid Optimization Framework                         ==
# ===================================================================

def run_hybrid_optimization(jobs, num_jobs, num_machines, cycles, sa_runs, sa_steps):
    """執行完整的混合式優化流程。"""
    sampler = MLInverseSampler(num_jobs=num_jobs)
    overall_best_makespan = float('inf')
    for cycle in range(cycles):
        start_perms = sampler.sample(sa_runs)
        cycle_solutions = []
        for i in range(sa_runs):
            perm, makespan = simulated_annealing(jobs, num_machines, start_perm=start_perms[i], steps=sa_steps)
            cycle_solutions.append((perm, makespan))
            if makespan < overall_best_makespan:
                overall_best_makespan = makespan
        cycle_solutions.sort(key=lambda x: x[1])
        elites = [p for p, m in cycle_solutions[:5]]
        sampler.train(elites)
    return overall_best_makespan, sampler

# ===================================================================
# == Part 5: Evaluation Helpers & Comparison Framework           ==
# ===================================================================

def kendall_distance(p, q):
    """計算兩個排序 p 和 q 之間的肯德爾距離 (Kendall Tau Distance)。"""
    tau, _ = kendalltau(p, q)
    if np.isnan(tau): return 1.0
    return 1.0 - ((tau + 1.0) / 2.0)

def load_variance(perm, jobs):
    """計算一個排序中，各工作總加工時長的變異數。"""
    loads = [sum(jobs[job_id]['proc']) for job_id in perm]
    return np.var(loads)

def compare_methods(num_jobs, num_machines, num_baseline_runs=10):
    """
    對「基準 SA」和「混合式 ML-SA」進行公平比較。
    """
    print("=" * 60)
    print("⚔️  演算法比較：基準 SA (Baseline SA) vs. 混合式 ML-SA (Hybrid ML-SA)")
    print("=" * 60)
    
    jobs = create_random_instance(num_jobs, num_machines)
    print(f"已生成一個用於比較的通用問題實例 ({num_jobs} 個工作, {num_machines} 台機器)。\n")

    # 調整參數以確保總計算量大致相等
    # 總計算量 ≈ 運行次數 * 每次運行的步數
    baseline_steps = 50000
    hybrid_cycles = 4
    hybrid_runs_per_cycle = 5
    hybrid_steps = (baseline_steps * num_baseline_runs) // (hybrid_cycles * hybrid_runs_per_cycle)
    print(f"為公平起見，總計算量已校準 (總 SA 步數約為 {baseline_steps * num_baseline_runs})。\n")

    # --- 方法 1: 基準 SA ---
    print(f"--- 執行方法 1: 基準 SA ({num_baseline_runs} 次獨立運行) ---")
    baseline_results = []
    start_time = time.time()
    for i in range(num_baseline_runs):
        _, makespan = simulated_annealing(jobs, num_machines, steps=baseline_steps)
        baseline_results.append(makespan)
        print(f"  運行 {i+1}/{num_baseline_runs}... 找到 Makespan: {makespan}")
    baseline_time = time.time() - start_time

    # --- 方法 2: 混合式 ML-SA ---
    print(f"\n--- 執行方法 2: 混合式 ML-SA ({hybrid_cycles} 輪循環) ---")
    start_time = time.time()
    hybrid_best_makespan, trained_sampler = run_hybrid_optimization(
        jobs, num_jobs, num_machines,
        cycles=hybrid_cycles, sa_runs=hybrid_runs_per_cycle, sa_steps=hybrid_steps
    )
    hybrid_time = time.time() - start_time
    
    # --- 結果分析與輸出 ---
    print("\n" + "=" * 60)
    print("📊 評估結果分析")
    print("=" * 60)

    print(f"\n方法 1: 基準 SA")
    print(f"  ⏱️  總耗時: {baseline_time:.2f} 秒")
    print(f"  🏆 最佳 Makespan: {np.min(baseline_results)}")
    print(f"  📈 平均 Makespan: {np.mean(baseline_results):.2f}")
    print(f"  📉 結果標準差: {np.std(baseline_results):.2f}")

    print(f"\n方法 2: 混合式 ML-SA")
    print(f"  ⏱️  總耗時: {hybrid_time:.2f} 秒")
    print(f"  🏆 最佳 Makespan: {hybrid_best_makespan}")
    
    # 使用評估工具分析 ML 採樣器的特性
    samples = trained_sampler.sample(50)
    distances = [kendall_distance(samples[i], samples[j]) for i in range(len(samples)) for j in range(i + 1, len(samples))]
    print(f"  🧬 ML 採樣器生成解的多樣性 (平均肯德爾距離): {np.mean(distances):.3f}")
    print("=" * 60)


if __name__ == '__main__':
    # 主程式入口，直接調用比較函式
    compare_methods(num_jobs=25, num_machines=8, num_baseline_runs=10)