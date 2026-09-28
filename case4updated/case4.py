import csv
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

# Global configuration parameters
Q = 0.99
ALPHA = 0.3
N1 = 5000
N2 = 20000
N3 = 50000  # Final deep step extension for stubborn points

def precompute_L3(n_steps, q=Q, alpha=ALPHA):
    """Precomputes the q-variant fractional coefficients up to n_steps."""
    L1 = np.array([q**j for j in range(n_steps + 2)])
    L2 = (q**alpha) * L1
    L3_list = [1.0]
    for t in range(n_steps):
        next_val = L3_list[-1] * (1.0 - L2[t]) / (1.0 - L1[t + 1])
        L3_list.append(next_val)
    return np.array(L3_list)

def evaluate_single_param(args):
    """
    Evaluates a single coordinate point using vectorized NumPy dot products.
    Runs concurrently across parallel CPU cores.
    """
    r1, r2, n_steps, L3, final_tier = args
    a_minus_1 = (r1 - 1.0) + 1j * r2
    
    # Initialize trajectory array
    x = np.zeros(n_steps + 2, dtype=complex)
    x[0] = 1.0 + 0j
    
    unbounded = False
    for t in range(n_steps + 1):
        # Clean vectorized convolution sum
        conv_sum = np.dot(L3[:t+1], x[t::-1])
        x[t+1] = 1.0 + a_minus_1 * conv_sum
        
        # Early exit check for fast divergence
        if np.abs(x[t+1]) > 1e6:
            unbounded = True
            break
            
    point = [r1, r2]
    if unbounded:
        return 'orange', point
    
    # Check standard tail magnitude limit over the last 1000 points
    max_tail_val = np.max(np.abs(x[n_steps-1000:n_steps+1]))
    if max_tail_val < 1e-5:
        return 'blue', point
        
    if final_tier:
        # Final tier fallback: check if envelope is expanding or contracting
        return 'blue' if np.abs(x[n_steps+1]) < 1.0 else 'orange', point
            
    return 'green', point

def run_simulation():
    r1_vals = [round(-0.5 + i * 0.05, 2) for i in range(45)]
    r2_vals = [round(-1.2 + i * 0.05, 2) for i in range(49)]
    full_grid = [[r1, r2] for r1 in r1_vals for r2 in r2_vals]
    
    num_cores = multiprocessing.cpu_count()
    print(f"Starting hardware parallel processing using {num_cores} logical CPU cores.")
    
    blue, orange, green = [], [], []
    
    # --- Tier 1: Initial Quick Scan (n = 5000) ---
    print(f"\n--- Tier 1: Parallel scanning full grid with n = {N1} ---")
    L3_n1 = precompute_L3(N1)
    # FIXED: Explicitly unpack coordinates as separate pt[0] and pt[1] scalar inputs
    tasks_n1 = [(pt[0], pt[1], N1, L3_n1, False) for pt in full_grid]
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(evaluate_single_param, tasks_n1)
        
    for label, pt in results:
        if label == 'blue': blue.append(pt)
        elif label == 'orange': orange.append(pt)
        elif label == 'green': green.append(pt)
    print(f"Tier 1 Results -> Blue: {len(blue)}, Orange: {len(orange)}, Green: {len(green)}")
    
    # --- Tier 2: Deep Trajectory Extension (n = 20000) ---
    if green:
        print(f"\n--- Tier 2: Extending {len(green)} green points to n = {N2} ---")
        L3_n2 = precompute_L3(N2)
        tasks_n2 = [(pt[0], pt[1], N2, L3_n2, False) for pt in green]
        
        with multiprocessing.Pool(processes=num_cores) as pool:
            results = pool.map(evaluate_single_param, tasks_n2)
            
        green = []
        for label, pt in results:
            if label == 'blue': blue.append(pt)
            elif label == 'orange': orange.append(pt)
            elif label == 'green': green.append(pt)
        print(f"Tier 2 Results -> Blue: {len(blue)}, Orange: {len(orange)}, Green: {len(green)}")
        
    # --- Tier 3: Final Max Depth + Trend Test (n = 50000) ---
    final_green = []
    if green:
        print(f"\n--- Tier 3: Pushing final {len(green)} points to n = {N3} ---")
        L3_n3 = precompute_L3(N3)
        tasks_n3 = [(pt[0], pt[1], N3, L3_n3, True) for pt in green]
        
        with multiprocessing.Pool(processes=num_cores) as pool:
            results = pool.map(evaluate_single_param, tasks_n3)
            
        for label, pt in results:
            if label == 'blue': blue.append(pt)
            elif label == 'orange': orange.append(pt)
            elif label == 'green': final_green.append(pt)
        print(f"Tier 3 Results -> Blue: {len(blue)}, Orange: {len(orange)}, Final Green: {len(final_green)}")
    
    # --- Save CSV Outputs ---
    def save_to_csv(filename, data):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data)
            
    print("\nWriting generated classifications to CSV files...")
    save_to_csv('blue.csv', blue)
    save_to_csv('orange.csv', orange)
    save_to_csv('green.csv', final_green)
    
    # --- Plotting the Stability Map ---
    print("Generating stability region map...")
    plt.figure(figsize=(10, 8))
    
    if orange:
        orange_arr = np.array(orange)
        plt.scatter(orange_arr[:, 0], orange_arr[:, 1], color='orange', s=15, label='Orange (Unbounded)')
    if blue:
        blue_arr = np.array(blue)
        plt.scatter(blue_arr[:, 0], blue_arr[:, 1], color='blue', s=15, label='Blue (Stable / Decayed)')
    if final_green:
        green_arr = np.array(final_green)
        plt.scatter(green_arr[:, 0], green_arr[:, 1], color='green', s=15, label='Green (Truly Indeterminate)')
        
    plt.title(f"Parallel Accelerated Stability Map ($q={Q}$, $\\alpha={ALPHA}$)", fontsize=14)
    plt.xlabel("$r_1 = \\mathrm{Re}(a)$", fontsize=12)
    plt.ylabel("$r_2 = \\mathrm{Im}(a)$", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axvline(1, color='black', linewidth=0.8, linestyle='--')
    
    plt.savefig('multiprocess_stability_region.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Process complete.")

if __name__ == '__main__':
    run_simulation()
