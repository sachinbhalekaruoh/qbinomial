import numpy as np
import csv
import matplotlib.pyplot as plt

def evaluate_parameters(param_list, n_steps, precomputed_L3=None):
    """
    Evaluates trajectories for a given list of parameters [r1, r2].
    Returns three lists: blue, orange, and green.
    """
    # Precompute q-binomial coefficients for this specific n_steps
    if precomputed_L3 is None:
        q = 0.9
        alpha = 0.3
        L1 = np.array([q**j for j in range(n_steps + 2)])
        L2 = (q**alpha) * L1

        L3_list = [1.0]
        for t in range(n_steps):
            next_val = L3_list[-1] * (1.0 - L2[t]) / (1.0 - L1[t + 1])
            L3_list.append(next_val)
        L3 = np.array(L3_list)
    else:
        L3 = precomputed_L3

    blue_res = []
    orange_res = []
    green_res = []

    for r1, r2 in param_list:
        a_minus_1 = (r1 - 1) + 1j * r2
        x = np.zeros(n_steps + 2, dtype=complex)
        x[0] = 1.0 + 0j
        
        unbounded = False
        for t in range(n_steps + 1):
            conv_sum = np.dot(L3[:t+1], x[t::-1])
            x[t+1] = 1.0 + a_minus_1 * conv_sum
            
            if np.abs(x[t+1]) > 1e8:
                orange_res.append([r1, r2])
                unbounded = True
                break
                
        if not unbounded:
            # Check maximum amplitude of the last 501 points to see if it decayed
            max_tail_val = np.max(np.abs(x[n_steps-500:n_steps+1]))
            if max_tail_val < 1e-6:
                blue_res.append([r1, r2])
            else:
                green_res.append([r1, r2])
                
    return blue_res, orange_res, green_res

# --- Phase 1: Initial Scan (n = 5000) ---
n1 = 5000
print(f"--- Phase 1: Running initial scan with n = {n1} ---")

# Generate the full grid of parameters
r1_vals = [round(-0.4 + i * 0.05, 2) for i in range(40)]
r2_vals = [round(-1.2 + i * 0.05, 2) for i in range(49)]
full_grid = [[r1, r2] for r1 in r1_vals for r2 in r2_vals]

blue, orange, green = evaluate_parameters(full_grid, n1)
print(f"Initial counts -> Blue: {len(blue)}, Orange: {len(orange)}, Green (Unsettled): {len(green)}")

# --- Phase 2: Resolving Green Points (n = 20000) ---
if len(green) > 0:
    n2 = 20000
    print(f"\n--- Phase 2: Re-evaluating {len(green)} green points with n = {n2} ---")
    
    # Run the unresolved points through a deeper timeline
    deep_blue, deep_orange, final_green = evaluate_parameters(green, n2)
    
    # Merge the newly resolved parameters into the primary lists
    blue.extend(deep_blue)
    orange.extend(deep_orange)
    print(f"Resolution complete -> Moved {len(deep_blue)} to Blue, {len(deep_orange)} to Orange. Left in Green: {len(final_green)}")
else:
    final_green = []
    print("\nNo green points remained to resolve.")

# --- Save Results ---
def save_to_csv(filename, data):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

print("\nSaving final classified sets to CSV files...")
save_to_csv('blue.csv', blue)
save_to_csv('orange.csv', orange)
save_to_csv('green.csv', final_green)

# --- Plotting the Final Map ---
print("Generating final stability map...")
plt.figure(figsize=(10, 8))

if orange:
    orange_arr = np.array(orange)
    plt.scatter(orange_arr[:, 0], orange_arr[:, 1], color='orange', s=15, label='Orange (Unbounded)')
if blue:
    blue_arr = np.array(blue)
    plt.scatter(blue_arr[:, 0], blue_arr[:, 1], color='blue', s=15, label='Blue (Decay / Stable)')
if final_green:
    green_arr = np.array(final_green)
    plt.scatter(green_arr[:, 0], green_arr[:, 1], color='green', s=15, label='Green (Still Unsettled)')

plt.title(f"Final Stability Region Map (Multi-Phase Resolution)", fontsize=14)
plt.xlabel("$r_1 = \\mathrm{Re}(a)$", fontsize=12)
plt.ylabel("$r_2 = \\mathrm{Im}(a)$", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(1, color='black', linewidth=0.8, linestyle='--')

plt.savefig('final_stability_region.png', dpi=300, bbox_inches='tight')
plt.show()
print("Done! Final plot saved as 'final_stability_region.png'.")
