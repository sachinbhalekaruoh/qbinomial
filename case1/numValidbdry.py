import numpy as np
import csv
import matplotlib.pyplot as plt

# Parameters as defined in the paper
q = 0.3
alpha = 0.2
n = 5000

print("Precomputing q-binomial coefficients (L3)...")
L1 = np.array([q**j for j in range(n + 2)])
L2 = (q**alpha) * L1

L3_list = [1.0]
for t in range(n):
    next_val = L3_list[-1] * (1.0 - L2[t]) / (1.0 - L1[t + 1])
    L3_list.append(next_val)
L3 = np.array(L3_list)

# Initialize classification lists
blue = []
orange = []
green = []

# Generate precise parameter grids
r1_vals = [round(-0.5 + i * 0.05, 2) for i in range(45)]
r2_vals = [round(-1.2 + i * 0.05, 2) for i in range(49)]

print("Evaluating trajectories...")
for r1 in r1_vals:
    for r2 in r2_vals:
        a_minus_1 = (r1 - 1) + 1j * r2
        x = np.zeros(n + 2, dtype=complex)
        x[0] = 1.0 + 0j
        
        unbounded = False
        for t in range(n + 1):
            conv_sum = np.dot(L3[:t+1], x[t::-1])
            x[t+1] = 1.0 + a_minus_1 * conv_sum
            
            if np.abs(x[t+1]) > 1e8:
                orange.append([r1, r2])
                unbounded = True
                break
                
        if not unbounded:
            max_tail_val = np.max(np.abs(x[n-500:n+1]))
            if max_tail_val < 1e-6:
                blue.append([r1, r2])
            else:
                green.append([r1, r2])

# Function to save data to CSV
def save_to_csv(filename, data):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

print("Saving results to CSV files...")
save_to_csv('blue.csv', blue)
save_to_csv('orange.csv', orange)
save_to_csv('green.csv', green)

print("Generating the stability region plot...")
plt.figure(figsize=(10, 8))

# Plot each classified region with distinct colors
if orange:
    orange_arr = np.array(orange)
    plt.scatter(orange_arr[:, 0], orange_arr[:, 1], color='orange', s=15, label='Orange (Unbounded/Growth)')
if blue:
    blue_arr = np.array(blue)
    plt.scatter(blue_arr[:, 0], blue_arr[:, 1], color='blue', s=15, label='Blue (Decay)')
if green:
    green_arr = np.array(green)
    plt.scatter(green_arr[:, 0], green_arr[:, 1], color='green', s=15, label='Green (Neither)')

# Formatting the plot
plt.title(f"Stability Region for $q={q}$, $\\alpha={alpha}$", fontsize=14)
plt.xlabel("$r_1 = \\mathrm{Re}(a)$", fontsize=12)
plt.ylabel("$r_2 = \\mathrm{Im}(a)$", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right')
plt.axhline(0, color='black', linewidth=0.8, linestyle='-')
plt.axvline(1, color='black', linewidth=0.8, linestyle='--') # Shifts reference around a=1

# Show and save the image
plt.savefig('stability_region.png', dpi=300, bbox_inches='tight')
plt.show()
print("Done! Plot saved as 'stability_region.png'.")
