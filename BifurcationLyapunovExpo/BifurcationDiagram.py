import numpy as np
import csv
import os
import matplotlib.pyplot as plt

# 1. System Parameters
q = 0.3
alpha = 0.6
nMax = 3000       # INCREASED: Gives the fractional memory time to settle steady-state
nPlot = 50        # Number of final settled points to display on the diagram
nNorm = 10        # Renormalization interval for Lyapunov computation
x0 = 0.8          # Initial state condition

# Get directory where the script is located to guarantee exact file saving
script_dir = os.path.dirname(os.path.abspath(__file__))

print("Precomputing q-binomial weights...")
phi = [1.0]
for m in range(1, nMax + 1):
    ratio = (1.0 - q**(alpha + m - 1)) / (1.0 - q**m)
    phi.append(phi[-1] * ratio)
phiWeights = np.array(phi)

# Sweep range r from 0.9 to 3.7 with a step-size of 0.02 to ensure rapid execution
r_vals = np.arange(0.9, 3.71, 0.02)

# Storage for both plots
r_bifurcation = []
x_bifurcation = []
spectrum_data = []

print("Sweeping parameter r (Calculating Bifurcations & Lyapunov Exponents)...")
for r in r_vals:
    xHist = np.zeros(nMax + 1)
    aHist = np.zeros(nMax + 1)
    
    # Correct array element positioning
    xHist[0] = x0  
    aHist[0] = 1.0  
    
    leSum = 0.0
    unbounded = False
    
    for t in range(nMax):
        # 1. State Mapping Convolution
        fx_minus_x = r * xHist[:t+1] * (1.0 - xHist[:t+1]) - xHist[:t+1]
        xHist[t+1] = x0 + np.dot(phiWeights[:t+1], fx_minus_x[::-1])
        
        # Guard against divergence/explosion bounds
        if np.isnan(xHist[t+1]) or abs(xHist[t+1]) > 1e3:
            unbounded = True
            break
            
        # 2. Tangent Space Mapping Convolution
        df_minus_1 = r * (1.0 - 2.0 * xHist[:t+1]) - 1.0
        tangent_terms = df_minus_1 * aHist[:t+1]
        aHist[t+1] = 1.0 + np.dot(phiWeights[:t+1], tangent_terms[::-1])
        
        # Periodic Renormalization
        if (t + 1) % nNorm == 0:
            rescale = abs(aHist[t+1])
            if rescale > 1e-15:
                leSum += np.log(rescale)
                aHist[:t+2] /= rescale
            else:
                aHist[t+1] = 1.0
                
    if not unbounded:
        # Collect Bifurcation points (extract the settled tail end)
        tail = xHist[-nPlot:]
        r_bifurcation.extend([r] * len(tail))
        x_bifurcation.extend(tail)
        
        # Collect Lyapunov Exponent
        spectrum_data.append([round(r, 2), leSum / nMax])
    else:
        spectrum_data.append([round(r, 2), np.nan])

# Export the Lyapunov Exponent Data to CSV
csv_path = os.path.join(script_dir, 'lyapunov_spectrum_final.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['r', 'Max_Lyapunov_Exponent'])
    writer.writerows(spectrum_data)
print(f"CSV data successfully exported to: {csv_path}")

# --- Generate the Stacked Graph Layout ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

# Plot 1: Bifurcation Diagram (Enhanced scatter marker size for visibility)
ax1.scatter(r_bifurcation, x_bifurcation, color='black', s=4, alpha=0.5, marker='o')
ax1.set_title(f"Fractional Logistic Map System Dynamics ($q={q}, \\alpha={alpha}$)", fontsize=14)
ax1.set_ylabel("State Attractor ($x$)", fontsize=12)
ax1.set_ylim(-0.05, 1.05)
ax1.grid(True, linestyle=':', alpha=0.5)

# Plot 2: Lyapunov Exponent Spectrum
spectrum_arr = np.array(spectrum_data, dtype=float)
valid_mask = ~np.isnan(spectrum_arr[:, 1])

ax2.plot(spectrum_arr[valid_mask, 0], spectrum_arr[valid_mask, 1], color='darkmagenta', linewidth=1.5)
ax2.axhline(0, color='red', linestyle='--', linewidth=1.0, label='Chaos Threshold ($\lambda=0$)')
ax2.set_xlabel("Control Parameter ($r$)", fontsize=12)
ax2.set_ylabel("Max Lyapunov Exponent ($\lambda$)", fontsize=12)
ax2.set_xlim(0.9, 3.7)
ax2.grid(True, linestyle=':', alpha=0.5)
ax2.legend(loc='lower left')

# Adjust presentation spacing and save image file
plt.tight_layout()
image_path = os.path.join(script_dir, 'system_dynamics_final.png')
plt.savefig(image_path, dpi=300, bbox_inches='tight')
print(f"Combined graph visualization successfully saved to: {image_path}")

plt.show()
