import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# 1. Base Configurations
alpha = 0.3       # Fixed fractional order
nMax = 2000       # High iterations to settle fractional memory transients
nPlot = 100       # Number of final points to check for periodicity
nNorm = 10        # Renormalization interval for LE
x0 = 0.8          # Initial state condition

script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Define Parameter Grid Ranges (r vs q)
r_vals = np.arange(1.0, 4.21, 0.02)   # Sweep r up to 4.2 to capture the full landscape
q_vals = np.arange(0.02, 0.99, 0.02)  # Sweep q from near-zero to near-one

# Integer classifications matrix
classification_matrix = np.zeros((len(q_vals), len(r_vals)))

print(f"Starting 2D rq-plane scan for alpha = {alpha}...")

# 3. Double-Loop Sweep over r and q
for q_idx, q in enumerate(q_vals):
    # Precompute q-binomial weights for the active row's q value
    phi = [1.0]
    for m in range(1, nMax + 1):
        ratio = (1.0 - q**(alpha + m - 1)) / (1.0 - q**m)
        phi.append(phi[-1] * ratio)
    phiWeights = np.array(phi)
    
    for r_idx, r in enumerate(r_vals):
        xHist = np.zeros(nMax + 1)
        aHist = np.zeros(nMax + 1)
        
        xHist[0] = x0
        aHist[0] = 1.0
        leSum = 0.0
        unbounded = False
        
        for t in range(nMax):
            # State convolution mapping
            fx_minus_x = r * xHist[:t+1] * (1.0 - xHist[:t+1]) - xHist[:t+1]
            xHist[t+1] = x0 + np.dot(phiWeights[:t+1], fx_minus_x[::-1])
            
            # Divergence check
            if np.isnan(xHist[t+1]) or abs(xHist[t+1]) > 1e3:
                unbounded = True
                break
                
            # Tangent map convolution mapping
            df_minus_1 = r * (1.0 - 2.0 * xHist[:t+1]) - 1.0
            tangent_terms = df_minus_1 * aHist[:t+1]
            aHist[t+1] = 1.0 + np.dot(phiWeights[:t+1], tangent_terms[::-1])
            
            # Renormalization
            if (t + 1) % nNorm == 0:
                rescale = abs(aHist[t+1])
                if rescale > 1e-15:
                    leSum += np.log(rescale)
                    aHist[:t+2] /= rescale
                else:
                    aHist[t+1] = 1.0
                    
        # --- BEHAVIOR CLASSIFICATION LOGIC ---
        if unbounded:
            classification_matrix[q_idx, r_idx] = 0
        else:
            max_le = leSum / nMax
            if max_le > 0.005:  # Confirmed Chaotic Regime
                classification_matrix[q_idx, r_idx] = 5
            else:
                tail_rounded = np.round(xHist[-nPlot:], 3)
                unique_states = len(np.unique(tail_rounded))
                
                if unique_states == 1:
                    classification_matrix[q_idx, r_idx] = 1  # Period 1
                elif unique_states == 2:
                    classification_matrix[q_idx, r_idx] = 2  # Period 2
                elif (unique_states == 3) or (unique_states == 4):
                    classification_matrix[q_idx, r_idx] = 3  # Period 3 or 4
                else:
                    classification_matrix[q_idx, r_idx] = 4  # Higher Periods

# 4. Configure Custom Colors and Categorical Labels
colors_list = ['black', 'royalblue', 'saddlebrown', 'forestgreen', 'darkorange', 'crimson']
cmap_custom = ListedColormap(colors_list)

# FIXED: Added raw string format r'...' to satisfy LaTeX escape characters seamlessly
labels = ['Diverged', 'Period 1 (Fixed)', 'Period 2', 'Period 3 / 4', 'Higher Periods', r'Chaos ($\lambda > 0$)']

bounds_list = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
norm = BoundaryNorm(bounds_list, cmap_custom.N)

# 5. Generate and Format Figure Render
plt.figure(figsize=(13, 8))
extent_list = [r_vals[0], r_vals[-1], q_vals[0], q_vals[-1]]

mesh = plt.imshow(classification_matrix, extent=extent_list, origin='lower', aspect='auto', cmap=cmap_custom, norm=norm)

# Generated programmatically to prevent format bugs
cbar_ticks = np.arange(6)
cbar = plt.colorbar(mesh, ticks=cbar_ticks)
cbar.ax.set_yticklabels(labels, fontsize=11)
cbar.set_label('System Dynamic Regimes', fontsize=12, labelpad=10)

# FIXED: Replaced standard strings with raw strings for math blocks
plt.title(f"2D Dynamics Phase Map in $rq$-Plane for Fixed $\\alpha = {alpha}$", fontsize=14, pad=15)
plt.xlabel("Control Parameter ($r$)", fontsize=12)
plt.ylabel(r"q-Variant Binomial Scaling Parameter ($q$)", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.3, color='white')

# Save phase diagram output image
image_path = os.path.join(script_dir, 'rq_plane_bifurcation_map.png')
plt.savefig(image_path, dpi=300, bbox_inches='tight')
print(f"2D rq phase classification map successfully generated and stored to: {image_path}")

plt.show()
