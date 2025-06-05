import numpy as np
import matplotlib.pyplot as plt

# Generate random data for the heatmap (4x4)
N = 10
data = np.random.rand(N, N)
data = data * 3
np.fill_diagonal(data, 8.0)
data[4,1] = 9
data[8,3] = 7
data[2,9] = 8
data[3,4] = 8

data[4,4] = 0
data[3,3] = 0
data[1,1] = 0
data[7,7] = 3
data[8,8] = 4

# Create the heatmap
plt.figure(figsize=(N, N))
plt.imshow(data, cmap='viridis', interpolation='nearest')

# Remove axis ticks for a clean look
plt.xticks([])
plt.yticks([])

# Optional: Add gridlines
plt.grid(visible=False)

# Show or save the figure
plt.tight_layout()
plt.savefig("random_heatmap.png", dpi=300)
# plt.show()
