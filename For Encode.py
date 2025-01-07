import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

start=time.time()

# Fresnel Propagation Function (Phase-only Hologram)
def fresnel_phase_hologram(x0, y0, z0, X, Y, z, k):
    r = np.sqrt((X - x0)**2 + (Y - y0)**2 + z**2)
    return np.exp(1j * k * r)

# Parameters
wavelength = 0.633  # Wavelength in microns
k = 2 * np.pi / wavelength  # Wave number
z = 1.0  # Propagation distance in microns

# # Define 3D Cube Points
cube_size = 20
# original_points = []
# for x in [-cube_size/2, cube_size/2]:
#     for y in [-cube_size/2, cube_size/2]:
#         for z0 in [-cube_size/2, cube_size/2]:
#             original_points.append((x, y, z0))

# Define the number of points along each axis (grid resolution)
num_points = 40  # This will give you 5 points along each axis (adjust as needed)

# Create a 3D grid of points inside the cube
x_vals = np.linspace(-cube_size / 2, cube_size / 2, num_points)
y_vals = np.linspace(-cube_size / 2, cube_size / 2, num_points)
z_vals = np.linspace(-cube_size / 2, cube_size / 2, num_points)

# Generate all combinations of x, y, z coordinates
original_points = [(x, y, z) for x in x_vals for y in y_vals for z in z_vals]


# Observation Plane
grid_size = 300
x = np.linspace(-50, 50, grid_size)
y = np.linspace(-50, 50, grid_size)
X, Y = np.meshgrid(x, y)

# Compute Phase-only Hologram
hologram_phase = np.zeros_like(X, dtype=complex)
for (x0, y0, z0) in original_points:
    hologram_phase += fresnel_phase_hologram(x0, y0, z0, X, Y, z, k)

# Extract phase information
hologram_phase_only = np.angle(hologram_phase)

"""mod into 0-2"""
arr_r_mod=np.mod(hologram_phase_only,2)
arr_g_mod=np.mod(hologram_phase_only,2)
arr_b_mod=np.mod(hologram_phase_only,2)    

"""Map phase to gray level for diff laser"""
arr_r_modified=arr_r_mod*(255/1.85)
arr_g_modified=arr_g_mod*(255/2.63)
arr_b_modified=arr_b_mod*(255/3.55)
# Display Phase-only Hologram
plt.figure(figsize=(6, 6))
plt.imshow(arr_r_modified, cmap='gray', extent=(-50, 50, -50, 50))
plt.title("Phase-only Hologram")
plt.colorbar(label="Phase (radians)")
plt.show()

# Simulate Reconstruction
z_planes = np.linspace(500_000, 1_500_000, 50)  # From 0.5 meters to 1.5 meters (500,000 to 1,500,000 microns)
  # Different reconstruction depths
volume_reconstruction = np.zeros((len(z_planes), grid_size, grid_size))

# Apply inverse Fourier transform for reconstruction at different z planes
for z_idx, z in enumerate(z_planes):
    # Apply complex exponential phase field for reconstruction
    hologram_field = np.exp(1j * hologram_phase_only)
    
    # Perform the inverse FFT
    reconstructed_field = np.fft.ifft2(hologram_field)
    
    # Calculate the intensity at this plane
    volume_reconstruction[z_idx] = np.abs(reconstructed_field)**2

# Extract Reconstruction Points
threshold = 0.01 * volume_reconstruction.max()
z_coords, y_coords, x_coords = np.where(volume_reconstruction > threshold)
z_real = z_planes[z_coords]
x_real = x[x_coords]
y_real = y[y_coords]



# Plot Original Cube and Reconstruction
fig = plt.figure(figsize=(12, 6))

# Original Cube
ax1 = fig.add_subplot(121, projection='3d')
for (x0, y0, z0) in original_points:
    ax1.scatter(x0, y0, z0, c='blue', s=50, label="Cube Corner Points")
ax1.set_title("Original Cube (3D)")
ax1.set_xlabel("X (microns)")
ax1.set_ylabel("Y (microns)")
ax1.set_zlabel("Z (microns)")
ax1.set_xlim(-cube_size, cube_size)
ax1.set_ylim(-cube_size, cube_size)
ax1.set_zlim(-cube_size, cube_size)
ax1.view_init(elev=20, azim=30)

# Reconstructed Cube
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(x_real, y_real, z_real, c=volume_reconstruction[z_coords, y_coords, x_coords], cmap='hot', s=2)
ax2.set_title("Reconstructed Cube (3D)")
ax2.set_xlabel("X (microns)")
ax2.set_ylabel("Y (microns)")
ax2.set_zlabel("Z (microns)")
ax2.view_init(elev=20, azim=30)

plt.tight_layout()
plt.show()

end=time.time()
print(f"time comsuming {end-start}s")