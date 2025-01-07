import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Fresnel Propagation Function (for the phase hologram)
def fresnel_propagation(x0, y0, z0, X, Y, z, k):
    r = np.sqrt((X - x0)**2 + (Y - y0)**2 + (z - z0)**2)  # Distance from each point to observation plane
    return np.exp(1j * k * r) / r  # Fresnel diffraction formula for phase

# Inverse Fresnel Propagation Function (for the reconstruction)
def inverse_fresnel_propagation(x, y, z0, k, z):
    r = np.sqrt(x**2 + y**2 + (z - z0)**2)  # Distance from hologram to 3D points
    return np.exp(1j * k * r) / r  # Inverse Fresnel diffraction formula

# Parameters
wavelength = 0.633  # Wavelength in microns
k = 2 * np.pi / wavelength  # Wave number
z = 1.0  # Propagation distance from cube to hologram (in microns)

# Define 3D Cube Points (Corners of the Cube)
cube_size = 10  # Cube side length in microns
original_points = []
for x in [-cube_size/2, cube_size/2]:
    for y in [-cube_size/2, cube_size/2]:
        for z0 in [-cube_size/2, cube_size/2]:
            original_points.append((x, y, z0))

# Observation Plane (Hologram plane)
grid_size = 200  # Resolution of the observation grid
x = np.linspace(-50, 50, grid_size)  # X-coordinate grid
y = np.linspace(-50, 50, grid_size)  # Y-coordinate grid
X, Y = np.meshgrid(x, y)  # Create the 2D meshgrid for the hologram

# Compute the Phase-only Hologram (Interference of light from the cube's corners)
hologram_field = np.zeros_like(X, dtype=complex)
for (x0, y0, z0) in original_points:
    hologram_field += fresnel_propagation(x0, y0, z0, X, Y, z, k)

# Extract phase information for the hologram
hologram_phase_only = np.angle(hologram_field)

# Display Phase-only Hologram
plt.figure(figsize=(6, 6))
plt.imshow(hologram_phase_only, cmap='gray', extent=(-50, 50, -50, 50))
plt.title("Phase-only Hologram")
plt.colorbar(label="Phase (radians)")
plt.show()

# Simulate Reconstruction (Inverse Fresnel Propagation)
z_planes = np.linspace(0.5, 1.5, 50)  # Simulate reconstruction at various depths
volume_reconstruction = np.zeros((len(z_planes), grid_size, grid_size))

# Reconstruct the volume from the phase hologram by backpropagating
for z_idx, z_val in enumerate(z_planes):
    field_reconstruction = np.exp(1j * hologram_phase_only)  # Use the phase-only hologram for the reconstruction
    reconstructed_field = np.zeros_like(X, dtype=complex)

    for (x0, y0, z0) in original_points:
        # Use inverse Fresnel propagation for each point in the cube
        reconstructed_field += inverse_fresnel_propagation(X, Y, z0, k, z_val)

    # Record the reconstructed intensity (the absolute square of the field)
    volume_reconstruction[z_idx] = np.abs(reconstructed_field)**2

# Extract Reconstruction Points
threshold = 0.1 * volume_reconstruction.max()  # Set a threshold to extract points above a certain intensity
z_coords, y_coords, x_coords = np.where(volume_reconstruction > threshold)
z_real = z_planes[z_coords]
x_real = x[x_coords]
y_real = y[y_coords]

# Plot Original Cube and Reconstructed Cube
fig = plt.figure(figsize=(12, 6))

# Plot the original cube
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

# Plot the reconstructed cube
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(x_real, y_real, z_real, c=volume_reconstruction[z_coords, y_coords, x_coords], cmap='hot', s=2)
ax2.set_title("Reconstructed Cube (3D)")
ax2.set_xlabel("X (microns)")
ax2.set_ylabel("Y (microns)")
ax2.set_zlabel("Z (microns)")
ax2.view_init(elev=20, azim=30)

plt.tight_layout()
plt.show()
