# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 11:11:43 2024

@author: gaosh
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define constants
wavelength = 633e-9  # Wavelength of light in meters (e.g., 633nm for red laser)
k = 2 * np.pi / wavelength  # Wave number

# Define cube vertices in 3D space (in meters)
cube_vertices = np.array([
    [0, 0, 0], [1e-3, 0, 0], [1e-3, 1e-3, 0], [0, 1e-3, 0],  # Bottom face
    [0, 0, 1e-3], [1e-3, 0, 1e-3], [1e-3, 1e-3, 1e-3], [0, 1e-3, 1e-3]  # Top face
])

# Define the hologram plane (2D grid)
holo_size = 5e-3  # Hologram plane size (5mm x 5mm)
num_pixels = 512  # Resolution of the hologram
x = np.linspace(-holo_size / 2, holo_size / 2, num_pixels)
y = np.linspace(-holo_size / 2, holo_size / 2, num_pixels)
X, Y = np.meshgrid(x, y)

# Distance to the hologram plane
z_reconstruction = np.linspace(0.05, 0.15, 100)  # 5cm to 15cm reconstruction depths
holo_field = np.zeros((num_pixels, num_pixels), dtype=complex)

# Calculate the hologram field by summing contributions from all cube points
for vertex in cube_vertices:
    r = np.sqrt((X - vertex[0])**2 + (Y - vertex[1])**2 + z_reconstruction[0]**2)  # Initial distance
    holo_field += np.exp(1j * k * r) / r  # Wavefront from point source

# Initialize 3D volume for reconstructed intensity
volume = np.zeros((len(z_reconstruction), num_pixels, num_pixels))

# Perform reconstruction at each depth plane
for i, z in enumerate(z_reconstruction):
    phase_factor = np.exp(1j * k * z) * np.exp(1j * (k / (2 * z)) * (X**2 + Y**2))
    recon_plane = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(holo_field * phase_factor)))
    volume[i] = np.abs(recon_plane)**2

# Normalize the volume
volume /= volume.max()

# Plot 3D reconstruction
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Extract the 3D positions of the bright spots in the reconstructed volume
threshold = 0.2  # Intensity threshold for detecting points
positions = np.argwhere(volume > threshold)

# Convert voxel indices to physical coordinates
z_vals, y_vals, x_vals = positions.T
x_coords = x_vals * holo_size / num_pixels - holo_size / 2
y_coords = y_vals * holo_size / num_pixels - holo_size / 2
z_coords = z_reconstruction[z_vals]

# Plot the 3D points
ax.scatter(x_coords * 1e3, y_coords * 1e3, z_coords * 1e3, c=z_coords, cmap='hot', s=10)

# Label axes
ax.set_title('3D Reconstruction of Cube')
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
plt.show()

