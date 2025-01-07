# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 16:55:41 2024

@author: gaosh
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
wavelength = 633e-9  # Wavelength of light (meters)
pixel_pitch = 10e-6  # Pixel pitch of the hologram (meters)
grid_size = 256      # Grid size (NxN)
z_planes = [0.01, 0.02, 0.03]  # Depths in meters

# Step 1: Create Intensity Maps for a 3D Object
x = np.linspace(-1, 1, grid_size)
y = np.linspace(-1, 1, grid_size)
X, Y = np.meshgrid(x, y)

intensity_slices = [
    np.exp(-((X**2 + Y**2) / 0.1**2)),               # Near plane
    np.exp(-(((X - 0.2)**2 + (Y + 0.2)**2) / 0.1**2)),  # Mid plane
    np.exp(-(((X + 0.3)**2 + (Y - 0.3)**2) / 0.1**2)),  # Far plane
]

# Step 2: Compute Hologram Using Fresnel Diffraction
def fresnel_propagation(field, z, wavelength, pixel_pitch):
    k = 2 * np.pi / wavelength  # Wave number
    fx = np.fft.fftfreq(grid_size, pixel_pitch)
    fy = np.fft.fftfreq(grid_size, pixel_pitch)
    FX, FY = np.meshgrid(fx, fy)
    H = np.exp(1j * k * z) * np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))
    return np.fft.ifft2(np.fft.fft2(field) * H)

hologram_complex = np.zeros((grid_size, grid_size), dtype=complex)
for z, intensity in zip(z_planes, intensity_slices):
    hologram_complex += fresnel_propagation(np.sqrt(intensity), z, wavelength, pixel_pitch)

hologram_intensity = np.abs(hologram_complex)**2
hologram_phase = np.angle(hologram_complex)
hologram_phase_fft = np.angle(np.fft.fftshift(np.fft.fft2(hologram_complex)))

# Step 3: Reconstruct Using IFFT
reconstructed_slices = []
for z in z_planes:
    reconstructed_field = fresnel_propagation(hologram_complex, -z, wavelength, pixel_pitch)
    reconstructed_slices.append(np.abs(reconstructed_field)**2)

current_field_GS = np.fft.ifft2(np.fft.ifftshift(np.exp(1j * hologram_phase_fft)))

# Step 4: 3D Visualization
fig = plt.figure(figsize=(12, 8))

# 3D plots for reconstructed slices
for i, (z, reconstructed) in enumerate(zip(z_planes, reconstructed_slices)):
    ax = fig.add_subplot(2, len(z_planes), i + 1, projection='3d')
    ax.plot_surface(X, Y, reconstructed, cmap='viridis', edgecolor='none')
    ax.set_title(f"Reconstruction at z={z:.2f} m")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Intensity")

# 3D plot for GS result
ax_gs = fig.add_subplot(2, 1, 2, projection='3d')
ax_gs.plot_surface(X, Y, np.abs(current_field_GS)**2, cmap='viridis', edgecolor='none')
ax_gs.set_title("Reconstruction Using GS")
ax_gs.set_xlabel("X")
ax_gs.set_ylabel("Y")
ax_gs.set_zlabel("Intensity")

plt.tight_layout()
plt.show()
# 3D plot for all z reconstructions in one plot with x, y, z axes
fig = plt.figure(figsize=(12, 8))

ax = fig.add_subplot(111, projection='3d')

# Create meshgrid for X, Y, and Z values
X, Y = np.meshgrid(x, y)

# Loop through each z-plane and add the intensity values
for i, (z, reconstructed) in enumerate(zip(z_planes, reconstructed_slices)):
    # Flatten the intensity and coordinate arrays for plotting
    X_flat, Y_flat = np.meshgrid(x, y)
    Z_flat = np.full_like(X_flat, z)  # Set Z values to the current z-plane for all (X, Y) points
    Intensity_flat = reconstructed.flatten()  # Flatten the intensity map

    # Plot the surface for the current z-plane
    ax.scatter(X_flat.flatten(), Y_flat.flatten(), Z_flat.flatten(), c=Intensity_flat, cmap='viridis', marker='o', s=1)

ax.set_title("3D Reconstruction at Different z-Planes")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.tight_layout()
plt.show()

