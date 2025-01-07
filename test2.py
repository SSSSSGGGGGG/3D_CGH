# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 16:48:58 2024

@author: gaosh
"""
import numpy as np
import matplotlib.pyplot as plt

lambda_r = 0.680e-6  # Red wavelength
lambda_g = 0.532e-6  # Green wavelength (reference)
lambda_b = 0.461e-6  # Blue wavelength

kr = 2 * np.pi / lambda_r
# Original 2D image
image_size = 512  # Size of the square image
image = np.zeros((image_size, image_size))
image[200:300, 200:300] = 1  # Simple square object

# Compute 2D FFT to get the hologram
fft_hologram = np.fft.fftshift(np.fft.fft2(image))
phase_hologram_2d = np.angle(fft_hologram)  # Phase-only hologram

# Display the phase hologram
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.title("Phase-Only Hologram (2D FFT)")
plt.imshow(phase_hologram_2d, cmap="seismic")
plt.colorbar()
plt.show()

# Voxel Grid (3D object representation)
voxel_grid = np.zeros((20, image_size, image_size))
voxel_grid[10, 200:300, 200:300] = 1  # A 3D object slice

# Compute Wave Field by Summing Fresnel Contributions from Each Slice
wave_field = np.zeros((image_size, image_size), dtype=complex)
z_distances = np.linspace(0.01, 0.1, 20)  # Slice depths in meters

for z_idx, z in enumerate(z_distances):
    slice_wave = voxel_grid[z_idx] * np.exp(1j * kr * z)  # Add phase shift
    wave_field += slice_wave

# Compute the Phase Hologram
phase_hologram_3d = np.angle(wave_field)

# Display the Phase Hologram for 3D
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Voxel Grid Object (Single Slice)")
plt.imshow(voxel_grid[10], cmap="gray")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.title("Phase-Only Hologram (3D Wave Field)")
plt.imshow(phase_hologram_3d, cmap="seismic")
plt.colorbar()
plt.show()
