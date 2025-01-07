# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 16:55:41 2024

@author: gaosh
"""

import numpy as np
import matplotlib.pyplot as plt

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
hologram_phase_fft=np.angle(np.fft.fftshift(np.fft.fft2(hologram_complex)))

# Step 3: Reconstruct Using IFFT
reconstructed_slices = []
for z in z_planes:
    reconstructed_field = fresnel_propagation(hologram_complex, -z, wavelength, pixel_pitch)
    reconstructed_slices.append(np.abs(reconstructed_field)**2)
current_field_GS = np.fft.ifft2(np.fft.ifftshift(np.exp(1j * hologram_phase_fft)))
# Step 4: Visualization
# Input Intensity Slices
plt.figure(figsize=(12, 4))
for i, intensity in enumerate(intensity_slices):
    plt.subplot(1, len(intensity_slices), i + 1)
    plt.imshow(intensity, cmap='gray')
    plt.title(f"Input Slice at z={z_planes[i]:.2f} m")
    plt.axis('off')
plt.show()

# Hologram Intensity
plt.figure(figsize=(6, 6))
plt.imshow(hologram_intensity, cmap='gray')
plt.title("Generated Hologram Intensity")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(abs(current_field_GS)**2, cmap='gray')
plt.title("reconstruction Generated Hologram angle")
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(hologram_phase_fft, cmap='gray')
plt.title("fft of Generated Hologram angle")
plt.colorbar()
plt.show()

# Reconstructed Slices
plt.figure(figsize=(12, 4))
for i, reconstructed in enumerate(reconstructed_slices):
    plt.subplot(1, len(reconstructed_slices), i + 1)
    plt.imshow(reconstructed, cmap='gray')
    plt.title(f"Reconstruction at z={z_planes[i]:.2f} m")
    plt.axis('off')
plt.show()
