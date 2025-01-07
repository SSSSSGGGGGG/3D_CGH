# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 01:25:36 2024

@author: gaosh
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftshift, fft2, ifft2

# Parameters
N = 100  # Resolution (pixels)
cube_size = 50  # Cube size in pixels
num_slices = 10  # Number of slices to represent the cube
wavelength = 633e-9  # Wavelength (meters)
z = 0.1  # Propagation distance to reconstruction plane (meters)
dx = 10e-6  # Pixel size (meters)
k = 2 * np.pi / wavelength  # Wave number

# Step 1: Define a 3D Cube (representing it as a stack of 2D slices)
cube = np.zeros((num_slices, N, N))  # A stack of slices representing the cube

# Simulate the cube by placing a square in the middle of each slice
for i in range(num_slices):
    start = (N - cube_size) // 2
    end = start + cube_size
    cube[i, start:end, start:end] = 1  # Square projection in each slice

# Step 2: Compute Fourier Transforms (Fraunhofer diffraction) for Each Slice
fourier_phase_slices = []
for i in range(num_slices):
    fourier_plane_field = fftshift(fft2(cube[i]))
    fourier_phase = np.angle(fourier_plane_field)
    fourier_phase_slices.append(np.mod(fourier_phase, 2 * np.pi))  # Wrap phase for SLM

# Step 3: Combine the Phase Information and Apply Inverse FFT

# Combine the phase information across all slices by summing them (or averaging)
combined_phase = np.sum(fourier_phase_slices, axis=0)  # Sum phase information from all slices

# Step 4: Apply Inverse FFT to simulate reconstruction at the target plane
reconstructed_field = np.exp(1j * combined_phase)  # Apply the phase information to the complex field
reconstructed_cube = ifft2(reconstructed_field)  # Inverse FFT to get the reconstructed light field

# Step 5: Compute the Intensity of the Reconstructed Field
intensity_reconstructed = np.abs(reconstructed_cube)**2

# Step 6: Visualize the Reconstruction

plt.figure(figsize=(12, 6))

# Show the original cube and the intensity of the reconstruction
for i in range(3):  # Show first 3 slices of the original and reconstructed results
    plt.subplot(3, 2, 2*i+1)
    plt.imshow(cube[i], cmap='gray')
    plt.title(f"Original Slice {i+1}")
    plt.colorbar()

    plt.subplot(3, 2, 2*i+2)
    plt.imshow(intensity_reconstructed, cmap='hot')
    plt.title(f"Reconstructed Intensity at Slice {i+1}")
    plt.colorbar()

plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftshift, fft2, ifft2
from mpl_toolkits.mplot3d import Axes3D

# Parameters
N = 256  # Resolution (pixels)
cube_size = 20  # Cube size in pixels
num_slices = 10  # Number of slices to represent the cube
wavelength = 633e-9  # Wavelength (meters)
z = 0.1  # Propagation distance to reconstruction plane (meters)
dx = 10e-6  # Pixel size (meters)
k = 2 * np.pi / wavelength  # Wave number

# Step 1: Define a 3D Cube (representing it as a stack of 2D slices)
cube = np.zeros((num_slices, N, N))  # A stack of slices representing the cube

# Simulate the cube by placing a square in the middle of each slice
for i in range(num_slices):
    start = (N - cube_size) // 2
    end = start + cube_size
    cube[i, start:end, start:end] = 1  # Square projection in each slice

# Step 2: Compute Fourier Transforms (Fraunhofer diffraction) for Each Slice
fourier_phase_slices = []
for i in range(num_slices):
    fourier_plane_field = fftshift(fft2(cube[i]))
    fourier_phase = np.angle(fourier_plane_field)
    fourier_phase_slices.append(np.mod(fourier_phase, 2 * np.pi))  # Wrap phase for SLM

# Step 3: Combine the Phase Information and Apply Inverse FFT

# Combine the phase information across all slices by summing them (or averaging)
combined_phase = np.sum(fourier_phase_slices, axis=0)  # Sum phase information from all slices

# Step 4: Apply Inverse FFT to simulate reconstruction at the target plane
reconstructed_field = np.exp(1j * combined_phase)  # Apply the phase information to the complex field
reconstructed_cube = ifft2(reconstructed_field)  # Inverse FFT to get the reconstructed light field

# Step 5: Compute the Intensity of the Reconstructed Field
intensity_reconstructed = np.abs(reconstructed_cube)**2

# Step 6: Plotting the Original Cube, Reconstructed Cube, and Phase Information

# Plot the original cube
fig = plt.figure(figsize=(15, 8))

# Plot original cube
ax1 = fig.add_subplot(131, projection='3d')
X = np.arange(N)
Y = np.arange(N)
X, Y = np.meshgrid(X, Y)

for i in range(num_slices):
    ax1.plot_surface(X, Y, cube[i], rstride=1, cstride=1, color=plt.cm.viridis(np.random.rand()), alpha=0.7)

ax1.set_title("Original Cube")
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Intensity')

# Plot the reconstructed cube (intensity)
ax2 = fig.add_subplot(132, projection='3d')
for i in range(num_slices):
    ax2.plot_surface(X, Y, np.abs(reconstructed_cube)**2, rstride=1, cstride=1, color=plt.cm.viridis(np.random.rand()), alpha=0.7)

ax2.set_title("Reconstructed Cube")
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Intensity')

# Plot the phase of the reconstructed field
ax3 = fig.add_subplot(133, projection='3d')
phase_reconstructed = np.angle(reconstructed_cube)
ax3.plot_surface(X, Y, phase_reconstructed, rstride=1, cstride=1, cmap='hsv', alpha=0.7)

ax3.set_title("Phase of Reconstructed Cube")
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Phase')

plt.tight_layout()
plt.show()
