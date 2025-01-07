# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 15:26:36 2024

@author: gaosh
"""

import numpy as np
import matplotlib.pyplot as plt

# Define wavelengths (in meters, for example)
lambda_r = 0.680e-6  # Red wavelength
lambda_g = 0.532e-6  # Green wavelength (reference)
lambda_b = 0.461e-6  # Blue wavelength

kr = 2 * np.pi / lambda_r  # Wave number
z = 0.5  # Propagation distance (meters)
grid_size = 1024  # Resolution of hologram
pixel_pitch = 4.5e-6  # Pixel pitch of SLM (6.4 µm)
x = np.linspace(-grid_size//2, grid_size//2 - 1, grid_size) * pixel_pitch
y = np.linspace(-grid_size//2, grid_size//2 - 1, grid_size) * pixel_pitch
X, Y = np.meshgrid(x, y)

def fresnelKernel(x,y,z,wavlength):
    r2=x**2+y**2
    return np.exp(1j*kr/(2*z)*r2)

def fresnelPropagation(u0,z,wavelength):
    kernel=fresnelKernel(X, Y, z, wavelength)
    u1=u0*kernel
    u2=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u1)))
    return u2

def hologram_f_pointCloud(points):
    hologram=np.zeros((grid_size,grid_size),dtype=complex)
    for x,y,z in points:
        x_idx=int(x/pixel_pitch+grid_size//2)
        y_idy=int(y/pixel_pitch+grid_size//2)
        if 0<=x_idx<grid_size and 0 <=y_idy<grid_size:
            hologram[y_idy,x_idx]+=np.exp(1j * kr * z)
    return hologram

# Voxel Grid Input
def hologram_from_voxel_grid(voxels):
    hologram = np.zeros((grid_size, grid_size), dtype=complex)
    for z_slice, intensity in enumerate(voxels):
        z = z_slice * pixel_pitch
        hologram += intensity * np.exp(1j * kr * z)
    return hologram

# Depth Map Input
def hologram_from_depth_map(depth_map):
    depth_map_rescaled = depth_map / np.max(depth_map) * z
    hologram = np.exp(1j * kr * depth_map_rescaled)
    return hologram

point_C=[(0, 0, 0.1),  # (x, y, z) coordinates in meters
    (1e-4, -1e-4, 0.2),
    (-1e-4, 1e-4, 0.3),]
# 2. Voxel Grid (3D object represented as slices)
voxel_grid = np.zeros((10, grid_size, grid_size))
voxel_grid[5, grid_size//2-10:grid_size//2+10, grid_size//2-10:grid_size//2+10] = 1  # A slice

# 3. Depth Map (2D image, values represent depth)
depth_map = np.random.rand(grid_size, grid_size)
holo_point_cloud = hologram_f_pointCloud(point_C)
holo_voxel_grid = hologram_from_voxel_grid(voxel_grid)
holo_depth_map = hologram_from_depth_map(depth_map)
def plot_hologram(hologram, title):
    intensity = np.abs(hologram)
    plt.figure()
    plt.imshow(np.log(1 + intensity), cmap='gray')
    plt.title(title)
    plt.colorbar()
    plt.show()
    
plot_hologram(holo_point_cloud, "Hologram from Point Cloud")
plot_hologram(holo_voxel_grid, "Hologram from Voxel Grid")
plot_hologram(holo_depth_map, "Hologram from Depth Map")