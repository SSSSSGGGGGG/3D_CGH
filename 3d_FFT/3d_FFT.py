# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 15:05:31 2025

@author:Shang Gao 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn, fftshift
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Create or load 3D data cube
size=200
cube_em=np.zeros((size,size,size))
center=size//2
# cube_em[center, center, center] = 1

size_s=10
cube_bl=np.ones((size_s,size_s,size_s))
cube_em[center-size_s//2:center+size_s//2, center-size_s//2:center+size_s//2, center-size_s//2:center+size_s//2] = cube_bl
# For demonstration, only show a subset to reduce the number of points
x, y, z = np.indices((size, size, size))
mask_em = cube_em > 0.5  # Will be empty here
x2, y2, z2 = np.indices((size_s, size_s, size_s))
mask_bl = cube_bl > 0.5  # Will be full

# print("cube_bl shape:", cube_bl.shape)
# print("Slice shape in cube_em:", cube_em[
#     center - size_s // 2 : center + size_s // 2,
#     center - size_s // 2 : center + size_s // 2,
#     center - size_s // 2 : center + size_s // 2
# ].shape)

fig=plt.figure()
# Subplot 1: full cube with embedded small cube
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(x[mask_em], y[mask_em], z[mask_em], c=cube_em[mask_em], cmap="hot", s=5)
ax.set_title("Large Cube with Embedded Small Cube")
plt.colorbar(sc, ax=ax)
ax.set_xlim(0, size)
ax.set_ylim(0, size)
ax.set_zlim(0, size)
plt.show()

# fig2=plt.figure()
# # Subplot 2: sparse full cube
# ax2 = fig2.add_subplot(111, projection="3d")
# sc2 = ax2.scatter(x2[mask_bl], y2[mask_bl], z2[mask_bl], c=cube_bl[mask_bl], cmap="hot", s=5)
# ax2.set_title("Filled Cube (Sparse)")
# plt.colorbar(sc2, ax=ax2)
# ax2.set_xlim(0, size_s)
# ax2.set_ylim(0, size_s)
# ax2.set_zlim(0, size_s)
# plt.show()

# cube = np.random.rand(size, size, size)
rand=np.random.uniform(size, size, size)
rand_2pi=rand*2*np.pi
# cube=cube*rand_2pi
# # Step 2: FFT
# fft_cube = fftn(cube_em)
field=np.sqrt(cube_em)*rand_2pi
fft_cube = fftshift(fftn(field))
max_=abs(fft_cube).max()
mask_fft=fft_cube>0.05*max_
fig2=plt.figure()
# Subplot 2: sparse full cube
ax2 = fig2.add_subplot(111, projection="3d")
sc2 = ax2.scatter(x[mask_fft], y[mask_fft], z[mask_fft], c=abs(fft_cube)[mask_fft], cmap="hot", s=5)
ax2.set_title("FFT Cube")
plt.colorbar(sc2, ax=ax2)
ax2.set_xlim(0, size)
ax2.set_ylim(0, size)
ax2.set_zlim(0, size)
plt.show()

# plt.figure()
# plt.imshow(abs(fft_cube)[:,:,100],cmap="hot")
# plt.show()

# Step 3: Extract phase only
phase = np.angle(fft_cube)
phase_only_cube = np.exp(1j * phase)

# Step 4: Inverse FFT using phase only
reconstructed = ifftn(abs(fft_cube)*phase_only_cube)

# Step 5: Intensity (squared magnitude)
intensity = np.abs(reconstructed)#**2
h,w,t=intensity.shape
x3, y3, z3 = np.indices((h,w,t))
re_mask=intensity>0.0005*max_
fig3=plt.figure()
ax3 = fig3.add_subplot(111, projection="3d")
sc3 = ax3.scatter(x3[re_mask], y3[re_mask], z3[re_mask], c=intensity[re_mask], cmap="hot", s=5)
ax2.set_title("FFT Cube")
plt.colorbar(sc3, ax=ax3)
ax2.set_xlim(0, size)
ax2.set_ylim(0, size)
ax2.set_zlim(0, size)
plt.show()
# # Step 6: Threshold for visualization
# threshold = 0.7 * intensity.max()
# coords = np.array(np.where(intensity > threshold)).T
# values = intensity[intensity > threshold]

# # Step 7: 3D Visualization
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=values, cmap='inferno', s=2)

# ax.set_title("3D iFFT from Phase Only")
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.colorbar(sc, label="Reconstructed Magnitude (|iFFT|)")#|iFFT|²
# plt.tight_layout()
# plt.show()

# coords2 = np.array(np.where(cube > threshold)).T
# values2 = cube[cube > threshold]
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(coords2[:, 0], coords2[:, 1], coords2[:, 2], c=values2, cmap='inferno', s=2)

# ax.set_title("3D iFFT from Phase Only")
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.colorbar(sc, label="Original)")#|iFFT|²
# plt.tight_layout()
# plt.show()


