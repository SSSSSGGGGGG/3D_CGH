# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 15:04:59 2025

@author:Shang Gao 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn, fftshift

size = 10
cube_rgb = np.zeros((size, size, size, 3), dtype=np.float32)  # Last dimension is for RGB

# Set some voxels to color
cube_rgb[5, 5, 5] = [1.0, 0.0, 0.0]   # Red
cube_rgb[5, 5, 6] = [0.0, 0.0, 0.0]   # Green
cube_rgb[5, 5, 7] = [0.0, 0.0, 0.0]   # Blue

field=np.sqrt(cube_rgb)#*rand_2pi
fft_cube = fftshift(fftn(field))

max_=abs(fft_cube).max()
mask_fft=np.any(fft_cube>0.3*max_, axis=-1)

x, y, z = np.indices((size, size, size))
mask = np.any(cube_rgb > 0, axis=-1)  # Only plot colored voxels

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = cube_rgb[mask]  # shape: (N, 3)
sc = ax.scatter(x[mask], y[mask], z[mask], c=colors, s=50)

ax.set_title("3D RGB Cube")
plt.show()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
colors2 = (abs(fft_cube)[mask_fft])/1e5  # shape: (N, 3)
sc2 = ax2.scatter(x[mask_fft], y[mask_fft], z[mask_fft], c=colors2, s=50)

ax.set_title("FT 3D RGB Cube")
plt.show()

phase = np.angle(fft_cube)
phase_only_cube = np.exp(1j * phase)

# Step 4: Inverse FFT using phase only
reconstructed = ifftn(abs(fft_cube)*phase_only_cube)

# Step 5: Intensity (squared magnitude)
intensity = np.abs(reconstructed)#**2
h,w,t,co=intensity.shape
x3, y3, z3 = np.indices((h,w,t))
re_mask=np.any(intensity>0.00000005*max_,axis=-1)
fig3=plt.figure()
ax3 = fig3.add_subplot(111, projection="3d")
sc3 = ax3.scatter(x3[re_mask], y3[re_mask], z3[re_mask], c=intensity[re_mask]*1e-4, s=5)
ax2.set_title("FFT Cube")
plt.colorbar(sc3, ax=ax3)
ax2.set_xlim(0, h)
ax2.set_ylim(0, w)
ax2.set_zlim(0, t)
plt.show()