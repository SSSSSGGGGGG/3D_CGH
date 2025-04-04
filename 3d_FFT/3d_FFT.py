# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 15:05:31 2025

@author:Shang Gao 
"""

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.fft import fftn, ifftn, fftshift, ifftshift

# im = plt.imread("C:/Users/Laboratorio/3D_CGH/3d_FFT/fl_one_s.png")[:,:,:3]  # Example 3D data
# im=np.array(im)# im_shift=fftshift(im)
# h,w,c=im.shape
# rand=np.random.rand(h,w,c)
# rand_2pi=rand*2*np.pi

# p=1
# field=np.exp(1j*rand_2pi)*np.sqrt(im)**p
# print(f"p={p}, avg R={np.average(abs(field[:,:,0]))}, G={np.average(abs(field[:,:,1]))}, B={np.average(abs(field[:,:,2]))}")
# fft_nd = fftn(field)
# phase=np.angle(fft_nd)

# ifft_nd = ifftn(np.exp(1j*phase))
# # ifft_nd = ifftshift(ifftn(np.exp(1j*phase)))

# # plt.figure()
# # plt.imshow(im)
# # plt.show()

# # plt.figure()
# # plt.imshow(abs(fft_nd))
# # plt.show()

# # plt.figure()
# # plt.imshow(phase)
# # plt.show()

# plt.figure()
# plt.imshow(abs(ifft_nd)**2*10e5)

# plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Create or load 3D data cube
cube = np.random.rand(640, 640, 640)
rand=np.random.rand(640, 640, 640)
rand_2pi=rand*2*np.pi
cube=cube*rand_2pi
# Step 2: FFT
fft_cube = fftn(cube)

# Step 3: Extract phase only
phase = np.angle(fft_cube)
phase_only_cube = np.exp(1j * phase)

# Step 4: Inverse FFT using phase only
reconstructed = ifftn(phase_only_cube)

# Step 5: Intensity (squared magnitude)
intensity = np.abs(reconstructed)#**2

# Step 6: Threshold for visualization
threshold = 0.7 * intensity.max()
coords = np.array(np.where(intensity > threshold)).T
values = intensity[intensity > threshold]

# Step 7: 3D Visualization
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=values, cmap='inferno', s=2)

ax.set_title("3D iFFT from Phase Only")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.colorbar(sc, label="Reconstructed Intensity (|iFFT|²)")
plt.tight_layout()
plt.show()


