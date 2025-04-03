# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 15:05:31 2025

@author:Shang Gao 
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn, fftshift, ifftshift

im = plt.imread("C:/Users/Laboratorio/3D_CGH/3d_FFT/fl_one_s.png")[:,:,:3]  # Example 3D data
# im_shift=fftshift(im)
h,w,c=im.shape
rand=np.random.rand(h,w,c)
rand_2pi=rand*2*np.pi

p=3
field=np.exp(1j*rand_2pi)*np.sqrt(im)**p
print(f"p={p}, avg R={np.average(abs(field[:,:,0]))}, G={np.average(abs(field[:,:,1]))}, B={np.average(abs(field[:,:,2]))}")
fft_nd = fftn(field)
phase=np.angle(fft_nd)

ifft_nd = ifftn(np.exp(1j*phase))
# ifft_nd = ifftshift(ifftn(np.exp(1j*phase)))

# plt.figure()
# plt.imshow(im)
# plt.show()

# plt.figure()
# plt.imshow(abs(fft_nd))
# plt.show()

# plt.figure()
# plt.imshow(phase)
# plt.show()

plt.figure()
plt.imshow(abs(ifft_nd)**2*10e5)

plt.show()
