# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 10:52:16 2024

@author: gaosh
"""

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

# Define the vertices of the cube
vertices = np.array([
    [0, 0, 0],  # Vertex 0
    [1, 0, 0],  # Vertex 1
    [1, 1, 0],  # Vertex 2
    [0, 1, 0],  # Vertex 3
    [0, 0, 1],  # Vertex 4
    [1, 0, 1],  # Vertex 5
    [1, 1, 1],  # Vertex 6
    [0, 1, 1]   # Vertex 7
])

# Define the faces of the cube using the vertices
faces = [
    [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
    [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top face
    [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front face
    [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back face
    [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right face
    [vertices[0], vertices[3], vertices[7], vertices[4]]   # Left face
]

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create a 3D polygon collection for the faces
poly3d = Poly3DCollection(faces, alpha=0.5, edgecolor='k')
ax.add_collection3d(poly3d)

# Set the limits of the plot
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

# Add labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Display the plot
plt.show()
