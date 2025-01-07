# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 10:55:32 2024

@author: gaosh
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the vertices of the cube
vertices = [
    [0, 0, 0],  # Vertex 0
    [1, 0, 0],  # Vertex 1
    [1, 1, 0],  # Vertex 2
    [0, 1, 0],  # Vertex 3
    [0, 0, 1],  # Vertex 4
    [1, 0, 1],  # Vertex 5
    [1, 1, 1],  # Vertex 6
    [0, 1, 1]   # Vertex 7
]

# Define the edges of the cube by connecting vertices
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom square
    (4, 5), (5, 6), (6, 7), (7, 4),  # Top square
    (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
]

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the vertices as points
for vertex in vertices:
    ax.scatter(*vertex, color='blue')

# Plot the edges as lines
for edge in edges:
    start, end = edge
    x = [vertices[start][0], vertices[end][0]]
    y = [vertices[start][1], vertices[end][1]]
    z = [vertices[start][2], vertices[end][2]]
    ax.plot(x, y, z, color='black')

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
