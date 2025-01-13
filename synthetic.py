import numpy as np
import pandas as pd

# Hexagon parameters
center_x, center_y = 128, 128  # Center of the hexagon
radius = 80  # Radius of the hexagon (chosen to fit within the 256x256 grid)
num_vertices = 6  # Number of vertices for a hexagon

# Parameters for velocity and loops
velocity_multiplier = 15  # Increase this to simulate faster movement
num_loops = 3  # Number of times to loop around the hexagon

# Calculate the angles for each vertex (in radians)
angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)

# Generate hexagon vertices
hexagon_coords = [(center_x + radius * np.cos(angle), center_y + radius * np.sin(angle)) for angle in angles]

# Number of steps between each vertex (smoothness), scaled by velocity_multiplier
steps_between_vertices = max(1, int(50 / velocity_multiplier))

# List to store all the points moving along the hexagonal path
path_points = []

# Function to linearly interpolate between two points
def interpolate_points(p1, p2, steps):
    return [(p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])) for t in np.linspace(0, 1, steps)]

# Interpolate between each consecutive vertex pair and assign a timestamp
timestamp = 0
for loop in range(num_loops):  # Loop multiple times around the hexagon
    for i in range(len(hexagon_coords)):
        start_vertex = hexagon_coords[i]
        end_vertex = hexagon_coords[(i + 1) % num_vertices]  # Wrap around to the first vertex
        segment_points = interpolate_points(start_vertex, end_vertex, steps_between_vertices)
        
        # Assign a timestamp of 1 to each point
        for j in range(1, len(segment_points)):
            timestamp += 1  # Increment timestamp by 1 for each point
            path_points.append((segment_points[j][0], segment_points[j][1]))

# Convert to a DataFrame for better presentation
df_hex_path = pd.DataFrame(path_points, columns=["x", "y"])

# Display the first few generated coordinates with timestamps
print(df_hex_path.head())

# Save to CSV if needed
df_hex_path.to_csv("hexagon_path_with_velocity_and_loops.csv", index=False)
