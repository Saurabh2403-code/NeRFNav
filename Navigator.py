import numpy as np
import heapq
import matplotlib.pyplot as plt
from simulator import drone
import matplotlib.pyplot as plt

class Node:
    def __init__(self, coordinates, parent):
        self.coordinates = coordinates
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0
        
    def __lt__(self, other):
        return self.f < other.f
    
def heuristic(c1, c2):
    return np.sqrt(((c1[0]-c2[0])**2) + ((c1[1]-c2[1])**2) + ((c1[2]-c2[2])**2))

def get_neighbors(current_node, grid_shape, collision_map):
    neighbors = []
    x, y, z = current_node.coordinates

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                
                new_x, new_y, new_z = x + dx, y + dy, z + dz
                
                if (new_x < 0 or new_x >= grid_shape[0] or 
                    new_y < 0 or new_y >= grid_shape[1] or 
                    new_z < 0 or new_z >= grid_shape[2]):
                    continue
                
                if collision_map[new_x, new_y, new_z] == 1:
                    continue
                    
                neighbors.append(Node((new_x, new_y, new_z), current_node))
    return neighbors
collision_map = np.load("map.npy") 
grid_shape = collision_map.shape  
print(f"Map Loaded. Shape: {grid_shape}")

start = (35, 25, 35)
end = (30, 105, 100)

if collision_map[start] == 1 or collision_map[end] == 1:
    print(" ERROR: Start or Goal is inside a wall! Move them.")
    exit()

open_list = []
done_set = set()
start_node = Node(start, None)
start_node.g = 0
start_node.h = heuristic(start, end)
start_node.f = start_node.g + start_node.h

heapq.heappush(open_list, start_node)

print(f"Starting Search from {start} to {end}...")

count = 0 
path_found = False 

while len(open_list) > 0:
    count += 1
    if count % 5000 == 0: 
        best_node = open_list[0]
        print(f"Searching... Step {count}. Distance to Goal: {best_node.h:.2f}")

    current_node = heapq.heappop(open_list)
    
    if current_node.coordinates == end:
        print("Path Found!")
        path_found = True
        
        path = []
        curr = current_node
        while curr is not None:
            path.append(curr.coordinates)
            curr = curr.parent
        path = path[::-1] # Reverse
        print(f"Path Length: {len(path)} steps")
        break
    done_set.add(current_node.coordinates)
    
    neighbors = get_neighbors(current_node, grid_shape, collision_map)
    
    for neighbor in neighbors:
        if neighbor.coordinates in done_set:
            continue
            
        neighbor.g = current_node.g + 1
        neighbor.h = heuristic(neighbor.coordinates, end)
        neighbor.f = neighbor.g + neighbor.h
        
        heapq.heappush(open_list, neighbor)

if not path_found:
    print("No path found. The drone is trapped or the goal is unreachable.")


distance=0.0
for i in range(len(path)-1):
    dist=heuristic(path[i],path[i+1])
    distance+=dist
print(distance)
def chaikin(c1,c2):
    """
    c1 and c2 are co ordinates in which I have to smoothen the path,now
    in between c1 and c1 we will define two new points q and r,such that
    q=(3/4)*c1+(1/4)*c2
    r=(1/4)c1+(3/4)*c2
    """
    c1=np.array(c1)
    c2=np.array(c2)
    return tuple((3/4)*c1+(1/4)*c2),tuple((1/4)*c1+(3/4)*c2)
def corner(path,n):
    for j in range(n):
        out=[path[0]]
        for i in range(len(path)-1):
            new_point=chaikin(path[i],path[i+1])
            out.append(new_point[0])
            out.append(new_point[1])
        out.append(path[-1])
        path=out
    return path
smooth_path=corner(path,5)

# if path_found:
#     path_x = [p[0] for p in smooth_path]
#     path_y = [p[1] for p in smooth_path]
#     path_z = [p[2] for p in smooth_path]

#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot(path_x, path_y, path_z, color='red', linewidth=3, label='Drone Path')
#     ax.scatter(*start, color='green', s=100, label='Start')
#     ax.scatter(*end, color='blue', s=100, label='Goal')

#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     plt.legend()
#     plt.title("Mission Complete")
#     plt.show()


start_pos = smooth_path[0]
drone = drone(mass=1.0, velocity=[0,0,0], position=start_pos)

Kp = 1.5  
Kd = 1.0  
dt = 0.1  

real_path = []  

current_target_idx = 0
max_steps = 3000 

print(f"Taking off! Tracking {len(smooth_path)} waypoints...")

for step in range(max_steps):
    if current_target_idx < len(smooth_path):
        target = np.array(smooth_path[current_target_idx])
    else:

        target = np.array(smooth_path[-1])
        
    error = target - drone.position
    dist_to_target = np.linalg.norm(error)
    

    if dist_to_target < 0.5 and current_target_idx < len(smooth_path) - 1:
        current_target_idx += 1
        
    force = (Kp * error) - (Kd * drone.velocity)
    
    drone.update(force, dt)
    real_path.append(drone.position.copy())
    
    if current_target_idx == len(smooth_path) - 1 and dist_to_target < 0.1:
        print(f"Mission Complete at step {step}!")
        break


real_path = np.array(real_path)
smooth_path_np = np.array(smooth_path) 
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.plot(smooth_path_np[:,0], smooth_path_np[:,1], smooth_path_np[:,2], 
        color='red', linestyle='--', linewidth=2, label='Planned Trajectory')

ax.plot(real_path[:,0], real_path[:,1], real_path[:,2], 
        color='blue', linewidth=3, label='Actual Flight')
is_obstacle=collision_map==1
ax.voxels(is_obstacle, edgecolor='k', alpha=0.05) 

ax.scatter(*start, color='green', s=100, label='Start')
ax.scatter(*end, color='purple', s=100, label='Goal')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Simulated Drone Flight")
plt.legend()
plt.savefig('Visual path with the rendered object.png',dpi=300,bbox_inches='tight')
plt.show()