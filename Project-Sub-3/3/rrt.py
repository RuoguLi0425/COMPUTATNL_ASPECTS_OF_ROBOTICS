import sim
import pybullet as p
import numpy as np
import random

random.seed(0)
MAX_ITERS = 10000
delta_q = 0.5
steer_goal_p = 0.01
move_speed = 0.01
def visualize_path(q_1, q_2, env, color=[0, 1, 0]):
    """
    Draw a line between two positions corresponding to the input configurations
    :param q_1: configuration 1
    :param q_2: configuration 2
    :param env: environment
    :param color: color of the line, please leave unchanged.
    """
    # obtain position of first point
    env.set_joint_positions(q_1)
    point_1 = p.getLinkState(env.robot_body_id, 9)[0]
    # obtain position of second point
    env.set_joint_positions(q_2)
    point_2 = p.getLinkState(env.robot_body_id, 9)[0]
    # draw line between points
    p.addUserDebugLine(point_1, point_2, color, 1.0)

def execute_path(path_conf, env):
    # ========= TODO: Problem 3 ========
    # 1. Execute the path while visualizing the location of joint 5
    #    (see Figure 2 in homework manual)
    #    You can get the position of joint 5 with:
    #         p.getLinkState(env.robot_body_id, 9)[0]
    #    To visualize the position, you should use sim.SphereMarker
    #    (Hint: declare a list to store the markers)
    # 2. Drop the object (Hint: open gripper, step the simulation, close gripper)
    # 3. Return the robot to original location by retracing the path

    # Create an empty list to store sphere markers
    sphere_markers = []

    # Move the robot along the path while visualizing joint 5's position with a sphere marker
    for conf in path_conf:
        env.move_joints(conf, speed=move_speed)
        body_pos = p.getLinkState(env.robot_body_id, 9)[0]
        sphere_markers.append(sim.SphereMarker(body_pos))

    # Open the gripper, step the simulation, and close the gripper to drop the object
    env.open_gripper()
    p.stepSimulation()
    env.close_gripper()

    # Move the robot back to its original location by reversing the path
    for conf in reversed(path_conf):
        env.move_joints(conf, speed=move_speed)

    # ==================================
    return None

def rrt(q_init, q_goal, MAX_ITERS, delta_q, steer_goal_p, env):
    """
    :param q_init: initial configuration
    :param q_goal: goal configuration
    :param MAX_ITERS: max number of iterations
    :param delta_q: steer distance
    :param steer_goal_p: probability of steering towards the goal
    :returns path: series of joint angles
    """
    # ========== PART 3 =========
    # TODO: Implement RRT code here. This function should return a list of joint configurations
    # that the robot should take in order to reach q_goal starting from q_init

    # Convert the initial and goal configurations to tuples
    q_init = tuple(q_init)
    q_goal = tuple(q_goal)

    # Initialize the set of vertices, the set of edges, and the parent map
    V = {q_init}
    E = set()
    parent_map = {q_init: None}

    # Loop through the maximum number of iterations
    for i in range(MAX_ITERS):
        # Sample a new configuration randomly
        q_rand = SemiRandomSample(steer_goal_p, q_goal)

        # Find the nearest vertex in the tree to the random configuration
        q_nearest = Nearest(V, q_rand)

        # Steer from the nearest vertex towards the random configuration
        q_new = Steer(q_nearest, q_rand, delta_q)

        # Check if the new vertex is obstacle-free
        if ObstacleFree(q_nearest, q_new, env):
            # Add the new vertex to the set of vertices
            # the edge connecting it to the nearest vertex to the set of edges
            V.add(q_new)
            E.add((q_nearest, q_new))

            # Update the parent map with the new vertex and its parent
            parent_map[q_new] = q_nearest

            # Visualize the new edge
            visualize_path(q_nearest, q_new, env)

            # If the new vertex is close enough to the goal
            # add the goal vertex and the edge connecting it to the new vertex
            if Distance(q_new, q_goal) < delta_q:
                V.add(q_goal)
                E.add((q_new, q_goal))
                parent_map[q_goal] = q_new

                # Build the path from the initial configuration to the goal configuration and return it
                print("Finding path...")
                path = build_path(q_init, q_goal, parent_map)
                print("Found path...")
                return path

    return None


def SemiRandomSample(steer_goal_p, q_goal):
    # Randomly sample between q_goal and completely random configuration
    arr_random_sample = np.random.choice([q_goal, None], p=[steer_goal_p, 1 - steer_goal_p])

    # If a random configuration is chosen, generate a new completely random configuration
    if arr_random_sample is None:
        q_rand = np.random.random_sample((6)) * 2 * np.pi - np.pi  # range -pi to pi
    # If q_goal is chosen, set the configuration to q_goal
    else:
        q_rand = arr_random_sample

    return q_rand

def Nearest(vertices, target_vertex):
    """Find the vertex in 'vertices' that is closest to the 'target_vertex'"""
    target_vertex_np = np.array(target_vertex)
    vertices_np = np.array(list(vertices))

    # Calculate the distance between all vertices and the target vertex
    distances = np.linalg.norm(vertices_np - target_vertex_np, axis=1)

    # Get the index of the vertex with the smallest distance
    min_index = np.argmin(distances)

    # Return the vertex with the smallest distance
    return tuple(vertices_np[min_index])


def Steer(q_nearest, q_rand, delta_q):
    # Convert input configurations to numpy arrays
    q_nearest = np.array(q_nearest)
    q_rand = np.array(q_rand)

    # Calculate the distance between the two configurations
    dist = Distance(q_nearest, q_rand)

    # If the distance is less than or equal to delta_q, return q_rand
    if dist <= delta_q:
        q_new = q_rand
    # If the distance is greater than delta_q, move delta_q distance from q_nearest towards q_rand
    else:
        dir_vec = (q_rand - q_nearest) / dist * delta_q
        q_new = q_nearest + dir_vec

    # Convert the result back to a tuple and return
    return tuple(q_new)


def ObstacleFree(q_nearest, q_new, env, num_checks=20):
    q_nearest = np.array(q_nearest)
    q_new = np.array(q_new)

    # Check if the q_new has collided with an obstacle
    if env.check_collision(q_new):
        return False

    # Check num_checks intermediate points between q_nearest and q_new
    for i in range(0, 1, num_checks):
        interp_ratio = i / num_checks
        q_interp = (1 - interp_ratio) * q_nearest + interp_ratio * q_new
        if env.check_collision(q_interp):
            return False

    return True

def Distance(q_new, q_goal):
    # Calculate the Euclidean distance between two joint configurations

    # param q_new: The first joint configuration
    # param q_goal: The second joint configuration
    # return: The Euclidean distance between q_new and q_goal
    return np.sqrt(sum([(a - b) ** 2 for a, b in zip(q_new, q_goal)]))

def build_path(q_init, q_goal, parent_map):
    # Initialize empty list to store the path
    Path_List = []
    # Start from the goal configuration
    current = q_goal
    # While the current configuration is not the initial configuration
    while current != q_init:
        # Append the current configuration to the path list
        Path_List.append(current)
        # Update the current configuration to the parent of the current configuration
        current = parent_map[current]
    # Append the initial configuration to the path list
    Path_List.append(q_init)
    # Reverse the path list to get the path from the initial configuration to the goal configuration
    Path_List.reverse()
    # Return the path list
    return Path_List
