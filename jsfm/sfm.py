import jax.numpy as jnp
from jax import jit, vmap, lax, debug

# TODO: Add possibility to make a step without obstacles (for now a dummy obstacle is needed)

@jit
def compute_edge_closest_point(reference_point:jnp.ndarray, edge:jnp.ndarray, current_closest_point:jnp.ndarray, current_min_distance:float):
    """
    This function computes the closest point of the edge to the reference point and confronts it with the current closest point and min dist to the obstacle.
    Finally it overweites the closest point and the min distance to the obstacle in the current ones.
    
    args:
    - reference_point: shape is (2,) in the form (px, py)
    - edge: shape is (2, 2) where each edge includes its two vertices (p1, p2) composed by two coordinates (x, y)
    - current_closest_point: shape is (2,) in the form (cx, cy)
    - current_min_distance: min distance to the current closest point

    output:
    - closest_points: shape is (2,) in the form (cx, cy)
    - min_distances: min distance to the closest point
    """
    a = edge[0]
    b = edge[1]
    t = (jnp.dot(reference_point - a, b - a)) / (jnp.linalg.norm(b - a) ** 2)
    t_lb = lax.cond(t>0,lambda x: x,lambda x: 0.,t)
    t_star = lax.cond(t_lb<1,lambda x: x,lambda x: 1.,t_lb)
    h = a + t_star * (b - a)
    dist = jnp.linalg.norm(h - reference_point)
    closest_point = lax.cond(dist < current_min_distance, lambda x: h, lambda x: x, current_closest_point)
    min_distance = lax.cond(dist < current_min_distance, lambda x: dist, lambda x: x, current_min_distance)
    return closest_point, min_distance

@jit
def compute_obstacle_closest_point(reference_point:jnp.ndarray, obstacle:jnp.ndarray) -> jnp.ndarray:
    """
    This function computes the closest point of the obstacle to the reference point
    
    args:
    - reference_point: shape is (2,) in the form (px, py)
    - obstacle: shape is (n_edges, 2, 2) where each obs contains one of its edges (min. 3 edges) and each edge includes its two vertices (p1, p2) composed by two coordinates (x, y)

    output:
    - closest_point: shape is (2,) in the form (cx, cy)
    """
    closest_point = jnp.zeros((2,))
    min_distance = jnp.float32(10000.)
    closest_point, min_distance = lax.fori_loop(0, len(obstacle), lambda i, vals: compute_edge_closest_point(reference_point, obstacle[i], vals[0], vals[1]), (closest_point, min_distance))
    return closest_point
vectorized_compute_obstacle_closest_point = vmap(compute_obstacle_closest_point, in_axes=(None, 0))

@jit
def pairwise_social_force(human_state:jnp.ndarray, other_human_state:jnp.ndarray, parameters:jnp.ndarray, other_human_parameters:jnp.ndarray):
    """
    This function computes the social force between a pair of humans

    args:
    - human_state: shape is (4,) in the form (px, py, vx, vy)
    - other_humans_state: shape is (4,) in the form (px, py, vx, vy)
    - parameters: shape is (19,) in the form (radius, mass, v_max, tau, Ai, Aw, Bi, Bw, Ci, Cw, Di, Dw, k1, k2, k0, kd, alpha, k_lambda, safety_space)
    - other_humans_parameters: shape is (19,) in the form (radius, mass, v_max, tau, Ai, Aw, Bi, Bw, Ci, Cw, Di, Dw, k1, k2, k0, kd, alpha, k_lambda, safety_space)

    output:
    - social_force: shape is (2,) in the form (fx, fy)
    """
    rij = parameters[0] + other_human_parameters[0] + parameters[18] + other_human_parameters[18]
    diff = human_state[:2] - other_human_state[:2]
    dist = jnp.linalg.norm(diff)
    nij = diff / dist
    real_dist = rij - dist
    tij = jnp.array([-nij[1], nij[0]])
    human_linear_velocity = human_state[2:4]
    other_human_linear_velocity = other_human_state[2:4]
    delta_vij = jnp.dot(other_human_linear_velocity - human_linear_velocity, tij)
    pairwise_social_force = lax.cond(real_dist > 0, lambda x: x * (parameters[4] * jnp.exp(real_dist / parameters[6]) + parameters[12] * real_dist) * nij + (parameters[8] * jnp.exp(real_dist / parameters[10]) + parameters[13] * real_dist * delta_vij) * tij, lambda x: x * (parameters[4] * jnp.exp(real_dist / parameters[6])) * nij + (parameters[8] * jnp.exp(real_dist / parameters[10])) * tij, jnp.ones((2,)))
    return pairwise_social_force

@jit
def compute_obstacle_force(human_state:jnp.ndarray, obstacle:jnp.ndarray, parameters:jnp.ndarray):
    """
    This function computes the obstacle force between a human and an obstacle.
    
    args:
    - human_state: shape is (4,) in the form (px, py, vx, vy)
    - obstacle: shape is (2,) in the form (ox, oy)
    - parameters: shape is (19,) in the form (radius, mass, v_max, tau, Ai, Aw, Bi, Bw, Ci, Cw, Di, Dw, k1, k2, ko, kd, alpha, k_lambda, safety_space)

    output:
    - obstacle_force: shape is (2,) in the form (fx, fy)
    """
    diff = human_state[:2] - obstacle
    dist = jnp.linalg.norm(diff)
    niw = diff / dist
    tiw = jnp.array([-niw[1], niw[0]])
    linear_velocity = human_state[2:4]
    delta_viw = - jnp.dot(linear_velocity, tiw)
    real_dist = parameters[0] - dist + parameters[18]
    obstacle_force = lax.cond(real_dist > 0, lambda x: x * (parameters[5] * jnp.exp(real_dist / parameters[7]) + parameters[12] * real_dist) * niw + (-parameters[9] * jnp.exp(real_dist / parameters[11]) - parameters[13] * real_dist) * delta_viw * tiw, lambda x: x * (parameters[5] * jnp.exp(real_dist / parameters[7])) * niw + (-parameters[9] * jnp.exp(real_dist / parameters[11])) * delta_viw * tiw, jnp.ones((2,)))
    return obstacle_force
vectorized_compute_obstacle_force = vmap(compute_obstacle_force, in_axes=(None, 0, None))

@jit
def single_update(idx:int, humans_state:jnp.ndarray, human_goal:jnp.ndarray, parameters:jnp.ndarray, obstacles:jnp.ndarray, dt:float) -> jnp.ndarray:
    """
    This functions makes a step in time (of length dt) for a single human using the Headed Social Force Model (HSFM) with 
    global force guidance for torque and sliding component on the repulsive forces.

    args:
    - idx: human index in the state, goal and parameter vectors
    - humans_state: shape is (n_humans, 4) in the form is (px, py, vx, vy)
    - humans_goal: shape is (2,) in the form (gx, gy)
    - parameters: shape is (n_humans, 19) in the form (radius, mass, v_max, tau, Ai, Aw, Bi, Bw, Ci, Cw, Di, Dw, k1, k2, ko, kd, alpha, k_lambda, safety_space)
    - obstacles: shape is (n_obstacles, n_edges, 2, 2) where each obs contains one of its edges (min. 3 edges) and each edge includes its two vertices (p1, p2) composed by two coordinates (x, y)
    - dt: sampling time for the update
    
    output:
    - updated_human_state: shape is (4,) in the form (px, py, vx, vy)
    """
    self_state = humans_state[idx]
    self_parameters = parameters[idx]
    # Desired force computation
    linear_velocity = self_state[2:4]
    diff = human_goal - self_state[:2]
    dist = jnp.linalg.norm(diff)
    desired_force =  lax.cond(dist > self_parameters[0],lambda x: x * (self_parameters[1] * (((diff / dist) * self_parameters[2]) - linear_velocity) / self_parameters[3]),lambda x: x * 0,jnp.ones((2,)))
    # Social force computation
    social_force = lax.fori_loop(0, len(humans_state), lambda j, acc: lax.cond(j != idx, lambda acc: acc + pairwise_social_force(self_state, humans_state[j], self_parameters, parameters[j]), lambda acc: acc, acc), jnp.zeros((2,)))
    # Obstacle force computation
    closest_points = vectorized_compute_obstacle_closest_point(self_state[:2], obstacles)
    obstacle_force = jnp.sum(vectorized_compute_obstacle_force(self_state, closest_points, self_parameters), axis=0) / len(obstacles)
    # Global force computation
    global_force = desired_force + social_force + obstacle_force
    # Update
    updated_human_state = jnp.zeros((4,))
    updated_human_state = updated_human_state.at[0].set(self_state[0] + dt * linear_velocity[0])
    updated_human_state = updated_human_state.at[1].set(self_state[1] + dt * linear_velocity[1])
    updated_human_state = updated_human_state.at[2].set(self_state[2] + dt * (global_force[0] / self_parameters[1]))
    updated_human_state = updated_human_state.at[3].set(self_state[3] + dt * (global_force[1] / self_parameters[1]))
    # Bound linear velocity
    updated_human_state = updated_human_state.at[2:4].set(
        lax.cond(
            jnp.linalg.norm(updated_human_state[2:4]) > self_parameters[2], 
            lambda x: (x / jnp.linalg.norm(x)) * self_parameters[2], 
            lambda x: x, 
            updated_human_state[2:4]))
    # DEBUGGING
    # debug.print("\n")
    # debug.print("jax.debug.print(closest_points) -> {x}", x=closest_points)
    # debug.print("jax.debug.print(min_distances) -> {x}", x=min_distances)
    # debug.print("jax.debug.print(desired_force) -> {x}", x=desired_force)
    # debug.print("jax.debug.print(social_force) -> {x}", x=social_force)
    # debug.print("jax.debug.print(obstacle_force) -> {x}", x=obstacle_force)
    # debug.print("jax.debug.print(global_force) -> {x}", x=global_force)
    # debug.print("jax.debug.print(updated_human_state) -> {x}", x=updated_human_state)
    return updated_human_state
vectorized_single_update = vmap(single_update, in_axes=(0, None, 0, None, None, None))

@jit
def step(humans_state:jnp.ndarray, humans_goal:jnp.ndarray, parameters:jnp.ndarray, obstacles:jnp.ndarray, dt:float) -> jnp.ndarray:
    """
    This functions makes a step in time (of length dt) for the humans' state using the Headed Social Force Model (HSFM) with 
    global force guidance for torque and sliding component on the repulsive forces.

    args:
    - humans_state: shape is (n_humans, 4) where each row is (px, py, vx, vy)
    - humans_goal: shape is (n_humans, 2) where each row is (gx, gy)
    - parameters: shape is (n_humans, 19) where each row is (radius, mass, v_max, tau, Ai, Aw, Bi, Bw, Ci, Cw, Di, Dw, k1, k2, ko, kd, alpha, k_lambda, safety_space)
    - obstacles: shape is (n_obstacles, n_edges, 2, 2) where each obs contains one of its edges (min. 3 edges) and each edge includes its two vertices (p1, p2) composed by two coordinates (x, y)
    - dt: sampling time for the update
    
    output:
    - updated_humans_state: shape is (n_humans, 4) where each row is (px, py, vx, vy)
    """
    idxs = jnp.arange(len(humans_state))
    updated_humans_state = vectorized_single_update(idxs, humans_state, humans_goal, parameters, obstacles, dt)
    return updated_humans_state
