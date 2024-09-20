from jax import numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import os
from jsfm.sfm import step
from jsfm.utils import *

# Hyperparameters
room_half_length = 7
dt = 0.01
end_time = 15
humans_state = np.array([[7.,0.,0.,0.],
                         [6.8,0.8,0.,0.],
                         [6.8,-0.8,0.,0.],
                         [6.5,1.5,0.,0.],
                         [6.5,-1.5,0.,0.]])
static_obstacles = jnp.array([[[[-0.1,0.5],[0.1,0.5]],[[0.1,0.5],[0.1,3]],[[0.1,3],[-0.1,3]],[[-0.1,3],[-0.1,0.5]]],
                              [[[-0.1,-0.5],[0.1,-0.5]],[[0.1,-0.5],[0.1,-3]],[[0.1,-3],[-0.1,-3]],[[-0.1,-3],[-0.1,-0.5]]]])


# Initial conditions
humans_goal = np.zeros((len(humans_state), 2))
for i in range(len(humans_state)):
    # Goal: (gx, gy)
    humans_goal[i,0] = -room_half_length
    humans_goal[i,1] = 0.
humans_state = jnp.array(humans_state)
humans_parameters = get_standard_humans_parameters(len(humans_state))
humans_goal = jnp.array(humans_goal)

# Dummy step - Warm-up (we first compile the JIT functions to avoid counting compilation time later)
_ = step(humans_state, humans_goal, humans_parameters, static_obstacles, dt)

# Simulation 
steps = int(end_time/dt)
print(f"\nAvailable devices: {jax.devices()}\n")
print(f"Starting simulation... - Simulation time: {steps*dt} seconds\n")
start_time = time.time()
all_states = np.empty((steps+1, len(humans_state), 4), np.float32)
all_states[0] = humans_state
for i in range(steps):
    humans_state = step(humans_state, humans_goal, humans_parameters, static_obstacles, dt)
    all_states[i+1] = humans_state
end_time = time.time()
print("Simulation done! Computation time: ", end_time - start_time)
all_states = jax.device_get(all_states) # Transfer data from GPU to CPU for plotting (only at the end)

# Plot
COLORS = list(mcolors.TABLEAU_COLORS.values())
print("\nPlotting...")
figure, ax = plt.subplots(figsize=(10,10))
ax.axis('equal')
ax.set(xlabel='X',ylabel='Y',xlim=[-room_half_length-1,room_half_length+1],ylim=[-room_half_length-1,room_half_length+1])
for h in range(len(humans_state)): 
    ax.plot(all_states[:,h,0], all_states[:,h,1], color=COLORS[h%len(COLORS)], linewidth=0.5, zorder=0)
    ax.scatter(humans_goal[h,0], humans_goal[h,1], marker="*", color=COLORS[h%len(COLORS)], zorder=2)
    for k in range(0,steps+1,int(3/dt)):
        circle = plt.Circle((all_states[k,h,0],all_states[k,h,1]),humans_parameters[h,0], edgecolor=COLORS[h%len(COLORS)], facecolor="white", fill=True, zorder=1)
        ax.add_patch(circle)
        num = int(k*dt) if (k*dt).is_integer() else (k*dt)
        ax.text(all_states[k,h,0],all_states[k,h,1], f"{num}", color=COLORS[h%len(COLORS)], va="center", ha="center", size=10, zorder=1, weight='bold')
for o in static_obstacles: ax.fill(o[:,:,0],o[:,:,1], facecolor='black', edgecolor='black', zorder=3)
figure.savefig(os.path.join(os.path.dirname(__file__),".images",f"example2.png"), format='png')
plt.show()