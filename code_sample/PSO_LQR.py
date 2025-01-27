import numpy as np
import time
from Fuzzy_controller import FuzzyController
from Suspension_system import *
from LQR_controller import *
from tqdm import tqdm
import matplotlib.pyplot as plt

# Real-time simulation loop
def simulation_PSO(seat, tyre, road, controller=None, simulation_length: int = 5, tau: float = 0.002):
    """
    Simulates the real-time motion of the seat, tyre, and road based on the system's state.
    
    Args:
    - seat, tyre, road: Objects representing the physical system
    - controller: The controller object (if None, actuator = 0)
    - simulation_length: The length of the simulation in seconds (default: 5s)
    - tau: The time step for the simulation (default: 0.002s)
    """
    t = 0  # Initialize time
    y_positions = []
    y_speeds = []
    y_accs = []
    times = []
    actuator_forces = []
    road_states = []
    tyre_states = []
    seat_states = []
    actuator_force = []

    while t < simulation_length:
        # Calculate actuator force from controller if provided
        if controller:         
            # Compute actuator force using the fuzzy controller
            # actuator = controller.compute_actuator_force(seat_error, seat_error_rate)
            actuator = controller.compute_actuator_force(seat.state, tyre.state, road.state)
        else:
            actuator = 0  # Default actuator input (no controller)


        # Update system dynamics using EOM
        EOM_suspension(seat, tyre, road, actuator, tau)
        # Retrieve current seat states: position, velocity, acceleration
        positions, speeds, accelerations = seat.state

        times.append(t)
        y_positions.append(positions[1])
        y_speeds.append(speeds[1])
        y_accs.append(accelerations[1])
        actuator_forces.append(actuator)
        road_states.append(road.state)
        tyre_states.append(tyre.state)
        seat_states.append(seat.state)
        bump = False

        # # Update visualizations (real-time)
        # update_dot(positions)  # Update dot position (only the x-y position)
        # update_speed_accel(t, speeds, accelerations)  # Update speed & acceleration plots
        
        # # Pause to simulate real-time processing (tau = delay)
        # plt.pause(tau)

        if bump:
            y_road, y_dot_road = road_bump(t)
        else:
            y_road, y_dot_road = potholes_road(t)
        
        road.update_state(*[[0, y_road], [0, y_dot_road], [0, 0]])

        # Increment time by the step size tau
        t += tau

    y_positions = np.array(y_positions)
    y_speeds = np.array(y_speeds)

    road_positions = np.fromiter((road_posi[0][1] for road_posi in road_states), dtype=float)

    seat_errors = y_positions - road_positions

    return seat_errors, y_speeds

class Particle:
    def __init__(self, Q, R, velocity_Q, velocity_R):
        self.Q = Q  # The particle's current Q matrix (diagonal)
        self.R = R  # The particle's current R matrix (diagonal)
        self.velocity_Q = velocity_Q  # The velocity for the diagonal of Q
        self.velocity_R = velocity_R  # The velocity for the diagonal of R
        self.best_Q = Q.copy()  # Personal best Q matrix (diagonal)
        self.best_R = R.copy()  # Personal best R matrix (diagonal)
        self.best_cost = float('inf')  # Best cost found so far

    def update_personal_best(self, new_cost):
        """Update personal best if new cost is lower."""
        if new_cost < self.best_cost:
            self.best_Q = self.Q.copy()
            self.best_R = self.R.copy()
            self.best_cost = new_cost

# Fitness function to evaluate the performance of Q, R matrices in LQR
def fitness_function(Q, R, velocity):

    driver_seat = Seat()
    tyre = Tyre()
    road = Road()

    # print(f'{Q = }')
    # print(f'{R = }')

    if velocity:
        # Instantiate the lqr controller
        LQR_controller = LQRcontroller(state_space_suspension_velocity(), Q_R_LQR_suspension_velocity(), velocity = True)

    else:
        # Instantiate the lqr controller
        LQR_controller = LQRcontroller(state_space_suspension(), Q_R_LQR_suspension())

    seat_positions, seat_speeds = simulation_PSO(driver_seat, tyre, road, controller=LQR_controller)
    
    # Define a performance metric (e.g., tracking error)
    performance_metric = np.sum(seat_positions**2) + np.sum(seat_speeds**2)  # Example: sum of squared errors

    del driver_seat, tyre, road, LQR_controller, seat_positions, seat_speeds

    return performance_metric

def plot_Q_swarm(all_Q, dot, ax_dot):
        # Extract Q values for plotting
        x_all = [Q[0, 0] for Q in all_Q]  # Assuming you want to plot the (0,0) elements
        y_all = [Q[1, 1] for Q in all_Q]  # and the (1,1) elements from the Q matrices

        # Update plot data with new positions of the particles
        dot.set_data(x_all, y_all)  # Set new data for the particles' positions
        ax_dot.relim()  # Recalculate the limits based on the new data
        ax_dot.autoscale_view()  # Autoscale the plot to fit new data
        plt.draw()  # Redraw the plot
        plt.pause(0.01)  # Pause to allow the plot to update

# PSO to tune Q and R matrices
def particle_swarm_optimization(num_particles, num_iterations, velocity = False):
    if velocity:
        dim = 5
    else:
        dim = 4
    particles = initialize_particles(num_particles, dim)  # Initialize positions and velocities
    global_best = None
    global_best_Q = None
    global_best_R = None

        # Create the plot for real-time visualization
    fig, ax_dot = plt.subplots(figsize=(10, 7))
    ax_dot.set_xlim(-1000, 1000000)  # Adjust the limits based on expected range of Q values
    ax_dot.set_ylim(-1000, 1000000)
    dot, = ax_dot.plot([], [], 'ro')  # Create a red dot for particle positions
    ax_dot.set_title('Swarm behavior')

    # Initialize the plot before starting the iterations
    plt.ion()  # Turn on interactive mode
    plt.show()  # Show the plot window
    
    for num_ite in tqdm(range(num_iterations), desc="PSO Optimization Progress", ncols=100):
        all_Q = []
        for particle in tqdm(particles, desc="Particles Simulation Progress", ncols=80):
        # for particle in particles:
            fitness_value = fitness_function(particle.Q, particle.R, velocity)
            particle.update_personal_best(fitness_value)
            all_Q.append(particle.Q)
            # # Update personal and global bests
            # if fitness_value < particle.best_cost:
            #     particle.update_personal_best(new_cost)
            #     particle.best_cost = fitness_value
            #     particle.best_Q = particle.Q
            #     particle.best_R = particle.R
            
            if global_best is None or fitness_value < global_best:
                global_best = fitness_value
                global_best_Q = particle.Q
                global_best_R = particle.R

        # Update particles' velocities and positions
        update_particles(particles, global_best_Q, global_best_R, num_ite, dim)

        plot_Q_swarm(all_Q, dot, ax_dot)

    print(f'Best results of the PSO: \n{global_best_Q = } \n\n& \n\n{global_best_R = } \n\n& \n\n{global_best = }')

    # Path to results file
    Results_PSO = r'C:\Users\tugdu\Documents\Deep_reinforcement_learning\results_pso.txt'

    with open(Results_PSO, 'a') as f:  # Open in 'a' mode to append to the file
        f.write("Best Q matrix:\n")
        f.write("\n".join(["\t".join([f"{elem:.4f}" for elem in row]) for row in global_best_Q]))  # Format Q matrix
        f.write(f"\n\nBest R value: {global_best_R[0,0]}\n")
        f.write(f"\nBest score (fitness): {global_best}\n")  # Add score
        f.write(f"###############################")  # Add score
        # Keep the plot open until the user presses Enter
        input("Press Enter to close the plot...")

    return global_best_Q, global_best_R

# Initialize particle positions and velocities (for diagonal matrices)
def initialize_particles(num_particles, dim):
    particles = []
    for _ in range(num_particles):
        # Q is dimxdim diagonal matrix, R is 1x1 diagonal matrix
        Q = np.diag(np.random.uniform(1, 500000, size=dim))  # Random diagonal elements for Q

        if dim > 4:
            Q[4, 4] = np.random.uniform(0.000001, 0.1)
            R = np.diag(np.random.uniform(0.0001, 0.1, size=1))  # Random diagonal element for R
        else:
            R = np.diag(np.random.uniform(0.1, 100, size=1))
        
        # Initialize random velocities for the diagonal elements
        velocity_Q = np.diag((np.random.rand(dim)-0.5))*Q  # Velocity for each diagonal element of Q (as a diagonal matrix)
        velocity_R = np.diag(np.random.rand(1)*0.5-0.25)*R*0.25/0.5  # Velocity for the diagonal element of R (as a diagonal matrix)

        particles.append(Particle(Q, R, velocity_Q, velocity_R))
    return particles

# Function to normalize and scale velocity updates
def normalize_and_scale(velocity, particle_value, min_factor=0.10, max_factor=0.15):
    magnitude = np.linalg.norm(velocity)
    particle_magnitude = np.linalg.norm(particle_value)
    
    # Scale the velocity magnitude relative to the particle's current value
    min_magnitude = particle_magnitude * min_factor
    max_magnitude = particle_magnitude * max_factor

    if magnitude > 0:
        scale_factor = np.clip(magnitude, min_magnitude, max_magnitude) / magnitude
        velocity *= scale_factor
    
    return velocity

# Update particle positions with normalized velocity updates
def update_particles(particles, global_best_Q, global_best_R, num_ite, dim):
    for particle in particles:
        if num_ite < 100:
            inertia = 1.0  # Keeps the particle moving in its current direction
            personal_influence = 0.8  # Pull towards the personal best
            global_influence = 0.2  # Pull towards the global best
            
            # Generate exploration factors for randomness in velocity
            exploration_factor_Q = 0.5 + np.random.rand(dim)  # Random values in range [0.5, 1.5]
            exploration_factor_R = 0.5 + np.random.rand(1)  # For R (1x1 matrix)
                
            random_direction_factor = 0.2

            min_factor=0.02

            max_factor=0.15

        # Adjust influence factors after a certain number of iterations
        else:
            inertia = 0.5
            personal_influence = 0.8
            global_influence = 0.9
            exploration_factor_Q = 0.8 + np.random.rand(dim) * 0.4  # Range [0.8, 1.2]
            exploration_factor_R = 0.8 + np.random.rand(1) * 0.4

            random_direction_factor = 0.1

            min_factor=0.001

            max_factor=0.1

        # Add random directional influence (using sign) to velocity
        random_direction_Q = np.sign(np.random.rand(dim) - 0.5)  # Values in {-1, 1}
        random_direction_R = np.sign(np.random.rand(1) - 0.5)  # For R (1x1 matrix)

        # Calculate velocity update (before scaling)
        velocity_update_Q = inertia * particle.velocity_Q + \
                                personal_influence * (particle.best_Q - particle.Q) + \
                                global_influence * (global_best_Q - particle.Q) + \
                                + random_direction_Q * np.abs(particle.Q) * random_direction_factor  # Add random direction
        
        velocity_update_R = inertia * particle.velocity_R + \
                                personal_influence * (particle.best_R - particle.R) + \
                                global_influence * (global_best_R - particle.R) + \
                                random_direction_R * np.abs(particle.R) * random_direction_factor  # Add random direction

        # Normalize and scale velocity updates to ensure they are bounded by 10-15%
        velocity_update_Q = normalize_and_scale(velocity_update_Q, particle.Q, min_factor, max_factor)
        velocity_update_R = normalize_and_scale(velocity_update_R, particle.R, min_factor, max_factor)

        # Apply velocity updates to Q and R (diagonal matrices)

        particle.Q += velocity_update_Q
        particle.R += velocity_update_R  

        particle.velocity_Q = velocity_update_Q
        particle.velocity_R = velocity_update_R

        # Ensure Q and R remain positive on diagonal (non-negative constraint)
        particle.Q[np.diag_indices_from(particle.Q)] = np.clip(np.diag(particle.Q), a_min=0, a_max=None)
        particle.R[np.diag_indices_from(particle.R)] = np.clip(np.diag(particle.R), a_min=1e-3, a_max=None)
