import numpy as np
import time
from Fuzzy_controller import FuzzyController, Customed_FuzzyController
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
    def __init__(self, control_ini, velocity_ini):
        self.control_params = {}
        self.velocity_params = {}
        self.best_params = {}

        # Initialize control parameters
        for key, value in control_ini.items():
            self.control_params[key] = value
            # Store the best parameters as a copy of control parameters
            self.best_params[key] = value.copy() if hasattr(value, 'copy') else value

        # Initialize velocity parameters
        for key, value in velocity_ini.items():
            self.velocity_params[key] = value

        # Initialize the best cost
        self.best_cost = float('inf')

    def __str__(self):
        return (f"Particle(control_params={self.control_params}, "
                f"velocity_params={self.velocity_params}, "
                f"best_params={self.best_params}, best_cost={self.best_cost})")

    def update_personal_best(self, new_cost):
        """Update personal best if new cost is lower."""
        if new_cost < self.best_cost:
            # Update the best control parameters
            for key, value in self.control_params.items():
                # Store the best parameters as a copy if possible
                self.best_params[key] = value.copy() if hasattr(value, 'copy') else value

            self.best_cost = new_cost

def init_bell_PSO(value_range, granularity):
        maxi_input = max(value_range)
        mini_input = min(value_range)
        rang = maxi_input - mini_input

        # Adjust max and min values in one line using a ternary operation
        maxi_input *= 1.25 if maxi_input > 0 else 0.75
        mini_input *= 0.75 if mini_input > 0 else 1.25

        dimension = granularity *2 + 1
        width_ran = np.random.rand(dimension)*rang
        slope_ran = np.random.rand(dimension)*10
        center_ran = mini_input + np.random.rand(dimension)*rang
        center_ran.sort()

        # Combine parameters into the bell_para list using zip
        bell_para = list(zip(width_ran, slope_ran, center_ran))      

        return bell_para


# Fitness function to evaluate the performance of Q, R matrices in LQR
def fitness_function(control_params, ranges):
    input_params = []
    output_params = []
    for key, value in control_params.items():
        if 'output' in key:
            output_params.append({'name': key, 'universe': ranges[key]['universe'], 'membership_params': value})
            output_gran = ranges[key]['granularity']

        else: 
            input_params.append({'name': key, 'universe': ranges[key]['universe'], 'membership_params': value})
            input_gran = ranges[key]['granularity']

    driver_seat = Seat()
    tyre = Tyre()
    road = Road()

    Fuzzy_controller = Customed_FuzzyController(input_params, output_params, input_gran, output_gran)

    seat_positions, seat_speeds = simulation_PSO(driver_seat, tyre, road, controller=Fuzzy_controller)
    
    # Define a performance metric (e.g., tracking error)
    performance_metric = np.sum(seat_positions**2) + np.sum(seat_speeds**2)  # Example: sum of squared errors

    del driver_seat, tyre, road, Fuzzy_controller, seat_positions, seat_speeds

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
def Fuzzy_particle_swarm_optimization(num_particles, num_iterations, params):
    particles = initialize_particles(num_particles, params)  # Initialize positions and velocities
    global_best = None
    global_best_params = None
    ranges = {}
    for key, value in params.items():
        ranges[key] = {'universe': value['universe'], 'granularity': value['granularity']}

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
        all_params = []
        for particle in tqdm(particles, desc="Particles Simulation Progress", ncols=80):
        # for particle in particles:
            fitness_value = fitness_function(particle.control_params, ranges)
            particle.update_personal_best(fitness_value)
            all_params.append(particle.control_params)
            
            if global_best is None or fitness_value < global_best:
                global_best = fitness_value
                global_best_params = particle.best_params


        # Update particles' velocities and positions
        update_particles(particles, global_best_params, num_ite)

        plot_Q_swarm(all_params, dot, ax_dot)

    print(f'Best results of the PSO: \n{global_best_params = } \n\n& \n\n{global_best = }')

    # Path to results file
    Results_PSO = r'C:\Users\Tugdual\Desktop\Deep_reinforcement_learning\results_pso_fuzzy.txt'

    with open(Results_PSO, 'a') as f:  # Open in 'a' mode to append to the file
        f.write("Best Q matrix:\n")
        f.write("\n".join(["\t".join([f"{elem:.4f}" for elem in row]) for row in global_best_params]))  # Format Q matrix
        f.write(f"\nBest score (fitness): {global_best}\n")  # Add score
        f.write(f"###############################")  # Add score
        # Keep the plot open until the user presses Enter
        input("Press Enter to close the plot...")

    return global_best_params

def initialize_particles(num_particles, params):
    particles = []
    
    for _ in range(num_particles):
        control_params = {}
        velocity_params = {}

        for key, para in params.items():
            # Initialize control_params using init_bell_PSO
            control_params[key] = init_bell_PSO(para['universe'], para['granularity'])
            # control_params[key] = {'name': key, 'universe': para['range'], 'membership_params': bell_para, 'granularity': para['granularity']}
            
            # Generate random velocity for control_params
            control_param_array = np.array(control_params[key])
            velocity_params[key] = control_param_array * (np.random.rand(*control_param_array.shape) * 0.5 - 0.25)
        
        particles.append(Particle(control_params, velocity_params))
    
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
def update_particles(particles, global_best_params, num_ite):
    for particle in particles:
        if num_ite < 100:
            inertia = 1.0  # Keeps the particle moving in its current direction
            personal_influence = 0.8  # Pull towards the personal best
            global_influence = 0.2  # Pull towards the global best
                
            random_direction_factor = 0.2

            min_factor=0.02

            max_factor=0.15

        # Adjust influence factors after a certain number of iterations
        else:
            inertia = 0.5
            personal_influence = 0.8
            global_influence = 0.9

            random_direction_factor = 0.1

            min_factor=0.001

            max_factor=0.1

        for key, value in particle.control_params.items():
                # Add random directional influence (using sign) to velocity
            random_direction = np.sign(np.random.rand(value.shape) - 0.5)  # Values in {-1, 1}
            velocity_update_control = inertia * particle.velocity_params[key] + \
                                personal_influence * (particle.best_params[key] - value) + \
                                global_influence * (global_best_params[key] - value) + \
                                + random_direction * np.abs(value) * random_direction_factor
            
            velocity_update_control = normalize_and_scale(velocity_update_control, particle.control_params[key], min_factor, max_factor)
            # Update the control params and velocity in a separate step to avoid confusion
            particle.control_params[key] = value + velocity_update_control
            particle.velocity_params[key] = velocity_update_control 

        # # Ensure Q and R remain positive on diagonal (non-negative constraint)
        # particle.Q[np.diag_indices_from(particle.Q)] = np.clip(np.diag(particle.Q), a_min=0, a_max=None)
        # particle.R[np.diag_indices_from(particle.R)] = np.clip(np.diag(particle.R), a_min=1e-3, a_max=None)
