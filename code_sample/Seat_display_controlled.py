# Importing required libraries
import matplotlib.pyplot as plt
import numpy as np
import time
from Fuzzy_controller import FuzzyController, Customed_FuzzyController
# from Fuzzy_controller_customed import Customed_FuzzyController
from utile import *
from Suspension_system import *
from LQR_controller import *
from PSO_LQR import *
from PSO_FUZZY import *
# pip install scipy
# pip install numpy
# pip install matplotlib
# pip install scikit-fuzzy
# pip install pip
# pip install control
# pip install tqdm
# pip install networkx

# Real-time simulation loop
def simulate_real_time_motion(seat, tyre, road, controller=None, controller_type = 'None',simulation_length: int = 5, tau: float = 0.002, bump = False):
    """
    Simulates the real-time motion of the seat, tyre, and road based on the system's state.
    
    Args:
    - seat, tyre, road: Objects representing the physical system
    - controller: The controller object (if None, actuator = 0)
    - simulation_length: The length of the simulation in seconds (default: 30s)
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

    y_positions, y_speeds = np.array(y_positions), np.array(y_speeds)
    
    performance_metric = np.sum(y_positions**2) + np.sum(y_speeds**2)
    
    # final plot 
    plot_results(times, y_positions, y_speeds, y_accs, actuator_forces)  # positions, speed & acceleration plots

    plot_road(times, road_states)

    plot_relative(times, seat_states, tyre_states, road_states, actuator_forces)

    controller_results_path = r'C:\Users\Tugdual\Desktop\Deep_reinforcement_learning\controller_results.txt'

    with open(controller_results_path, 'a') as f:
        f.write(f'Controller type: {controller_type}\n')
        f.write(f'Many potholes: {not bump}\n')
        # f.write(f'Controller parameters: {controller}\n')
        f.write(f'Performance: {performance_metric}\n')
        f.write(f"###############################")  
        f.write(f"###############################\n")  



    print(f'Performance of the controller: {performance_metric = }')

def main():
    training = True
    controller_type = 'lqr'
    controller = None
    velocity = False

    ## PSO 
    num_particles = 10
    num_iterations = 15
    
    input_granularity = 1
    output_granularity = 1

    if training:
        if 'fuzzy' in controller_type:
            parameters = {}
            seat_error_range = np.arange(-0.1, 0.11, 0.01)
            seat_error_rate_range = np.arange(-0.75, 0.76, 0.075)
            actuator_force_range = np.arange(-100, 101, 10)

            parameters['input_seat_error'] = {'universe': seat_error_range, 'granularity': input_granularity}
            parameters['input_seat_error_rate'] = {'universe': seat_error_rate_range, 'granularity': input_granularity}
            parameters['output_actuator_force'] = {'universe': actuator_force_range, 'granularity': output_granularity}
        
            Fuzzy_particle_swarm_optimization(num_particles, num_iterations, parameters)

        elif 'lqr' in controller_type:
            particle_swarm_optimization(num_particles, num_iterations, velocity)

    else:
        driver_seat = Seat()
        tyre = Tyre()
        road = Road()

        if controller_type == 'lqr':
            # Instantiate the lqr controller
            controller = LQRcontroller(state_space_suspension(), Q_R_LQR_suspension())

        elif controller_type == 'lqr_velo':
            # Instantiate the lqr controller
            controller = LQRcontroller(state_space_suspension_velocity(), Q_R_LQR_suspension_velocity(), velocity = True)

        elif controller_type == 'fuzzy':
            # Instantiate the fuzzy controller
            controller = Customed_FuzzyController()

        elif controller_type == 'customed_fuzzy':
            ## input = {para1: {'universe', 'name', membership_params}}
            gran = 1

            # Instantiate the fuzzy controller
            controller = Customed_FuzzyController(input_gran = gran, output_gran = gran)

            seat_error_range = np.arange(-0.1, 0.11, 0.01)
            seat_error_rate_range = np.arange(-0.75, 0.76, 0.075)
            actuator_force_range = np.arange(-100, 101, 10)

            seat_para = [       
                [0.05, 1, -0.1],  # negative
                [0.05, 1, 0],      # zero
                [0.05, 1, 0.1]    # positive
            ]
            seat_rate_para = [
                [0.5, 1, -0.75],  # negative
                [0.5, 1, 0],     # zero
                [0.5, 1, 0.75]    # positive
            ]
            input_params = [
                {'name': 'seat_error', 'universe': seat_error_range, 'membership_params': seat_para},
                {'name': 'seat_error_rate', 'universe': seat_error_rate_range, 'membership_params': seat_rate_para}
            ]

            force_para = [
                [50, 2, -100],  # negative
                [50, 2, 0],    # zero
                [50, 2, 100]    # positive
            ]
            output_params = [
                {'name': 'actuator_force', 'universe': actuator_force_range, 'membership_params': force_para}
            ]
        
            customed_controller = Customed_FuzzyController(input_params, output_params, input_gran = gran, output_gran = gran)

            # Plot the membership functions
            controller.plot_membership_functions()

            controller.precompute_lookup_table()
            controller.plot_control_surface()


            # Get actuator force using the lookup table for real-time usage
            actuator_force = controller.get_actuator_force_from_lookup(0.02, 0.5)
            print(f"Actuator Force from Lookup Table: {actuator_force}")

        # Run the simulation
        simulate_real_time_motion(driver_seat, tyre, road, controller=controller, controller_type=controller_type)



    plt.show()

if __name__ == '__main__':
    main()
