import matplotlib.pyplot as plt
import numpy as np

# Function to plot the results
def plot_results(times: list[float], positions: list[float], speeds: list[float], 
                 accelerations: list[float], actuator_forces: list[float]):
    
    # Create subplots
    fig, axs = plt.subplots(4, 1, figsize=(8, 8))
    fig.tight_layout(pad=3)
    
    # Plot position
    axs[0].set_title("Position Over Time")
    axs[0].plot(times, positions, 'g')
    
    # Plot speed
    axs[1].set_title("Speed Over Time")
    axs[1].plot(times, speeds, 'b')
    
    # Plot acceleration
    axs[2].set_title("Acceleration Over Time")
    axs[2].plot(times, accelerations, 'r')
    
    # Plot actuator force
    axs[3].set_title("Actuator Force Over Time")
    axs[3].plot(times, actuator_forces, 'm')
    
    # Automatically scale axes based on the data
    for ax in axs:
        ax.set_xlim(min(times), max(times))  # Set the x-axis limits automatically
        ax.autoscale(enable=True, axis='y')  # Automatically scale the y-axis

# Function to plot seat real displacement
def plot_relative(times: list[float], seat_states, tyre_states, road_states, actuator_forces):

    seat_positions = [row[0][1] for row in seat_states]
    seat_speeds = [row[1][1] for row in seat_states]
    tyre_positions = [row[0][1] for row in tyre_states]
    tyre_speeds = [row[1][1] for row in tyre_states]
    road_ys = [row[0][1] for row in road_states]
    road_yspeeds = [row[1][1] for row in road_states]

    # Convert lists to NumPy arrays to allow element-wise operations
    seat_positions = np.array(seat_positions)
    tyre_positions = np.array(tyre_positions)
    road_ys = np.array(road_ys)
    
    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(8, 8))
    fig.tight_layout(pad=3)
    
    # Plot position
    axs[0].set_title("Seat Position relative Over Time")
    axs[0].plot(times, seat_positions - tyre_positions, 'g')
    
    # # Plot speed
    # axs[1].set_title("Seat speed relative Over Time")
    # axs[1].plot(times, speeds, 'b')
    
    # Plot acceleration
    axs[1].set_title("Tyre position relative Over Time")
    axs[1].plot(times, tyre_positions - road_ys, 'r')
    
    # Plot actuator force
    axs[2].set_title("Actuator Force Over Time")
    axs[2].plot(times, actuator_forces, 'm')
    
    # Automatically scale axes based on the data
    for ax in axs:
        ax.set_xlim(min(times), max(times))  # Set the x-axis limits automatically
        ax.autoscale(enable=True, axis='y')  # Automatically scale the y-axis

def plot_road(times:list, road_states:list):

    road_ys = [row[0][1] for row in road_states]
    road_yspeeds = [row[1][1] for row in road_states]
    road_yaccs = [row[2][1] for row in road_states]

    fig2, axs2 = plt.subplots(3, 1, figsize=(8, 8))
    fig2.tight_layout(pad=3)

        # Plot actuator force
    axs2[0].set_title("Road displacement")
    axs2[0].plot(times, road_ys, 'm')

        # Plot actuator force
    axs2[1].set_title("Road speed")
    axs2[1].plot(times, road_yspeeds, 'g')

        # Plot actuator force
    axs2[2].set_title("Road accelerations")
    axs2[2].plot(times, road_yaccs, 'b')
    
    # Automatically scale axes based on the data
    for ax in axs2:
        ax.set_xlim(min(times), max(times))  # Set the x-axis limits automatically
        ax.autoscale(enable=True, axis='y')  # Automatically scale the y-axis

def main():
    # Example test data
    times = np.linspace(0, 10, 100)  # Example time data
    positions = np.sin(times) * 0.05  # Example position data
    speeds = np.cos(times) * 0.1  # Example speed data
    accelerations = -np.sin(times) * 2  # Example acceleration data
    actuator_forces = np.cos(times) * 1.5  # Example actuator force data

    # Call the plot function
    plot_results(times, positions, speeds, accelerations, actuator_forces)

if __name__ == '__main__':
    main()


# # First figure for the dot representing seat position
# fig1, ax_dot = plt.subplots(figsize=(6, 4))
# ax_dot.set_xlim(-10, 10)
# ax_dot.set_ylim(-1, 1) # Keep the y-axis tight to make it look like a thin axis
# dot, = ax_dot.plot([], [], 'ro')
# ax_dot.set_title('Seat Position')

# # Second figure for speed and acceleration
# fig2, (ax_speed, ax_accel) = plt.subplots(2, 1, figsize=(6, 8))
# fig2.tight_layout(pad=3)

# # Speed plot
# ax_speed.set_title("Speed Over Time")
# ax_speed.set_xlim(0, 50) # Example time range (0-50 seconds)
# ax_speed.set_ylim(-10, 10) # Example speed range
# line_speed, = ax_speed.plot([], [], 'b')

# # Acceleration plot
# ax_accel.set_title("Acceleration Over Time")
# ax_accel.set_xlim(0, 50) # Example time range (0-50 seconds)
# ax_accel.set_ylim(-10, 10) # Example acceleration range
# line_accel, = ax_accel.plot([], [], 'g')

# # Initialize data for speed and acceleration plots
# time_data = []
# speed_data = []
# accel_data = []

# # Function to update the dot position
# def update_dot(positions):
#     x_position, y_position = positions
#     dot.set_data([x_position], [y_position])
#     ax_dot.draw_artist(dot)
#     fig1.canvas.flush_events()

# # Function to update speed and acceleration over time
# def update_speed_accel(t, speeds, accels):
#     x_speed, y_speed = speeds
#     x_accel, y_accel = accels
#     time_data.append(t)
#     speed_data.append(y_speed)
#     accel_data.append(y_accel)
    
#     # Update speed line
#     line_speed.set_data(time_data, speed_data)
#     ax_speed.relim()
#     ax_speed.autoscale_view()  # Rescale if needed
    
#     # Update acceleration line
#     line_accel.set_data(time_data, accel_data)
#     ax_accel.relim()
#     ax_accel.autoscale_view()  # Rescale if needed
    
#     fig2.canvas.draw()
