import numpy as np


# Bump parameters
b_i = 0.1  # Bump intensity

## Set the model parameters of the Active Suspension.
K_s = 900;      ## or 1040 Suspension Stiffness (N/m) 
K_us = 1250;   ## or 2300 Tyre stiffness (N/m)
M_s = 2.45;     ## or 2.5 Sprung Mass (kg) 
M_us = 1;      ## (mu) or 1.150 Unsprung Mass (kg)
B_s = 7.5;      ## Suspension Inherent Damping coefficient (sec/m)
B_us = 5;       ## Tyre Inhenrent Damping coefficient (sec/m)

# Seat class
class Seat:
    def __init__(self, state: list[list[float]] = [[0, 0], [0, 0], [0, 0]]):
        """
        Initializes the seat object with a default state of position, velocity, and acceleration.
        Args:
        state: A 2D list where the first sublist is [x, y], second is [x_dot, y_dot], and third is [x_acc, y_acc].
        """
        self.state = state

    def update_state(self, position: list[float], velocity: list[float], acceleration: list[float]):
        """
        Updates the state of the seat with new position, velocity, and acceleration.
        Args:
        position: [x, y]
        velocity: [x_dot, y_dot]
        acceleration: [x_acc, y_acc]
        """
        self.state = [position, velocity, acceleration]

    def get_state(self) -> list[list[float]]:
        """
        Returns the current state of the seat.
        """
        return self.state

# Tyre class (renamed to "Tyre" to match conventional spelling)
class Tyre:
    def __init__(self, state: list[list[float]] = [[0, 0], [0, 0], [0, 0]]): ## [[0, -0.128], [0, 0], [0, 0]]
        """
        Initializes the tyre object with a default state.
        """
        self.state = state

    def update_state(self, position: list[float], velocity: list[float], acceleration: list[float]):
        """
        Updates the state of the tyre with new position, velocity, and acceleration.
        """
        self.state = [position, velocity, acceleration]

    def get_state(self) -> list[list[float]]:
        """
        Returns the current state of the tyre.
        """
        return self.state

# Road class
class Road:
    def __init__(self, state: list[list[float]] = [[0, 0], [0, 0], [0, 0]]): ## [[0, -0.115-0.128], [0, 0], [0, 0]]
        """
        Initializes the road object with a default state.
        """
        self.state = state

    def update_state(self, position: list[float], velocity: list[float], acceleration: list[float]):
        """
        Updates the state of the road with new position, velocity, and acceleration.
        """
        self.state = [position, velocity, acceleration]

    def get_state(self) -> list[list[float]]:
        """
        Returns the current state of the road.
        """
        return self.state

def EOM_suspension(seat: Seat, tyre: Tyre, road: Road, actuator, tau: float):
    # Extracting states (Position, Velocity)
    Z_s, Z_s_dot = seat.state[0][1], seat.state[1][1]  # Seat position and velocity
    Z_us, Z_us_dot = tyre.state[0][1], tyre.state[1][1]  # Tyre position and velocity
    Z_r, Z_r_dot = road.state[0][1], road.state[1][1]  # Road position and velocity
    F_c = actuator  # Actuator force (constant or function input)

    Z_us_dot_dot = (-F_c - B_s * (Z_us_dot - Z_s_dot) - B_us * (Z_us_dot - Z_r_dot) - K_s * (Z_us - Z_s) - K_us * (Z_us - Z_r))/M_us
    Z_s_dot_dot = (F_c - B_s * (Z_s_dot - Z_us_dot) - K_s * (Z_s - Z_us))/M_s

    seat_accelerations = [0, Z_s_dot_dot]
    tyre_accelerations = [0, Z_us_dot_dot]
    road_accelerations = [0, 0]

    # Updating states using the Euler integrator
    seat.update_state(*euler_integral(seat_accelerations, seat.state, tau))
    tyre.update_state(*euler_integral(tyre_accelerations, tyre.state, tau))
    road.update_state(*euler_integral(road_accelerations, road.state, tau))  # Assuming road has no acceleration


def euler_integral(accelerations: float, state: list[list[float]], tau: float) -> list[list[float]]:
    """
    Euler integrator to update position, velocity, and acceleration.
    Args:
    - acceleration: New calculated acceleration
    - state: Current state of the system [[x, y], [x_dot, y_dot], [x_acc, y_acc]]
    - tau: Time step (default 0.1)
    
    Returns:
    - Updated state [[x_new, y_new], [x_dot_new, y_dot_new], [x_acc, y_acc]]
    """

    position_new = [state[0][i] + tau * state[1][i] for i in range(2)]  # Update position
    velocity_new = [state[1][i] + tau * accelerations[i] for i in range(2)]  # Update velocity
    acceleration_new = accelerations  # No update on acceleration; it's derived

    return [position_new, velocity_new, acceleration_new]

def potholes_road(t):
    y = 0.02 * np.cos(10*t)
    y_dot = - 0.2 * np.sin(10*t)
    return y, y_dot

# Bump profile and derivative
def road_bump(t):
    if 1 < t <= 1.3:
        y = b_i * (t - 1.0)
        y_dot = b_i
    elif 1.3 < t <= 1.6:
        y = b_i * (1.6 - t)
        y_dot = -b_i
    else:
        y = 0
        y_dot = 0
    
    return y, y_dot
    
def state_space_suspension():
    # This section sets the A,B,C and D matrices for the Active Suspension model.
    # (Check the matrices based on your own calculations)

    A = np.array([[0, 1, 0, -1],
        [-K_s/M_s, -B_s/M_s, 0, B_s/M_s],
        [0, 0, 0, 1],
        [K_s/M_us, B_s/M_us, -K_us/M_us, -(B_s+B_us)/M_us]])

    B = np.array([[0] , [1/M_s] , [0] , [-1/M_us]])
    C = np.array([[1, 0, 0, 0] , [-K_s/M_s, -B_s/M_s, 0, B_s/M_s] ])
    D = np.array([[0], [1/M_s]])
    E = np.array([[0], [0], [-1], [B_us/M_us]])

## Double matrices
    # B = np.array([[0,  0] , [0, 1/M_s] , [-1,  0] , [B_us/M_us, -1/M_us]])
    # C = np.array([[1, 0, 0, 0] , [-K_s/M_s, -B_s/M_s, 0, B_s/M_s] ])
    # D = np.array([[0, 0], [0, 1/M_s]])

    return A, B, C, D

def state_space_suspension_velocity():

    A = np.array([[0, 1, 0, -1],
        [-K_s/M_s, -B_s/M_s, 0, B_s/M_s],
        [0, 0, 0, 1],
        [K_s/M_us, B_s/M_us, -K_us/M_us, -(B_s+B_us)/M_us]])

    B = np.array([[0] , [1/M_s] , [0] , [-1/M_us]])
    C = np.array([[1, 0, 0, 0] , [-K_s/M_s, -B_s/M_s, 0, B_s/M_s] ])
    D = np.array([[0], [1/M_s]])
    E = np.array([[0], [0], [-1], [B_us/M_us]])

    A_new = np.block([
        [A, B],
        [np.zeros((1, A.shape[1])), np.zeros((1, 1))]
    ])
    
    B_new = np.vstack([np.zeros((A.shape[0], 1)), np.array([[1]])])

        # Append a zero column to C to account for the added state (u)
    C_new = np.hstack([C, np.array([[0], [1/M_s]])])
    
    # D remains the same in this case, but it could be adjusted if needed
    D_new = 0  # Modify if necessary depending on your system

    return A_new, B_new, C_new, D_new

def Q_R_LQR_suspension_velocity(Q = None, R = None):
    if Q is None or R is None: 
        Q_lqr = np.array([[5.44897534e+05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00],
       [0.00000000e+00, 6.40454449e+04, 0.00000000e+00, 0.00000000e+00,
        0.00000000e+00],
       [0.00000000e+00, 0.00000000e+00, 5.07931439e+05, 0.00000000e+00,
        0.00000000e+00],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.28132386e+05,
        0.00000000e+00],
       [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
        4.50918480e-02]])
        R_lqr = np.array(0.0956169)

    else:
        Q_lqr = Q
        R_lqr = R
    return Q_lqr, R_lqr

def Q_R_LQR_suspension(Q = None, R = None):
    if Q is None or R is None: 
        # Q_lqr = np.array([[1/0.000009, 0, 0, 0],
        #     [0, 1, 0, 0],
        #     [0, 0, 1/0.000001, 0],
        #     [0, 0, 0, 1]])
        # R_lqr = np.array(1000/(38.33**2))

    #     Q_lqr = np.array([[ 939.55567469,  0,  0,
    #      0],
    #    [ 0,  5842.4740375,  0,
    #      0],
    #    [ 0,  0, 2415.07530014,
    #      0],
    #    [ 0,  0,  0,
    #     0]])

    #     R_lqr = np.array(0.001)

    #     Q_lqr = np.array([[ 17568.23816065,  0,  0,
    #      0],
    #    [ 0,  104960.27379016,  0,
    #      0],
    #    [ 0,  0, 15144.24890316,
    #      0],
    #    [ 0,  0,  0,
    #     0]])

        # R_lqr = np.array(0.03234082)

    #     Q_lqr = np.array([[ 90626.54815757,  0,  0,
    #      0],
    #    [ 0,  100108.52528358,  0,
    #      0],
    #    [ 0,  0, 110.71830098,
    #      0],
    #    [ 0,  0,  0,
    #     0]])

    #     R_lqr = np.array(0.09645807)

        # Q_lqr = np.array([[ 32777.1798417,  0,  0,
        #     0],
        # [ 0,  569485.17768715,  0,
        #     0],
        # [ 0,  0, 0,
        #     0],
        # [ 0,  0,  0,
        #     0]])

        # R_lqr = np.array(0.20342518)

        Q_lqr = np.array([[ 310626.48242021,  0,  0,
            0],
        [ 0,  363733.8818427,  0,
            0],
        [ 0,  0, 17589.36844349,
            0],
        [ 0,  0,  0,
            0]])

        R_lqr = np.array(0.33061567)

#         global_best_Q = array([[208894.73404767,      0.        ,      0.        ,
#              0.        ],
#        [     0.        , 238529.54245309,      0.        ,
#              0.        ],
#        [     0.        ,      0.        ,   6795.68642709,
#              0.        ],
#        [     0.        ,      0.        ,      0.        ,
#              0.        ]])
# &
# global_best_R = array([[0.22233647]])
    
    else:
        Q_lqr = Q
        R_lqr = R
    return Q_lqr, R_lqr


def pid_actuator(t:float):
    control_input = 0
    return control_input
