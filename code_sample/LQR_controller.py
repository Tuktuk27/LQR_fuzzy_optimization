import numpy as np
import control as ct

class LQRcontroller:
    def __init__(self, state_space, weight_matrices, velocity = False, tau = 0.002):

        A, B, C, D = state_space

        Q, R = weight_matrices

        self.velocity = velocity
        self.tau = tau

        self.actuator_force = 0
        self.actuator_velocity = 0

        if velocity: 
            self.compute_actuator_force = self.compute_velocity
        else:
            self.compute_actuator_force = self.compute_force

        self.sys = ct.ss(A,B,C,D)

        # self.K_lqr = [86.6125, 37.8421, -410.7621, -7.1974]
        self.K_lqr, self.S, self.E = ct.lqr(self.sys, Q, R)


    def compute_force(self, seat_state, tyre_state, road_state):
        seat_error = seat_state[0][1] - tyre_state[0][1]
        tyre_error = tyre_state[0][1] - road_state[0][1]
        # seat_error = seat_state[0][1] 
        # tyre_error = tyre_state[0][1] 
        seat_error_speed = seat_state[1][1]
        tyre_error_speed = tyre_state[1][1]

        state = [seat_error, seat_error_speed, tyre_error, tyre_error_speed]

        actuator_force = np.matmul(-self.K_lqr, state)
        actuator_force = actuator_force.item()  # Extract the single scalar value

        # Saturation logic
        actuator_force = np.clip(actuator_force, -100, 100)

        return actuator_force

    def compute_velocity(self, seat_state, tyre_state, road_state):

        input_force =  self.actuator_force + self.tau * self.actuator_velocity

        seat_error = seat_state[0][1] - tyre_state[0][1]
        tyre_error = tyre_state[0][1] - road_state[0][1]
        # seat_error = seat_state[0][1] 
        # tyre_error = tyre_state[0][1] 
        seat_error_speed = seat_state[1][1]
        tyre_error_speed = tyre_state[1][1]
        state = [seat_error, seat_error_speed, tyre_error, tyre_error_speed, input_force]

        self.actuator_velocity = np.matmul(-self.K_lqr, state)
        self.actuator_velocity = self.actuator_velocity.item()  # Extract the single scalar value

        # Saturation logic
        self.actuator_velocity = np.clip(self.actuator_velocity, -100000, 100000)

        return input_force
