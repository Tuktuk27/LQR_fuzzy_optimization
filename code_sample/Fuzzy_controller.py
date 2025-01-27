import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from collections import deque
import itertools


class FuzzyController:
    def __init__(self):
        # Define fuzzy input variables: seat error and seat error rate
        self.seat_error = ctrl.Antecedent(np.arange(-10, 11, 1), 'seat_error')
        self.seat_error_rate = ctrl.Antecedent(np.arange(-5, 6, 1), 'seat_error_rate')
        
        # Define fuzzy output variable: actuator force
        self.actuator_force = ctrl.Consequent(np.arange(-100, 101, 1), 'actuator_force')
        
        # Define membership functions for input: seat error
        self.seat_error['negative'] = fuzz.trimf(self.seat_error.universe, [-10, -10, 0])
        self.seat_error['zero'] = fuzz.trimf(self.seat_error.universe, [-2, 0, 2])
        self.seat_error['positive'] = fuzz.trimf(self.seat_error.universe, [0, 10, 10])
        
        # Define membership functions for input: seat error rate
        self.seat_error_rate['negative'] = fuzz.trimf(self.seat_error_rate.universe, [-5, -5, 0])
        self.seat_error_rate['zero'] = fuzz.trimf(self.seat_error_rate.universe, [-1, 0, 1])
        self.seat_error_rate['positive'] = fuzz.trimf(self.seat_error_rate.universe, [0, 5, 5])
        
        # Define membership functions for output: actuator force
        self.actuator_force['negative'] = fuzz.trimf(self.actuator_force.universe, [-100, -100, 0])
        self.actuator_force['zero'] = fuzz.trimf(self.actuator_force.universe, [-10, 0, 10])
        self.actuator_force['positive'] = fuzz.trimf(self.actuator_force.universe, [0, 100, 100])
        
        # Define fuzzy rules
        rule1 = ctrl.Rule(self.seat_error['negative'] & self.seat_error_rate['negative'], self.actuator_force['positive'])
        rule2 = ctrl.Rule(self.seat_error['negative'] & self.seat_error_rate['zero'], self.actuator_force['positive'])
        rule3 = ctrl.Rule(self.seat_error['zero'] & self.seat_error_rate['zero'], self.actuator_force['zero'])
        rule4 = ctrl.Rule(self.seat_error['positive'] & self.seat_error_rate['positive'], self.actuator_force['negative'])
        rule5 = ctrl.Rule(self.seat_error['positive'] & self.seat_error_rate['zero'], self.actuator_force['negative'])
        
        # New rules for zero error but non-zero error rate
        rule6 = ctrl.Rule(self.seat_error['zero'] & self.seat_error_rate['positive'], self.actuator_force['negative'])
        rule7 = ctrl.Rule(self.seat_error['zero'] & self.seat_error_rate['negative'], self.actuator_force['positive'])
        
        # Create control system
        self.control_system = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)
        
    def compute_actuator_force(self, seat_state, tyre_state, road_state):
        """
        This function computes the actuator force based on the fuzzy rules.
        
        Args:
        - seat_error_value: The current seat displacement error
        - seat_error_rate_value: The current rate of change of the seat displacement
        
        Returns:
        - actuator force (float): The computed actuator force
        """
        # Pass input values to the fuzzy simulation
        self.simulation.input['seat_error'] = seat_state[0][1]  # Assume seat position is the first state
        self.simulation.input['seat_error_rate'] = seat_state[1][1]  # Assume seat velocity is the second state
        
        # Compute the output (actuator force)
        self.simulation.compute()
        
        # Return the computed actuator force
        return self.simulation.output['actuator_force']

class Customed_FuzzyController:
    def __init__(self, input_params = None, output_params = None, input_gran = 1, output_gran = 1): ## granularity = 1 --> 3 level, = 2 --> 5 levels, = 3 --> 7 levels...
        '''
        input = {para1: {'universe', 'name', membership_params}}
        '''
        self.input_gran = input_gran
        self.output_gran = output_gran

        # Default parameters for nonlinearity
        self.non_linear_params = {'a': 0.1, 'b': 0.0, 'c': 1.0, 'd': 1.0}
        # Determine the number of dimensions from input_params
        self.num_input = len(input_params)

        if input_params is None or output_params is None:
            input_params, output_params = self.default_parameters()
            print('\n ############## \n Warning !! Using by default parameters \n ############## \n')

        ## P as Positive, N as Negative, M as Medium or Mean
        self.granularity_level_standard = ['XXXN', 'XXN', 'XN', 'N', 'M', 'P', 'XP', 'XXP', 'XXXP']

        self.input_labels, self.input_values = self.create_labels(self.input_gran)
        self.output_labels, self.output_values = self.create_labels(self.output_gran, True)

        # Dictionaries to store antecedents and consequents
        self.antecedents = {}
        self.consequents = {}

        # Create antecedents for input parameters
        for param in input_params:
            # Dynamically create antecedent and assign to self
            antecedent = ctrl.Antecedent(param['universe'], param['name'])
            setattr(self, f'{param["name"]}_antecedent', antecedent)
            # Store in dictionary for later use
            self.antecedents[param['name']] = antecedent
            # Create membership functions for the antecedent
            self.create_membership_functions(antecedent, param['universe'], param['membership_params'], self.input_labels)

        # Create consequents for output parameters
        for param in output_params:
            # Dynamically create consequent and assign to self
            consequent = ctrl.Consequent(param['universe'], param['name'])
            setattr(self, f'{param["name"]}_consequent', consequent)
            # Store in dictionary for later use
            self.consequents[param['name']] = consequent
            # Create membership functions for the consequent
            self.create_membership_functions(consequent, param['universe'], param['membership_params'], self.output_labels)



        self.set_rules_matrices()

        self.create_dynamic_rules()

        self.lookup_table = self.precompute_lookup_table()
    
    def default_parameters(self):
        # Define default input parameters for bell equation: width, slope, center
        # use         self.input_gran and self.output_gran (basically 1, so negative, zero or positive, but could be 2 or more, so for 2: strong negative, negative, zero positive, strong positive)
        
        seat_error_range = np.arange(-0.1, 0.11, 0.01)
        seat_error_rate_range = np.arange(-0.75, 0.76, 0.075)
        actuator_force_range = np.arange(-100, 101, 10)


        seat_para = self.generate_bell_matrix(seat_error_range, self.input_gran)
        seat_rate_para = self.generate_bell_matrix(seat_error_rate_range, self.input_gran)

        input_params = [
            {'name': 'seat_error', 'universe': seat_error_range, 'membership_params': seat_para},
            {'name': 'seat_error_rate', 'universe': seat_error_rate_range, 'membership_params': seat_rate_para}
        ]

        # Define default output parameters
        force_para = self.generate_bell_matrix(actuator_force_range, self.output_gran)

        output_params = [
            {'name': 'actuator_force', 'universe': actuator_force_range, 'membership_params': force_para}
        ]
        
        return input_params, output_params
    
    def generate_bell_matrix(self, value_range, granularity):
        maxi_input = max(value_range)
        mini_input = min(value_range)
        bell_para = []
        for i in range(granularity *2 + 1):
            width = float((maxi_input - mini_input)/6)
            slope = 2
            center = float(mini_input + (i * (maxi_input - mini_input) / ((granularity *2 + 1) - 1)))
            bell_para.append([width, slope, center])
                
        return bell_para
    
    def create_labels(self, gran, actuator = False): 
        # Find middle index
        mid_index = len(self.granularity_level_standard) // 2

        # Slicing out elements around 'M'
        left_slice = self.granularity_level_standard[mid_index - gran:mid_index]
        right_slice = self.granularity_level_standard[mid_index + 1:mid_index + 1 + gran]

        # Concatenate left slice, 'M', and right slice
        labels = left_slice + ['M'] + right_slice

        mid_id = len(labels) // 2

        # Calculate negative values (elements before mid_index)
        values_neg = [-(mid_id - i) / mid_id for i in range(mid_id)]

        # Calculate positive values (elements after mid_index)
        values_pos = [(i - mid_id) / mid_id for i in range(mid_id + 1, len(labels))]

        # Combine negative, zero, and positive values
        values = values_neg + [0] + values_pos

        ## the Force is actually negative to the position or speed so we reverse the actuator values matrix
        if actuator: 
            # Reverse both labels and values for actuator case
            labels.reverse()
            values.reverse()

        return labels, values

    def create_membership_functions(self, antecedent_or_consequent, universe, params, labels):
        """Create and assign membership functions in the universe."""
        for i, label in enumerate(labels):
            antecedent_or_consequent[label] = fuzz.gbellmf(universe, *params[i])


    def set_rules_matrices(self, rule_matrix = None, operator_matrix=None, weights_matrix=None):
        """Set the rules matrices: operator, rule and weigths matrices"""

        if operator_matrix is None and self.operator_matrix is None:
            self.operator_matrix = np.full((self.input_gran*2 + 1, self.input_gran*2 + 1), 'AND')  # Initialize with default 'AND' operators
        else:
            self.operator_matrix = operator_matrix  
        
        if weights_matrix is None and self.weights_matrix is None:
            self.weights_matrix = np.ones((self.input_gran*2 + 1, self.input_gran*2 + 1))
        else:
            self.weights_matrix = weights_matrix

        if rule_matrix is None and self.rule_matrix is None:
            # Initialize rule matrix
            self.rule_matrix = np.zeros([len(self.input_labels)] * self.num_input)
            # Create a dynamic iteration for the rule_matrix
            for index_tuple in itertools.product(range(len(self.input_labels)), repeat=self.num_input):
                # Unpack the index_tuple dynamically
                value_sum = sum(self.input_values[idx] for idx in index_tuple)
                
                # Assign value based on sum and size of the rule_matrix
                self.rule_matrix[index_tuple] = value_sum / self.num_input
        else:
            self.rule_matrix = rule_matrix

    def set_nonlinear_params(self, a, b, c, d=1):
        """Update non-linear parameters for the RL agent."""
        self.non_linear_params = {'a': a, 'b': b, 'c': c, 'd': d}

    def non_linear_weight(self, x):
        """Non-linear transformation based on a, b, c parameters."""
        a = self.non_linear_params['a']
        b = self.non_linear_params['b']
        c = self.non_linear_params['c']
        d = self.non_linear_params['d']
        return np.exp(a * x ** 3 + b * x ** 2 + c * x + d)

    def apply_operator(self, operator, seat_error_label, seat_error_rate_label):
        """Apply the selected operator dynamically."""
        if operator == 'AND':
            return self.seat_error_antecedent[seat_error_label] & self.seat_error_rate_antecedent[seat_error_rate_label]
        elif operator == 'OR':
            return self.seat_error_antecedent[seat_error_label] | self.seat_error_rate_antecedent[seat_error_rate_label]
        elif operator == 'NOT':
            return ~self.seat_error_antecedent[seat_error_label] & self.seat_error_rate_antecedent[seat_error_rate_label]
        elif operator == 'NOR':
            return ~(self.seat_error_antecedent[seat_error_label] | self.seat_error_rate_antecedent[seat_error_rate_label])
        elif operator == 'NAND':
            return ~(self.seat_error_antecedent[seat_error_label] & self.seat_error_rate_antecedent[seat_error_rate_label])
        elif operator == 'XOR':
            return self.seat_error_antecedent[seat_error_label] ^ self.seat_error_rate_antecedent[seat_error_rate_label]
        elif operator == 'XNOR':
            return ~(self.seat_error_antecedent[seat_error_label] ^ self.seat_error_rate_antecedent[seat_error_rate_label])
        else:
            raise ValueError(f"Unknown operator: {operator}")

    def create_dynamic_rules(self):
        """Dynamically create fuzzy rules with non-linear weighting."""
        rules = []
        for i, seat_error_label in enumerate(self.input_labels):
            for j, seat_error_rate_label in enumerate(self.input_labels):
                action = self.rule_matrix[i, j]
                operator = self.operator_matrix[i, j]
                weight = self.weights_matrix[i, j]

                # Non-linear weight transformation
                # transformed_weight = self.non_linear_weight(weight)
                transformed_weight = weight

                action_output_idx = (np.abs(action - np.array(self.output_values))).argmin()

                consequence = self.output_labels[action_output_idx]

                # Apply dynamic operator
                rule_condition = self.apply_operator(operator, seat_error_label, seat_error_rate_label)

                # Create the rule with the transformed weight
                rule = ctrl.Rule(rule_condition, self.actuator_force_consequent[consequence]) 
                rules.append(rule)

        # Create control system and simulation based on the rules
        self.control_system = ctrl.ControlSystem(rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)


    def plot_membership_functions(self):
        """
        Plot all membership functions present in the fuzzy system using the `.view()` method.
        This will loop through all antecedents and consequents and plot each in a separate window.
        """

        fuzzy_variables = list(self.antecedents.values()) + list(self.consequents.values())

        for fuzzy_var in fuzzy_variables:
            fuzzy_var.view()

    def compute_actuator_force(self, seat_state, tyre_state, road_state = None):
        """
        This function computes the actuator force based on the fuzzy rules.
        """

        # Pass input values to the fuzzy simulation
        self.simulation.input['seat_error'] = seat_state[0][1] #- tyre_state[0][1]
        self.simulation.input['seat_error_rate'] = seat_state[1][1]
        
        # Compute the output (actuator force)
        self.simulation.compute()
        
        # Return the computed actuator force
        return self.simulation.output['actuator_force'] 

    def plot_control_surface(self, seat_states = None, tyre_states = None):
        """
        Plot the control surface of the fuzzy controller to visualize how the controller responds
        to different combinations of seat error and seat error rate.
        """

        if seat_states is None or tyre_states is None:
            seat_states, tyre_states = self.lookup_init_para()
            print('Missing states range. Using default values.')
        else:
            self.seat_states, self.tyre_states = seat_states, tyre_states

        num_rows = len(seat_states)
        num_cols = len(seat_states[0])

        min_seat_error = 0
        min_seat_error_rate = 0
        max_seat_error = 0
        max_seat_error_rate = 0

        z = np.zeros((num_rows, num_cols))

        # Compute actuator force for each combination of seat_state and tyre_state
        for i in range(num_rows):
            for j in range(num_cols):
                z[i, j] = self.compute_actuator_force(seat_states[i][j], tyre_states[i][j])
                min_seat_error = min(min_seat_error, seat_states[i][j][0][1])
                min_seat_error_rate = min(min_seat_error_rate, seat_states[i][j][1][1])
                max_seat_error = max(max_seat_error, seat_states[i][j][0][1])
                max_seat_error_rate = max(max_seat_error_rate, seat_states[i][j][1][1])

        # Create a meshgrid based on the seat error and seat error rate ranges
        seat_error_range = np.linspace(min_seat_error, max_seat_error, len(seat_states))
        seat_error_rate_range = np.linspace(min_seat_error_rate, max_seat_error_rate, len(seat_states[0]))
        
        X, Y = np.meshgrid(seat_error_range, seat_error_rate_range)

        # Plotting the control surface
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, z, cmap='viridis')

        ax.set_xlabel('Seat Error')
        ax.set_ylabel('Seat Error Rate')
        ax.set_zlabel('Actuator Force')
        ax.set_title('Control Surface')

        plt.show()

    def precompute_lookup_table(self, seat_states = None, tyre_states = None):
        """
        Precompute a lookup table for the control surface to speed up real-time control.
        """
        if seat_states is None or tyre_states is None:
            seat_states, tyre_states = self.lookup_init_para()
            print('Missing states range. Using default values.')
        else:
            self.seat_states, self.tyre_states = seat_states, tyre_states
        # Get the number of rows and columns from seat_states (assuming 2D array-like structure)
        num_rows = len(seat_states)        # Number of rows in seat_states
        num_cols = len(seat_states[0])     # Number of columns in seat_states

        # Initialize the lookup table with the correct dimensions
        self.lookup_table = np.zeros((num_rows, num_cols))

        # Fill the lookup table (assuming you use a nested loop)
        for i in range(num_rows):
            for j in range(num_cols):
                self.lookup_table[i, j] = self.compute_actuator_force(seat_states[i][j], tyre_states[i][j])

        return self.lookup_table
    
    def lookup_init_para(self):
        lookup_granularity = 20
        seat_error_range_lookup = np.linspace(-0.1, 0.1, lookup_granularity)  # Adjust range if needed
        seat_error_rate_range_lookup = np.linspace(-0.75, 0.75, lookup_granularity)  # Adjust range if needed

        # Initialize lookup tables for seat and tyre states as arrays of size (20, 20)
        self.seat_states_lookup = np.zeros((lookup_granularity, lookup_granularity, 2, 2))  # 2x2 for each entry
        self.tyre_states_lookup = np.zeros((lookup_granularity, lookup_granularity, 2, 2))  # Same structure for tyres

        # Populate seat_states_lookup and tyre_states_lookup with values
        for i in range(lookup_granularity):
            for j in range(lookup_granularity):
                self.seat_states_lookup[i, j] = np.array([[0, seat_error_range_lookup[i]], [0, seat_error_rate_range_lookup[j]]])
                self.tyre_states_lookup[i, j] = np.array([[0, 0], [0, 0]])  # Populate based on actual logic

        # Optionally return these values if needed, though now they are stored in `self`
        return self.seat_states_lookup, self.tyre_states_lookup


    def get_actuator_force_from_lookup(self, seat_error_val, seat_error_rate_val):
        """
        Retrieve precomputed actuator force from the lookup table based on the inputs.
        """
        if self.lookup_table is None: 
            self.precompute_lookup_table()

        # Extract y values from the first row and all columns
        seat_error_values = np.array([row[0][0][1] for row in self.seat_states_lookup])

        # Extract ydot values from all rows but only the first column
        seat_error_rate_values = np.array([state[1][1] for state in self.seat_states_lookup[0]])

        # Convert seat_error_val and seat_error_rate_val to nearest index using precomputed lookup values
        seat_error_idx = (np.abs(np.array(seat_error_values) - seat_error_val)).argmin()
        seat_error_rate_idx = (np.abs(np.array(seat_error_rate_values) - seat_error_rate_val)).argmin()

        # Retrieve the precomputed actuator force
        return self.lookup_table[seat_error_idx, seat_error_rate_idx]

