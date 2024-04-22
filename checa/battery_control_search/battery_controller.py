import numpy as np
import heapq
import time


class TreeNode:
    def __init__(self, state, parent=None, action=None, level=0, cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.level = level
        self.cost = cost
        self.children = []

    def __lt__(self, other):
        # This method is used by the heapq module to compare nodes
        return self.cost < other.cost

    def add_child(self, child):
        self.children.append(child)

    def is_leaf(self):
        return len(self.children) == 0


class BatteryController:
    def __init__(self, battery_capacity, tau, min_soc_per_hour, dt=0.1, max_soc=1.0, balance_type='A'):
        self.capacity = battery_capacity
        self.tau = tau
        self.dt = dt
        self.max_soc = max_soc
        self.min_soc_per_hour = min_soc_per_hour
        self.balance_type = balance_type

        self.root = None
        self.nodes_expanded = 0
        self.best_final_cost = float('inf')
        self.best_final_node = None
        self.future_balances = None
        self.obj_balance = None

    def fit(self, state):
        self.root = None
        self.nodes_expanded = 0
        self.best_final_cost = float('inf')
        self.best_final_node = None

        self.obj_balance = state[0]
        self.future_balances = state[2:]

    def expand_node(self, hour, node):
        last_soc = node.state[node.level]
        max_action = self.max_soc - last_soc
        used_hour = (hour + 1 + node.level) % 24
        min_action = self.min_soc_per_hour[str(used_hour)] - last_soc

        possible_actions = self.get_possible_actions(min_action, max_action)
        for action in possible_actions:
            new_node_state, new_cost = self.transition_model(node.state, action, node.level)
            cost = node.cost + new_cost
            new_node = TreeNode(new_node_state, node, action, node.level + 1, cost)
            node.add_child(new_node)

    def get_possible_actions(self, lower_bound, upper_bound):
        # Determine possible actions
        num_steps = int((upper_bound - lower_bound) / self.dt) + 1
        return np.linspace(lower_bound, upper_bound, num_steps+1)

    def transition_model(self, node_state, action, level):
        """
        node_state: [soc at t, soc at t+1 ... soc at t+tau, t balance, t+1 balance, t+2 balance, ..., t+tau balance]
                        0           1               tau        tau+1     tau+2      tau+2            2*tau + 1
        action: action to take at time t+level
        """
        new_state = node_state.copy()

        last_soc = node_state[level]
        last_balance = node_state[level + self.tau + 1]
        pred_balance = self.future_balances[level + 1]

        new_soc = last_soc + action

        new_balance = pred_balance + action * self.capacity

        new_state[level+1] = new_soc
        new_state[level + self.tau + 2] = new_balance

        # Compute cost, as the abs(last_balance - new_balance)
        if self.balance_type == 'A':
            cost = np.abs(self.obj_balance - new_balance)
        elif self.balance_type == 'B':
            cost = np.abs(last_balance - new_balance)
        else:  # self.balance_type == 'C':
            cost = np.abs((self.obj_balance + pred_balance)/2 - new_balance)

        """if new_balance < self.obj_balance:  # Lower than last balance -
            cost = np.abs(self.obj_balance - new_balance)
        else:
            cost = np.abs(last_balance - new_balance)"""
        return new_state, cost

    def search(self, state, hour):
        """
        state: [Avg balance, Cur SOC, Cur balance, t+1 balance, t+2 balance, ..., t+tau balance]
        node_state: [soc at t, soc at t+1 ... soc at t+tau, t balance, t+1 balance, t+2 balance, ..., t+tau balance]
                        0           1               tau        tau+1     tau+2      tau+2            2*tau + 1
        """
        self.fit(state)
        # Get root node state33
        node_state = np.zeros(2*(self.tau + 1))
        node_state[0] = state[1] # SOC at t
        node_state[self.tau+1] = state[2] # Balance at t
        self.root = TreeNode(node_state)
        # Use a priority queue to expand the most promising nodes first
        priority_queue = []
        heapq.heappush(priority_queue, (self.root.cost, self.root))
        final_level_nodes = []
        start_time = time.time()
        while priority_queue:
            current_cost, current_node = heapq.heappop(priority_queue)

            if current_cost > self.best_final_cost:
                continue

            self.nodes_expanded += 1

            if current_node.level == self.tau:
                #print(f"{self.nodes_expanded} nodes expanded. Final level node found with cost {current_cost:.3f}")
                final_level_nodes.append(current_node)
                if current_cost < self.best_final_cost:
                    self.best_final_cost = current_cost
                    self.best_final_node = current_node

                if (self.nodes_expanded >= 10000) or (time.time() - start_time > 5):  # Max 5 seconds
                    break
                continue

            self.expand_node(hour, current_node)
            for child in current_node.children:
                heapq.heappush(priority_queue, (child.cost, child))

        #plot_tree(self.root)
        return self.find_actions(self.best_final_node), self.best_final_cost

    def find_actions(self, final_node):
        socs = final_node.state[:self.tau+1]

        actions = []
        for i in range(self.tau):
            actions.append(socs[i+1] - socs[i])
        return actions

