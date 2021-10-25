# This gridworld was created by Ken Arnold and Allen Schmaltz as part
# of the 2015 offering of CS282 at Harvard; I've adapted things just
# slightly to make for a very simple tutorial.  

import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random

# Maze state is represented as a 2-element NumPy array: (Y, X). Increasing Y is South.

# Possible actions, expressed as (delta-y, delta-x).
maze_actions = {
    'N': np.array([-1, 0]),
    'S': np.array([1, 0]),
    'E': np.array([0, 1]),
    'W': np.array([0, -1]),
    'P': np.array([0, 0])
}

def onehot(s, size):
    if s == 0:
        onehot_s = [1] + [0] * (size - 1 - s)
    else:
        onehot_s = [0] * s + [1] + [0] * (size - 1 - s)
    return onehot_s

class Maze(object):
    """
    Simple wrapper around a NumPy 2D array to handle flattened indexing and staying in bounds.
    """
    def __init__(self, topology):
        self.topology = parse_topology(topology)
        self.flat_topology = self.topology.ravel()
        self.shape = self.topology.shape

    def in_bounds_flat(self, position):
        return 0 <= position < np.product(self.shape)

    def in_bounds_unflat(self, position):
        return 0 <= position[0] < self.shape[0] and 0 <= position[1] < self.shape[1]

    def get_flat(self, position):
        if not self.in_bounds_flat(position):
            raise IndexError("Position out of bounds: {}".format(position))
        return self.flat_topology[position]

    def get_unflat(self, position):
        if not self.in_bounds_unflat(position):
            raise IndexError("Position out of bounds: {}".format(position))
        return self.topology[tuple(position)]

    def flatten_index(self, index_tuple):
        return np.ravel_multi_index(index_tuple, self.shape)

    def unflatten_index(self, flattened_index):
        return np.unravel_index(flattened_index, self.shape)

    def flat_positions_containing(self, x):
        return list(np.nonzero(self.flat_topology == x)[0])

    def flat_positions_not_containing(self, x):
        return list(np.nonzero(self.flat_topology != x)[0])

    def unflat_positions_containing(self, x):
        flat_positions = list(np.nonzero(self.flat_topology == x)[0])
        unflat_positions = [self.unflatten_index(flat_position) for flat_position in flat_positions]
        return unflat_positions

    def unflat_positions_not_containing(self, x):
        flat_positions = list(np.nonzero(self.flat_topology != x)[0])
        unflat_positions = [self.unflatten_index(flat_position) for flat_position in flat_positions]
        return unflat_positions
    
    @property
    def start_coords(self):
        return [self.unflatten_index(x) for x in self.flat_positions_containing('o')]

    @property
    def goal_coords(self):
        return [self.unflatten_index(x) for x in self.flat_positions_containing('*') + self.flat_positions_containing('$')]

    def __str__(self):
        return '\n'.join(''.join(row) for row in self.topology.tolist())

    def __repr__(self):
        return 'Maze({})'.format(repr(self.topology.tolist()))

    def __len__(self):
        return len(self.topology)

    def __getitem__(self, i):
        return self.topology[i]

def parse_topology(topology):
    return np.array([list(row) for row in topology])

def move_avoiding_walls(maze, position, action):
    """
    Return the new position after moving, and the event that happened ('hit-wall' or 'moved').

    Works with the position and action as a (row, column) array.
    """
    # Compute new position
    new_position = position + action

    # Compute collisions with walls, including implicit walls at the ends of the world.
    if not maze.in_bounds_unflat(new_position) or maze.get_unflat(new_position) == '#':
        return position, 'hit-wall'

    if action[0] == 0 and action[1] == 1:
        return tuple(new_position), 'not moved'
    else:
        return tuple(new_position), 'moved'

class GridWorld(gym.Env):
    """
    A simple task in a maze: get to the goal.

    Parameters
    ----------

    maze : list of strings or lists
        maze topology (see below)

    rewards: dict of string to number. default: {'*': 10}.
        Rewards obtained by being in a maze grid with the specified contents,
        or experiencing the specified event (either 'hit-wall' or 'moved'). The
        contributions of content reward and event reward are summed. For
        example, you might specify a cost for moving by passing
        rewards={'*': 10, 'moved': -1}.

    terminal_markers: sequence of chars, default '*'
        A grid cell containing any of these markers will be considered a
        "terminal" state.

    action_error_prob: float
        With this probability, the requested action is ignored and a random
        action is chosen instead.

    Notes
    -----

    Maze topology is expressed textually. Key:
     '#': wall
     '.': open (really, anything that's not '#')
     '*': goal
     'o': origin
     'X': pitfall
    """

    def __init__(self, maze, reward_fn, terminal_markers='', action_error_prob=0, directions="NSEWP", state_embedding='unflatten'):

        self.maze_dimensions = (len(maze), len(maze[0]))

        self.maze = Maze(maze) if not isinstance(maze, Maze) else maze
        self.reward_fn = reward_fn
        self.terminal_markers = terminal_markers
        self.action_error_prob = action_error_prob

        self.actions = [maze_actions[direction] for direction in directions]
        self.num_actions = len(self.actions)
        self.state = None
        self.state_flatten = None
        self.state_onehot = None
        self.state_shape = self.maze.shape
        self.state_embedding = state_embedding # unflatten or onehot
        self.num_states = self.maze.shape[0] * self.maze.shape[1]
        self.action_space = spaces.Discrete(5)
        if state_embedding == 'unflatten':
            self.observation_space = spaces.Box(low=0, high=10, shape=(2,))
            self.observe     = self.observe_unflatten
            self.is_terminal = self.is_terminal_unflatten
            self.is_cliff    = self.is_cliff_unflatten
        elif state_embedding == 'flatten':
            self.observation_space = space.Box(low=0, shape=(1,))
            self.observe     = self.observe_flatten
            self.is_terminal = self.is_terminal_flatten
            self.is_cliff    = self.is_cliff_flatten
        elif state_embedding == 'onehot':
            self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_states,))
            self.observe     = self.observe_onehot
            self.is_terminal = self.is_terminal_onehot
            self.is_cliff    = self.is_cliff_onehot
        else:
            raise NotImplementedError('Unimplemented embedding method!')

        self.number_steps = 0
        self.max_number_steps = 20
        self.reset()
        
    def __repr__(self):
        return 'GridWorld(maze={maze!r}, reward_fn={reward_fn}, terminal_markers={terminal_markers}, action_error_prob={action_error_prob})'.format(**self.__dict__)

    def reset(self):
        """
        Reset the position to a starting position (an 'o'), chosen at random.
        """
        options = self.maze.unflat_positions_containing('o')
        self.state = options[np.random.choice(len(options)) ]
        self.state_flatten = self.maze.flatten_index(self.state)
        self.state_onehot = onehot(self.state_flatten, self.num_states)
        self.number_steps = 0
        return self.observe()

    """Check if the given state is a terminal state."""
    def is_terminal_unflatten(self, state):
        return (self.maze.get_unflat(state) in self.terminal_markers) or (self.number_steps >= self.max_number_steps)

    def is_terminal_flatten(self, state):
        return (self.maze.get_flat(state) in self.terminal_markers) or (self.number_steps >= self.max_number_steps)

    def is_terminal_onehot(self, state):
        state_flatten = state.index(1)
        return (self.maze.get_flat(state_flatten) in self.terminal_markers) or (self.number_steps >= self.max_number_steps)

    """Check if the given state is a cliff state."""
    def is_cliff_unflatten(self, state):
        return self.maze.get_unflat(state) == 'X'

    def is_cliff_flatten(self, state):
        return self.maze.get_flat(state) == 'X'

    def is_cliff_onehot(self, state):
        state_flatten = state.index(1)
        return self.maze.get_flat(state_flatten) == 'X'

    def observe_unflatten(self):
        return self.state

    def observe_flatten(self):
        return self.state_flatten

    def observe_onehot(self):
        return self.state_onehot

    def step(self, action_idx):
        """Perform an action (specified by index), yielding a new state and reward."""
        # In the absorbing end state, nothing does anything.
        if self.is_terminal_unflatten(self.state):
            return self.observe(), 0, True, {}

        if self.action_error_prob and np.random.rand() < self.action_error_prob:
            # finale! this would be just pick any direction
            # changed it to be slip 90-degrees in some direction
            action_idx = np.random.choice(self.num_actions)
            # if np.random.rand() < .5:
            #     if action_idx == 0 or action_idx == 1:
            #         action_idx = 2
            #     else: 
            #         action_idx = 0
            # else: 
            #     if action_idx == 0 or action_idx == 1:
            #         action_idx = 3
            #     else: 
            #         action_idx = 1
            
        self.number_steps += 1

        # reward = self.reward_fn(self.state, action_idx)

        action = self.actions[action_idx]
        new_state, result = move_avoiding_walls(self.maze, self.state, action)

        # reward = self.reward_fn(self.state, action_idx) # reward_fn is a function of the state and action
        reward = self.reward_fn(new_state) # reward_fn is just a function of the new state

        done = self.is_terminal_unflatten(new_state)

        self.state = new_state
        self.state_flatten = self.maze.flatten_index(new_state)
        self.state_onehot = onehot(self.state_flatten, self.num_states)
        return self.observe(), reward, done, {}

    def close(self):
        pass
