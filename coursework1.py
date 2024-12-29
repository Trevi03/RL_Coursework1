import numpy as np
import random
import matplotlib.pyplot as plt # Graphical library
import matplotlib
import math

# WARNING: fill in these two functions that will be used by the auto-marking script
# [Action required]

def get_CID():
  return "02145518" # Return your CID (add 0 at the beginning to ensure it is 8 digits long)

def get_login():
  return "txy21" # Return your short imperial login

# This class is used ONLY for graphics
# YOU DO NOT NEED to understand it to work on this coursework
class GraphicsMaze(object):

  def __init__(self, shape, locations, default_reward, obstacle_locs, absorbing_locs, absorbing_rewards, absorbing, disp_env):

    self.shape = shape
    self.locations = locations
    self.absorbing = absorbing

    # Walls
    self.walls = np.zeros(self.shape)
    for ob in obstacle_locs:
      self.walls[ob] = 20

    # Rewards
    self.rewarders = np.ones(self.shape) * default_reward
    for i, rew in enumerate(absorbing_locs):
      self.rewarders[rew] = 10 if absorbing_rewards[i] > 0 else -10

    # Print the map to show it
    if disp_env:
      self.paint_maps()

  def paint_maps(self):
    """
    Print the Maze topology (obstacles, absorbing states and rewards)
    input: /
    output: /
    """
    plt.figure(figsize=(15,10))
    plt.imshow(self.walls + self.rewarders)
    plt.show()

  def paint_state(self, state):
    """
    Print one state on the Maze topology (obstacles, absorbing states and rewards)
    input: /
    output: /
    """
    states = np.zeros(self.shape)
    states[state] = 30
    plt.figure(figsize=(15,10))
    plt.imshow(self.walls + self.rewarders + states)
    plt.show()

  def draw_deterministic_policy(self, Policy):
    """
    Draw a deterministic policy
    input: Policy {np.array} -- policy to draw (should be an array of values between 0 and 3 (actions))
    output: /
    """
    plt.figure(figsize=(15,10))
    plt.imshow(self.walls + self.rewarders) # Create the graph of the Maze
    for state, action in enumerate(Policy):
      if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any action
        continue
      arrows = [r"$\uparrow$",r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"] # List of arrows corresponding to each possible action
      action_arrow = arrows[action] # Take the corresponding action
      location = self.locations[state] # Compute its location on graph
      plt.text(location[1], location[0], action_arrow, ha='center', va='center') # Place it on graph
    plt.show()

  def draw_policy(self, Policy):
    """
    Draw a policy (draw an arrow in the most probable direction)
    input: Policy {np.array} -- policy to draw as probability
    output: /
    """
    deterministic_policy = np.array([np.argmax(Policy[row,:]) for row in range(Policy.shape[0])])
    self.draw_deterministic_policy(deterministic_policy)

  def draw_value(self, Value):
    """
    Draw a policy value
    input: Value {np.array} -- policy values to draw
    output: /
    """
    plt.figure(figsize=(15,10))
    plt.imshow(self.walls + self.rewarders) # Create the graph of the Maze
    for state, value in enumerate(Value):
      if(self.absorbing[0, state]): # If it is an absorbing state, don't plot any value
        continue
      location = self.locations[state] # Compute the value location on graph
      plt.text(location[1], location[0], round(value,2), ha='center', va='center') # Place it on graph
    plt.show()

  def draw_deterministic_policy_grid(self, Policies, title, n_columns, n_lines):
    """
    Draw a grid representing multiple deterministic policies
    input: Policies {np.array of np.array} -- array of policies to draw (each should be an array of values between 0 and 3 (actions))
    output: /
    """
    plt.figure(figsize=(20,8))
    for subplot in range (len(Policies)): # Go through all policies
      ax = plt.subplot(n_columns, n_lines, subplot+1) # Create a subplot for each policy
      ax.imshow(self.walls+self.rewarders) # Create the graph of the Maze
      for state, action in enumerate(Policies[subplot]):
        if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any action
          continue
        arrows = [r"$\uparrow$",r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"] # List of arrows corresponding to each possible action
        action_arrow = arrows[action] # Take the corresponding action
        location = self.locations[state] # Compute its location on graph
        plt.text(location[1], location[0], action_arrow, ha='center', va='center') # Place it on graph
      ax.title.set_text(title[subplot]) # Set the title for the graph given as argument
    plt.show()

  def draw_policy_grid(self, Policies, title, n_columns, n_lines):
    """
    Draw a grid representing multiple policies (draw an arrow in the most probable direction)
    input: Policy {np.array} -- array of policies to draw as probability
    output: /
    """
    deterministic_policies = np.array([[np.argmax(Policy[row,:]) for row in range(Policy.shape[0])] for Policy in Policies])
    self.draw_deterministic_policy_grid(deterministic_policies, title, n_columns, n_lines)

  def draw_value_grid(self, Values, title, n_columns, n_lines):
    """
    Draw a grid representing multiple policy values
    input: Values {np.array of np.array} -- array of policy values to draw
    output: /
    """
    plt.figure(figsize=(20,8))
    for subplot in range (len(Values)): # Go through all values
      ax = plt.subplot(n_columns, n_lines, subplot+1) # Create a subplot for each value
      ax.imshow(self.walls+self.rewarders) # Create the graph of the Maze
      for state, value in enumerate(Values[subplot]):
        if(self.absorbing[0,state]): # If it is an absorbing state, don't plot any value
          continue
        location = self.locations[state] # Compute the value location on graph
        plt.text(location[1], location[0], round(value,1), ha='center', va='center') # Place it on graph
      ax.title.set_text(title[subplot]) # Set the title for the graoh given as argument
    plt.show()

  # Added myself
  def draw_value_heatmap(self,values,title="Input Title"):
    """
    Draw a policy value
    input: Value {np.array} -- policy values to draw
    output: /
    """
    minval = float(math.floor(min(values)))
    maxval = float(math.ceil(max(values)))

    matrix = np.full((13,10),minval-1)
    plt.figure(figsize=(15,10))
    
    # Blue scale
    colors = plt.cm.Blues(np.linspace(0.2, 1, 10))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("mycmap", colors) 

    # # Yellow, Green, Blue Seq cmap
    # colors = plt.cm.YlGnBu(np.linspace(0.1, 1, 10))
    # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("mycmap", colors) 

    cmap.set_under('yellow')
    cmap.set_over('purple')
    cmap.set_bad('limegreen')
    
    for state, value in enumerate(values):
      location = self.locations[state] # Compute the value location on graph
      if location==(2, 0):
        matrix[location[0],location[1]] = np.nan
        continue
      elif(self.absorbing[0, state]): # If it is an absorbing state, don't plot any value
        matrix[location[0],location[1]] = maxval+1
        continue
      matrix[location[0],location[1]] = round(value,2)
      plt.text(location[1], location[0], round(value,2), ha='center', va='center') # Place it on graph
    plt.imshow(matrix,cmap=cmap,vmin=minval,vmax=maxval)
    plt.title(title)
    # plt.imshow(self.walls + self.rewarders) # Create the graph of the Maze
    plt.show()

# This class define the Maze environment
class Maze(object):

  # [Action required]
  def __init__(self, 
               prob_success = 0.96,
               gamma = 0.82,
               goal = 0,
               disp_env = True):
    """
    Maze initialisation. y = 1 , z = 8
    input: / p = 0.8+0.02*(9-y) = 0.96  , gamma = 0.8+0.02*y = 0.82
    output: / 
    """

    # [Action required]
    # Properties set from the CID
    self._prob_success = prob_success # float
    self._gamma = gamma # float
    self._goal = goal # integer (0 for R0, 1 for R1, 2 for R2, 3 for R3)

    # Build the maze
    self._build_maze(disp_env)


  # Functions used to build the Maze environment
  # You DO NOT NEED to modify them
  def _build_maze(self,disp_env):
    """
    Maze initialisation.
    input: /
    output: /
    """

    # Properties of the maze
    self._shape = (13, 10)
    self._obstacle_locs = [
                          (1,0), (1,1), (1,2), (1,3), (1,4), (1,7), (1,8), (1,9), \
                          (2,1), (2,2), (2,3), (2,7), \
                          (3,1), (3,2), (3,3), (3,7), \
                          (4,1), (4,7), \
                          (5,1), (5,7), \
                          (6,5), (6,6), (6,7), \
                          (8,0), \
                          (9,0), (9,1), (9,2), (9,6), (9,7), (9,8), (9,9), \
                          (10,0)
                         ] # Location of obstacles
    self._absorbing_locs = [(2,0), (2,9), (10,1), (12,9)] # Location of absorbing states
    self._absorbing_rewards = [ (500 if (i == self._goal) else -50) for i in range (4) ]
    self._starting_locs = [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8), (0,9)] #Reward of absorbing states
    self._default_reward = -1 # Reward for each action performs in the environment
    self._max_t = 500 # Max number of steps in the environment

    # Actions
    self._action_size = 4
    self._direction_names = ['N','E','S','W'] # Direction 0 is 'N', 1 is 'E' and so on

    # States
    self._locations = []
    for i in range (self._shape[0]):
      for j in range (self._shape[1]):
        loc = (i,j)
        # Adding the state to locations if it is no obstacle
        if self._is_location(loc):
          self._locations.append(loc)
    self._state_size = len(self._locations)

    # Neighbours - each line is a state, ranked by state-number, each column is a direction (N, E, S, W)
    self._neighbours = np.zeros((self._state_size, 4))

    for state in range(self._state_size):
      loc = self._get_loc_from_state(state)

      # North
      neighbour = (loc[0]-1, loc[1]) # North neighbours location
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('N')] = self._get_state_from_loc(neighbour)
      else: # If there is no neighbour in this direction, coming back to current state
        self._neighbours[state][self._direction_names.index('N')] = state

      # East
      neighbour = (loc[0], loc[1]+1) # East neighbours location
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('E')] = self._get_state_from_loc(neighbour)
      else: # If there is no neighbour in this direction, coming back to current state
        self._neighbours[state][self._direction_names.index('E')] = state

      # South
      neighbour = (loc[0]+1, loc[1]) # South neighbours location
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('S')] = self._get_state_from_loc(neighbour)
      else: # If there is no neighbour in this direction, coming back to current state
        self._neighbours[state][self._direction_names.index('S')] = state

      # West
      neighbour = (loc[0], loc[1]-1) # West neighbours location
      if self._is_location(neighbour):
        self._neighbours[state][self._direction_names.index('W')] = self._get_state_from_loc(neighbour)
      else: # If there is no neighbour in this direction, coming back to current state
        self._neighbours[state][self._direction_names.index('W')] = state

    # Absorbing
    self._absorbing = np.zeros((1, self._state_size))
    for a in self._absorbing_locs:
      absorbing_state = self._get_state_from_loc(a)
      self._absorbing[0, absorbing_state] = 1

    # Transition matrix
    self._T = np.zeros((self._state_size, self._state_size, self._action_size)) # Empty matrix of domension S*S*A
    for action in range(self._action_size):
      for outcome in range(4): # For each direction (N, E, S, W)
        # The agent has prob_success probability to go in the correct direction
        if action == outcome:
          prob = 1 - 3.0 * ((1.0 - self._prob_success) / 3.0) # (theoritically equal to self.prob_success but avoid rounding error and garanty a sum of 1)
        # Equal probability to go into one of the other directions
        else:
          prob = (1.0 - self._prob_success) / 3.0

        # Write this probability in the transition matrix
        for prior_state in range(self._state_size):
          # If absorbing state, probability of 0 to go to any other states
          if not self._absorbing[0, prior_state]:
            post_state = self._neighbours[prior_state, outcome] # Post state number
            post_state = int(post_state) # Transform in integer to avoid error
            self._T[prior_state, post_state, action] += prob

    # Reward matrix
    self._R = np.ones((self._state_size, self._state_size, self._action_size)) # Matrix filled with 1
    self._R = self._default_reward * self._R # Set default_reward everywhere
    for i in range(len(self._absorbing_rewards)): # Set absorbing states rewards
      post_state = self._get_state_from_loc(self._absorbing_locs[i])
      self._R[:,post_state,:] = self._absorbing_rewards[i]

    # Creating the graphical Maze world
    self._graphics = GraphicsMaze(self._shape, self._locations, self._default_reward, self._obstacle_locs, self._absorbing_locs, self._absorbing_rewards, self._absorbing,disp_env)

    # Reset the environment
    self.reset()


  def _is_location(self, loc):
    """
    Is the location a valid state (not out of Maze and not an obstacle)
    input: loc {tuple} -- location of the state
    output: _ {bool} -- is the location a valid state
    """
    if (loc[0] < 0 or loc[1] < 0 or loc[0] > self._shape[0]-1 or loc[1] > self._shape[1]-1):
      return False
    elif (loc in self._obstacle_locs):
      return False
    else:
      return True


  def _get_state_from_loc(self, loc):
    """
    Get the state number corresponding to a given location
    input: loc {tuple} -- location of the state
    output: index {int} -- corresponding state number
    """
    return self._locations.index(tuple(loc))


  def _get_loc_from_state(self, state):
    """
    Get the state number corresponding to a given location
    input: index {int} -- state number
    output: loc {tuple} -- corresponding location
    """
    return self._locations[state]

  # Getter functions used only for DP agents
  # You DO NOT NEED to modify them
  def get_T(self):
    return self._T

  def get_R(self):
    return self._R

  def get_absorbing(self):
    return self._absorbing

  # Getter functions used for DP, MC and TD agents
  # You DO NOT NEED to modify them
  def get_graphics(self):
    return self._graphics

  def get_action_size(self):
    return self._action_size

  def get_state_size(self):
    return self._state_size

  def get_gamma(self):
    return self._gamma

  # Functions used to perform episodes in the Maze environment
  def reset(self):
    """
    Reset the environment state to one of the possible starting states
    input: /
    output:
      - t {int} -- current timestep
      - state {int} -- current state of the envionment
      - reward {int} -- current reward
      - done {bool} -- True if reach a terminal state / 0 otherwise
    """
    self._t = 0
    self._state = self._get_state_from_loc(self._starting_locs[random.randrange(len(self._starting_locs))])
    self._reward = 0
    self._done = False
    return self._t, self._state, self._reward, self._done

  def step(self, action):
    """
    Perform an action in the environment
    input: action {int} -- action to perform
    output:
      - t {int} -- current timestep
      - state {int} -- current state of the envionment
      - reward {int} -- current reward
      - done {bool} -- True if reach a terminal state / 0 otherwise
    """

    # If environment already finished, print an error
    if self._done or self._absorbing[0, self._state]:
      print("Please reset the environment")
      return self._t, self._state, self._reward, self._done

    # Drawing a random number used for probaility of next state
    probability_success = random.uniform(0,1)

    # Look for the first possible next states (so get a reachable state even if probability_success = 0)
    new_state = 0
    while self._T[self._state, new_state, action] == 0:
      new_state += 1
    assert self._T[self._state, new_state, action] != 0, "Selected initial state should be probability 0, something might be wrong in the environment."

    # Find the first state for which probability of occurence matches the random value
    total_probability = self._T[self._state, new_state, action]
    while (total_probability < probability_success) and (new_state < self._state_size-1):
     new_state += 1
     total_probability += self._T[self._state, new_state, action]
    assert self._T[self._state, new_state, action] != 0, "Selected state should be probability 0, something might be wrong in the environment."

    # Setting new t, state, reward and done
    self._t += 1
    self._reward = self._R[self._state, new_state, action]
    self._done = self._absorbing[0, new_state] or self._t > self._max_t
    self._state = new_state

    return self._t, self._state, self._reward, self._done
  
  # defined for Q3
  def is_terminal(self, state):  
    return self._absorbing[0, state]==1

# This class define the Dynamic Programing agent
class DP_agent(object):

  # [Action required]
  # WARNING: make sure this function can be called by the auto-marking script
  def solve(self, env):
    """
    Solve a given Maze environment using Dynamic Programming
    input: env {Maze object} -- Maze to solve
    output:
      - policy {np.array} -- Optimal policy found to solve the given Maze environment
      - V {np.array} -- Corresponding value function
    """

    # Initialisation (can be edited)
    threshold = 0.0001
    delta = threshold
    gamma = env.get_gamma()
    absorbing = env.get_absorbing()
    state_size = env.get_state_size()
    T = env.get_T()
    R = env.get_R()

    policy = np.zeros((state_size, env.get_action_size()))
    V = np.zeros(env.get_state_size())
    epochs = 0
    
    # Ensure gamma value is valid
    assert (gamma <=1) and (gamma >= 0), "Discount factor should be in [0, 1]."

    while delta >= threshold:
      delta = 0 # Reinitialise delta value
      epochs+=1
      # For each state
      for s0 in range(state_size):

        # If not an absorbing state
        if not absorbing[0, s0]:
                  
          # Store the previous value for that state
          v = V[s0] 

          # Compute Q value
          Q = np.zeros(4) # Initialise with value 0
          for s1 in range(state_size):
            Q += T[s0, s1,:] * (R[s0, s1, :] + gamma * V[s1])
                
          # Set the new value to the maximum of Q
          V[s0]= np.max(Q) 

          # Compute the new delta
          delta = max(delta, np.abs(v - V[s0]))

    for s0 in range(state_size):
      # Compute the Q value
      Q = np.zeros(4)
      for s1 in range(state_size):
        Q += T[s0, s1,:] * (R[s0, s1, :] + gamma * V[s1])
            
      # The action that maximises the Q value gets probability 1
      policy[s0, np.argmax(Q)] = 1 
    
    return policy, V
  
# This class define the Monte-Carlo agent
# On-policy first-visit Monte Carlo with decaying epsilon
class MC_agent(object):
  def __init__(self, epsilon=1.0,max_ep=5000,decay_rate=0.9995):
    self.epsilon = epsilon  # 1 the agent acts randomly, 0 the agent always takes the current greedy actions
    self.max_ep = max_ep
    self.decay_rate = decay_rate

  def generate_episode(self,env, policy):
    """
    Generate an epiosode based of policy.
    input:
      - Q {np.array} -- action-value function (state_size, action_size)
      - policy -- Policy for each episode iteration
      - env 
    output:
      - trajectory {np.array} -- List of states
      - rewards {np.array} -- List of rewards corresponding to list of states
      - actions {np.array} -- List of action corresponding to each initial state
    """
     
    done = False
    trajectory = []
    rewards = []
    actions = []
    t, current_state, reward, done = env.reset()
    # keep walking until absorbing states or reach max step
    while not done:
      # Generate random next step
      action = np.random.choice(range(env.get_action_size()), p=policy[current_state,:])
      trajectory.append(current_state) # update trajectory list
      actions.append(action) # update actions corresponding trace

      # Take a step and check if absorbed
      t, current_state, reward, done = env.step(action)
      rewards.append(reward)
    return trajectory, rewards, actions

  def update_Q(self,env,Q,N, trajectory, actions,rewards):
    """
    Generate an epiosode based of policy.
    input:
      - Q {np.array} -- action-value function (state_size, action_size)
      - N {np.array} -- visit counter function (state_size, action_size)
      - trajectory {np.array} -- List of states for each episode
      - rewards {np.array} -- List of rewards corresponding to list of states per episode
      - actions {np.array} -- List of action corresponding to each initial state per episode
      - policy -- Policy for each episode iteration
      - env 
    output:
      - new_Q {np.array} -- updated Q
      - N {np.array} -- Updated visit counter
    """
    gamma = env.get_gamma()
    new_Q = np.copy(Q)

    # agent experience
    number_of_steps = len(trajectory)

    state_action_visited = []

    # update Q according to experience
    for i in range(number_of_steps):
      s = trajectory[i] # current state
      a = actions[i]    # current action
      if (s, a) in state_action_visited:
        # If state has been visited with action before
        continue
      else:
        G = 0
        for index,reward in enumerate(rewards[i:]):
          # G = gamma * G + reward
          G = G + reward*gamma**index
        # use first time MC update
        N[s, a]+=1  # update the visit counter to this state
        new_Q[s, a] = Q[s, a] + (1/N[s, a]) *(G - Q[s, a])
        state_action_visited.append((s, a))
    return new_Q, N

  def update_policy(self, env, Q,k):
    """
    Epsilon-greedy policy improvement based on a given Q-function and epsilon.
    input:
      - Q {np.array} -- action-value function (state_size, action_size)
      - epsilon {float} -- The probability to select a random action, between 0 and 1.
      - env 
    output:
      - policy {np.array}
    """
    state_size = env.get_state_size()
    action_size = env.get_action_size()
    policy = np.zeros((state_size, action_size))

    decayed_epsilon = self.epsilon*(self.decay_rate**k)
    # update policy
    for s in range(state_size):
      optimal_action = np.argmax(Q[s, :])
      for a in range(action_size):
        if a==optimal_action:
          policy[s][a] = 1-decayed_epsilon+(decayed_epsilon/action_size)
        else:
          policy[s][a] = decayed_epsilon/action_size

    return policy

  # [Action required]
  # WARNING: make sure this function can be called by the auto-marking script
  def solve(self, env):
    """
    Solve a given Maze environment using Monte Carlo learning
    input: env {Maze object} -- Maze to solve
    output:
      - policy {np.array} -- Optimal policy
      - values {list of np.array} -- Final value function
      - total_rewards {list of float} -- Total non-discounted sum of reward for each episode
    """

    # Initialisation (can be edited)
    Q = np.random.rand(env.get_state_size(), env.get_action_size())
    N = np.zeros((env.get_state_size(), env.get_action_size()))
    policy = self.update_policy(env,Q,1)

    values = []
    total_rewards = []
    ep = 0

    while ep<self.max_ep:
      ep+=1
      # Generate an episode from policy
      trajectory, rewards, actions = self.generate_episode(env=env,policy=policy)
      Q, N = self.update_Q(env=env,Q=Q,N=N,trajectory=trajectory,actions=actions,rewards=rewards)
      policy = self.update_policy(env,Q,k=ep)
      total_rewards.append(sum(rewards))
    values = np.max(Q,1)

    return policy, values, total_rewards

class MC_agent_Q3:
  def __init__(self, epsilon=0.4, decay_rate=0.995, temperature=1.0, starting_epsilon=1.0):
    # Initialize parameters for exploration strategies
    self.epsilon = epsilon  # For epsilon-greedy
    self.decay_rate = decay_rate  # For decaying epsilon
    self.starting_epsilon = starting_epsilon
    self.temperature = temperature  # For Softmax

  def choose_action(self, state, q_values, strategy='fixed_epsilon', episode=1):
    # Chooses an action based on the specified exploration strategy.
    if strategy == 'fixed_epsilon':
        return self.epsilon_greedy_action(state, q_values, epsilon=self.epsilon)
    elif strategy == 'decaying_epsilon':
        # Student should modify decayed_epsilon as per requirement
        decayed_epsilon = self.starting_epsilon*(self.decay_rate ** episode)
        return self.epsilon_greedy_action(state, q_values, epsilon=decayed_epsilon)
    elif strategy == 'softmax':
        # Implement softmax strategy logic
        return self.softmax_action(state, q_values, temperature=self.temperature)

  # Placeholder for epsilon-greedy strategy
  def epsilon_greedy_action(self, state, q_values, epsilon):
    # Implement epsilon-greedy selection logic
    if np.random.rand() < epsilon:
        return np.random.choice(range(len(q_values[state])))
    return np.argmax(q_values[state])

  # Placeholder for softmax strategy
  def softmax_action(self, state, q_values, temperature):
    # Implement softmax selection logic here
    preferences = q_values[state] / temperature
    exp_preferences = np.exp(preferences - np.max(preferences))
    probabilities = exp_preferences / np.sum(exp_preferences)
    return np.random.choice(range(len(q_values[state])), p=probabilities)

  # Main solve function with adjustable strategies
  def solve(self, maze, strategy='fixed_epsilon', num_episodes=1000):
    gamma = maze.get_gamma()
    q_values = np.zeros((maze.get_state_size(), maze.get_action_size()))
    policy = np.zeros((maze.get_state_size(), maze.get_action_size()))
    N = np.zeros((maze.get_state_size(), maze.get_action_size()))
    total_rewards = []

    for episode in range(1, num_episodes + 1):
        t, state, reward, done = maze.reset()
        rewards = []
        state_visited = []
        action_done = []
        

        # Generate Episode
        while not done: # define is_terminal()
            # Use chosen strategy in each episode
            action = self.choose_action(state, q_values, strategy=strategy, episode=episode)
            t, next_state, reward, done = maze.step(action)
            # Placeholder for Q-value update logic 
            # - get a list of rewards, states and action per episode
            rewards.append(reward)
            state_visited.append(state)
            action_done.append(action)

            state = next_state
        
        # Update Q values per episode using first-visit MC
        state_action_visited = []

        for i in range(len(state_visited)):
            current_state = state_visited[i]
            current_action = action_done[i]
            if (current_action,current_state) in state_action_visited:
                continue
            else:
                G = 0
                for ind, rew in enumerate(rewards[i:]):
                    G = G + rew * gamma**ind
                N[current_state, current_action]+=1
                q_values[current_state, current_action] += (1/N[current_state, current_action]) *(G - q_values[current_state, current_action])
                state_action_visited.append((current_state, current_action))
        total_rewards.append(sum(rewards))

    for s in range(maze.get_state_size()):
    # The action that maximises the Q value gets probability 1
        policy[s, np.argmax(q_values[s,:])] = 1 
        
    values = np.max(q_values,axis=1)
        
    return policy, values, total_rewards


if __name__=='__main__':
  ### Question 0: Defining the environment
  print("Creating the Maze:\n")
  maze = Maze(disp_env=False)  # defaults: prob_success = 0.96, gamma = 0.82
  # T = maze.get_T()

  ### Question 1: Dynamic programming
  dp_agent = DP_agent()
  dp_policy, dp_value = dp_agent.solve(maze)

  print("Results of the DP agent:\n")
  maze.get_graphics().draw_policy(dp_policy)
  # maze.get_graphics().draw_value(dp_value)
  maze.get_graphics().draw_value_heatmap(dp_value,"p = 0.96, gamma = 0.82")

  ## Effect of gamma and p
  # test gamma values
  gamma_val = 0.2
  maze = Maze(gamma=gamma_val,disp_env=False)   # defaults: prob_success = 0.96, gamma = 0.82
  dp_agent = DP_agent()
  dp_policy, dp_value = dp_agent.solve(maze)

  maze.get_graphics().draw_policy(dp_policy)
  title = "".join(["p = 0.96, gamma = ",str(gamma_val)])
  print("Heatmap of the DP agent result:\n")
  maze.get_graphics().draw_value_heatmap(dp_value,title)

  # test p-values
  p_val = 0.25   # [0.1, 0.25, 0.96]
  maze = Maze(prob_success=p_val,disp_env=False)   # defaults: prob_success = 0.96, gamma = 0.82
  dp_agent = DP_agent()
  dp_policy, dp_value = dp_agent.solve(maze)

  maze.get_graphics().draw_policy(dp_policy)
  title = "".join(["p = ",str(p_val),", gamma = 0.96"])
  print("Heatmap of the DP agent result:\n")
  maze.get_graphics().draw_value_heatmap(dp_value,title)



  ### Question 2: Monte-Carlo learning
  # maze = Maze(disp_env=False)
  # ep = 500

  maze = Maze(disp_env=False)
  ep = 5000

  mc_agent = MC_agent(max_ep=ep,epsilon=0.1)
  mc_policy, mc_values, total_rewards = mc_agent.solve(maze)

  print("Results of the MC agent:\n")
  title = "".join(["decayed epsilon = ",str(mc_agent.epsilon)," x ",str(mc_agent.decay_rate),"^(ep)"])
  maze.get_graphics().draw_policy(mc_policy)
  maze.get_graphics().draw_value_heatmap(mc_values,title)

  plt.plot(np.arange(0, ep, 1),total_rewards)
  plt.xlabel("episode number")
  plt.ylabel("Total non-discounted Reward")
  plt.show()



  ## Learning Curve
  runs = 25
  eps = 5000
  n_total_mc_rewards = np.zeros((runs, eps))
  mc_agent = MC_agent(max_ep=ep)

  print("Complete run: ",end="")
  for run in range(runs):
      mc_policy, mc_values, total_mc_rewards = mc_agent.solve(maze)
      n_total_mc_rewards[run] = total_mc_rewards
      print(" "+str(run+1),end="")
  print()

  # print value and policy
  print("Results of the MC agent:\n")
  #maze.get_graphics().draw_policy(mc_policy)
  #maze.get_graphics().draw_value(mc_values[-1])

  # Q2 x and y values for plots
  x = [i for i in range(1, eps+1)]
  y = np.mean(n_total_mc_rewards, axis=0)
  y_std = np.std(n_total_mc_rewards, axis=0)

  # Q2 plot for mean and std deviation graphs
  plt.plot(x, y, label="mean", color='blue',linewidth=0.8)
  plt.fill_between(x, y-y_std, y+y_std, label="std deviation", alpha=0.4, facecolor="red")
  plt.legend(loc="lower right")
  plt.xlabel("Episode Number")  
  plt.ylabel("Total Non-discounted Sum of Rewards")  
  title = "".join(["decayed epsilon = ",str(mc_agent.epsilon)," x ",str(mc_agent.decay_rate),"^(ep)"])
  plt.title(title)
  plt.show()



  # Question 3: Exploration Strategies for Monte-Carlo (MC) Agent
  # This is to get you started .. but ofcourse you can change
  ## make sure this section DOES NOT run on automarker

  # Testing each exploration strategy and plotting results
  mc_agent = MC_agent_Q3(epsilon=0.4)
  # strategies = ['fixed_epsilon', 'decaying_epsilon', 'softmax']
  strategies = ['fixed_epsilon']
  results = {}

  # Loop for each exploration strategy, with placeholders for results and learning curves
  for strategy in strategies:
      mc_policy, mc_values, total_rewards = mc_agent.solve(maze, strategy=strategy,num_episodes=5000)
      results[strategy] = total_rewards  # Store rewards for learning curve plotting
      print("Computation completed: "+strategy)
      maze.get_graphics().draw_policy(mc_policy)
      maze.get_graphics().draw_value_heatmap(mc_values,strategy)

  # Plot learning curves for exploration strategies
  import matplotlib.pyplot as plt

  c = ['r','b','g']
  ind = 0
  plt.figure(figsize=(10, 6))
  for strategy, rewards in results.items():
      plt.plot(rewards, label=strategy,alpha=0.8,color=c[ind])
      ind+=1

  plt.xlabel('Episodes')
  plt.ylabel('Total Rewards')
  plt.legend()
  plt.title('Learning Curves for Different Exploration Strategies')
  plt.show()


  ## Learning Curve
  runs = 25
  eps = 1000
  n_total_mc_rewards = np.zeros((runs, eps))
  mc_agent = MC_agent_Q3()

  # ['fixed_epsilon', 'decaying_epsilon', 'softmax']
  strat = 'decaying_epsilon'

  print("Complete run: ",end="")
  for run in range(runs):
      mc_policy, mc_values, total_mc_rewards = mc_agent.solve(maze,strategy=strat,num_episodes=eps)
      n_total_mc_rewards[run] = total_mc_rewards
      print(" "+str(run+1),end="")
  print()

  # Q2 x and y values for plots
  x = [i for i in range(1, eps+1)]
  y = np.mean(n_total_mc_rewards, axis=0)
  y_std = np.std(n_total_mc_rewards, axis=0)

  # Q2 plot for mean and std deviation graphs
  plt.plot(x, y, label="mean", color='blue')
  plt.fill_between(x, y-y_std, y+y_std, label="std deviation", alpha=0.4, facecolor="red")
  plt.legend(loc="lower right")
  plt.xlabel("Episode Number")  
  plt.ylabel("Total Non-discounted Sum of Rewards")  
  if strat == 'fixed_epsilon':
      plt.title("epsilon = "+str(mc_agent.epsilon))
  elif strat == 'decaying_epsilon':
      plt.title(" decayed epsilon = "+str(mc_agent.starting_epsilon)+" x "+str(mc_agent.decay_rate)+"^(ep)")
  else:
      plt.title("softmax, temp = "+str(mc_agent.temperature))
  plt.show()

  ## Sensitivity Analysis
  mc_policy, mc_values, total_mc_rewards = mc_agent.solve(maze,strategy='decaying_epsilon',num_episodes=1000)
  # print value and policy
  print("Results of the MC agent:\n")
  maze.get_graphics().draw_policy(mc_policy)
  maze.get_graphics().draw_value_heatmap(mc_values)





