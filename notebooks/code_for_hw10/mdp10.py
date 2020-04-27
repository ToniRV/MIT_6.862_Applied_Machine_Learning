import pdb
import random
import numpy as np
from dist import uniform_dist, delta_dist, mixture_dist, DDist
from util import argmax_with_val, argmax
from torch import nn
import torch

class MDP:
    # Needs the following attributes:
    # states: list or set of states
    # actions: list or set of actions
    # discount_factor: real, greater than 0, less than or equal to 1
    # start: optional instance of DDist, specifying initial state dist
    #    if it's unspecified, we'll use a uniform over states
    # These are functions:
    # transition_model: function from (state, action) into DDist over next state
    # reward_fn: function from (state, action) to real-valued reward

    def __init__(self, states, actions, transition_model, reward_fn,
                     discount_factor = 1.0, start_dist = None):
        self.states = states
        self.actions = actions
        self.transition_model = transition_model
        self.reward_fn = reward_fn
        self.discount_factor = discount_factor
        self.start = start_dist if start_dist else uniform_dist(states)

    # Given a state, return True if the state should be considered to
    # be terminal.  You can think of a terminal state as generating an
    # infinite sequence of zero reward.
    def terminal(self, s):
        return False

    # Randomly choose a state from the initial state distribution
    def init_state(self):
        return self.start.draw()

    # Simulate a transition from state s, given action a.  Return
    # reward for (s,a) and new state, drawn from transition.  If a
    # terminal state is encountered, sample next state from initial
    # state distribution
    def sim_transition(self, s, a):
        return (self.reward_fn(s, a),
                self.init_state() if self.terminal(s) else
                    self.transition_model(s, a).draw())

    def state2vec(self, s):
        '''
        Return one-hot encoding of state s; used in neural network agent implementations
        '''
        v = np.zeros((1, len(self.states)))
        v[0,self.states.index(s)] = 1.
        return v

# Perform value iteration on an MDP, also given an instance of a q
# function.  Terminate when the max-norm distance between two
# successive value function estimates is less than eps.
# interactive_fn is an optional function that takes the q function as
# argument; if it is not None, it will be called once per iteration,
# for visuzalization

# The q function is typically an instance of TabularQ, implemented as a
# dictionary mapping (s, a) pairs into Q values This must be
# initialized before interactive_fn is called the first time.

def value_iteration(mdp, q, eps = 0.01, max_iters = 1000):
    # Your code here (COPY FROM HW9)
    raise NotImplementedError('value_iteration')

# Given a state, return the value of that state, with respect to the
# current definition of the q function
def value(q, s):
    # Your code here (COPY FROM HW9)
    raise NotImplementedError('value')

# Given a state, return the action that is greedy with reespect to the
# current definition of the q function
def greedy(q, s):
    # Your code here (COPY FROM HW9)
    raise NotImplementedError('greedy')

def epsilon_greedy(q, s, eps = 0.5):
    if random.random() < eps:  # True with prob eps, random action
        # Your code here (COPY FROM HW9)
        raise NotImplementedError('epsilon_greedy')
    else:
        # Your code here (COPY FROM HW9)
        raise NotImplementedError('epsilon_greedy')

class TabularQ:
    def __init__(self, states, actions):
        self.actions = actions
        self.states = states
        self.q = dict([((s, a), 0.0) for s in states for a in actions])
    def copy(self):
        q_copy = TabularQ(self.states, self.actions)
        q_copy.q.update(self.q)
        return q_copy
    def set(self, s, a, v):
        self.q[(s,a)] = v
    def get(self, s, a):
        return self.q[(s,a)]
    def update(self, data, lr):
        # Your code here
        raise NotImplementedError('TabularQ.update')

#### Test for update method of tabularQ
def test_update_method():
  q = TabularQ([0,1,2,3],['b','c'])
  q.update([(0, 'b', 50), (2, 'c', 30)], 0.5)
  q.update([(0, 'b', 25)], 0.5)
  if q.get(0, 'b') == 25.0 and q.get(2, 'c') == 15.0: 
    print('PASSED')
    return
  print('FAILED')
  return

# uncomment the line below to test your update method implementation from TabularQ
# test_update_method()

def Q_learn(mdp, q, lr=.1, iters=100, eps = 0.5, interactive_fn=None):
    # Your code here
    raise NotImplementedError('Q_learn')
    for i in range(iters):
        # include this line in the iteration, where i is the iteration number
        if interactive_fn: interactive_fn(q, i)
    pass

#### Test for Q_learn
def tinyTerminal(s):
    return s==4
def tinyR(s, a):
    if s == 1: return 1
    elif s == 3: return 2
    else: return 0
def tinyTrans(s, a):
    if s == 0:
        if a == 'a':
            return DDist({1 : 0.9, 2 : 0.1})
        else:
            return DDist({1 : 0.1, 2 : 0.9})
    elif s == 1:
        return DDist({1 : 0.1, 0 : 0.9})
    elif s == 2:
        return DDist({2 : 0.1, 3 : 0.9})
    elif s == 3:
        return DDist({3 : 0.1, 0 : 0.5, 4 : 0.4})
    elif s == 4:
        return DDist({4 : 1.0})
      
def testQ():
    tiny = MDP([0, 1, 2, 3, 4], ['a', 'b'], tinyTrans, tinyR, 0.9)
    tiny.terminal = tinyTerminal
    q = TabularQ(tiny.states, tiny.actions)
    qf = Q_learn(tiny, q)
    ret = list(qf.q.items())
    expected = [((0, 'a'), 0.6649739221724159), ((0, 'b'), 0.1712369526453748), 
                ((1, 'a'), 0.7732751316011999), ((1, 'b'), 1.2034912054227331), 
                ((2, 'a'), 0.37197205380133874), ((2, 'b'), 0.45929063274463033), 
                ((3, 'a'), 1.5156163024818292), ((3, 'b'), 0.8776852768653631), 
                ((4, 'a'), 0.0), ((4, 'b'), 0.0)]
    ok = True
    for (s,a), v in expected:
      qv = qf.get(s,a)
      if abs(qv-v) > 1.0e-5:
        print("Oops!  For (s=%s, a=%s) expected %s, but got %s" % (s, a, v, qv))
        ok = False
    if ok:
      print("Tests passed!")

# uncomment the 2 lines below to test your Q_learn method
# random.seed(0)
# testQ()      

# Simulate an episode (sequence of transitions) of at most
# episode_length, using policy function to select actions.  If we find
# a terminal state, end the episode.  Return accumulated reward a list
# of (s, a, r, s') where s' is None for transition from terminal state.
# Also return an animation if draw=True.
def sim_episode(mdp, episode_length, policy, draw=False):
    episode = []
    reward = 0
    s = mdp.init_state()
    all_states = [s]
    for i in range(int(episode_length)):
        a = policy(s)
        (r, s_prime) = mdp.sim_transition(s, a)
        reward += r
        if mdp.terminal(s):
            episode.append((s, a, r, None))
            break
        episode.append((s, a, r, s_prime))
        if draw:
            mdp.draw_state(s)
        s = s_prime
        all_states.append(s)
    animation = animate(all_states, mdp.n, episode_length) if draw else None
    return reward, episode, animation

# Create a matplotlib animation from all states of the MDP that
# can be played both in colab and in local versions.
def animate(states, n, ep_length):
    try:
        from matplotlib import animation, rc
        import matplotlib.pyplot as plt
        from google.colab import widgets

        plt.ion()
        plt.figure(facecolor="white")
        fig, ax = plt.subplots()
        plt.close()

        def animate(i):
            if states[i % len(states)] == None or states[i % len(states)] == 'over':
                return
            ((br, bc), (brv, bcv), pp, pv) = states[i % len(states)]
            im = np.zeros((n, n+1))
            im[br, bc] = -1
            im[pp, n] = 1
            ax.cla()
            ims = ax.imshow(im, interpolation = 'none',
                        cmap = 'viridis',
                        extent = [-0.5, n+0.5,
                                    -0.5, n-0.5],
                        animated = True)
            ims.set_clim(-1, 1)
        rc('animation', html='jshtml')
        anim = animation.FuncAnimation(fig, animate, frames=ep_length, interval=100)
        return anim
    except:
        # we are not in colab, so the typical animation should work
        return None

# Return average reward for n_episodes of length episode_length
# while following policy (a function of state) to choose actions.
def evaluate(mdp, n_episodes, episode_length, policy):
    score = 0
    length = 0
    for i in range(n_episodes):
        # Accumulate the episode rewards
        r, e, _ = sim_episode(mdp, episode_length, policy)
        score += r
        length += len(e)
        # print('    ', r, len(e))
    return score/n_episodes, length/n_episodes

def Q_learn_batch(mdp, q, lr=.1, iters=100, eps=0.5,
                  episode_length=10, n_episodes=2,
                  interactive_fn=None):
    # Your code here
    raise NotImplementedError('Q_learn_batch')
    for i in range(iters):
        # include this line in the iteration, where i is the iteration number
        if interactive_fn: interactive_fn(q, i)
    pass


#### Test for Q_learn_batch
def testBatchQ():
    tiny = MDP([0, 1, 2, 3, 4], ['a', 'b'], tinyTrans, tinyR, 0.9)
    tiny.terminal = tinyTerminal
    q = TabularQ(tiny.states, tiny.actions)
    qf = Q_learn_batch(tiny, q)
    ret = list(qf.q.items())
    expected = [((0, 'a'), 4.7566600197286535), ((0, 'b'), 3.993296047838986), 
                ((1, 'a'), 5.292467934685342), ((1, 'b'), 5.364014782870985), 
                ((2, 'a'), 4.139537149779127), ((2, 'b'), 4.155347555640753), 
                ((3, 'a'), 4.076532544818926), ((3, 'b'), 4.551442974149778), 
                ((4, 'a'), 0.0), ((4, 'b'), 0.0)]

    ok = True
    for (s,a), v in expected:
      qv = qf.get(s,a)
      if abs(qv-v) > 1.0e-5:
        print("Oops!  For (s=%s, a=%s) expected %s, but got %s" % (s, a, v, qv))
        ok = False
    if ok:
      print("Tests passed!")
      
      return list(qf.q.items())

# uncomment the 2 lines below to test your Q_learn_batch method
# random.seed(0)
# testBatchQ()



def make_nn(state_dim, num_hidden_layers, num_units):
    '''
    model = Sequential()
    model.add(Dense(num_units, input_dim = state_dim, activation='relu'))
    for i in range(num_hidden_layers-1):
        model.add(Dense(num_units, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer=Adam())
    return model
    '''
    model = []
    model += [nn.Linear(state_dim, num_units), nn.ReLU()]
    for i in range(num_hidden_layers-1):
        model += [nn.Linear(num_units, num_units), nn.ReLU()]
    model += [nn.Linear(num_units, 1)]
    model = nn.Sequential(*model)
    return model

class NNQ:
    def __init__(self, states, actions, state2vec, num_layers, num_units,
                 lr=1e-2, epochs=1):
        self.running_loss = 0. # To keep a running average of the loss
        self.running_one = 0. # idem
        self.num_running = 0.001 # idem
        self.lr = lr
        self.actions = actions
        self.states = states
        self.state2vec = state2vec
        self.epochs = epochs
        state_dim = state2vec(states[0]).shape[1] # a row vector

        self.models = None        # Your code here

    def predict(self, model, s):
      return model(torch.FloatTensor(self.state2vec(s))).detach().numpy()

    def get(self, s, a):
        # Your code here
        raise NotImplementedError('NNQ.get')

    def fit(self, model, X,Y, epochs=None, dbg=None):
      if epochs is None: epochs = self.epochs
      train = torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.FloatTensor(Y))
      train_loader = torch.utils.data.DataLoader(train, batch_size=256,shuffle=True)
      opt = torch.optim.SGD(model.parameters(), lr=self.lr)
      for epoch in range(epochs):
        for (X,Y) in train_loader:
          opt.zero_grad()
          loss = torch.nn.MSELoss()(model(X), Y)
          loss.backward()
          self.running_loss = self.running_loss*(1.-self.num_running) + loss.item()*self.num_running
          self.running_one = self.running_one*(1.-self.num_running) + self.num_running
          opt.step()
      if dbg is True or (dbg is None and np.random.rand()< (0.001*X.shape[0])):
        print('Loss running average: ', self.running_loss/self.running_one)

    def update(self, data, lr, dbg=None):
        # Your code here: train the model for every action
        # Remember to check there is actually data to train on!
        raise NotImplementedError('NNQ.update')


# Code for tests
def tinyTerminal(s):
    return s==4
def tinyR(s, a):
    if s == 1: return 1
    elif s == 3: return 2
    else: return 0
def tinyTrans(s, a):
    if s == 0:
        if a == 'a':
            return DDist({1 : 0.9, 2 : 0.1})
        else:
            return DDist({1 : 0.1, 2 : 0.9})
    elif s == 1:
        return DDist({1 : 0.1, 0 : 0.9})
    elif s == 2:
        return DDist({2 : 0.1, 3 : 0.9})
    elif s == 3:
        return DDist({3 : 0.1, 0 : 0.5, 4 : 0.4})
    elif s == 4:
        return DDist({4 : 1.0})
def tinyTrans2(s, a):
    if s == 0:
        return DDist({1 : 1.0})
    elif s == 1:
        return DDist({2 : 1.0})
    elif s == 2:
        return DDist({3 : 1.0})
    elif s == 3:
        return DDist({4 : 1.0})
    elif s == 4:
        return DDist({4 : 1.0})

def test_NNQ(data):
    tiny = MDP([0, 1, 2, 3, 4], ['a', 'b'], tinyTrans, tinyR, 0.9)
    tiny.terminal = tinyTerminal
    q = NNQ(tiny.states, tiny.actions, tiny.state2vec, 2, 10)
    q.update(data, 1)
    ret =  [q.get(s,a) for s in q.states for a in q.actions]
    expect = [np.array([[-0.07211456]]), np.array([[-0.19553234]]),
              np.array([[-0.21926211]]), np.array([[0.01699455]]),
              np.array([[-0.26390356]]), np.array([[0.06374809]]),
              np.array([[0.0340214]]), np.array([[-0.18334733]]),
              np.array([[-0.438375]]), np.array([[-0.13844737]])]
    cnt = 0
    ok = True
    for s in q.states:
      for a in q.actions:
        if not np.all(np.abs(ret[cnt]-expect[cnt]) < 1.0e0):
          print("Oops, for s=%s, a=%s expected %s but got %s" % (s, a, expect[cnt], ret[cnt]))
          ok = False
        cnt += 1
    if ok:
      print("Output looks generally ok")
    return q

def test_NNQ2(data):
    np.random.seed(0)
    torch.manual_seed(0)
    tiny = MDP([0, 1, 2, 3, 4], ['a', 'b'], tinyTrans2, tinyR, 0.9)
    tiny.terminal = tinyTerminal
    q = NNQ(tiny.states, tiny.actions, tiny.state2vec, 2, 10)
    q.update(data, 1)
    return [q.get(s,a).item(0) for s in q.states for a in q.actions]


# Uncomment the following lines to test your NNQ implementation

# test_NNQ([(0,'a',0.3),(1,'a',0.1),(0,'a',0.1),(1,'a',0.5)])
# print(test_NNQ2([(0,'a',0.3),(1,'a',0.1),(0,'a',0.1),(1,'a',0.5)]))
