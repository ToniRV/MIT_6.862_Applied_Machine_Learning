from util import *

class SM:
    start_state = None  # default start state

    def transition_fn(self, s, x):
        '''s:       the current state
           x:       the given input
           returns: the next state'''
        raise NotImplementedError

    def output_fn(self, s):
        '''s:       the current state
           returns: the corresponding output'''
        raise NotImplementedError

    def transduce(self, input_seq):
        '''input_seq: the given list of inputs
           returns:   list of outputs given the inputs'''
        # Your code here
        pass


class Accumulator(SM):
    start_state = 0

    def transition_fn(self, s, x):
        return s + x

    def output_fn(self, s):
        return s


class Binary_Addition(SM):
    start_state = None # Change

    def transition_fn(self, s, x):
        # Your code here
        pass

    def output_fn(self, s):
        # Your code here
        pass


class Reverser(SM):
    start_state = None # Change

    def transition_fn(self, s, x):
        # Your code here
        pass

    def output_fn(self, s):
        # Your code here
        pass


class RNN(SM):
    def __init__(self, Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2, start_state):
        # Your code here
        pass

    def transition_fn(self, s, x):
        # Your code here
        pass

    def output_fn(self, s):
        # Your code here
        pass

Wsx =  None            # Your code here
Wss =  None            # Your code here
Wo =  None             # Your code here
Wss_0 =  None          # Your code here
Wo_0 =  None           # Your code here
f1 =  None             # Your code here, e.g. lambda x : x
f2 =  None             # Your code here
start_state = None     # Your code here
acc_sign = RNN(Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2, start_state)

Wsx =  None            # Your code here
Wss =  None            # Your code here
Wo =  None             # Your code here
Wss_0 =  None          # Your code here
Wo_0 =  None           # Your code here
f1 =  None             # Your code here, e.g. lambda x : x
f2 =  None             # Your code here
start_state = None     # Your code here
auto = RNN(Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2, start_state)
