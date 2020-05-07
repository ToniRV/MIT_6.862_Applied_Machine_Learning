from util import quadratic_loss, \
    quadratic_linear_gradient, tanh, tanh_gradient, softmax, NLL, \
    NLL_softmax_gradient, sigmoid, sigmoid_gradient
import numpy as np
import sm

# Based on an implementation by Michael Sun

# l : input_dim
# m : state_dim
# n : output_dim

class RNN:
    weight_scale = .01
    def __init__(self, l, m, n, loss_fn, f2, dloss_f2, step_size=0.1, f1 = tanh, df1 = tanh_gradient, init_state = None):
        self.input_dim = l
        self.hidden_dim = m
        self.output_dim = n
        self.loss_fn = loss_fn
        self.dloss_f2 = dloss_f2
        self.step_size = step_size
        self.f1 = f1
        self.f2 = f2
        self.df1 = df1
        # Initial state is all zeros
        self.init_state = np.zeros((self.hidden_dim, 1))
        self.hidden_state = self.init_state
        # Initialize weight matrices
        self.Wsx = np.random.random((m, l)) * self.weight_scale
        self.Wss = np.random.random((m, m)) * self.weight_scale
        self.Wo =  np.random.random((n, m)) * self.weight_scale
        self.Wss0 = np.random.random((m, 1)) * self.weight_scale
        self.Wo0 = np.random.random((n, 1)) * self.weight_scale

    # Just one step of forward propagation.  x and y are for a single time step
    # Depends on self.hidden_state and reassigns it
    # Returns predicted output, loss on this output, and dLoss_dz2
    def forward_propagation(self, x):
        new_state = self.f1(np.dot(self.Wsx, x) +
                            np.dot(self.Wss, self.hidden_state) + self.Wss0)
        z2 = np.dot(self.Wo, new_state) + self.Wo0
        p = self.f2(z2)
        self.hidden_state = new_state
        return p

    def forward_prop_loss(self, x, y):
        p = self.forward_propagation(x)
        loss = self.loss_fn(p, y)
        dL_dz2 = self.dloss_f2(p, y)
        return p, loss, dL_dz2

    # Back propgation through time
    # xs is matrix of inputs: l by k
    # dL_dz2 is matrix of output errors:  1 by k
    # states is matrix of state values: m by k
    def bptt(self, xs, dLtdz2, states):
        dWsx = np.zeros_like(self.Wsx)
        dWss = np.zeros_like(self.Wss)
        dWo = np.zeros_like(self.Wo)
        dWss0 = np.zeros_like(self.Wss0)
        dWo0 = np.zeros_like(self.Wo0)
        # Derivative of future loss (from t+1 forward) wrt state at time t
        # initially 0;  will pass "back" through iterations
        dFtdst = np.zeros((self.hidden_dim, 1))
        k = xs.shape[1]
        # Technically we are considering time steps 1..k, but we need
        # to index into our xs and states with indices 0..k-1
        for t in range(k-1, -1, -1):
            # Get relevant quantities
            xt = xs[:, t:t+1]
            st = states[:, t:t+1]
            stm1 = states[:, t-1:t] if t-1 >= 0 else self.init_state
            dLtdz2t = dLtdz2[:, t:t+1]
            # Compute gradients step by step
            # ==> Use self.df1(st) to get dfdz1;
            # ==> Use self.Wo, self.Wss, etc. for weight matrices
            # derivative of loss at time t wrt state at time t
            dLtdst = None        # Your code
            raise Exception("bptt implementation incomplete")   # comment this out
            # derivatives of loss from t forward
            dFtm1dst = None            # Your code
            dFtm1dz1t = None           # Your code
            dFtm1dstm1 = None          # Your code
            # gradients wrt weights
            dLtdWo = None              # Your code
            dLtdWo0 = None             # Your code
            dFtm1dWss = None           # Your code
            dFtm1dWss0 = None          # Your code
            dFtm1dWsx = None           # Your code
            # Accumulate updates to weights
            dWsx += dFtm1dWsx
            dWss += dFtm1dWss
            dWss0 += dFtm1dWss0
            dWo += dLtdWo
            dWo0 += dLtdWo0
            # pass delta "back" to next iteration
            dFtdst = dFtm1dstm1
        return dWsx, dWss, dWo, dWss0, dWo0

    def sgd_step(self, xs, dLdz2s, states,
                 gamma1 = 0.9, gamma2 = 0.999, fudge = 1.0e-8):
        dWsx, dWss, dWo, dWss0, dWo0 = self.bptt(xs, dLdz2s, states)
        self.Wsx -= self.step_size * dWsx
        self.Wss -= self.step_size * dWss
        self.Wo -= self.step_size * dWo
        self.Wss0 -= self.step_size * dWss0
        self.Wo0 -= self.step_size * dWo0

    def reset_hidden_state(self):
        self.hidden_state = self.init_state

    def forward_seq(self, x, y):
        k = x.shape[1]
        dLdZ2s = np.zeros((self.output_dim, k))
        states = np.zeros((self.hidden_dim, k))
        train_error = 0.0
        self.reset_hidden_state()
        for j in range(k):
            p, loss, dLdZ2 = self.forward_prop_loss(x[:, j:j+1], y[:, j:j+1])
            dLdZ2s[:, j:j+1] = dLdZ2
            states[:, j:j+1] = self.hidden_state
            train_error += loss
        return train_error/k, dLdZ2s, states

    # For now, training_seqs will be a list of pairs of np arrays.
    # First will be l x k second n x k where k is the sequence length
    # and can be different for each pair
    def train_seq_to_seq(self, training_seqs, epochs = 100000,
                         print_interval = None):
        if print_interval is None: print_interval = int(epochs / 10)
        num_seqs = len(training_seqs)
        total_train_err = 0
        for epoch in range(epochs):
            i = np.random.randint(num_seqs)
            x, y = training_seqs[i]
            avg_seq_train_error, dLdZ2s, states = self.forward_seq(x, y)

            # grads = self.bptt(x, dLdZ2s, states)
            # grads_n = self.num_grad(lambda : forward_seq(x, y, dLdZ2s,
            # states)[0])
            # compare_grads(grads, grads_n)

            self.sgd_step(x, dLdZ2s, states)
            total_train_err += avg_seq_train_error
            if (epoch % print_interval) == 0 and epoch > 0:
                print('training error', total_train_err / print_interval)
                total_train_err = 0

    def num_grad(self, f, delta=0.001):
        out = []
        for W in (self.Wsx, self.Wss, self.Wo, self.Wss0, self.Wo0):
            Wn = np.zeros(W.shape)
            out.append(Wn)
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    wi = W[i,j]
                    W[i,j] = wi - delta
                    fxm = f()
                    W[i,j] = wi + delta
                    fxp = f()
                    W[i,j] = wi
                    Wn[i,j] = (fxp - fxm)/(2*delta)
        return out

    # Return a state machine made out of these weights
    def sm(self):
        return sm.RNN(self.Wsx, self.Wss, self.Wo, self.Wss0, self.Wo0,
                      self.f1, self.f2)

    # Assume that input and output are same dimension
    def gen_seq(self, init_sym, seq_len, codec):
        assert self.input_dim == self.output_dim
        assert self.f2 == softmax
        result  = []
        self.reset_hidden_state()
        x = codec.encode(init_sym)
        for _ in range(seq_len):
            p = self.forward_propagation(x)
            x = np.random.multinomial(1, p.T[0])
            print(p)
            print(x)
            result.append(codec.decode(x))
        print('end')
        return result

def compare_grads(g, gn):
    names = ('Wsx', 'Wss', 'Wo', 'Wss0', 'Wo0')
    for i in range(len(g)):
        diff = np.max(np.abs(g[i]-gn[i]))
        if diff > 0.001:
            print('Diff in', names[i], 'is', diff)
            print('Analytical')
            print(g[i])
            print('Numerical')
            print(gn[i])
            input('Go?')

############################################################################
#
# One-hot encoding/decoding
#
############################################################################

class OneHotCodec:
    def __init__(self, alphabet):
        pairs = list(enumerate(alphabet))
        self.n = len(pairs)
        self.coder = dict([(c, i) for (i, c) in pairs])
        self.decoder = dict(pairs)

    # Take a symbol, return a one-hot vector
    def encode(self, c):
        return self.encode_index(self.coder[c])

    # Take an index, return a one-hot vector
    def encode_index(self, i):
        v = np.zeros((self.n, 1))
        v[i, 0] = 1
        return v

    # Take a one-hot vector, return a symbol
    def decode(self, v):
        return self.decoder[int(np.nonzero(v)[0])]

    # Take a probability vector, return max likelihood symbol
    def decode_max(self, v):
        return self.decoder[np.argmax(v)]

    def encode_seq(self, cs):
        return np.hstack([self.encode(c) for c in cs])

############################################################################
#
#  Testing
#
############################################################################

np.random.seed(0) # set the random seed to ensure the output is the same across students
np.seterr(over='raise')

def linear_accumulator_test(num_epochs = 10000,
                            num_seqs = 100, seq_length =5,
                            step_size = .01):
    data = []
    for _ in range(num_seqs):
        x = np.random.random((1, seq_length)) - 0.5
        y = np.zeros((1, seq_length))
        for j in range(seq_length):
            y[0, j] = x[0, j] + (0.0 if j == 0 else y[0, j-1])
        data.append((x, y))
    rnn = RNN(1, 1, 1, quadratic_loss, lambda z: z, quadratic_linear_gradient,
              step_size, lambda z: z, lambda z: 1)
    rnn.train_seq_to_seq(data, num_epochs)
    print(rnn.Wsx); print(rnn.Wss); print(rnn.Wo); print(rnn.Wss0); print(rnn.Wo0)


def delay_num_test(delay = 1, num_epochs = 10000,
               num_seqs = 10000, seq_length = 10,
               step_size = .005):

    # In case we want to initialize.  Now just for delay = 1
    #Wsx = np.array([[1.], [0.]])
    #Wss = np.array([[0., 0.],
    #              [1., 0.]])
    #Wo = np.array([[0., 1.]])
    #Wss0 = np.array([[0.], [0.0]])
    #Wo0 = np.array([[0.]])

    data = []
    for _ in range(num_seqs):
        vals = np.random.random((1, seq_length))
        x = np.hstack([vals, np.zeros([1, delay])])
        y = np.hstack([np.zeros((1, delay)), vals])
        data.append((x, y))
    m = (delay + 1) * 2
    rnn = RNN(1, m, 1, quadratic_loss, lambda z: z, quadratic_linear_gradient,
              step_size, lambda z: z, lambda z: 1)
    # Wsx = Wsx, Wo = Wo, Wss = Wss, Wo0 = Wo0, Wss0 = Wss0)
    rnn.train_seq_to_seq(data, num_epochs)
    assert np.all(np.isclose(rnn.Wsx, np.array([[0.00856855],
        [0.01936238],
        [0.01382334],
        [0.00771265]])))
    assert np.all(np.isclose(rnn.Wss, np.array([[0.01505222, 0.02059889, 0.01594291, 0.00558242],
        [0.00824307, 0.01741733, 0.01864768, 0.01079751],
        [0.01577334, 0.0124164 , 0.0171516 , 0.0071486 ],
        [0.00722321, 0.00497032, 0.00514737, 0.00979516]])))
    assert np.all(np.isclose(rnn.Wo, np.array([[0.02522786, 0.01744193, 0.01995478, 0.0084726 ]])))
    assert np.all(np.isclose(rnn.Wss0, np.array([[0.01502946],
        [0.01181406],
        [0.02051297],
        [0.00716202]])))
    assert np.all(np.isclose(rnn.Wo0, np.array([[0.42198824]])))
    print('Test passed!') 
    
    print(rnn.Wsx); print(rnn.Wss); print(rnn.Wo); print(rnn.Wss0); print(rnn.Wo0)
    mm = rnn.sm()
    print(mm.transduce([np.array([[float(v)]]) \
                        for v in [3, 4, 5, 1, 2, -1, 6]]))


def delay_char_test(delay = 1, num_epochs = 10000,
                    alphabet = tuple(range(10)),
                    num_seqs = 10000, seq_length = 4, step_size = .001):

    # In case we want to initialize.  Now just for delay = 1, n = 2
    """
    Wsx = np.array([[1., 0.],
                    [0., 1.],
                    [0., 0.],
                    [0., 0.]])
    Wss = np.array([[0., 0., 0., 0.],
                    [0., 0., 0., 0.],
                    [1., 0., 0., 0.],
                    [0., 1., 0., 0.]])
    Wo = np.array([[0., 0., 1., 0.],
                   [0., 0., 0., 1.]])
    Wss0 = np.array([[0.], [0.], [0.], [0.]])
    Wo0 = np.array([[0.], [0.]])
    """
    codec = OneHotCodec(alphabet)
    n = codec.n
    data = []
    for _ in range(num_seqs):
        rand_seq = np.random.random_integers(0, n-1, seq_length)
        vals = codec.encode_seq(rand_seq)
        pad = codec.encode_seq(alphabet[0:1] * delay)
        x = np.hstack([vals, pad])
        y = np.hstack([pad, vals])
        data.append((x, y))

    m = (delay + 1) * n
    # f1, df1 = lambda z: z, lambda z: 1
    # f1, df1 = sigmoid, sigmoid_gradient
    f1, df1 = tanh, tanh_gradient
    loss, f2, dLdf2 = NLL, softmax, NLL_softmax_gradient
    # loss, f2, dLdf2 = quadratic_loss, lambda x: x, quadratic_linear_gradient
    rnn =  RNN(n, m, n, loss, f2, dLdf2,
               step_size, f1, df1)
               # fill buffer up with first char
               #init_state = np.vstack([codec.encode(alphabet[0])]*(delay + 1)),
               #Wsx = Wsx, Wo = Wo, Wss = Wss, Wo0 = Wo0, Wss0 = Wss0)
    rnn.train_seq_to_seq(data, num_epochs)

    # Demo
    mm = rnn.sm()
    vin = [codec.encode(c) for c in [0, 1, 1, 0, 0, 2, 1, 2, 0, 1, 1]]
    vout = mm.transduce(vin)
    cout = [codec.decode_max(v) for v in vout]
    print(cout)

# interpret first bit as lowest order;  may leave off highest-order bit
# s1 and s2 are 1 x k
# return 1 x k
def bin_add(s1, s2):
    k = s1.shape[1]
    result = np.zeros((1, k))
    carry = 0
    for j in range(k):
        tot = s1[0, j] + s2[0, j] + carry
        result[0,j] = tot % 2
        carry = 1 if tot > 1 else 0
    return result

def binary_addition_test(num_seqs = 1000, seq_length = 5, num_epochs = 50000,
                         step_size = 0.01, num_hidden = 8):

    data = []
    for _ in range(num_seqs):
        s1 = np.random.random_integers(0, 1, (1, seq_length))
        s2 = np.random.random_integers(0, 1, (1, seq_length))
        x = np.vstack([s1, s2])
        y = bin_add(s1, s2)
        data.append((x, y))

    l = 2 # two input dimensions
    m = num_hidden
    n = 1 # one output dimension

    #f1 = lambda x: x; df1 = lambda x: 1
    f1 = sigmoid; df1 = sigmoid_gradient
    loss = quadratic_loss
    f2 = lambda z: z; dldz2 = quadratic_linear_gradient

    rnn =  RNN(l, m, n, loss, f2, dldz2, step_size, f1, df1)
    rnn.train_seq_to_seq(data, num_epochs)

    mm = rnn.sm()
    n1 = '01101'
    n2 = '01111'
    # answer is:    11100
    a = [np.array([[float(d1), float(d2)]]).T for d1, d2 in zip(n1, n2)]
    vin = list(reversed(a))
    vout = mm.transduce(vin)
    print(vin)
    print(vout)


def seq_prediction_test(num_epochs = 10000, step_size = 0.01,
                        num_seqs = 100, seq_length = 5,
                        num_hidden = 10):

    # find unique tokens.  hacked for now.
    alphabet = ('a', 'b', 'c', 'd', 'e', 'f')

    codec = OneHotCodec(alphabet + ('end',))
    data = []
    al = len(alphabet)
    for _ in range(num_seqs):
        b = np.random.randint(al)
        char_seq = [alphabet[(b + i) % al] for i in range(seq_length)]
        vals = codec.encode_seq(char_seq)
        pad = codec.encode_seq(['end'])
        y = np.hstack([vals, pad])
        x = np.hstack([pad, vals])
        data.append((x, y))

    l = codec.n
    m = num_hidden
    n = codec.n
    f1 = tanh; df1 = tanh_gradient
    loss = NLL
    f2 = softmax; dldz2 = NLL_softmax_gradient

    rnn =  RNN(l, m, n, loss, f2, dldz2, step_size, f1, df1)
    rnn.train_seq_to_seq(data, num_epochs)

    # test by generating random strings
    for _ in range(5):
        print([codec.decode_max(c) for c in
                         rnn.gen_seq('end', seq_length, codec)])


############################################################################
#
#  Testing
#
############################################################################

#delay_num_test(delay = 1, seq_length = 10, num_epochs = 10000, step_size =.005)

#linear_accumulator_test()


#delay_char_test(delay = 1, seq_length = 5,
#                num_epochs = 200000, step_size =.01)

# Works with 4 hidden, does not work with 3 hidden
#binary_addition_test(seq_length = 5, num_seqs = 200, num_epochs = 200000,
#                     step_size = 0.05, num_hidden = 4)

#seq_prediction_test(num_epochs = 50000, num_hidden = 20, step_size = 0.05)
