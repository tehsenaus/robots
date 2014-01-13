import numpy
import theano
import theano.tensor as TT

# Recurrent Neural Net
class RNN:
    def __init__(self, n, nin, nout):
        # number of hidden units
        self.n = n
        # number of input units
        self.nin = nin
        # number of output units
        self.nout = nout


        # recurrent weights as a shared variable
        W = self.W = theano.shared(numpy.random.uniform(size=(n, n), low=-.01, high=.01))
        # input to hidden layer weights
        W_in = self.W_in = theano.shared(numpy.random.uniform(size=(nin, n), low=-.01, high=.01))
        # hidden to output layer weights
        W_out = self.W_out = theano.shared(numpy.random.uniform(size=(n, nout), low=-.01, high=.01))


        # recurrent function (using tanh activation function) and linear output
        # activation function
        def step(u_t, h_tm1, W, W_in, W_out):
            h_t = TT.tanh(TT.dot(u_t, W_in) + TT.dot(h_tm1, W))
            y_t = TT.dot(h_t, W_out)
            return h_t, y_t

        # input (where first dimension is time)
        u = self.__u = TT.matrix()
        # initial hidden state of the RNN
        h0 = self.__h0 = TT.vector()

        # the hidden state `h` for the entire sequence, and the output for the
        # entrie sequence `y` (first dimension is always time)
        [h, y], _ = theano.scan(step,
                                sequences=u,
                                outputs_info=[h0, None],
                                non_sequences=[W, W_in, W_out])
        self.__y = y

        self.__run = theano.function([u, h0], [h, y])

    class Instance:
        def __init__(self, rnn, h0):
            self.rnn = rnn
            self.h = h0

        def run(self, u):
            hs, ys = self.rnn.run(u, self.h)
            self.h = hs[-1]
            return ys

        def step(self, x):
            return self.run([x])[0]

    def createInstance(self, h0=None):
        if h0 is None:
            h0 = numpy.zeros(self.n)
        return self.Instance(self, h0)

    def randomize(self):
        self.W.set_value(numpy.random.uniform(size=(self.n, self.n), low=-1., high=1.))
        self.W_in.set_value(numpy.random.uniform(size=(self.nin, self.n), low=-1., high=1.))
        self.W_out.set_value(numpy.random.uniform(size=(self.n, self.nout), low=-1., high=1.))
        return self

    def run(self, u, h0):
        return self.__run(u, h0)
    
    # Trains the RNN
    # 
    def train(self, u, t, h0, lr):
        self.initTraining()

        return self.__train(h0, u, t, lr)

    def initTraining(self):
        try:
            self.__train
        except AttributeError:

            print "Initializing training..."

            # target (where first dimension is time)
            t = TT.matrix()
            # learning rate
            lr = TT.scalar()

            # error between output and target
            error = ((self.__y - t) ** 2).sum()
            # gradients on the weights using BPTT
            gW, gW_in, gW_out = TT.grad(error, [self.W, self.W_in, self.W_out])
            # training function, that computes the error and updates the weights using
            # SGD.
            self.__train = theano.function([self.__h0, self.__u, t, lr],
                                 error,
                                 updates={self.W: self.W - lr * gW,
                                         self.W_in: self.W_in - lr * gW_in,
                                         self.W_out: self.W_out - lr * gW_out})

            print "... training initialized"


if __name__ == "__main__":
    rnn = RNN(50, 5, 5)

    n = 1000
    u = numpy.random.uniform(size=(n, rnn.nin), low=-1., high=1.)
    t = numpy.random.uniform(size=(n, rnn.nout), low=-.01, high=.1)

    for i in range(50):
        print rnn.train(u, t, numpy.zeros(rnn.n), 0.005)
