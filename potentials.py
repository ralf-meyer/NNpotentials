import tensorflow as _tf
import numpy as _np
from itertools import combinations_with_replacement

#def nn_layer(input_tensor, input_dim, output_dim, act = _tf.nn.selu,
 # initial_bias = None, name = "layer"):
#    with _tf.name_scope(name):
#        #weights = _tf.Variable(_tf.random_uniform([input_dim, output_dim],
#        #    minval = -1./_np.sqrt(input_dim), maxval = 1./_np.sqrt(input_dim),
#        #    dtype = _tf.float64), name = "W")
#        weights = _tf.Variable(_tf.random_normal([input_dim, output_dim],
#            stddev = 1./_np.sqrt(input_dim), dtype = _tf.float64), name = "W")
#        _tf.add_to_collection(_tf.GraphKeys.REGULARIZATION_LOSSES, weights)
#        if initial_bias == None:
#            biases = _tf.Variable(0.1*_tf.ones([output_dim],
#                dtype = _tf.float64), name = "b")
#        else:
#            biases = _tf.Variable(initial_bias, name = "b")
#        preactivate = _tf.matmul(input_tensor, weights) + biases
#        if act == None:
#            activations = preactivate
#        else:
#            activations = act(preactivate)
#        _tf.summary.histogram("weights", weights)
#        _tf.summary.histogram("biases", biases)
#        _tf.summary.histogram("activations", activations)
#        return activations, weights, biases

def nn_layer(input_tensor, input_dim, output_dim, act = _tf.nn.selu,
  initial_bias = None, name = "layer"):
    with _tf.variable_scope(name):
        #weights = _tf.Variable(_tf.random_uniform([input_dim, output_dim],
        #    minval = -1./_np.sqrt(input_dim), maxval = 1./_np.sqrt(input_dim),
        #    dtype = _tf.float64), name = "W")
        weights = _tf.get_variable("w", dtype = _tf.float64,
            shape = [input_dim, output_dim], initializer = _tf.random_normal_initializer(
            stddev = 1./_np.sqrt(input_dim), dtype = _tf.float64),
            collections = [_tf.GraphKeys.MODEL_VARIABLES,
                            _tf.GraphKeys.REGULARIZATION_LOSSES,
                            _tf.GraphKeys.GLOBAL_VARIABLES])
        if initial_bias == None:
            biases = _tf.get_variable("b", dtype = _tf.float64, shape = [output_dim],
                initializer = _tf.constant_initializer(0.01, dtype = _tf.float64),
                collections = [_tf.GraphKeys.MODEL_VARIABLES,
                            _tf.GraphKeys.GLOBAL_VARIABLES])
        else:
            biases = _tf.get_variable("b", dtype = _tf.float64, shape = [output_dim],
                initializer = _tf.constant_initializer(initial_bias, dtype = _tf.float64),
                collections = [_tf.GraphKeys.MODEL_VARIABLES,
                            _tf.GraphKeys.GLOBAL_VARIABLES])
        preactivate = _tf.matmul(input_tensor, weights) + biases
        if act == None:
            activations = preactivate
        else:
            activations = act(preactivate)
        _tf.summary.histogram("weights", weights)
        _tf.summary.histogram("biases", biases)
        _tf.summary.histogram("activations", activations)
        return activations, weights, biases

def poly_cutoff(input_tensor, cut_a, cut_b):
    with _tf.name_scope("PolyCutoff"):
        return (1.0 - 10.0 * ((input_tensor-cut_a)/(cut_b-cut_a))**3 +
            15.0 * ((input_tensor-cut_a)/(cut_b-cut_a))**4 -
            6.0 * ((input_tensor-cut_a)/(cut_b-cut_a))**5) * \
            _tf.cast(input_tensor < cut_b, dtype = _tf.float64) * \
            _tf.cast(input_tensor > cut_a, dtype = _tf.float64) +\
            1.0 * _tf.cast(input_tensor < cut_a, dtype = _tf.float64)

class AbstractANN():
    def __init__(self):
        self.output = _tf.constant(0.0, dtype = _tf.float64)

class EAMpotential():
    def __init__(self, atom_types):
        self.target = _tf.placeholder(shape = (None,), dtype = _tf.float64,
            name = "target")
        self.atom_types = atom_types

        self.ANNs = {}
        self.atom_maps = {}

        for t in self.atom_types:
            self.atom_maps[t] = _tf.sparse_placeholder(shape = (None, None),
                dtype = _tf.float64, name = "{}_map".format(t))

    def _post_setup(self):
        self.E_predict = _tf.reduce_sum([
            _tf.sparse_tensor_dense_matmul(self.atom_maps[t],
            self.ANNs[t].output) for t in self.atom_types], axis = [0, 2],
            name = "E_prediction")

        # Tensorflow operation to initialize the variables of the atomic networks
        #self.init_vars = [a.init_vars for a in self.ANNs.itervalues()]

        self.num_atoms =  _tf.reduce_sum(
            [_tf.sparse_reduce_sum(m, axis = 1) for m in self.atom_maps.itervalues()],
            axis = 0, name = "NumberOfAtoms")
        # Tensorflow operation that calculates the sum squared error per atom.
        # Note that the whole error per atom is squared.
        with _tf.name_scope("SSE"):
            self.sse = _tf.reduce_sum(_tf.square((self.target - self.E_predict)/
                self.num_atoms), axis = 0, name = "SumSquaredError")
            _tf.summary.scalar("SSE", self.sse, family = "performance")
        with _tf.name_scope("RMSE"):
            self.rmse = _tf.sqrt(_tf.losses.mean_squared_error(self.target,
                self.E_predict, weights = 1.0/self.num_atoms**2))
            _tf.summary.scalar("RMSE", self.rmse, family = "performance")

        self.variables = _tf.get_collection(_tf.GraphKeys.MODEL_VARIABLES,
            scope = _tf.get_default_graph().get_name_scope())
        self.saver = _tf.train.Saver(self.variables)


class BPAtomicNN():
    def __init__(self, input_dim, layers = [20], offset = 0):
        self.input = _tf.placeholder(shape = (None, input_dim),
            dtype = _tf.float64, name = "ANN_input")
        hidden_layers = []
        hidden_vars = []
        for i, n in enumerate(layers):
            if i == 0:
                layer, weights, bias = nn_layer(
                    self.input, input_dim, n, name = "hiddenLayer_%d"%(i+1))
            else:
                layer, weights, bias = nn_layer(hidden_layers[-1], layers[i-1],
                    n, name = "hiddenLayer_%d"%(i+1))
            hidden_layers.append(layer)
            hidden_vars.append(weights)
            hidden_vars.append(bias)
        self.output, out_weights, out_bias = nn_layer(hidden_layers[-1],
            layers[-1], 1, act = None, initial_bias = _np.array([offset],
            dtype = _np.float64), name = "outputLayer")
        #self.variables = hidden_vars + [out_weights, out_bias]
        #self.init_vars = _tf.variables_initializer(self.variables,
        #    name = "ANN_initializer")

class BPpotential(EAMpotential):
    def __init__(self, atom_types, input_dims, layers = None, offsets = None):
        with _tf.variable_scope("BPpot"):
            EAMpotential.__init__(self, atom_types)

            if layers == None:
                layers = [20]*len(self.atom_types)
            if offsets == None:
                offsets = [0.0]*len(self.atom_types)

            for (t, dims, lays, offs) in zip(atom_types, input_dims, layers, offsets):
                with _tf.variable_scope("{}_ANN".format(t), reuse = _tf.AUTO_REUSE):
                    self.ANNs[t] = BPAtomicNN(dims, lays, offs)
            EAMpotential._post_setup(self)

class EAMAtomicNN():
    def __init__(self, atom_types, offset = 0.0, name = "ANN"):
        self.atom_types = atom_types
        self.name = name

        self.pairPot = {}
        self.F = _tf.identity
        self.rho = {}

        self.inputs = {}
        self.b_maps = {}
        for t in atom_types:
            self.inputs[t] = _tf.placeholder(shape = (None, 1),
                dtype = _tf.float64, name = "ANN_input_{}".format(t))
            self.b_maps[t] = _tf.sparse_placeholder(
                dtype = _tf.float64, name = "b_map_{}".format(t))
        self.offset = _tf.Variable(offset, dtype = _tf.float64, name = "offset",
            collections = [_tf.GraphKeys.MODEL_VARIABLES, _tf.GraphKeys.GLOBAL_VARIABLES])
        _tf.summary.scalar("offset", self.offset, family = "modelParams")
        #self.variables = [self.offset]

    def _post_setup(self):
        self.sum_rho = _tf.reduce_sum(
            [_tf.sparse_tensor_dense_matmul(self.b_maps[t], self.rho[t])
            for t in self.atom_types], axis = 0, name = "SumRho")
        _tf.summary.histogram("SumRho", self.sum_rho)
        with _tf.variable_scope(self.name+"_EmbeddingFunc", reuse = _tf.AUTO_REUSE):
            self.F_out = self.F(self.sum_rho)
        self.output = _tf.add(_tf.reduce_sum(
                [_tf.sparse_tensor_dense_matmul(self.b_maps[t], self.pairPot[t])
                for t in self.atom_types], axis = 0, name = "SumPairPot") + \
            self.F_out, self.offset, name = "AtomicEnergy")
        #self.init_vars = _tf.variables_initializer(self.variables)

class SMATBpotential(EAMpotential):
    def __init__(self, atom_types, initial_params = None, offsets = None,
        cut_a = 5.4, cut_b = 8.1, pair_trainable = True, rho_trainable = True,
        r0_trainable = False):
        with _tf.variable_scope("SMATB"):
            EAMpotential.__init__(self, atom_types)

            A = {}
            xi = {}
            p = {}
            q = {}
            r0 = {}
            pairPot = {}
            rho = {}
            for t1, t2 in combinations_with_replacement(atom_types, r = 2):
                t12 = tuple(sorted([t1, t2]))
                A[t12], p[t12], r0[t12], pairPot[t12] = SMATBpotential.build_pairPot(
                    t12, initial_params, cut_a, cut_b, r0_trainable, pair_trainable)
                xi[t12], q[t12], _, rho[t12] = SMATBpotential.build_rho(
                    t12, initial_params, cut_a, cut_b, r0_trainable, rho_trainable)

            if offsets == None:
                offsets = [0.0]*len(self.atom_types)

            for t1, offset in zip(self.atom_types, offsets):
                with _tf.name_scope("{}_ANN".format(t1)):
                    self.ANNs[t1] = EAMAtomicNN(atom_types, offset)
                    self.ANNs[t1].F = lambda rho: -_tf.sqrt(rho)

                    for t2 in self.atom_types:
                        t12 = tuple(sorted([t1, t2]))
                        with _tf.variable_scope("{}_{}_PairPot".format(*t12), reuse = _tf.AUTO_REUSE):
                            self.ANNs[t1].pairPot[t2] = pairPot[t12](
                                self.ANNs[t1].inputs[t2])
                        with _tf.variable_scope("{}_{}_rho".format(*t12), reuse = _tf.AUTO_REUSE):
                            self.ANNs[t1].rho[t2] = rho[t12](
                                self.ANNs[t1].inputs[t2])
                        #self.ANNs[t1].variables.extend(
                        #    [A[t12], xi[t12], p[t12], q[t12], r0[t12]])

                    self.ANNs[t1]._post_setup()
            EAMpotential._post_setup(self)

    @staticmethod
    def build_pairPot(t12, initial_params, cut_a, cut_b, r0_trainable, pair_trainable):
        """ builds the pair potential for a given atom type tuple
            t12 = (t1, t2)
            returns: parameters A, p, r0 and the corresponding pair potential as
                    a function of r
        """
        with _tf.variable_scope("{}_{}_PairPot".format(*t12), reuse = _tf.AUTO_REUSE):
            if ("A", t12[0], t12[1]) in initial_params:
                A_init = initial_params[("A", t12[0], t12[1])]
            else:
                A_init = 0.2
            A = _tf.get_variable("A_{}_{}".format(*t12), dtype = _tf.float64,
                initializer = _tf.constant(A_init, dtype = _tf.float64),
                trainable = pair_trainable,
                collections = [_tf.GraphKeys.MODEL_VARIABLES,
                                _tf.GraphKeys.GLOBAL_VARIABLES])
            _tf.summary.scalar("A_{}_{}".format(*t12), A, family = "modelParams")

            if ("p", t12[0], t12[1]) in initial_params:
                p_init = initial_params[("p", t12[0], t12[1])]
            else:
                p_init = 9.2
            p = _tf.get_variable("p_{}_{}".format(*t12), dtype = _tf.float64,
                initializer = _tf.constant(p_init, dtype = _tf.float64),
                trainable = pair_trainable,
                collections = [_tf.GraphKeys.MODEL_VARIABLES,
                                _tf.GraphKeys.GLOBAL_VARIABLES])
            _tf.summary.scalar("p_{}_{}".format(*t12), p, family = "modelParams")

        with _tf.variable_scope("{}_{}_r0".format(*t12), reuse = _tf.AUTO_REUSE):
            if ("r0", t12[0], t12[1]) in initial_params:
                r0_init = initial_params[("r0", t12[0], t12[1])]
            else:
                r0_init = 2.7
            r0 = _tf.get_variable("r0_{}_{}".format(*t12), dtype = _tf.float64,
                initializer = _tf.constant(r0_init, dtype = _tf.float64),
                trainable = r0_trainable,
                collections = [_tf.GraphKeys.MODEL_VARIABLES,
                                _tf.GraphKeys.GLOBAL_VARIABLES])
            if r0_trainable:
                _tf.summary.scalar("r0_{}_{}".format(*t12), r0,
                    family = "modelParams")

        pairPot = lambda r: A * _tf.exp(
            -p*(r/r0 - 1)) * poly_cutoff(r, cut_a, cut_b)
        return A, p, r0, pairPot

    @staticmethod
    def build_rho(t12, initial_params, cut_a, cut_b, r0_trainable, rho_trainable):
        """ builds the rho contribution for a given atom type tuple
            t12 = (t1, t2)
            returns: parameters xi, q, r0 and the corresponding rho contribution
                    as a function of r
        """
        with _tf.variable_scope("{}_{}_rho".format(*t12), reuse = _tf.AUTO_REUSE):
            if ("xi", t12[0], t12[1]) in initial_params:
                xi_init = initial_params[("xi", t12[0], t12[1])]
            else:
                xi_init = 1.6
            xi = _tf.get_variable("xi_{}_{}".format(*t12), dtype = _tf.float64,
                initializer = _tf.constant(xi_init, dtype = _tf.float64),
                trainable = rho_trainable,
                collections = [_tf.GraphKeys.MODEL_VARIABLES,
                                _tf.GraphKeys.GLOBAL_VARIABLES])
            _tf.summary.scalar("xi_{}_{}".format(*t12), xi, family = "modelParams")

            if ("q", t12[0], t12[1]) in initial_params:
                q_init = initial_params[("q", t12[0], t12[1])]
            else:
                q_init = 3.5
            q = _tf.get_variable("q_{}_{}".format(*t12), dtype = _tf.float64,
                initializer = _tf.constant(q_init, dtype = _tf.float64),
                trainable = rho_trainable,
                collections = [_tf.GraphKeys.MODEL_VARIABLES,
                                _tf.GraphKeys.GLOBAL_VARIABLES])
            _tf.summary.scalar("q_{}_{}".format(*t12), q,
                family = "modelParams")
        with _tf.variable_scope("{}_{}_r0".format(*t12), reuse = _tf.AUTO_REUSE):
            if ("r0", t12[0], t12[1]) in initial_params:
                r0_init = initial_params[("r0", t12[0], t12[1])]
            else:
                r0_init = 2.7
            r0 = _tf.get_variable("r0_{}_{}".format(*t12), dtype = _tf.float64,
                initializer = _tf.constant(r0_init, dtype = _tf.float64),
                trainable = r0_trainable,
                collections = [_tf.GraphKeys.MODEL_VARIABLES,
                                _tf.GraphKeys.GLOBAL_VARIABLES])
            if r0_trainable:
                _tf.summary.scalar("r0_{}_{}".format(*t12), r0,
                    family = "modelParams")

        rho = lambda r: xi**2 * _tf.exp(
            -2.0*q*(r/r0 - 1)) * poly_cutoff(r, cut_a, cut_b)
        return xi, q, r0, rho

class NNEpotential(EAMpotential):
    def __init__(self, atom_types, layers = [20], initial_params = None, cut_a = 5.4, cut_b = 8.1,
        r0_trainable = False, pair_trainable = True, rho_trainable = True, offsets = None):
        with _tf.variable_scope("NNEpot"):
            EAMpotential.__init__(self, atom_types)

            A = {}
            xi = {}
            p = {}
            q = {}
            r0 = {}
            pairPot = {}
            rho = {}

            for t1, t2 in combinations_with_replacement(atom_types, r = 2):
                t12 = tuple(sorted([t1, t2]))
                A[t12], p[t12], r0[t12], pairPot[t12] = SMATBpotential.build_pairPot(
                    t12, initial_params, cut_a, cut_b, r0_trainable, pair_trainable)
                xi[t12], q[t12], _, rho[t12] = SMATBpotential.build_rho(
                    t12, initial_params, cut_a, cut_b, r0_trainable, rho_trainable)

            if offsets == None:
                offsets = [0.0]*len(self.atom_types)

            for t1, offset in zip(self.atom_types, offsets):
                with _tf.variable_scope("{}_ANN".format(t1), reuse = _tf.AUTO_REUSE):
                    self.ANNs[t1] = EAMAtomicNN(atom_types, offset)
                    def F(rho):
                        for i, n in enumerate(layers):
                            if i == 0:
                                layer, weights, bias = nn_layer((rho - 30.0)/15.0, 1, n,
                                    name = "hiddenLayer_%d"%(i+1))
                            else:
                                # Use previous layer as input
                                layer, weights, bias = nn_layer(layer,
                                    layers[i-1], n, name = "hiddenLayer_%d"%(i+1))
                        output, _, _ = nn_layer(layer, layers[-1], 1,
                            act = None, name = "outputLayer")
                        return -_tf.sqrt(rho)*output
                    self.ANNs[t1].F = F

                    for t2 in self.atom_types:
                        t12 = tuple(sorted([t1, t2]))
                        with _tf.name_scope("{}_{}_PairPot".format(*t12)):
                            self.ANNs[t1].pairPot[t2] = pairPot[t12](
                                self.ANNs[t1].inputs[t2])
                        with _tf.name_scope("{}_{}_rho".format(*t12)):
                            self.ANNs[t1].rho[t2] = rho[t12](
                                self.ANNs[t1].inputs[t2])

                    self.ANNs[t1]._post_setup()
            EAMpotential._post_setup(self)

class NNERHOpotential(EAMpotential):
    def __init__(self, atom_types, layers = [20], layers2 = [20],
        initial_params = None, r0_trainable = False, pair_trainable = True,
        cut_a = 5.4, cut_b = 8.1, offsets = None):
        with _tf.variable_scope("NNERHOpot", reuse = _tf.AUTO_REUSE):
            EAMpotential.__init__(self, atom_types)

            A = {}
            xi = {}
            p = {}
            q = {}
            r0 = {}
            pairPot = {}
            rho = {}

            for t1, t2 in combinations_with_replacement(atom_types, r = 2):
                t12 = tuple(sorted([t1, t2]))
                A[t12], p[t12], r0[t12], pairPot[t12] = SMATBpotential.build_pairPot(
                    t12, initial_params, cut_a, cut_b, r0_trainable, pair_trainable)

                def rho_temp(r):
                    for i, n in enumerate(layers2):
                        if i == 0:
                            layer, weights, bias = nn_layer(r/r0[t12]-1, 1, n,
                                name = "hiddenLayer_%d"%(i+1))
                        else:
                            # Use previous layer as input
                            layer, weights, bias = nn_layer(layer,
                                layers[i-1], n, name = "hiddenLayer_%d"%(i+1))
                    output, _, _ = nn_layer(layer, layers[-1], 1,
                        act = None, name = "outputLayer")
                    return _tf.exp(output) * poly_cutoff(r, cut_a, cut_b)

                rho[t12] = rho_temp

            if offsets == None:
                offsets = [0.0]*len(self.atom_types)

            for t1, offset in zip(self.atom_types, offsets):
                with _tf.name_scope("{}_ANN".format(t1)):
                    self.ANNs[t1] = EAMAtomicNN(atom_types, offset, "%s"%t1)
                    def F(rho):
                        for i, n in enumerate(layers):
                            if i == 0:
                                layer, weights, bias = nn_layer(rho, 1, n,
                                    name = "hiddenLayer_%d"%(i+1))
                            else:
                                # Use previous layer as input
                                layer, weights, bias = nn_layer(layer,
                                    layers[i-1], n, name = "hiddenLayer_%d"%(i+1))
                        output, _, _ = nn_layer(layer, layers[-1], 1,
                            act = None, name = "outputLayer")
                        return -_tf.sqrt(rho)*output
                    self.ANNs[t1].F = F

                    for t2 in self.atom_types:
                        t12 = tuple(sorted([t1, t2]))
                        with _tf.variable_scope("{}_{}_PairPot".format(*t12), reuse = _tf.AUTO_REUSE):
                            self.ANNs[t1].pairPot[t2] = pairPot[t12](
                                self.ANNs[t1].inputs[t2])
                        with _tf.variable_scope("{}_{}_rho".format(*t12), reuse = _tf.AUTO_REUSE):
                            self.ANNs[t1].rho[t2] = rho[t12](
                                self.ANNs[t1].inputs[t2])
                        self.ANNs[t1].variables.extend(
                            [A[t12], p[t12], r0[t12]])

                    self.ANNs[t1]._post_setup()
            EAMpotential._post_setup(self)

def calculate_eam_maps(_Gs, _types):
    batchsize = len(_Gs)
    r_Ni_Ni = []
    r_Ni_Au = []
    r_Au_Ni = []
    r_Au_Au = []
    Ni_Ni_indices = []
    Ni_Au_indices = []
    Au_Ni_indices = []
    Au_Au_indices = []
    N_Ni = 0
    N_Au = 0
    Ni_indices = []
    Au_indices = []
    for i, (G_vec, t_vec) in enumerate(zip(_Gs, _types)):
        for Gi, ti in zip(G_vec, t_vec):
            if ti == 0:
                Ni_indices.append([i, N_Ni])
                for j in range(len(Gi[0])):
                    Ni_Ni_indices.append([N_Ni, len(r_Ni_Ni) + j])
                for j in range(len(Gi[1])):
                    Ni_Au_indices.append([N_Ni, len(r_Ni_Au) + j])
                N_Ni += 1
                r_Ni_Ni.extend(Gi[0])
                r_Ni_Au.extend(Gi[1])
            elif ti == 1:
                Au_indices.append([i, N_Au])
                for j in range(len(Gi[0])):
                    Au_Ni_indices.append([N_Au, len(r_Au_Ni) + j])
                for j in range(len(Gi[1])):
                    Au_Au_indices.append([N_Au, len(r_Au_Au) + j])
                N_Au += 1
                r_Au_Ni.extend(Gi[0])
                r_Au_Au.extend(Gi[1])

    # Cast into numpy arrays, also takes care of wrong dimesionality of empty lists
    Ni_Ni_indices = _np.array(Ni_Ni_indices, dtype = _np.int64).reshape((-1,2))
    Ni_Au_indices = _np.array(Ni_Au_indices, dtype = _np.int64).reshape((-1,2))
    Au_Ni_indices = _np.array(Au_Ni_indices, dtype = _np.int64).reshape((-1,2))
    Au_Au_indices = _np.array(Au_Au_indices, dtype = _np.int64).reshape((-1,2))
    b_map_Ni_Ni = _tf.SparseTensorValue(Ni_Ni_indices, [1.0]*len(r_Ni_Ni), [N_Ni, len(r_Ni_Ni)])
    b_map_Ni_Au = _tf.SparseTensorValue(Ni_Au_indices, [1.0]*len(r_Ni_Au), [N_Ni, len(r_Ni_Au)])
    b_map_Au_Ni = _tf.SparseTensorValue(Au_Ni_indices, [1.0]*len(r_Au_Ni), [N_Au, len(r_Au_Ni)])
    b_map_Au_Au = _tf.SparseTensorValue(Au_Au_indices, [1.0]*len(r_Au_Au), [N_Au, len(r_Au_Au)])
    Ni_indices = _np.array(Ni_indices, dtype = _np.int64).reshape((-1,2))
    Au_indices = _np.array(Au_indices, dtype = _np.int64).reshape((-1,2))
    Ni_map = _tf.SparseTensorValue(Ni_indices, [1.0]*N_Ni, [batchsize, N_Ni])
    Au_map = _tf.SparseTensorValue(Au_indices, [1.0]*N_Au, [batchsize, N_Au])
    r_Ni_Ni = _np.array(r_Ni_Ni).reshape((-1,1))
    r_Ni_Au = _np.array(r_Ni_Au).reshape((-1,1))
    r_Au_Ni = _np.array(r_Au_Ni).reshape((-1,1))
    r_Au_Au = _np.array(r_Au_Au).reshape((-1,1))

    return r_Ni_Ni, r_Ni_Au, r_Au_Ni, r_Au_Au, b_map_Ni_Ni, b_map_Ni_Au, b_map_Au_Ni, b_map_Au_Au, Ni_map, Au_map

def calculate_bp_maps(_Gs, _types):
    batchsize = len(_Gs)
    N_Ni = 0
    N_Au = 0
    Ni_indices = []
    Au_indices = []
    Ni_atoms = []
    Au_atoms = []
    for i, (G_vec, t_vec) in enumerate(zip(_Gs, _types)):
        Ni_atoms.append(_np.array(G_vec)[t_vec == 0])
        Au_atoms.append(_np.array(G_vec)[t_vec == 1])
        for ti in t_vec:
            if ti == 0:
                Ni_indices.append([i, N_Ni])
                N_Ni += 1
            elif ti == 1:
                Au_indices.append([i, N_Au])
                N_Au += 1
    # Cast into numpy arrays, also takes care of wrong dimesionality of empty lists
    Ni_indices = _np.array(Ni_indices, dtype = _np.int64).reshape((-1,2))
    Au_indices = _np.array(Au_indices, dtype = _np.int64).reshape((-1,2))
    Ni_maps = _tf.SparseTensorValue(Ni_indices, [1.0]*N_Ni, [batchsize, N_Ni])
    Au_maps = _tf.SparseTensorValue(Au_indices, [1.0]*N_Au, [batchsize, N_Au])
    return _np.concatenate(Ni_atoms), _np.concatenate(Au_atoms), Ni_maps, Au_maps
