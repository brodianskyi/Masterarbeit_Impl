import sys

import keras
import tensorflow as tf
from keras import activations
from keras import backend as K
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.layers import InputSpec
from keras.layers import Layer
from keras_contrib.utils.test_utils import to_tuple


class CRF(Layer):
    """
    !!!!!! KERAS_VERSION = "2.2.3" !!!!!!!
    Learn_Mode either "join" or "marginal"
    "loses.crf_nll" for "join" and
    "losses.categorical_crossentropy" or "losses.sparse_categorical_crossentropy" for "marginal" mode
    for convenience will be used "losses.crf_loss"
    -----------------------------
    Test_Mode either "Viterbi" or "Marginal"
    if "marginal mode" is used for learn_mode then "Marginal" - Test mode
    if "join" is used for learn_mode then "Viterby" - Test mode
    -----------------------------
    Sparse_Target - if "True" - labels are one-hot encoded Else - indices
    -----------------------------
    self.supports_masking = True -> Masking allows to handle variable length inputs,
    like padding with 0 all inputs sequences to the same length
    -----------------------------
    Use_Boundary = True - add start-end chain energies
    CRF function = energie function = E(y|x) = f(y,x)+f(y1,yi+1,x) -> Unary + pairwise potentials
    ------------------------------
    Use Bias -  whether the layer uses a bias vector, bias is additional units in the each layer allows to shift the
    activation function to the left or right
    ------------------------------
    Activation - activation function to use
    activation = linear -> y=kx
    activation is y=sum(weight*input)+bias
    -------------------------------
    --------Initializers-----------
    Kernel_Initializer - initialize for the "kernel" weights matrix
    used for linear transformation of the inputs
    -------------------------------
    Chain_Initializer - initialize "chain_kernel" weights matrix,
    used for CRF chain energy
    ------------------------------
    Bias_Initializer - Initializer for the bias vector
    ------------------------------
    Boundary_initializer - initializer for "left_boundary", "right_boundary"
    -----------------------------
    ---------Regularizers---------
    Kernel_regularizer - regularize for the "kernel" weights matrix
    ------------------------------
    Chain_regularizer - regularize "chain kernel" matrix, used for CRF chain energy
    ------------------------------
    Bias_regularizer - regularize function applied to the bias vector
    ------------------------------
    Boundary_regularizer - regularize function applied to the "left_boundary", "right_boundary"
    ------------------------------
    -----------Constrains----------
    Kernel_constrains - constraint function applied to "kernel"(linear transformation of the inputs) matrix
    ------------------------------
    Chain_constrains -  constraint function applied to "CRF chain energy"
    -----------------------------
    Bias_constrains - constraint function applied to the bias vector
    -----------------------------
    Boundary_constrains - constraint function applied to the "left_boundary", "right_boundary",
    add start-end chain energies
    ----------------------------
    ----------------------------
    Input_dim - input_shape, required when using the layer as the first layer in a model
    ----------------------------
    Unroll -  Unrolling can speed up a RNN, although it tends to be more memory-intensive
    Unrolling is only suitable for short sequences
    If false (by default) - symbolic loop will be used
    """

    def __init__(self, units,
                 learn_mode="join",
                 test_mode=None,
                 sparse_target=False,
                 use_boundary=True,
                 use_bias=True,
                 activation="linear",
                 kernel_initializer="glorot_uniform",
                 chain_initializer="orthogonal",
                 bias_initializer="zeros",
                 boundary_initializer="zeros",
                 kernel_regularizer=None,
                 chain_regularizer=None,
                 bias_regularizer=None,
                 boundary_regularizer=None,
                 kernel_constraint=None,
                 chain_constraint=None,
                 bias_constraint=None,
                 boundary_constraint=None,
                 input_dim=None,
                 unroll=False,
                 **kwargs):
        super(CRF, self).__init__(**kwargs)
        self.inter_ses = tf.InteractiveSession()
        self.oldStdout = sys.stdout
        self.output_file = open("output.txt", "w")
        self.supports_masking = True
        # units = n_otr_tags + 1 = 12
        self.units = units
        # learn_mode = "join"
        self.learn_mode = learn_mode
        assert self.learn_mode in ["join", "marginal"]
        self.test_mode = test_mode

        if self.test_mode is None:
            self.test_mode = "viterbi" if self.learn_mode == "join" else "marginal"
        else:
            assert self.test_mode in ["viterbi", "marginal"]
        # test_mode = "viterbi"; learn_mode = "join"
        self.sparse_target = sparse_target
        self.use_boundary = use_boundary
        self.use_bias = use_bias

        # activation = linear
        self.activation = activations.get(activation)
        # kernel_initializer = "glorot_uniform"
        self.kernel_initializer = initializers.get(kernel_initializer)
        # chain_initializer = "orthogonal"
        self.chain_initializer = initializers.get(chain_initializer)
        # boundary_initializer = "zeros"
        self.boundary_initializer = initializers.get(boundary_initializer)
        # bias_initializer = "zeros"
        self.bias_initializer = initializers.get(bias_initializer)
        # "None" for all
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.chain_regularizer = regularizers.get(chain_regularizer)
        self.boundary_regularizer = regularizers.get(boundary_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        # "None" for all
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.chain_constraint = constraints.get(chain_constraint)
        self.boundary_constraint = constraints.get(boundary_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        # unroll = "False"
        self.unroll = unroll

    def build(self, input_shape):
        """
        Create the layer weights
        """
        # input_shape =<class tuple> (None, 80, 50)
        # fix an inconsistency between "keras" and "tensorflow.keras"
        # if tf.keras -> input_shape is tuple with "Dimensions" objects
        # if keras -> input_shape is tuple of ints or "None"
        # keras.__name__ = keras
        print("---Used version of keras = ", keras.__name__)
        # keras -> input_shape do not change
        # input_shape = dt_src.input_shape
        input_shape = to_tuple(input_shape)
        # input_shape = output from Enbedding layer (batch_size, max_seq_len, embedding_dim)
        # input_shape = <class tuple>(None, 80, 50) -> (None, MAX_LEN = 80, Units_lstm = 50)
        # InputSpec - specifies the input_dim of every input to a layer
        # self.input_spec is defined in keras.layers.Layer
        # input_spec = <class "list">[InputSpec(shape=(None, 80, 50), ndim=3]
        # "rank" oder "n_dim" of the input -> e. g. for shape(3,3,4) rank = 3 -> 3 dimensional vector
        self.input_spec = [InputSpec(shape=input_shape)]
        # self.input_dim initialized in __init__
        # self.input_dim = 50
        # input_dim = embedding_dim
        self.input_dim = input_shape[-1]
        # initialize kernel_weights_matrix(linear transformation of the inputs)
        # shape = (input_dim, units_crf=n_tags+1) = (50, 12) dtype=float32_ref
        self.kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      name="kernel",
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        # initialize chain_weights_matrix, used for CRF chain energy functions
        # shape = (units/units) = (n_tags+1/n_tags+1)
        # shape = (12, 12) - (units(n_otr_tags)/units) - dtype=float32_ref
        self.chain_kernel = self.add_weight(shape=(self.units, self.units),
                                            name="chain_kernel",
                                            initializer=self.chain_initializer,
                                            regularizer=self.chain_regularizer,
                                            constraint=self.chain_constraint)
        # self.use_bias = True
        if self.use_bias:
            # shape = (12,) - dtype=float32_ref
            self.bias = self.add_weight(shape=(self.units,),
                                        name="bias",
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = 0

        # self.use_boundary = True
        if self.use_boundary:
            # shape = (12,) - dtype=float32_ref
            self.left_boundary = self.add_weight(shape=(self.units,),
                                                 name="left_boundary",
                                                 initializer=self.boundary_initializer,
                                                 regularizer=self.boundary_regularizer,
                                                 constraint=self.boundary_constraint)
            # shape = (12,) - dtype=float32_ref
            self.right_boundary = self.add_weight(shape=(self.units,),
                                                  name="right_boundary",
                                                  initializer=self.boundary_initializer,
                                                  regularizer=self.boundary_regularizer,
                                                  constraint=self.boundary_constraint)
        # This method must set self.built = True at the end
        self.built = True
        # then go to the method call

    def call(self, X, mask=None):
        """
        Layer logic implementation (learning_mode/test_mode)
        """
        # input_shape(from def build) = <class tuple>(None, 80, 50)
        # X = <Tensor, shape=(?,80,50), dtype=float32>
        # Input mask to CRF must have dim=2 if not None
        # mask<embedding> = shape=(?,80), dtype=bool>
        if mask is not None:
            assert K.ndim(mask) == 2

        # test_mode = "viterbi"; learn_mode = "join"
        if self.test_mode == "viterbi":
            test_output = self.viterbi_decoding(X, mask)
        else:
            test_output = self.get_marginal_prob(X, mask)

        self.uses_learning_phase = True
        if self.learn_mode == "join":
            # K.zeros_like - initilize a tensor of given shape with zeros
            # K.dot multiplies two tesor
            # train_output = <Tensor/zeros_like:0, shape=(?,80,12),dtype=float32>
            train_output = K.zeros_like(K.dot(X, self.kernel))
            out = K.in_train_phase(train_output, test_output)
        else:
            if self.test_mode == "viterbi":
                train_output = self.get_marginal_prob(X, mask)
                out = K.in_train_phase(train_output, test_output)
            else:
                out = test_output
        return out

    def compute_output_shape(self, input_shape):
        # modify the shape of input
        return input_shape[:2] + (self.units,)

    def viterbi_decoding(self, X, mask=None):
        print("------Viterby_decoding------")
        # X = <Tensor, shape=(?,80,50), dtype=float32>
        # mask = shape=(?,80), dtype=bool>
        # self.kernel= (input_dim, units_crf=n_tags+1) = (50, 12)- dtype=float32_ref (linear transform of the input)
        # self.bias.shape = (12,); dtype=float32_ref
        # ! activation=linear=kx+b ! - K.dot - multiplication of two tensor
        # ! input_energy = activation((X*self.kernel) + self.bias) = activation(kx+b)
        # input_energy shape=(?,80,12); dtype=float32
        # activation(K.dot( X=(?,80,50)_dtype=float32 * kernel(50, 12)- dtype=float32) + bias(12,); dtype=float32)=(?,80,12)_dtype=float32
        sys.stdout = self.output_file
        self.format_print("X", X)
        self.format_print("self.kernel", self.kernel)
        self.format_print("K.dot(X, kernel)", K.dot(X, self.kernel))
        self.format_print("self.bias", self.bias)
        self.format_print("(K.dot(X, self.kernel) + self.bias)", K.dot(X, self.kernel) + self.bias)
        input_energy = self.activation(K.dot(X, self.kernel) + self.bias)
        self.format_print("input_energy=activation(K.dot(X, self.kernel) + self.bias)", input_energy)
        if self.use_boundary:
            # input_energy - (?, 80, 12) float32
            input_energy = self.add_boundary_energy(
                input_energy, mask, self.left_boundary, self.right_boundary)
        self.format_print("input_energy add boundary energy", input_energy)
        argmin_tables = self.recursion(input_energy, mask, return_logZ=False)
        return input_energy

    def add_boundary_energy(self, energy, mask, start, end):
        # energy = activation((X*self.kernel) + self.bias) ->(?, 80, 12)
        # mask =  <shape=(?,80), dtype=bool>
        # start =  self.left_boundary -> (12,) float32
        # end = self.right_boundary -> (12,) float32
        # expand_dims(tensor, axis=0)
        # axis: Position where to add a new axis to the tensor
        # start_old_shape=(12,) -> expand_dims -> start_expand_shape=(1,1,12); dtype=float32
        self.format_print("start before expand", start)
        start = K.expand_dims(K.expand_dims(start, 0), 0)
        self.format_print("start after expand = left_boundary; start", start)
        # end_old_shape=(12,) -> expand_dims -> end_expand_shape=(1,1,12); dtype=float32
        self.format_print("end_before expand", end)
        end = K.expand_dims(K.expand_dims(end, 0), 0)
        self.format_print("end after expand=right_boundary; end", end)

        if mask is None:
            energy = K.concatenate([energy[:, :1, :] + start, energy[:, 1:, :]], axis=1)
            energy = K.concatenate([energy[:, :-1, :], energy[:, -1:, :] + end], axis=1)
        else:
            # we have a mask value
            # cast - convert a tesnsor-variable value from one type to another
            # cast -> bool to float32
            # expand_dims from (?,80) to -> mask<embedding> = (?,80,1)
            self.format_print("mask", mask)
            mask = K.expand_dims(K.cast(mask, K.floatx()))
            self.format_print("mask after expand", mask)
            # K.greater(x,y) - Element-wise compare value of (x,y)
            # (x > y) -> (IF mask[i] > shift_right(mask)[i] -> True)
            # K.greater returns -> ["True", "False",...] bool tensor with the same shape as mask=shift_right_mask
            # start_mask(?,80,1) return a tensor from bool to float -> ["True", "False",..] -> [1. 0.]
            start_mask = K.cast(K.greater(mask, self.shift_right(mask)), K.floatx())
            self.format_print("start_mask", start_mask)
            # do the same for the left shift
            # end_mask(?,80,1) return a tensor form bool to float -> ["True", "False"] -> [1. 0.]
            end_mask = K.cast(K.greater(self.shift_left(mask), mask), K.floatx())
            self.format_print("end_mask", end_mask)
            # energy = input_energy = activation; start = left_boundary; end = right_boundary
            energy = energy + start_mask * start
            self.format_print("energy+start", energy)
            energy = energy + end_mask * end
            self.format_print("energy+end", energy)
        return energy

    def recursion(self, input_energy, mask=None, go_backwards=False,
                  return_sequences=True, return_logZ=True, input_length=None):
        """
        Forward (alpha) or backward(beta) recursion
        If "return_logZ=True", compute the logZ, the normalization constant
        If "return_logZ=False", compute the Viterbi best path lookup table
        """
        # !------------------------------------------------------------!
        # prev_target_val` has shape = (B, F)
        # where B = batch_size, F = output feature dim
        # !-----------------------------------------------------------!
        # input_energy - (?, 80, 12) float32
        # chain_energy-shape = (12, 12) - (n_otr_tags+1)/(n_otr_tags+1) - dtype = float32_ref
        # number of features = n_otr_tags+1=n_units; shape = (F,F)
        chain_energy = self.chain_kernel
        self.format_print("chain_energy", chain_energy)
        # chain_energy_(1,12,12) -> expand to shape -> (1, F, F): F=num of output features. 1st F is for t-1, 2nd F for t
        chain_energy = K.expand_dims(chain_energy, 0)
        self.format_print("chain energy expand", chain_energy)
        # shape=(B, F), dtype = float32
        # take [0] element in each string -> from (?, 80, 12) to (?, 12)
        # [ [[1,2,3],[3,4,5]],[ [5,6,7],[7,8,9]] ] -> (2,2,3) -> [[1,2,3][5,6,7]] -> prev_target_val shape = (2,3)
        # previous_target_value to zero -> if we in position [1] init  [0] to zero
        prev_target_val = K.zeros_like(input_energy[:, 0, :])
        self.format_print("prev_target_val", prev_target_val)
        # go_backward = False
        if go_backwards:
            # [ [[1,2,3],[3,4,5]],[ [5,6,7],[7,8,9]] ] -> k.reverse, axis=1 -> [ [[3,4,5],[1,2,3]],[ [7,8,9],[5,6,7]] ]
            # reverse order in axis=1
            input_energy = K.reverse(input_energy, 1)
            if mask is not None:
                mask = K.reverse(mask, 1)
        # K.zeros_like(prev_target_val[:, :1]) -> prev_target_val shape = (2,3) -> take first element in each string
        # K.zeros_like(prev_target_val[:, :1]) -> from (2,3) to  K.zeros_like_shape - (2,1)
        # initial_state - array[ prev_target_val shape = (2,3), K.zeros_like_shape - (2,1)]
        # initial_state  array [shape(2,3), (2,1)]
        initial_states = [prev_target_val, K.zeros_like(prev_target_val[:, :1])]
        self.format_print("initial_states", initial_states)
        # chain_energy->(1, F, F)->shape(1,12,12)
        constants = [chain_energy]
        self.format_print("constants=[chain_energy]", constants)
        if mask is not None:
            self.format_print("mask", mask)
            self.format_print("K.zeros_like(mask[:, :1])", K.zeros_like(mask[:, :1]))
            mask2 = K.cast(K.concatenate([mask, K.zeros_like(mask[:, :1])], axis=1),
                           K.floatx())
            self.format_print("mask2", mask2)
            constants.append(mask2)
            self.format_print("constants", constants)

        def _step(input_energy_i, states):
            return self.step(input_energy_i, states, return_logZ)

        self.format_print("input_energy", input_energy)
        target_val_last, target_val_seq, _ = K.rnn(_step, input_energy,
                                                   initial_states,
                                                   constants=constants,
                                                   input_length=input_length,
                                                   unroll=self.unroll)
        return "recursion"

    def step(self, input_energy_t, states, return_logZ=True):
        # if return_logZ = False -> compute the Viterbi best path
        print("return_logZ in step()", return_logZ)
        self.format_print("input_energy_t", input_energy_t)
        self.format_print("states", states)
        self.format_print("states[:3]", states[:3])
        prev_target_val, i, chain_energy = states[:3]
        self.format_print("prev_target_val in step()", prev_target_val)
        self.format_print("i in step()", i)
        self.format_print("chain_energy in step()", chain_energy)
        t = K.cast(i[0, 0], dtype="int32")
        self.format_print("K.cast(i[0, 0], dtype=int32) = t", t)
        # states[:3].length = 3
        if len(states) > 3:
            if K.backend() == "theano":
                m = states[3][:, t:(t + 2)]
            else:
                # extract slice from a tensor -> (tensor, begin_position, size)
                m = K.slice(states[3], [0, t], [-1, 2])
            input_energy_t = input_energy_t * K.expand_dims(m[:, 0])
            # (1, F, F)*(B, 1, 1) -> (B, F, F)
            chain_energy = chain_energy * K.expand_dims(K.expand_dims(m[:, 0] * m[:, 1]))
        # return_logZ = False -> compute the Viterbi best path
        if return_logZ:
            # shapes: (1, B, F) + (B, F, 1) -> (B, F, F)
            energy = 0

        sys.stdout = self.oldStdout
        self.output_file.close()

        return 0

    @staticmethod
    def shift_left(x, offset=1):
        # x = mask
        assert offset > 0
        # Concatenates tensors along one dimension
        # [ [[1],[2],[3]], [[4],[5],[6]] ] -> K.concatenate -> [ [[2],[3],[0]], [[5],[6],[0]] ]
        # (after shifting still the same shape)
        return K.concatenate([x[:, offset:], K.zeros_like(x[:, :offset])], axis=1)

    @staticmethod
    def shift_right(x, offset=1):
        # x = mask = (?,80,1) = dtype = float32
        assert offset > 0
        # --------x[:, :1]------------
        # offset=1, take in each string all elements until [1] -> element[0] in each string
        # [ [[0][1][2]] [[3][4][5]] ] -> offset = 1 -> [ [[0]] [[3]] ]
        # -----------------------------------
        # -----------K.zeros_like(x[:, :offset])- initialize this tensor with zeros
        # [ [[0]] [[3]] ] -> [ [[0]] [[0]] ] - now shape is (2,2,1)
        # -----------------------------------
        # -------------x[:, :-1]----------
        # negative index -> [ [[0][1][2]] [[3][4][5]] ] ->  x[1,-1] = [5] - last element from the end[-1] in string[1]
        # x[:, :-1] = [ [[2]] [[5]] ]
        # ----------------- K.concatenate------------------
        # shape(2, 1, 4) + shape(2, 2, 4) -> exis = 1 sum over second dimension -> (2, 1+2, 4)
        # [ [[1],[2],[3]], [[4],[5],[6]] ] -> K.concatenate -> [ [[0],[1],[2]], [[0],[4],[5]] ]
        # (after shifting still the same shape)
        return K.concatenate([K.zeros_like(x[:, :offset]), x[:, :-offset]], axis=1)

    def format_print(self, variable_name, input_data):
        if isinstance(input_data, list):
            return print("-" * 75, "\n" + variable_name + ".length = ", len(input_data), "\n" + variable_name + " = " +
                         "\n", self.inter_ses.run(input_data), "\n", "-" * 75)
        else:
            # type = tensor
            if input_data.shape is not None:
                return print("-" * 75, "\n" + variable_name + ".shape = ", input_data.shape, "\n" + variable_name +
                             " = " + "\n", self.inter_ses.run(input_data), "\n", "-" * 75)
            else:
                print("Unknown type can not be printed!")
