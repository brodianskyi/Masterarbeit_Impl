import keras
from keras import activations
from keras import backend as K
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.layers import Layer
from keras.layers import InputSpec
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
        # input_shape = (None, 80, 50)
        # fix an inconsistency between "keras" and "tensorflow.keras"
        # if tf.keras -> input_shape is tuple with "Dimensions" objects
        # if keras -> input_shape is tuple of ints or "None"
        # keras.__name__ = keras
        print("---Used version of keras = ", keras.__name__)
        # keras -> input_shape do not change
        input_shape = to_tuple(input_shape)
        # input_shape =(None, 80, 50) -> (None, MAX_LEN = 80, Units_lstm = 50)
        # InputSpec - specifies the input_dim of every input to a layer
        # self.input_spec is defined in keras.layers.Layer
        # input_spec = <class "list">[InputSpec(shape=(None, 80, 50), ndim=3]
        # "rank" oder "n_dim" of the input -> e. g. for shape(3,3,4) rank = 3 -> 3 dimensional vector
        self.input_spec = [InputSpec(shape=input_shape)]
        # self.input_dim initialized in __init__
        # self.input_dim = 50
        self.input_dim = input_shape[-1]
        # initialize kernel_weights_matrix(linear transformation of the inputs)
        # shape = (input_dim, units_crf=n_tags+1) = (50, 12)
        self.kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      name="kernel",
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        # initialize chain_weights_matrix, used for CRF chain energy functions
        # shape = (units/units) = (n_tags+1/n_tags+1)
        # shape = (12, 12)
        self.chain_kernel = self.add_weight(shape=(self.units, self.units),
                                            name="chain_kernel",
                                            initializer=self.chain_initializer,
                                            regularizer=self.chain_regularizer,
                                            constraint=self.chain_constraint)
        # self.use_bias = True
        if self.use_bias:
            # shape = (12,)
            self.bias = self.add_weight(shape=(self.units,),
                                        name="bias",
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = 0

        # self.use_boundary = True
        if self.use_boundary:
            # shape = (12,)
            self.left_boundary = self.add_weight(shape=(self.units,),
                                                 name="left_boundary",
                                                 initializer=self.boundary_initializer,
                                                 regularizer=self.boundary_regularizer,
                                                 constraint=self.boundary_constraint)
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
        # X = <Tensor, shape=(?,80,50), dtype=float32>
        # Input mask to CRF must have dim=2 if not None
        # mask = <embedding/NotEqual:0, shape=(?,80), dtype=bool>
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
        # mask =  <embedding/NotEqual:0, shape=(?,80), dtype=bool>
        # self.kernel.shape = (input_dim, units_crf=n_tags+1) = (50, 12)- (linear transform of the input)
        # self.bias.shape = (12,)
        # ! activation=linear=kx+b ! - K.dot - multiplication of two tensor
        # ! input_energy = activation((X*self.kernel) + self.bias) = activation(kx+b)
        input_energy = self.activation(K.dot(X, self.kernel) + self.bias)
        if self.use_boundary:
            input_energy = self.add_boundary_energy(
                input_energy, mask, self.left_boundary, self.right_boundary)
        return input_energy

    def add_boundary_energy(self, energy, mask, start, end):
        # expand_dims(tensor, axis=0)
        # axis: Position where to add a new axis to the tensor
        # start_old_shape=() -> expand_dims -> start_expand_shape=(1,1,12); dtype=float32
        start = K.expand_dims(K.expand_dims(start, 0), 0)
        # end_old_shape=() -> expand_dims -> end_expand_shape=(1,1,12); dtype=float32
        end = K.expand_dims(K.expand_dims(end, 0), 0)
        if mask is None:
            energy = K.concatenate([energy[:, :1, :] + start, energy[:, 1:, :]], axis=1)
            energy = K.concatenate([energy[:, :-1, :], energy[:, -1:, :] + end], axis=1)
        else:
            # we have a mask value
            # cast -  convert a tesnsor-variable value from one type to another
            # cast -> bool to float32
            # expand_dims -> (?,80) -> (?,80,1)
            mask = K.expand_dims(K.cast(mask, K.floatx()))
            # K.greater(x,y) - Element-wise truth value of (x > y)
            # (x > y) -> (mask > shift_right(mask))
            start_mask = K.cast(K.greater(mask, self.shift_right(mask)), K.floatx())
            end_mask = K.cast(K.greater(self.shift_left(mask), mask), K.floatx())
            energy = energy + start_mask * start
            energy = energy + end_mask * end
        return energy

    @staticmethod
    def shift_left(x, offset=1):
        # x = mask
        assert offset > 0
        # Concatenates tensors along one dimension
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
        # ----------------- K.concatenate =
        return K.concatenate([K.zeros_like(x[:, :offset]), x[:, :-offset]], axis=1)
