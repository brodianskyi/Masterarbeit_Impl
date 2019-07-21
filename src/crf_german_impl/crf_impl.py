from keras import activations
from keras import backend as K
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.layers import Layer
from keras_contrib.utils.test_utils import to_tuple


class CRF(Layer):
    """
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
        self.units = units
        self.learn_mode = learn_mode
        assert self.learn_mode in ["join", "marginal"]
        self.test_mode = test_mode
        if self.test_mode is None:
            self.test_mode = "viterbi" if self.learn_mode == "join" else "marginal"
        else:
            assert self.test_mode in ["viterbi", "marginal"]
        self.sparse_target = sparse_target
        self.use_boundary = use_boundary

        self.activation = activations.get(activation)

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.chain_initializer = initializers.get(chain_initializer)
        self.boundary_initializer = initializers.get(boundary_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.chain_regularizer = regularizers.get(chain_regularizer)
        self.boundary_regularizer = regularizers.get(boundary_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.chain_constraint = constraints.get(chain_constraint)
        self.boundary_constraint = constraints.get(boundary_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.unroll = unroll

    def build(self, input_shape):
        input_shape = to_tuple(input_shape)

    def call(self, X, mask=None):
        # Input mask to CRF must have dim 2 if not None
        if mask is not None:
            assert K.ndim(mask) == 2

    def compute_output_shape(self, input_shape):
        return None

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        return None
