import numpy as np
from .defaults import DTYPE, rng

class Value:
    def __init__(self ,data,_op= '' , _children = ()) -> None:
      if isinstance(data, Value):
          self.data = np.array(data.data, dtype=np.float32)
      elif isinstance(data, np.ndarray):
          self.data = data.astype(np.float32)
      else:
          self.data = np.array(data, dtype=np.float32)

      self.grad = np.zeros_like(self.data, dtype=np.float32)

      # Graph tracking
      # self._backward = lambda: None
      self._prev = set(_children)
      self._op = _op

      self._backward = self.default_backward




    # --- MATRIX OPERATIONS ---
    def __matmul__(self, other):
      other = other if isinstance(other , Value) else Value(other)
      out = Value(self.data @ other.data , _op="@", _children=(self,other))

      def _backward():
        # local error (delta) = (Incoming error vector) * (1-y**2)
        #dL/dx = X.T @ delta
        self.grad += out.grad @ other.data.T
        # dL/dw = delta @ W.T
        other.grad += self.data.T @ out.grad
      out._backward = _backward
      return out


    def __add__(self,other):
      other = other if isinstance(other , Value) else Value(other)
      out = Value(np.add(self.data ,other.data) , _op="+", _children=(self,other))

      def _backward():
        # addition distribution law
        self.grad += out.grad
        # OPTIMIZATION: If 'other' was a row bias vector (1, O)
        # and 'out.grad' is a batch matrix (N, O), sum across the batch rows (axis=0)
        if other.data.shape != out.grad.shape:
            # Sum columns down, keep dimensions matching (1, O)
            other.grad += np.sum(out.grad, axis=0, keepdims=True)
        else:
            other.grad += out.grad
      out._backward = _backward
      return out

    def __pow__(self, other):
        """Element-wise power operation. Handles int or float exponents (e.g., value ** 2)."""
        if not isinstance(other, (int, float)):
            raise ValueError("Only int/float exponents are supported for now.")

        # 1. Compute element-wise power using standard NumPy
        out_data = self.data ** other
        out = Value(out_data, _children=(self,), _op=f'**{other}')

        def _backward():
            # Power rule derivative: d/dx (x^n) = n * x^(n-1)
            # Multiply element-wise by incoming out.grad
            self.grad += out.grad * (other * (self.data ** (other - 1)))

        out._backward = _backward
        return out

    def __sub__(self, other):
        """Element-wise matrix subtraction (self - other)."""
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, _children=(self, other), _op='-')

        def _backward():
            # d/dx (x - y) = 1  -> passes gradient straight through
            self.grad += out.grad
            # d/dy (x - y) = -1 -> passes negative gradient straight through
            # Handle broadcasting if 'other' is a different shape (like a bias)
            if other.data.shape != out.grad.shape:
                other.grad -= np.sum(out.grad, axis=0, keepdims=True)
            else:
                other.grad -= out.grad

        out._backward = _backward
        return out

    def sum(self):
      """Reduces an entire matrix down into a single scalar Value node."""
      total_sum = np.sum(self.data)
      out = Value(total_sum, _children=(self,), _op='sum')

      def _backward():
          # A summation passes the scalar gradient backward to EVERY element in the matrix
          # np.ones_like ensures the gradient broadcast dimensions match self.data perfectly
          self.grad += out.grad * np.ones_like(self.data)

      out._backward = _backward
      return out






    # --- ACTIVATION FUNCTION ---
    def tanh(self):
      t = np.tanh(self.data)
      out = Value(t, _children=(self,), _op='tanh')

      def _backward():
        # derivative of tanh is 1 - tanh **2
        self.grad += out.grad * (1 - t**2)
      out._backward =_backward
      return out



    # --- TOPOLOGICAL SORT BACKPROPAGATION ---
    def backward(self):

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Set the starting grad to 1.0 with the same matrix as its data
        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()

    @staticmethod
    def default_backward():
        return None



    def __repr__(self) -> str:
      return f"Value(data={self.data})"
    


class DenseLayer:
  def __init__(self, nin: int, nout: int) -> None:
        """
        nin:  Number of incoming features (e.g., 4)
        nout: Number of neurons in this layer (e.g., 4)
        """
        # (nin, nout) so that input (N x nin) @ weights (nin x nout) -> (N x nout)
        self.weights = Value(rng.uniform(-1.0, 1.0, size=(nin, nout)).astype(DTYPE))

        # row vector
        self.bias = Value(rng.uniform(-1.0, 1.0, size=(1, nout)).astype(DTYPE))

        # Track forward pass states as full tracking Value nodes
        self.X = None
        self.Y = None


  def forward(self, X:Value) -> Value:
    self.X  = X
    # y = X @ W + b ; (N x nin) @ (nin x nout) + (1 x nout) -> (N x nout)
    linear_output = X @ self.weights + self.bias

    # Activation function : (tanh)
    self.Y = linear_output.tanh()
    return self.Y


  def backward(self ,incoming_error: np.ndarray) -> np.ndarray:
    """
    incoming_error: Gradient passed down from the next layer (N x nout)
    """
    # 1. Local gradient (delta)
    delta = incoming_error * (1 - self.Y.data ** 2)

    # 2. Gradient of weights: (nin x N) @ (N x nout)-> (nin x nout)
    self.weights.grad += self.X.data.T @ delta

    # 3. Gradient of Bias: sum errors down the column across the batch -> (1 x nout)
    self.bias.grad += np.sum(delta, axis=0, keepdims=True)

    # 4. Input Gradient (dX): (N x nout) @ (nout x nin) -> (N x nin)
    # pass down as 'incoming_error'
    dx = delta @ self.weights.data.T

    return dx

  def parameters(self):
    return [self.weights ,self.bias]



  def __repr__(self) -> str:
        return f"DenseLayer(weights={self.weights.data.shape}, bias={self.bias.data.shape})"
  

class MLP:

  def __init__(self, shape: list, nouts: list) -> None:
      """
      shape: e.g., [batch_size, input_features] -> [1, 4]
      nouts: e.g., [4, 4, 1]
      """
      # Pick the input features index directly (index 1 is 4)
      input_features = shape[1]

      n = [input_features] + nouts

      # Create layers sequentially passing raw integer counts down!
      # Layer 0: DenseLayer(4, 4)
      # Layer 1: DenseLayer(4, 4)
      # Layer 2: DenseLayer(4, 1)
      self.layers = [DenseLayer(n[i], n[i+1]) for i in range(len(nouts))]

  def forward(self, X: Value) -> Value:
      """
      Sequentially runs the forward pass matrix tracking through all layers
      """
      out = X
      for layer in self.layers:
          out = layer.forward(out)
      return out

  def parameters(self):
      """
      Collects all underlying weight and bias Value objects for optimization loops
      """
      params = []
      for l in self.layers:
          params.extend(l.parameters())
      return params

  def zero_grad(self):
        """Resets gradients and ensures optimizer tracking states are initialized."""
        for p in self.parameters():
            p.grad = np.zeros_like(p.data, dtype=np.float32)

            # Initialize advanced optimizer memory buffers as zero-matrices
            if not hasattr(p, 'momentum'):
                p.momentum = np.zeros_like(p.data, dtype=np.float32)
            if not hasattr(p, 'velocity'):
                p.velocity = np.zeros_like(p.data, dtype=np.float32)



