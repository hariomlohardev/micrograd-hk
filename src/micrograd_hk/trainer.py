import numpy as np
import os
import pickle
import copy
from .engen import Value, MLP

class Trainer:
    def __init__(self, mlp :MLP, xs: np.ndarray, ys: np.ndarray):
        """
        mlp: MLP() Multi Layer Perceptron
        xs: Complete input data batch of shape (N x input_features)
        ys: Complete target labels matrix of shape (N x output_features)
        """
        self.mlp = mlp
        # Convert raw inputs and outputs straight into global Value matrix nodes
        self.X_batch = Value(xs)
        self.Y_batch = Value(ys)

        self.lr = 0.1
        self.loss_history = []
        self.t = 0  # Global step counter for Adam / Adam-M

    def train(self,
              lr: float = 0.1,
              iterations: int = 20,
              optimizer: str = "adam",
              iter_stp: int = None,
              show_progress: bool = True,
              save_checkpoint: bool = True,           # NEW: Controls whether to generate file
              save_filename: str = "trained_mlp.pkl", # NEW: Filename/location descriptor
              save_dir: str = None):                  # NEW: Base save directory location
        """
        Executes the training loop over the specified number of iterations using matrix blocks.
        Saves the underlying MLP framework upon training completion if requested.
        """
        self.lr = lr
        last_loss = 0.0
        y_pred = None

        # PERFORMANCE: Only update terminal UI ~100 times total to prevent I/O bottleneck
        update_interval = max(1, iterations // 100)

        # Import tqdm ONLY if requested
        if show_progress:
            try:
                from tqdm import tqdm
                # miniters minimizes background thread waking up, further boosting speed
                iterator = tqdm(range(iterations), desc="Training", unit="iter", miniters=update_interval)
            except ImportError:
                print("Note: 'tqdm' is not installed. Progress bar disabled.")
                show_progress = False
                iterator = range(iterations)
        else:
            iterator = range(iterations)

        for step in iterator:
            # 1. Fully Vectorized Forward Pass
            y_pred = self.mlp.forward(self.X_batch)

            # 2. Vectorized Mean Squared Error (MSE) Loss
            error = y_pred - self.Y_batch
            loss = (error ** 2).sum()

            # Extract loss data once
            last_loss = float(loss.data)
            self.loss_history.append(last_loss)

            # 3. Reset Gradients and Initialize Moment Trackers
            self.mlp.zero_grad()

            # 4. Backward Pass: Backpropagates through the entire matrix graph automatically
            loss.backward()

            # 5. Execute Vectorized Optimization Updates
            self.step_optimizer(optimizer=optimizer, iter_stp=iter_stp)

            # PERFORMANCE: Only format strings and update terminal periodically
            if show_progress and step % update_interval == 0:
                iterator.set_postfix(loss=f"{last_loss:.5f}")

        # Final UI update to guarantee the exact final loss is displayed
        if show_progress and hasattr(iterator, 'set_postfix'):
            iterator.set_postfix(loss=f"{last_loss:.5f}")

        # --- NEW FILE GENERATION LAYER ---
        # --- 100% SUCCESS DISK SAVING LAYER ---
        if save_checkpoint:
            if save_dir is None:
                save_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

            os.makedirs(save_dir, exist_ok=True)
            full_path = os.path.join(save_dir, save_filename)

            print(f"\nTraining completed. Saving network checkpoint parameters to: '{full_path}'")
            try:
                # 1. Extract ONLY the raw mathematical arrays, ignoring nested backward functions
                raw_parameter_state = []
                for p in self.mlp.parameters():
                    # Save a deep copy of just the raw numpy array numbers
                    raw_parameter_state.append(np.copy(p.data))

                # 2. Serialize this clean, lightweight numeric list (Guaranteed to never crash)
                with open(full_path, "wb") as f:
                    pickle.dump(raw_parameter_state, f, protocol=pickle.HIGHEST_PROTOCOL)
                    f.flush()
                    os.fsync(f.fileno())
                print("Parameters extracted and safely synced to hardware disk!")
            except Exception as e:
                print(f"Warning: Failed to auto-save file due to: {e}")

        return last_loss, y_pred


    @staticmethod
    def load_model(filepath: str, architecture_blueprint: object) -> object:
        """
        Reconstructs a saved model by initializing a fresh architecture blueprint
        and injecting the stored raw numerical weight/bias matrices.

        architecture_blueprint: An un-initialized or freshly initialized MLP model
                                matching the original structure (e.g. MLP(shape=[1,784], nouts=[16,16,1]))
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Could not locate saved checkpoint file at: '{filepath}'")

        print(f"Extracting matrix array weights from file: '{filepath}'...")
        with open(filepath, "rb") as f:
            saved_numpy_arrays = pickle.load(f)

        print("Injecting numerical arrays into network parameters...")
        model_parameters = architecture_blueprint.parameters()

        if len(model_parameters) != len(saved_numpy_arrays):
            raise ValueError("Architecture blueprint parameters mismatch with saved file structure!")

        # Overwrite the fresh initialization data with your heavy trained parameters
        for p, saved_array in zip(model_parameters, saved_numpy_arrays):
            p.data = np.copy(saved_array)
            # Reset internal optimizer states cleanly
            p.grad = np.zeros_like(p.data)

        print("Model state successfully populated! Ready for active inference loops.")
        return architecture_blueprint


    def step_optimizer(self, optimizer: str, iter_stp: int = None):
        opt = optimizer.lower().strip()
        if opt == "sgd":
            self.sgd()
        elif opt == "momentum":
            self.momentum()
        elif opt in ("rmsprop", "rsmprop"):
            self.rmsprop()
        elif opt == "adam":
            self.adam()
        elif opt == "adam_m":
            self.adam_m(iter_stp=iter_stp)
        else:
            raise ValueError(f"Unknown optimizer: '{optimizer}'.")

    # --- Vectorized Optimizers ---

    def sgd(self) -> None:
        """Vanilla Stochastic Gradient Descent operating on whole matrices."""
        for p in self.mlp.parameters():
            p.data -= self.lr * p.grad

    def momentum(self, beta1: float = 0.9) -> None:
        """SGD with Vectorized Momentum."""
        for p in self.mlp.parameters():
            p.momentum = beta1 * p.momentum + (1.0 - beta1) * p.grad
            p.data -= self.lr * p.momentum

    def rmsprop(self, beta2: float = 0.99, eps: float = 1e-8) -> None:
        """RMSprop Optimizer operating on whole matrices."""
        for p in self.mlp.parameters():
            p.velocity = beta2 * p.velocity + (1.0 - beta2) * (p.grad ** 2)
            p.data -= (self.lr * p.grad) / (np.sqrt(p.velocity) + eps)

    def adam(self, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        """Standard Adaptive Moment Estimation (Adam)."""
        self.t += 1
        for p in self.mlp.parameters():
            p.momentum = beta1 * p.momentum + (1.0 - beta1) * p.grad
            p.velocity = beta2 * p.velocity + (1.0 - beta2) * (p.grad ** 2)

            m_hat = p.momentum / (1.0 - beta1 ** self.t)
            v_hat = p.velocity / (1.0 - beta2 ** self.t)

            p.data -= (self.lr * m_hat) / (np.sqrt(v_hat) + eps)

    def adam_m(self, iter_stp, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam-M Variant with periodic counter resets."""
        self.t += 1
        if iter_stp is not None and self.t > iter_stp:
            self.t = 1  # Reset counter back to 1

        for p in self.mlp.parameters():
            p.momentum = p.momentum * beta1 + (1.0 - beta1) * p.grad
            p.velocity = (beta2 * p.velocity) + (1.0 - beta2) * (p.grad ** 2)

            # Applying modified bias correction factors
            momentum_corrected = p.momentum / (1.5 - beta1 ** self.t)
            velocity_corrected = p.velocity / (2.5 - beta2 ** self.t)


            # Fixed: eps moved outside of np.sqrt
            p.data -= (self.lr * momentum_corrected) / (np.sqrt(velocity_corrected) + eps)





def test_optimizers_with_plot(mlp: MLP, xs: list, ys: list):
    import matplotlib.pyplot as plt

    """
    Trains your matrix-driven MLP using Adam and Adam-M,
    prints the final predictions, and displays a convergence graph.
    """
    optimizers = ['adam', 'adam_m']
    all_histories = {}

    # 1. Convert raw python lists to correct NumPy matrix blocks
    # xs input shape becomes: (100 samples x 3 features)
    # ys target shape becomes: (100 samples x 1 output feature column)
    xs_matrix = np.array(xs, dtype=np.float32)
    ys_matrix = np.array(ys, dtype=np.float32).reshape(-1, 1)

    plt.figure(figsize=(10, 6))

    for opt in optimizers:
        # 2. Deep copy the baseline network model to keep initialization perfectly identical
        test_model = copy.deepcopy(mlp)

        # 3. Instantiate your matrix-backed trainer
        loss_evaluator = Trainer(test_model, xs_matrix, ys_matrix)

        # 4. Train using matrix calculations
        final_loss, y_pred = loss_evaluator.train(lr=0.01, iterations=3000, optimizer=opt, iter_stp=None,save_checkpoint=True)

        # Store loss records for the graph plotting step
        all_histories[opt] = loss_evaluator.loss_history

        print(f"\n##########  {opt.upper()}  ##########")
        print(f"Final Batch MSE Loss: {final_loss:.6f}")
        print("-" * 45)

        # 5. Extract matrix elements safely using index limits for display
        for i in range(10):  # Displaying the first 10 samples for quick checking
            target_val = ys_matrix[i, 0]
            prediction_val = y_pred.data[i, 0]
            print(f"Sample {i+1:02d} | Target: {target_val:+.1f} | Prediction: {prediction_val:+.3f}")
        print(f"... and {len(ys) - 10} more samples truncated.")

        # Plot the convergence line for this optimizer
        plt.plot(loss_evaluator.loss_history, label=opt.upper(), linewidth=2)

    # 6. Apply professional Matplotlib layout styling to the chart
    plt.title("Optimizer Convergence Comparison: ADAM vs. ADAM-M", fontsize=14, fontweight='bold')
    plt.xlabel("Training Iterations (Epochs)", fontsize=12)
    plt.ylabel("Loss Matrix Scale (MSE Sum)", fontsize=12)
    plt.yscale("log")  # Using log scale highlights the fine tuning near zero loss
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()

    # Render graph seamlessly on screen
    plt.show()

if __name__ == '__main__':
    # Real-World Kaggle UCI Heart Disease - 100 Balanced Samples
    # Features: [Age, BloodPressure, MaxHeartRate] (Normalized 0.0 to 1.0)
    xs = [
        [0.43, 0.39, 0.61], [0.63, 0.67, 0.54], [0.48, 0.76, 0.55], [0.45, 0.64, 0.73], [0.23, 0.24, 0.49],
        [0.88, 0.49, 0.83], [0.56, 0.41, 0.39], [0.40, 0.53, 0.35], [0.49, 0.48, 0.70], [0.52, 0.26, 0.57],
        [0.85, 0.60, 0.57], [0.83, 0.44, 0.75], [0.11, 0.41, 0.61], [0.49, 0.53, 0.24], [0.22, 0.15, 0.49],
        [0.50, 0.70, 0.78], [0.15, 0.47, 0.64], [0.41, 0.52, 0.57], [0.36, 0.19, 0.63], [0.49, 0.40, 0.56],
        [0.77, 0.77, 0.58], [0.66, 0.71, 0.52], [0.34, 0.55, 0.36], [0.55, 0.79, 0.67], [0.22, 0.23, 0.50],
        [0.88, 0.70, 0.70], [0.42, 0.69, 0.65], [0.53, 0.70, 0.56], [0.58, 0.72, 0.49], [0.17, 0.22, 0.63],
        [0.70, 0.75, 0.65], [0.55, 0.56, 0.61], [0.46, 0.40, 0.83], [0.19, 0.26, 0.45], [0.49, 0.63, 0.54],
        [0.38, 0.16, 0.47], [0.65, 0.67, 0.55], [0.70, 0.74, 0.62], [0.29, 0.33, 0.52], [0.46, 0.22, 0.25],
        [0.43, 0.48, 0.22], [0.75, 0.50, 0.48], [0.54, 0.37, 0.25], [0.55, 0.68, 0.66], [0.17, 0.43, 0.60],
        [0.43, 0.50, 0.50], [0.80, 0.50, 0.58], [0.74, 0.75, 0.60], [0.71, 0.70, 0.77], [0.45, 0.59, 0.60],
        [0.22, 0.15, 0.48], [0.19, 0.49, 0.55], [0.44, 0.45, 0.60], [0.16, 0.38, 0.37], [0.21, 0.32, 0.66],
        [0.27, 0.16, 0.26], [0.73, 0.57, 0.46], [0.55, 0.58, 0.22], [0.67, 0.36, 0.47], [0.55, 0.58, 0.31],
        [0.48, 0.62, 0.81], [0.66, 0.38, 0.79], [0.51, 0.51, 0.77], [0.70, 0.51, 0.45], [0.44, 0.20, 0.67],
        [0.55, 0.44, 0.35], [0.42, 0.27, 0.46], [0.79, 0.68, 0.62], [0.26, 0.30, 0.55], [0.47, 0.44, 0.81],
        [0.12, 0.44, 0.38], [0.36, 0.44, 0.34], [0.37, 0.25, 0.46], [0.28, 0.40, 0.51], [0.54, 0.58, 0.51],
        [0.57, 0.55, 0.55], [0.66, 0.54, 0.65], [0.87, 0.38, 0.59], [0.27, 0.51, 0.65], [0.51, 0.28, 0.43],
        [0.59, 0.79, 0.46], [0.88, 0.47, 0.51], [0.70, 0.37, 0.54], [0.55, 0.35, 0.58], [0.42, 0.46, 0.70],
        [0.46, 0.34, 0.58], [0.48, 0.45, 0.55], [0.15, 0.57, 0.22], [0.42, 0.44, 0.27], [0.55, 0.41, 0.77],
        [0.62, 0.38, 0.81], [0.42, 0.40, 0.65], [0.82, 0.72, 0.82], [0.49, 0.59, 0.80], [0.22, 0.19, 0.39],
        [0.62, 0.56, 0.84], [0.59, 0.42, 0.28], [0.56, 0.59, 0.69], [0.42, 0.32, 0.38], [0.27, 0.57, 0.46]
    ]

    ys = [
        1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0,
        1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0,
        1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
        1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
        1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
        -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0
    ]

    # Instantiate the structural MLP matching input dimensions:
    # 3 features input, mapping internal layers [8 neurons, 8 neurons] down to 1 regression output
    m = MLP(shape=[1, 3], nouts=[8, 8, 1])

    # Execute benchmark evaluations and produce the comparative graph
    test_optimizers_with_plot(m, xs, ys)

