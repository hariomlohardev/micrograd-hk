import matplotlib.pyplot as plt
import copy
import numpy as np
from micrograd_hk import Value , Trainer ,MLP

class TestTrainer():
    def test_optimizers_with_plot(mlp: MLP, xs: list, ys: list):
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
    TestTrainer.test_optimizers_with_plot(m, xs, ys)

