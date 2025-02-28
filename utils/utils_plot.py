# utils_plot.py

import matplotlib.pyplot as plt

def plot_training_history(history, run_name):
    """
    Plots the training and validation loss and MAE over iterations.

    Parameters:
    - history (dict): Dictionary containing 'train_loss', 'train_mae', 'valid_loss', 'valid_mae'.
    - run_name (str): Name of the training run, used for the plot title and saving the figure.
    """
    iterations = range(1, len(history['train_loss']) + 1)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Loss
    axs[0].plot(iterations, history['train_loss'], label='Training Loss')
    axs[0].plot(iterations, history['valid_loss'], label='Validation Loss')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('MSE Loss')
    axs[0].set_title('Training and Validation Loss')
    axs[0].legend()
    axs[0].grid(True)

    # Plot MAE
    axs[1].plot(iterations, history['train_mae'], label='Training MAE')
    axs[1].plot(iterations, history['valid_mae'], label='Validation MAE')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('MAE')
    axs[1].set_title('Training and Validation MAE')
    axs[1].legend()
    axs[1].grid(True)

    plt.suptitle(f'Training History: {run_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{run_name}_training_history.png")
    plt.show()
