import matplotlib.pyplot as plt


class ManouevrePlotter:
    def __init__(self):
        pass

    import matplotlib.pyplot as plt
import numpy as np

class ManouevrePlotter:
    def __init__(self):
        pass

    def plot_manouevre(self, original, reconstructed):
        """
        Eredeti és rekonstruált adatok külön plotba helyezése.
        
        Args:
            original (numpy.ndarray): Az eredeti adatok (shape: [seq_len, input_dim]).
            reconstructed (numpy.ndarray): A rekonstruált adatok (shape: [seq_len, input_dim]).
            time_steps (list or numpy.ndarray): Az időbeli minták (shape: [seq_len]). Ha None, akkor [0, 1, ..., seq_len-1] lesz.
        """
        seq_len, input_dim = original.shape
        time_steps = np.arange(seq_len)

        plt.figure(figsize=(12, 6))
        
        # Eredeti adatok plotolása
        plt.subplot(2, 1, 1)
        for i in range(input_dim):
            plt.scatter(time_steps, original[:, i], label=f"Feature {i+1}", alpha=0.7)
        plt.title("Original Data")
        plt.xlabel("Time Steps")
        plt.ylabel("Values")
        plt.legend(loc="upper right", fontsize="small")
        plt.grid(True)

        # Rekonstruált adatok plotolása
        plt.subplot(2, 1, 2)
        for i in range(input_dim):
            plt.scatter(time_steps, reconstructed[:, i], label=f"Feature {i+1}", alpha=0.7)
        plt.title("Reconstructed Data")
        plt.xlabel("Time Steps")
        plt.ylabel("Values")
        plt.legend(loc="upper right", fontsize="small")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        

