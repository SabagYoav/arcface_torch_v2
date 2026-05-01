import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_angle_distributions(df, bins=200):
    """
    df: a DataFrame with columns: yaw, pitch, roll
    bins: number of histogram bins
    """

    angles = ['roll', 'pitch', 'yaw']
    titles = ['roll angle', 'pitch angle', 'yaw angle']

    plt.figure(figsize=(15, 4))

    for i, (col, title) in enumerate(zip(angles, titles)):
        plt.subplot(1, 3, i+1)

        # Histogram
        hist, bin_edges = np.histogram(df[col], bins=bins, range=(-100, 100))
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Compute stats
        mean_val = df[col].mean()
        std_val = df[col].std()

        plt.plot(centers, hist)
        plt.title(title)
        plt.xlabel(title)
        plt.ylabel('number of pictures per degree')
        plt.xlim([-100, 100])

        # Add text box with mean and std
        text = f"mean = {mean_val:.2f}°\nstd = {std_val:.2f}°"
        plt.text(
            0.05, 0.95, text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')
        )

    plt.tight_layout()
    plt.savefig("train_set_pose_angle_distributions.png")


if __name__ == "__main__":

    df = pd.read_json("data_analytics/pose/pose_results.json")
    plot_angle_distributions(df)