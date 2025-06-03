import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


class RedundancyMetricsAnalyzer:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.output_dir = "Results/radundancy_comparison"
        os.makedirs(self.output_dir, exist_ok=True)


    def plot_boxplot(self, metric):
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=self.df, x="threshold", y=metric)
        plt.title(f"Boxplot of {metric} by Threshold")
        filepath = os.path.join(self.output_dir, f"boxplot_{metric}.png")
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()

    def plot_scatter(self, x_metric, y_metric):
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=self.df, x=x_metric, y=y_metric, hue="threshold")
        plt.title(f"{y_metric} vs {x_metric}")
        filepath = os.path.join(self.output_dir, f"scatter_{y_metric}_vs_{x_metric}.png")
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()

    def plot_metric_trend(self, metric):
        trend = self.df.groupby("threshold")[metric].mean().reset_index()
        plt.figure(figsize=(6, 4))
        sns.lineplot(data=trend, x="threshold", y=metric, marker="o")
        plt.title(f"Average {metric} vs Threshold")
        filepath = os.path.join(self.output_dir, f"trend_{metric}.png")
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()

    def get_stats_summary(self):
        return self.df.describe(include="all")


analyzer = RedundancyMetricsAnalyzer("Results/redundancy_comparison_results.csv")

for metric in [
    "jaccard_index",
    "precision",
    "recall",
    "f1_score",
    "overlap_coefficient",
]:
    analyzer.plot_boxplot(metric)
    analyzer.plot_metric_trend(metric)
    for y_metric in [
        "jaccard_index",
        "precision",
        "recall",
        "f1_score",
        "overlap_coefficient",
    ]:
        if metric == y_metric:
            continue
        analyzer.plot_scatter(metric, y_metric)

summary = analyzer.get_stats_summary()
print(summary)
