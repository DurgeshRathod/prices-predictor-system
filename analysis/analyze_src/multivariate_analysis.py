from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class MultivariateAnalysisTemplate(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame):
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)

    def generate_correlation_heatmap(self, df):
        pass

    def generate_pairplot(self, df):
        pass


class SimpleMultivariateAnalysis(MultivariateAnalysisTemplate):

    def generate_correlation_heatmap(self, df):
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="cool", linewidths=0.5)
        plt.title("Correlation Heatmap")
        plt.show()

    def generate_pairplot(self, df):
        sns.pairplot(df)
        plt.suptitle("Pairplot of Selected Features", y=1.02)
        plt.show()
