from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        pass


class NumericalVsNumericalAnalysisStrategy(BivariateAnalysisStrategy):

    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x=feature1, y=feature2, data=df
        )  # bins is the number of bars to show in histplot
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()


class CategoricalVsNumericalAnalysisStrategy(BivariateAnalysisStrategy):

    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        plt.figure(figsize=(12, 8))
        sns.boxplot(
            x=feature1, y=feature2, data=df
        )  # bins is the number of bars to show in histplot
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45)
        plt.show()


class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str):
        self._strategy.analyze(df, feature1, feature2)
