import unittest
import logging
from LogitUtil import logit
from PlotUtil import PlotUtil
import pandas as pd

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

"""
Interesting Python features:
"""

class TestPlotUtil(unittest.TestCase):
    # Return a tiny test dataframe
    def my_test_df(self):
        # Example dataframe from https://www.geeksforgeeks.org/python-pandas-dataframe-dtypes/
        df = pd.DataFrame({'Weight': [95, 188, 156, 45, 131],
                           'Name': ['Sam', 'Andrea', 'Alex', 'Robin', 'Kia'],
                           'Age': [14, 25, 25, 8, 21]})

        # Create and set the index
        index_ = [0, 1, 2, 3, 4]
        df.index = index_
        return df

    def setUp(self):
        logger.debug('Starting TestPlotUtil')
        self.df = self.my_test_df()
        self.plt = PlotUtil()

    @logit()
    def test_plot_and_stats(self):
        # self.plt.plot_and_stats(self.df, xlabel='Age', ylabel='Weight')
        pass

    @logit()
    def test_count_plot(self):
        # logger.debug(f'Head of DF: {self.df.head()}')
        self.plt.count_plot(df=self.df, xlabel='Weight', return_function_do_not_plot=False)

    @logit()
    def test_histogram_plot(self):
        self.plt.historgram_plot(df=self.df, xlabel='Age', bins=5, return_function_do_not_plot=False)

    @logit()
    def test_figure_plots(self):
<<<<<<< Updated upstream
        f1 = self.plt.count_plot(df=self.df, xlabel='Age', return_function_do_not_plot=True)
        f2 = self.plt.count_plot(df=self.df, xlabel='Weight', return_function_do_not_plot=True)
        self.plt.figure_plots(plots=[f1, f2], rows=1, cols=2)
=======
        x = np.linspace(0, 10, 100)
        f1 = "plt.plot(x, np.sin(x), '-g', label='sin(x)')"
        f2 = "plt.plot(x, np.cos(x), '-r', label='cos(x)')"
        g1 = "plt.text(0.5, 0.5, str((1, 1)),fontsize=18, ha='center')"
        g2 = "plt.text(0.5, 0.5, str((1, 2)),fontsize=18, ha='center')"
        self.pltu.figure_plots2(plots=[g1, g2], rows=1, cols=2)

        # f1 = self.pltu.count_plot(df=self.df, xlabel='Age', return_function_do_not_plot=True)
        # f2 = self.pltu.count_plot(df=self.df, xlabel='Weight', return_function_do_not_plot=True)
        # self.pltu.figure_plots(plots=[f1, f2], rows=1, cols=2)
>>>>>>> Stashed changes


<<<<<<< Updated upstream
=======
    def test_confusion_matrix_test_vs_predict(self):
        y_test = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 2, 1, 0, 1, 2, 1, 1, 2]
        self.pltu.confusion_matrix_test_vs_predict(y_test, y_pred, ['0', '1', '2'])

    def test_visualize_results(self):
        x = np.linspace(-5, 6, 100)
        y = x * -1
        z = [1 if m > n else 0 for m, n in zip(x,y)]

        pass
>>>>>>> Stashed changes

