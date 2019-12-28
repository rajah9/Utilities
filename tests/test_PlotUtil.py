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
        f1 = self.plt.count_plot(df=self.df, xlabel='Age', return_function_do_not_plot=True)
        f2 = self.plt.count_plot(df=self.df, xlabel='Weight', return_function_do_not_plot=True)
        self.plt.figure_plots(plots=[f1, f2], rows=1, cols=2)



