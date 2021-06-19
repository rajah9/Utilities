import logging
import unittest

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from LogitUtil import logit
from PlotUtil import PlotUtil

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
        self.pltu = PlotUtil(myplt=plt)

    @logit()
    def test_plot_and_stats(self):
        self.pltu.scatter_plot(self.df, xlabel_col_name='Age', ylabel_col_name='Weight')

    @logit()
    def test_count_plot(self):
        # logger.debug(f'Head of DF: {self.df.head()}')
        self.pltu.count_plot(df=self.df, xlabel='Weight', return_function_do_not_plot=False)

    @logit()
    def test_histogram_plot(self):
        self.pltu.histogram_plot(df=self.df, xlabel='Age', bins=5, return_function_do_not_plot=True)
        self.pltu.plt.show()

    @logit()
    def test_confusion_matrix_plot(self):
        logger.debug('subtest 1: Normal plot.')
        m = [[33,2,0,0,0,0,0,0,0,1,3],
        [3,31,0,0,0,0,0,0,0,0,0],
        [0,4,41,0,0,0,0,0,0,0,1],
        [0,1,0,30,0,6,0,0,0,0,1],
        [0,0,0,0,38,10,0,0,0,0,0],
        [0,0,0,3,1,39,0,0,0,0,4],
        [0,2,2,0,4,1,31,0,0,0,2],
        [0,1,0,0,0,0,0,36,0,2,0],
        [0,0,0,0,0,0,1,5,37,5,1],
        [3,0,0,0,0,0,0,0,0,39,0],
        [0,0,0,0,0,0,0,0,0,0,38]]
        # self.pltu.plt = plt
        self.pltu.confusion_matrix_plot(m, False)
        # Test 2. A 2 x 2 matrix with labels
        m2 = [[80, 20], [25, 75]]
        rows = ['T', 'F']
        cols = ['P', 'N']
        self.pltu.confusion_matrix_plot(m2, col_names=cols, row_names=rows, return_function_do_not_plot=False)
        # Test for a not-square matrix
        logger.debug('subtest 3: not a square matrix.')
        expected_log_message = 'Matrix is not square'
        with self.assertLogs(PlotUtil.__name__, level='DEBUG') as cm:
            x = m[2:]
            self.pltu.confusion_matrix_plot(x, False)
            self.assertTrue(next((True for line in cm.output if expected_log_message in line), False))
        # Test for returning a function
        m3 = [[80, 10, 5], [7, 75, 13], [12, 19, 91]]
        self.pltu.confusion_matrix_plot(m3, return_function_do_not_plot=True)
        self.pltu.plt.show()

    def test_confusion_matrix_test_vs_predict(self):
        y_test = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 2, 1, 0, 1, 2, 1, 1, 2]
        self.pltu.confusion_matrix_test_vs_predict(y_test, y_pred, ['0', '1', '2'])

    def test_visualize_results(self):
        x = np.linspace(-5, 6, 100)
        y = x * -1
        z = [1 if m > n else 0 for m, n in zip(x,y)]

        pass

    def test_text_plot(self):
        self.pltu.text_plot('hello, matplotlib', return_function_do_not_plot=False)

    @logit()
    def test_figure_plots(self):
        self.pltu.init_subplots(2,2)
        # Access the subplot (ax) property
        self.pltu.subplot[0,0] = self.pltu.text_plot('top left', return_function_do_not_plot=True)
        # Using the tested function to do the accessing
        x = self.pltu.text_plot('top right', return_function_do_not_plot=True)
        self.pltu.figure_plot3(x, 0, 1)
        y = self.pltu.text_plot('bot left', return_function_do_not_plot=True)
        self.pltu.figure_plot3(y, 1, 0)
        z = self.pltu.text_plot('bot right', return_function_do_not_plot=True)
        self.pltu.figure_plot3(z, 1, 1)
        self.pltu.plt.show()
        self.fail('overlaying all text boxes in the lower right')
