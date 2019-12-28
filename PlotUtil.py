import pandas as pd
import seaborn as sns
from scipy.stats import linregress
import logging
from LogitUtil import logit
from FileUtil import FileUtil
from matplotlib import pyplot as plt

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class PlotUtil:
    def plot_and_stats(self, df:pd.DataFrame, xlabel:str, ylabel:str, color:str='r'):
        sns.regplot(x=xlabel, y=ylabel, data=df, robust=True, color=color, x_jitter=1.0, y_jitter=1.0)

        slope, intercept, r, p, epsilon = linregress(df[xlabel], df[ylabel])
        logger.info('Main equation: y = %.3f x + %.3f' % (slope, intercept))
        logger.info('r^2 = %.4f' % (r * r))
        logger.info('p = %.4f' % (p))
        logger.info('std err: %.4f' % (epsilon))

    def count_plot(self, df:pd.DataFrame, xlabel:str, hue:str=None, return_function_do_not_plot:bool=True):
        param_dict = {'x': xlabel, 'data': df}
        if hue:
            param_dict['hue'] = hue

        if return_function_do_not_plot:
            return sns.countplot(**param_dict)
        sns.countplot(**param_dict)
        plt.show()

    def historgram_plot(self, df:pd.DataFrame, xlabel:str, bins:int=10, return_function_do_not_plot:bool=True):
        if return_function_do_not_plot:
            return df[xlabel].hist(bins=bins)
        df[xlabel].hist(bins=bins)
        plt.show()

    def null_heatmap(self, df:pd.DataFrame, color_scheme:str='Blues'):
        sns.heatmap(df.isnull(), yticklabels=True, cbar=False, cmap=color_scheme)
        plt.show()

    def boxplot(self, df:pd.DataFrame, xlabel:str, ylabel:str):
        sns.boxplot(x=xlabel, y=ylabel, data=df)
        plt.show()

    def figure_plots(self, plots:list, rows:int, cols:int, dims:list=[6,12]):
        """
        Put the given plots list into a single figure. Goes left to right, top to bottom.
        :param plots: list of sns plots to be invoked.
        :param rows: how many rows in the figure
        :param cols: how many columns in the figure
        :param dims: height, width in inches as a list.
        :return:
        """
        plt.figure(figsize=dims)
        subplot_count = len(plots)

        for f in plots:
            for r in range(rows):
                for c in range(cols):
                    which_subplot = subplot_count * 100 + (r+1) * 10 + c+1
                    plt.subplot(which_subplot)
                    logger.debug(f'About to create subplot {which_subplot}')
                    f
        plt.show()


    