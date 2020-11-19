import logging
import string
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

"""
Helpful references:
  Python Data Science Handbook by Jake VanderPlas: https://www.oreilly.com/library/view/python-data-science/9781491912126/ch04.html
  * "The plt.show() command does a lot under the hood, as it must interact with your system’s interactive graphical backend."
  * "any plt plot command will cause a figure window to open, and further commands can be run to update the plot."
  * "Plotting interactively within an IPython notebook can be done with the %matplotlib command"
    * %matplotlib notebook will lead to interactive plots embedded within the notebook
    * %matplotlib inline will lead to static images of your plot embedded in the notebook
  * "You can save a figure using the savefig()" e.g., fig.savefig('my_figure.png')
  * Matplotlib has dual interfaces: a convenient MATLAB-style state-based interface, and a more powerful object-oriented interface.
  In[10]: # First create a grid of plots
        # ax will be an array of two Axes objects
        fig, ax = plt.subplots(2)
        x = np.linspace(0, 10, 100)
        # Call plot() method on the appropriate object
        ax[0].plot(x, np.sin(x))
        ax[1].plot(x, np.cos(x));
        
  * In Matplotlib, the figure (an instance of the class plt.Figure) can be thought of as a single container that contains all the objects representing axes, graphics, text, and labels.
  * The axes (an instance of the class plt.Axes) is: a bounding box with ticks and labels, which will eventually contain the plot elements that make up our visualization.
  * Throughout this book, we’ll commonly use the variable name fig to refer to a figure instance, and ax to refer to an axes instance or group of axes instances.

    In[3]: fig = plt.figure()
           ax = plt.axes()
    
           x = np.linspace(0, 10, 1000)
           ax.plot(x, np.sin(x));
        
  * Alternatively, we can use the pylab interface and let the figure and axes be created for us in the background

    In[4]: plt.plot(x, np.sin(x));
  * If we want to create a single figure with multiple lines, we can simply call the plot function multiple times:

    In[5]: plt.plot(x, np.sin(x))
           plt.plot(x, np.cos(x));  
           
    
           
Interesting Python features:
* In confusion_matrix_plot, uses a all function to help determine if a matrix is square.
* Uses typing create Strings (= a list of strings).
* Has an integer division ceiling. -(-102 // 10) # 11
** See https://stackoverflow.com/questions/33299093/how-to-perform-ceiling-division-in-integer-arithmetic

Rough outline
0. Import libraries
  a. from PlotUtil import PlotUtil
  b. plu = PlotUtil()
1. Import dataset (see PandasUtil)
2. Visualize data
  a. plu.count_plot(df=df, xlabel="spam", return_function_do_not_plot=False)
  b. (See PandasUtil)
  c. (See PandasUtil)
  d. plu.correlation_plot(df=df)
  e. Pairs plot
    1. target_col = 'Kyphosis'
    2. cols_to_pair_plot = pu.get_df_headers(df)
    3. cols_to_pair_plot.remove(target_col)
    4. plu.pair_plot(df=df, target_col_name=target_col, col_names=cols_to_pair_plot)
"""

Strings = List[str]

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class PlotUtil:
    def __init__(self, myplt: plt=plt):
        self.plt = myplt
        self.fig = None
        self._ax_array = None
        self._subplot_rows = 0
        self._subplot_cols = 0

    # # Getters and setters for plt
    # @property
    # def plt(self):
    #     return self._p
    #
    # # Setter for plt
    # @plt.setter
    # def plt(self, p: plt) -> plt:
    #     self._p = p

    @property
    def subplot(self):
        return self._ax_array

    def init_subplots(self, num_rows:int = 1, num_cols:int = 1):
        """
        Initialize the subplots, providing num_rows (or num_rows x num_cols) subplots.
        :param num_rows: how many rows
        :param num_cols: how many cols; if None, create a one-dim array
        :return:
        """
        self.fig, self._ax_array = self.plt.subplots(nrows=num_rows, ncols=num_cols)
        self._subplot_rows = num_rows
        self._subplot_cols = num_cols
        return

    def figure_plot3(self, sub_plt, which_row:int = 1, which_col:int = 1):
        """
        Place the given subplot correctly
        :param sub_plt:
        :param which_row:
        :param which_col:
        :return:
        """
        if (which_col < 0) or (which_col >= self._subplot_cols):
            logger.warning(f'Column of {which_col} out of range; must be between 0 and {self._subplot_cols}')
            return

        if (which_row < 0) or (which_row >= self._subplot_rows):
            logger.warning(f'Row of {which_row} out of range; must be between 0 and {self._subplot_rows}')
            return

        self._ax_array[which_row, which_col] = sub_plt

    def scatter_plot(self, df: pd.DataFrame, xlabel_col_name: str, ylabel_col_name: str, zlabel_col_name:str=None, color: str = 'r'):
        """
        Choose either sns.regplot (if this is is a simple 2 d) or sns.scatterplot (if there is a z, used as hue).
        :param df:
        :param xlabel_col_name:
        :param ylabel_col_name:
        :param zlabel_col_name:
        :param color:
        :return:
        """
        logger.debug(f'column named {xlabel_col_name} : \n{df[xlabel_col_name].tail()}')
        logger.debug(f'column named {ylabel_col_name} : \n{df[ylabel_col_name]}.tail()')
        if zlabel_col_name:
            z = df[zlabel_col_name]
            sns.scatterplot(x=df[xlabel_col_name], y=df[ylabel_col_name], hue=z)
        else:
            sns.regplot(x=xlabel_col_name, y=ylabel_col_name, data=df, color=color, x_jitter=1.0, y_jitter=1.0)
        self.plt.show()

    def violin_plot(self, df: pd.DataFrame, xlabel_col_name: str, ylabel_col_name: str, return_function_do_not_plot: bool = True):
        """
        Choose either sns.regplot (if this is is a simple 2 d) or sns.scatterplot (if there is a z, used as hue).
        :param df:
        :param xlabel_col_name:
        :param ylabel_col_name:
        :param zlabel_col_name:
        :param color:
        :return:
        """
        if return_function_do_not_plot:
            return sns.violinplot(data=df, x=xlabel_col_name, y=ylabel_col_name)
        sns.violinplot(data=df, x=xlabel_col_name, y=ylabel_col_name)
        self.plt.show()

    @staticmethod
    def pair_plot(df: pd.DataFrame, target_col_name: str, col_names: List = None):
        """
        Gives you an overview of how the columns stack up with each other.
        Seaborn pairplot will "Plot pairwise relationships in a dataset."
        Following example will establish the target column and plot the remaining variables against it.
          target_col = 'Kyphosis'
          cols_to_pair_plot = pu.get_df_headers(df)
          cols_to_pair_plot.remove(target_col)
          plu.pair_plot(df=df, target_col_name=target_col, col_names=cols_to_pair_plot)
        :param df:
        :param target_col_name: binary column you're interested in
        :param col_names: other interesting columns, excluding target_col_name
        :return:
        """
        param_dict = {'data': df, 'hue': target_col_name}
        if col_names:
            param_dict['vars'] = col_names
        sns.pairplot(**param_dict)
        plt.show()

    def count_plot(self, df: pd.DataFrame, xlabel: str, hue: str = None, return_function_do_not_plot: bool = True):
        """
        Provide a bar chart of the counts.
        :param df:
        :param xlabel:
        :param hue:
        :param return_function_do_not_plot:
        :return:
        """
        param_dict = {'x': xlabel, 'data': df}
        if hue:
            param_dict['hue'] = hue

        if return_function_do_not_plot:
            return sns.countplot(**param_dict)
        sns.countplot(**param_dict)
        self.plt.show()

    def histogram_plot(self, df: pd.DataFrame, xlabel: str, bins: int = 10, color: str = 'b', range=None, return_function_do_not_plot: bool = True):
        def plot_histogram():
            param_dict = {'bins': bins, 'color': color}
            if range:
                param_dict['range'] = range
            return df[xlabel].hist(**param_dict)

        if return_function_do_not_plot:
            return plot_histogram()
        plot_histogram()
        self.plt.show()

    def historgram_plot(self, df: pd.DataFrame, xlabel: str, bins: int = 10, return_function_do_not_plot: bool = True):
        logger.warning('deprecated.')
        self.histogram_plot(df, xlabel, bins, return_function_do_not_plot)

    def null_heatmap(self, df: pd.DataFrame, color_scheme: str = 'Blues'):
        sns.heatmap(df.isnull(), yticklabels=True, cbar=False, cmap=color_scheme)
        self.plt.show()

    def heatmap(self, cm:confusion_matrix, is_annot:bool=True, format:str='d'):
        """
        Print out a heat map.
        Calling example (using the input of a confusion matrix as the input to this routine)
          y_predict_train = DataScienceUtil.model_predict(classifier=decision_tree_classifier, X_test=X_test)
          cm = DataScienceUtil.confusion_matrix(y_test, y_predict_train)
          plu.heatmap(cm)

        :param cm:
        :param is_annot:
        :param format:
        :return:
        """
        self.plt.figure(figsize=(10, 5))
        sns.heatmap(cm, annot=is_annot, fmt=format)
        self.plt.show()

    def correlation_plot(self, df:pd.DataFrame):
        """
        Gives you a big-picture plot of how the features correlate with each other.
        :param df:
        :return:
        """
        self.heatmap(df.corr(), is_annot=True, format='.2g')

    def boxplot(self, df: pd.DataFrame, xlabel: str, ylabel: str):
        sns.boxplot(x=xlabel, y=ylabel, data=df)
        self.plt.show()

    def figure_plots2(self, plots: list, rows: int = 1, cols: int = 1, dims: np.array = None):
        pass #TODO
        t = lambda x: exec(x)
        for i, f in enumerate(plots, start=0):
            plt.subplot(rows, cols, i+1)
            #f
            # t("plt.text(0.5, 0.5, str((1, 2)),fontsize=18, ha='center')")
            t(f)

        plt.show()

    def figure_plots(self, plots: list, rows: int = 1, cols: int = 1, dims: np.array = None):
        """
        Put the given plots list into a single figure. Goes left to right, top to bottom.
        :param plots: list of sns plots to be invoked.
        :param rows: how many rows in the figure
        :param cols: how many columns in the figure
        :param dims: height, width in inches as a list.
        :return:
        """
        size = dims or np.array([10,10])
        plt.figure(figsize=size)

        # for i in range(1, 7):
        #     plt.subplot(2, 3, i)
        #     plt.text(0.5, 0.5, str((2, 3, i)),
        #              fontsize=18, ha='center')
        for i, f in enumerate(plots, start=1):
            plt.subplot(rows, cols, i)
            #f
            plt.text(0.5, 0.5, str((rows, cols, i)),fontsize=18, ha='center')

        # fig, ax = plt.subplots(rows, cols)
        #
        # plot_index = 0
        # for row_idx in range(rows):
        #     for col_idx in range(cols):
        #         ax[row_idx, col_idx] = plots[plot_index]
        #         plot_index += 1

        plt.show()


    def confusion_matrix_test_vs_predict(self, y_test:list, y_predict:list, names : List = None):
        cm = confusion_matrix(y_test, y_predict)
        row_names = names or ['Act Pos', 'Act Neg']
        col_names = names or ['Pred Pos', 'Pred Neg']
        self.confusion_matrix_plot(square_matrix=cm, row_names=row_names, col_names=col_names, sns_fmt='d', return_function_do_not_plot=False)


    def confusion_matrix_plot(self, square_matrix: np.array, row_names: Strings = [], col_names: Strings = [], sns_fmt='',
                              return_function_do_not_plot: bool = True):
        def is_square(test_matrix: np.array):
            return all([len(row) == len(test_matrix) for row in test_matrix])

        def confusion_matrix():
            size = len(square_matrix)
            my_col_names = col_names or [i for i in string.ascii_uppercase][:size]
            my_rows_names = row_names or [i for i in string.printable][:size]
            df_cm = pd.DataFrame(square_matrix, index=my_col_names, columns=my_rows_names)
            self.plt.figure()
            sns.heatmap(df_cm, annot=True, fmt=sns_fmt)

        if not is_square(square_matrix):
            logger.warning('Matrix is not square. Returning')
            return

        if return_function_do_not_plot:
            return confusion_matrix()
        else:
            confusion_matrix()
            self.plt.show()

    @staticmethod
    def visualize_results(X_set, y_set, classifier, title:str=None, x_label:str=None, y_label:str=None):
        """
        Visualize train or test results. Restriction: only works for binary 0/1 data in two regions.
        typical call: PlotUtil.visualize_results(X_set=X_train, y_set=y_train, classifier=classifier,
          title="Facebook ad", x_label="Time", y_label="Salary")
        :param X_set:
        :param y_set:
        :param classifier:
        :param title:
        :param x_label:
        :param y_label:
        :return:
        """
        # Create a grid that covers the entire area. This is a rectangle divided into blue and magenta regions.
        # If this classifier is linear, the regions are triangles. If the classifier is K nearest neighbors, then it will be irregular.
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                             np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                     alpha = 0.75, cmap = ListedColormap(('magenta', 'blue')))
        # Set plot limits
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        # Scatter plot each data point
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('magenta', 'blue'))(i), label = j)
        if title:
            plt.title(title)
        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)
        plt.legend()
        plt.show()

    @staticmethod
    def kde_plots(df1: pd.DataFrame, df2: pd.DataFrame, df1_name:str = 'frame 1', df2_name:str = 'frame 2', plots_in_a_row:int = 4):
        """
        Kernel Density Estimation (KDE) is a non-parametric way to estimate the probability density function
        of a random variable. For a given variable, if the distributions are largely overlapping, then the
        variable will not be very useful. However, if they are largely distinct, then the variable might be useful.

        :param df1: dataFrame representing one outcome
        :param df2: dataFrame representing the other outcome
        :param df1_name:
        :param df2_name:
        :param plots_in_a_row: How many plots to be displayed in a row
        :return:
        """
        column_headers = list(df1.columns)

        plot_rows = -(-len(column_headers) // plots_in_a_row) # This odd construction gives a ceiling

        fig, ax = plt.subplots(plot_rows, plots_in_a_row, figsize=(18, 30))
        for i, column_header in enumerate(column_headers):
            plt.subplot(plot_rows, plots_in_a_row, i+1)
            sns.kdeplot(df1[column_header], bw=0.4, label=df1_name, shade=True, color="r", linestyle="--")
            sns.kdeplot(df2[column_header], bw=0.4, label=df2_name, shade=True, color="y", linestyle=":")
            plt.title(column_header, fontsize=12)
        plt.show()

    def text_plot(self, text: str, x_pos:float = 0.5, y_pos:float = 0.5, fontsize: int = 14, horiz_align: str = 'center',
                  return_function_do_not_plot: bool = True):
        """
        Plot the given string
        :param text:
        :param x_pos:
        :param y_pos:
        :param fontsize:
        :param horiz_align:
        :return:
        """
        if return_function_do_not_plot:
            return self.plt.text(x=x_pos, y=y_pos, s=text, fontsize=fontsize, ha=horiz_align)
        self.plt.text(x=x_pos, y=y_pos, s=text, fontsize=fontsize, ha=horiz_align)
        self.plt.show()
