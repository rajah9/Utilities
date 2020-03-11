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

<<<<<<< Updated upstream
        slope, intercept, r, p, epsilon = linregress(df[xlabel], df[ylabel])
        logger.info('Main equation: y = %.3f x + %.3f' % (slope, intercept))
        logger.info('r^2 = %.4f' % (r * r))
        logger.info('p = %.4f' % (p))
        logger.info('std err: %.4f' % (epsilon))

    def count_plot(self, df:pd.DataFrame, xlabel:str, hue:str=None, return_function_do_not_plot:bool=True):
=======
    # Setter for plt
    @plt.setter
    def plt(self, p: plt) -> plt:
        self._p = p

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

        :param df:
        :param col_names:
        :param target_col_name:
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
>>>>>>> Stashed changes
        param_dict = {'x': xlabel, 'data': df}
        if hue:
            param_dict['hue'] = hue

        if return_function_do_not_plot:
            return sns.countplot(**param_dict)
        sns.countplot(**param_dict)
<<<<<<< Updated upstream
        plt.show()
=======
        self.plt.show()

    def histogram_plot(self, df: pd.DataFrame, xlabel: str, bins: int = 10, color: str = 'b', range = None, return_function_do_not_plot: bool = True):
        def plot_histogram():
            return df[xlabel].hist(bins=bins, color=color, range=range)
>>>>>>> Stashed changes

    def historgram_plot(self, df:pd.DataFrame, xlabel:str, bins:int=10, return_function_do_not_plot:bool=True):
        if return_function_do_not_plot:
            return df[xlabel].hist(bins=bins)
        df[xlabel].hist(bins=bins)
        plt.show()

<<<<<<< Updated upstream
    def null_heatmap(self, df:pd.DataFrame, color_scheme:str='Blues'):
=======
    def historgram_plot(self, df: pd.DataFrame, xlabel: str, bins: int = 10, return_function_do_not_plot: bool = True):
        logger.warning('deprecated.')
        self.histogram_plot(df, xlabel, bins, return_function_do_not_plot)


    def null_heatmap(self, df: pd.DataFrame, color_scheme: str = 'Blues'):
>>>>>>> Stashed changes
        sns.heatmap(df.isnull(), yticklabels=True, cbar=False, cmap=color_scheme)
        plt.show()

    def boxplot(self, df:pd.DataFrame, xlabel:str, ylabel:str):
        sns.boxplot(x=xlabel, y=ylabel, data=df)
        plt.show()

<<<<<<< Updated upstream
    def figure_plots(self, plots:list, rows:int, cols:int, dims:list=[6,12]):
=======
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
>>>>>>> Stashed changes
        """
        Put the given plots list into a single figure. Goes left to right, top to bottom.
        :param plots: list of sns plots to be invoked.
        :param rows: how many rows in the figure
        :param cols: how many columns in the figure
        :param dims: height, width in inches as a list.
        :return:
        """
<<<<<<< Updated upstream
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
=======
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
>>>>>>> Stashed changes


<<<<<<< Updated upstream
    
=======
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
>>>>>>> Stashed changes
