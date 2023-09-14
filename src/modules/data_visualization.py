import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


# Global style setting
def set_global_style():
    '''
    Function to set global style for the plots
    '''
    sns.set_style('darkgrid')
    sns.set(font_scale=1.3)

def plot_missing_values(dataframe):
    '''
    Function to plot missing values
    '''
    msno.bar(dataframe)
    plt.savefig("output/figures/visualizing_missing_data.png", dpi=100)
    plt.show()


def plot_correlation(dataframe):
    '''
    Function to plot correlation
    '''
    sns.heatmap(dataframe.corr(), annot=True)
    plt.savefig("output/figures/visualizing_heatmap_data.png", dpi=100)
    plt.show()

def plot_distribution(dataframe, column):
    '''
    Function to plot distribution
    '''
    plt.figure(figsize=(14, 8))
    sns.distplot(dataframe[column])
    plt.savefig("output/figures/visualizing_distribution_data.png", dpi=100)
    plt.show()

def plot_outliers(dataframe):
    '''
    Function to plot outliers
    '''
    plt.figure(figsize=(14, 8))
    sns.boxplot(dataframe)

    plt.savefig("output/figures/visualizing_outliers_data.png", dpi=100)
    plt.show()

def plot_stock_price(dataframe, column):
    '''
    Function to plot stock price
    '''
    plt.figure(figsize=(14,5))
    plt.plot(dataframe[column])
    plt.title('Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

def plot_stock_prediction(train, test):
    '''
    Function to plot stock prediction
    '''
    plt.figure(figsize=(10, 8))
    plt.plot(train.index, train['adj close'])
    plt.plot(test.index, test[['adj close', 'Predictions']])
    plt.title('Apple Stock Close Price')
    plt.xlabel('date')
    plt.ylabel("adj lose")
    plt.legend(['Train', 'Test', 'Predictions'])
    plt.show()

def plot_pairplot(dataframe):
    '''
    Function to plot pairplot
    '''
    sns.pairplot(dataframe)
    plt.savefig("output/figures/visualizing_pairplot_data.png", dpi=100)
    plt.show()