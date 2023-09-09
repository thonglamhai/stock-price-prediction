import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno


# Global style setting
def set_global_style():
    sns.set_style('darkgrid')
    sns.set(font_scale=1.3)

def plot_missing_values(dataframe):
    msno.bar(dataframe)
    plt.savefig("output/figures/visualizing_missing_data.png", dpi=100)
    plt.show()


def plot_correlation(dataframe):
    sns.heatmap(dataframe.corr(), annot=True)
    plt.savefig("output/figures/visualizing_heatmap_data.png", dpi=100)
    plt.show()

def plot_distribution(dataframe, column):
    sns.distplot(dataframe[column])
    plt.savefig("output/figures/visualizing_distribution_data.png", dpi=100)
    plt.show()

def plot_outliers(dataframe):
    plt.figure(figsize=(14, 8))
    sns.boxplot(dataframe)

    plt.savefig("output/figures/visualizing_outliers_data.png", dpi=100)
    plt.show()

def plot_scatter(dataframe, column1, column2):
    sns.scatterplot(x=column1, y=column2, data=dataframe)
    plt.show()

def plot_stock_price(dataframe, column):
    plt.figure(figsize=(14,5))
    plt.plot(dataframe[column])
    plt.title('Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

def plot_stock_prediction(train, test):
    plt.figure(figsize=(10, 8))
    plt.plot(train.index, train['adj close'])
    plt.plot(test.index, test[['adj close', 'Predictions']])
    plt.title('Apple Stock Close Price')
    plt.xlabel('date')
    plt.ylabel("adj lose")
    plt.legend(['Train', 'Test', 'Predictions'])
    plt.show()

def plot_pairplot(dataframe):
    sns.pairplot(dataframe)
    plt.savefig("output/figures/visualizing_pairplot_data.png", dpi=100)
    plt.show()