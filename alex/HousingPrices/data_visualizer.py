import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn
import scipy.stats as st

TRAIN_DIR = 'data'
TEST_DIR = 'data'

class DataVisualizer:
    def visualize(self):
        # find which features have most missing values
        train = pandas.read_csv(TRAIN_DIR + '/train.csv')

        quantity = [f for f in train.columns if train.dtypes[f] != 'object']
        quantity.remove('SalePrice')
        quantity.remove('Id')

        #self.display_features_missing_values(train)
        #self.find_training_results_shape(train)
        #print(self.test_normality(train, quantity))
        #self.find_best_features_for_transformation(train, quantity)

        quality = [f for f in train.columns if train.dtypes[f] == 'object']

        #self.display_var_boxplots(train, quality)
        self.display_est_influence_categorical(train, quality)


    def display_est_influence_categorical(self, train, quality):
        a = self.anova(train, quality)
        a['disparity'] = numpy.log(1. / a['pval'].values)
        seaborn.barplot(data=a, x='feature', y='disparity')
        x = plt.xticks(rotation=90)
        plt.show()

    def display_var_boxplots(self, train, quality):
        for c in quality:
            train[c] = train[c].astype('category')
            if train[c].isnull().any():
                train[c] = train[c].cat.add_categories(['MISSING'])
                train[c] = train[c].fillna('MISSING')

        f = pandas.melt(train, id_vars=['SalePrice'], value_vars=quality)
        g = seaborn.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False, size=5)
        g = g.map(self.boxplot, "value", "SalePrice")
        plt.show()

    def find_best_features_for_transformation(self, train, quantity):
        f = pandas.melt(train, value_vars=quantity)
        g = seaborn.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False)
        g = g.map(seaborn.distplot, "value")
        plt.show()

    def display_features_missing_values(self, train):
        missing = train.isnull().sum()
        missing = missing[missing > 0]
        missing.sort_values(inplace=True)
        print("missing: {}".format(missing))
        plt.interactive(False)
        missing.plot.bar()
        plt.show()

    def find_training_results_shape(self, train):
        y = train['SalePrice']
        plt.figure(1); plt.title('Johnson SU')
        seaborn.distplot(y, kde=False, fit=st.johnsonsu)
        plt.show()
        plt.figure(2); plt.title('Normal')
        seaborn.distplot(y, kde=False, fit=st.norm)
        plt.show()
        plt.figure(3); plt.title('Log Normal')
        seaborn.distplot(y, kde=False, fit=st.lognorm)
        plt.show()

    def test_normality(self, train, quantity):
        test_normality = lambda x: st.shapiro(x.fillna(0))[1] < 0.01
        normal = pandas.DataFrame(train[quantity])
        normal = normal.apply(test_normality)
        return not normal.any()

    def boxplot(self, x, y, **kwargs):
        seaborn.boxplot(x=x, y=y)
        x = plt.xticks(rotation=90)

    def anova(self, frame, quality):
        anv = pandas.DataFrame()
        anv['feature'] = quality
        pvals = []
        for c in quality:
            samples = []
            for cls in frame[c].unique():
                s = frame[frame[c] == cls]['SalePrice'].values
                samples.append(s)
            pval = st.f_oneway(*samples)[1]
            pvals.append(pval)
        anv['pval'] = pvals
        return anv.sort_values('pval')

dv = DataVisualizer()
dv.visualize()