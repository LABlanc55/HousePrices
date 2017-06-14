import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

TRAIN_DIR = 'data'


class DataCleaner:
    def cluster(self):
        train = pandas.read_csv(TRAIN_DIR + '/train.csv')

        quantity = [f for f in train.columns if train.dtypes[f] != 'object']
        quantity.remove('SalePrice')
        quantity.remove('Id')
        quality = [f for f in train.columns if train.dtypes[f] == 'object']
        qual_encoded = self.encode_quality(train, quality)

        features = quantity + qual_encoded
        model = TSNE(n_components=2, random_state=0, perplexity=50)
        X = train[features].fillna(0.).values
        tsne = model.fit_transform(X)

        std = StandardScaler()
        s = std.fit_transform(X)
        pca = PCA(n_components=30)
        pca.fit(s)
        pc = pca.transform(s)
        kmeans = KMeans(n_clusters=5)
        kmeans.fit(pc)

        fr = pandas.DataFrame({'tsne1': tsne[:, 0], 'tsne2': tsne[:, 1], 'cluster': kmeans.labels_})
        sns.lmplot(data=fr, x='tsne1', y='tsne2', hue='cluster', fit_reg=False)
        print(np.sum(pca.explained_variance_ratio_))
        plt.show()

    def clean(self):
        train = pandas.read_csv(TRAIN_DIR + '/train.csv')

        quantity = [f for f in train.columns if train.dtypes[f] != 'object']
        quantity.remove('SalePrice')
        quantity.remove('Id')

        quality = [f for f in train.columns if train.dtypes[f] == 'object']

        qual_encoded = self.encode_quality(train, quality)

        # features = quantity + qual_encoded
        # self.spearman(train, features)
        # self.generate_heatmaps(train, quantity, qual_encoded)
        # self.gen_pairplots(train, quantity, qual_encoded)

        self.segment_prices(train, quantity, 200000)

    def visualize(self):
        train = pandas.read_csv(TRAIN_DIR+'/train.csv')

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        train_numerics = train.select_dtypes(include=numerics)
        train_numerics = self.drop_unused(train_numerics, ['Id'])
        print(list(train_numerics))
        #train_numerics = self.clean_ms_subclass(train_numerics)
        train_numerics = self.clean_lot_frontage(train_numerics)

        #train_strings = train.select_dtypes(exclude=numerics)
        #print(list(train_strings))
        #print(train_strings.MSZoning.astype('category'))

    def segment_prices(self, train, quantity, segment_value):
        features = quantity
        standard = train[train['SalePrice'] < segment_value]
        pricey = train[train['SalePrice'] >= segment_value]

        diff = pandas.DataFrame()
        diff['feature'] = features
        diff['difference'] = [
            (pricey[f].fillna(0.).mean() - standard[f].fillna(0.).mean()) / (standard[f].fillna(0.).mean()) for f in
            features]
        sns.barplot(data=diff, x='feature', y='difference')
        x = plt.xticks(rotation=90)
        plt.show()

    def gen_pairplots(self, train, quantity, qual_encoded):
        f = pandas.melt(train, id_vars=['SalePrice'], value_vars=quantity + qual_encoded)
        g = sns.FacetGrid(f, col='variable', col_wrap=2, sharex=False, sharey=False, size=5)
        g = g.map(self.pairplot, "value", "SalePrice")
        plt.show()

    def pairplot(self, x, y, **kwargs):
        ax = plt.gca()
        ts = pandas.DataFrame({'time': x, 'val': y})
        ts = ts.groupby('time').mean()
        ts.plot(ax=ax)
        plt.xticks(rotation=90)

    def generate_heatmaps(self, train, quantity, qual_encoded):
        self.gen_heatmap(graph_num=1, corr=train[quantity + ['SalePrice']].corr())
        self.gen_heatmap(graph_num=2, corr=train[qual_encoded + ['SalePrice']].corr())

        corr = pandas.DataFrame(np.zeros([len(quantity) + 1, len(qual_encoded) + 1]), index=quantity + ['SalePrice'],
                                columns=qual_encoded + ['SalePrice'])
        for q1 in quantity + ['SalePrice']:
            for q2 in qual_encoded + ['SalePrice']:
                corr.loc[q1, q2] = train[q1].corr(train[q2])

        self.gen_heatmap(graph_num=3, corr=corr)

    def gen_heatmap(self, graph_num, corr):
        plt.figure(graph_num)
        sns.heatmap(corr)
        x = plt.xticks(rotation=90)
        y = plt.yticks(rotation=0)
        plt.show()

    def encode_quality(self, train, quality):
        qual_encoded = []
        for q in quality:
            self.encode(train, q)
            qual_encoded.append(q + '_E')
        return qual_encoded

    def encode(self, frame, feature):
        ordering = pandas.DataFrame()
        ordering['val'] = frame[feature].unique()
        ordering.index = ordering.val
        ordering['spmean'] = frame[[feature, 'SalePrice']].groupby(feature).mean()['SalePrice']
        ordering = ordering.sort_values('spmean')
        ordering['ordering'] = range(1, ordering.shape[0]+1)
        ordering = ordering['ordering'].to_dict()

        for cat, o in ordering.items():
            frame.loc[frame[feature] == cat, feature+'_E'] = o

    def spearman(self, frame, features):
        spr = pandas.DataFrame()
        spr['feature'] = features
        spr['spearman'] = [frame[f].corr(frame['SalePrice'], 'spearman') for f in features]
        spr = spr.sort_values('spearman')
        plt.figure(figsize=(6, 0.25*len(features)))
        sns.barplot(data=spr, y='feature', x='spearman', orient='h')
        plt.show()

    def drop_unused(self, df, columns):
        return df.drop(columns, axis=1)

    def clean_ms_subclass(self, df):
        print(df.MSSubClass.astype('category'))  # 15 bins: should be treated as category or number?
        return df

    def clean_lot_frontage(self, df):
        print(df.LotFrontage.dropna())
        return df


dc = DataCleaner()
dc.cluster()
