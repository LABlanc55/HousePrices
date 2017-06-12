import pandas

TRAIN_DIR = 'data'


class DataCleaner:
    def visualize(self):
        train = pandas.read_csv(TRAIN_DIR+'/train.csv')
        print(train.shape)
        #print(list(train))

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        train_numerics = train.select_dtypes(include=numerics)
        train_numerics = self.drop_unused(train_numerics, ['Id'])
        print(list(train_numerics))
        #train_numerics = self.clean_ms_subclass(train_numerics)
        train_numerics = self.clean_lot_frontage(train_numerics)

        #train_strings = train.select_dtypes(exclude=numerics)
        #print(list(train_strings))
        #print(train_strings.MSZoning.astype('category'))


    def drop_unused(self, df, columns):
        return df.drop(columns, axis=1)

    def clean_ms_subclass(self, df):
        print(df.MSSubClass.astype('category'))  # 15 bins: should be treated as category or number?
        return df

    def clean_lot_frontage(self, df):
        print(df.LotFrontage.dropna())
        return df

dc = DataCleaner()
dc.visualize()