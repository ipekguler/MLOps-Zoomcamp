from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

@transformer
def transform(df):
    train_dicts = df['PULocationID', 'DOLocationID'].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df['duration'].values
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return dv,lr
