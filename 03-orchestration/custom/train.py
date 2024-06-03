from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

@custom
def transform_custom(df):

    categorical = ['PULocationID', 'DOLocationID']

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df['duration'].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    print(lr.intercept_)
    return dv,lr
