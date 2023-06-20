import numpy as np
from sklearn.compose import ColumnTransformer


def invertColumnTransformer(column_transformer, preprocessed_X):
    assert isinstance(column_transformer, ColumnTransformer)

    preprocessed_split = {
        name: [None] * len(cols) for name, _, cols in column_transformer.transformers_
    }

    for iCol in range(preprocessed_X.shape[1]):
        name, _, cols = [
            (n, t, list(c)) for n, t, c in column_transformer.transformers_ if iCol in c
        ].pop()
        preprocessed_split[name][cols.index(iCol)] = preprocessed_X[:, iCol]

    X = list()
    for name, algo, _ in column_transformer.transformers_:
        split = np.stack(preprocessed_split[name], axis=1)
        X.append(split if algo == "passthrough" else algo.inverse_transform(split))

    return np.concatenate(X, axis=1)
