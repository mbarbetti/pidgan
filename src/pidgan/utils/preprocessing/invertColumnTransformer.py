import numpy as np
from sklearn.compose import ColumnTransformer


def invertColumnTransformer(column_transformer, preprocessed_X):
    assert isinstance(column_transformer, ColumnTransformer)

    iCol = 0
    postprocessed_split = dict()
    for name, algo, cols in column_transformer.transformers_:
        preprocessed_cols = list()
        for _ in range(len(cols)):
            preprocessed_cols.append(preprocessed_X[:, iCol][:, None])
            iCol += 1
        preprocessed_block = np.concatenate(preprocessed_cols, axis=1)
        if algo == "passthrough":
            postprocessed_split[name] = preprocessed_block
        else:
            postprocessed_split[name] = algo.inverse_transform(preprocessed_block)

    X = [None] * preprocessed_X.shape[1]
    for name, _, cols in column_transformer.transformers_:
        for i, iCol in enumerate(cols):
            X[iCol] = postprocessed_split[name][:, i][:, None]

    return np.concatenate(X, axis=1)
