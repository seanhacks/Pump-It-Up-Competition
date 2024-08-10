import argparse
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from part1 import read_data, perform_preprocessing, CustomTransformer


def get_models():
    return [
         MLPClassifier(
            activation="tanh",
            learning_rate="adaptive",
            learning_rate_init=0.001,
            alpha=0.007477595542210561,
            momentum=0.6485614427118873,
            random_state=42,
        ),
        RandomForestClassifier(
            n_estimators=215,
            min_samples_leaf=2,
            min_samples_split=5,
            criterion="gini",
            max_features=0.5,
            random_state=42,
        ),
        HistGradientBoostingClassifier(
            learning_rate=0.805404040404043,
            min_samples_leaf=31,
            max_leaf_nodes=91,
            random_state=42,
        )
    ]
    
if __name__ == "__main__":
    TRAIN_VALUES = "../data/training_values.csv"
    TRAIN_LABELS = "../data/training_labels.csv"
    TEST_LABELS  = "../data/test_values.csv"

    models = get_models()
    encoders = [OneHotEncoder(sparse_output=False), OrdinalEncoder(), OrdinalEncoder()]
    output_files = ["./test_results_mlp.csv", "./test_results_rfc.csv", "./test_results_hgbc.csv"]

    for i in range(len(models)):
        train_features, train_labels, test_features = read_data(
            TRAIN_VALUES, 
            TRAIN_LABELS, 
            TEST_LABELS
        )
        train_labels = train_labels.values.ravel()
        test_ids = test_features["id"].values.copy()

        transformer = CustomTransformer()
        train_features = transformer.fit_transform(train_features)
        test_features = transformer.transform(test_features)

        # perform pre-processing on training and test features.
        column_preprocessor = perform_preprocessing(StandardScaler(), encoders[i])
        train_features = column_preprocessor.fit_transform(train_features, train_labels)
        test_features = column_preprocessor.transform(test_features)

        model = models[i]
        model.fit(train_features, train_labels)

        # Make predictions on the test features
        test_predictions = model.predict(test_features)
        output_df = pd.DataFrame({'id': test_ids,'status_group': test_predictions})
        output_df.to_csv(output_files[i], index=False)

