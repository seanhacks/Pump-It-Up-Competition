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

SEED = 42

def read_data(train_features, train_labels, test_features):
    """
    Reads the training data and labels from the specified
    file in the parameters of this function. Returns 
    the features and labels in separate pandas dataframes.
    """

    train_features_df = pd.read_csv(train_features)
    train_labels_df = pd.read_csv(train_labels).drop(columns="id")
    test_features_df = pd.read_csv(test_features)

    return train_features_df, train_labels_df, test_features_df 

def get_categorical_preprocessing(encoder : str):
    """
    Returns the type of pre-processing applied to the data.
    """
    encoder = encoder.lower()
    
    if encoder == "onehotencoder":
        return OneHotEncoder(sparse_output=False)
    elif encoder == "ordinalencoder":
        return OrdinalEncoder()
    elif encoder == "targetencoder":
        return TargetEncoder()

def get_numerical_preprocessing(preprocessor : str):
    """
    Returns the numerical preprocessor used on the data.
    """

    if preprocessor.lower() == "standardscaler":
        return StandardScaler()
    return None

def perform_preprocessing(scaler, encoder):
    """
    returns the transformer that will apply the preprocessing
    steps to the data.
    """
    if scaler is None:
        return ColumnTransformer(
            transformers=[
                ('cat', encoder, selector(dtype_include=["object"]))
            ],
            verbose=True
        )
    else:
        return ColumnTransformer(
            transformers=[
                ('num', scaler, selector(dtype_include=['float64', 'int64'])),
                ('cat', encoder, selector(dtype_include=['object']))
            ],
            verbose=True
        )

def get_classifier(classifier_type : str):
    """
    Gets the model that we want.
    """
    classifier_type = classifier_type.lower()

    if classifier_type == "logisticregression":
        # Needed to change max iter so that the model converged.
        return LogisticRegression(max_iter=500 ,random_state=SEED)
    elif classifier_type == "randomforestclassifier":
        return RandomForestClassifier(random_state=SEED, n_jobs=-1)
    elif classifier_type == "gradientboostingclassifier":
        return GradientBoostingClassifier(random_state=SEED)
    elif classifier_type == "histgradientboostingclassifier":
        return HistGradientBoostingClassifier(random_state=SEED)
    elif classifier_type == "mlpclassifier":
        return MLPClassifier(solver="sgd", learning_rate_init=1e-1, learning_rate="invscaling",random_state=SEED)


#https://www.andrewvillazon.com/custom-scikit-learn-transformers/
class CustomTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X : pd.DataFrame, y=None):

        # This is done in fit to ensure the minimum date is that of the
        # Training set when applying the transformations to the test set.
        X["date_recorded"] = pd.to_datetime(X["date_recorded"])
        self.min_date = X["date_recorded"].min() 
        self.median_construction_year = X[X['construction_year'] > 0]['construction_year'].median()
        return self

    def transform(self, X, y=None):

        # Alter format of date recorded.
        X["date_recorded"] = pd.to_datetime(X["date_recorded"])
        X["date_recorded"] = (X["date_recorded"] - self.min_date).dt.days

        # Find the minimum construction_year that is non-zero.
        # Then Find the median (more statistically stable than other metrics) 
        # for the constructions years above 0.
        # Replace the construction year values of 0 to this median.
        X['construction_year'] = X['construction_year'].replace(0, self.median_construction_year)

        # Remove cols with all unique values (avoid overfitting to these values.)
        X = X.drop(columns=['wpt_name', 'id'])

        # Remove all location markers except for longitude and latitude:
        X = X.drop(columns=['subvillage','basin','region','lga','ward'])

        # Remove scheme name as roughly 50% of the values are empty/missing.
        X = X.drop(columns=['scheme_name'])

        # extraction_type, extraction_type_class, extraction_type_group, all have very
        # similar values. We choose extraction_type as it has the highest cardinality of
        # the 3 and therefore may contain the most information.
        X = X.drop(columns=['extraction_type_class', 'extraction_type_group'])

        # payment_type and payment exactly the same.
        X = X.drop(columns=['payment_type'])

        # water quality and quality group the same.
        # larger was chosen
        X = X.drop(columns=['quality_group'])

        # quantity and quantity group the exact same!
        X = X.drop(columns=['quantity'])

        # source type kept as it groups together unknown and other. 
        X = X.drop(columns=['source_class', 'source'])

        # Larger set kept.
        X = X.drop(columns=['waterpoint_type_group'])

        # only one unique value, useless for predictions.
        X = X.drop(columns=['recorded_by'])

        # High cardinality 
        X = X.drop(columns=['funder', 'installer'])

        # Remove nans from 3 columns, public_meeting, scheme management, permit.
        X['public_meeting'] = X['public_meeting'].fillna(True)
        X['scheme_management'] = X['scheme_management'].fillna("Other")
        X['permit'] = X['permit'].fillna(True)

        return X

def run(catergorical, numerical, model_type, 
        train_file="../data/training_values.csv",
        train_labels_file="../data/training_labels.csv", 
        test_file="../data/test_values.csv",
        output_file="./out.csv"):
    
    # read data and store.
    train_features, train_labels, test_features = read_data(train_file, train_labels_file, test_file)
    #train_ids = train_features["id"].values.copy()
    test_ids = test_features["id"].values.copy()

    # get encoders and scalers.
    encoder = get_categorical_preprocessing(catergorical)
    scaler = get_numerical_preprocessing(numerical)

    # clean up date
    transformer = CustomTransformer()
    train_features = transformer.fit_transform(train_features)
    test_features = transformer.transform(test_features)

    # perform pre-processing on training and test features.
    column_preprocessor = perform_preprocessing(scaler, encoder)
    train_features = column_preprocessor.fit_transform(train_features, train_labels)
    test_features = column_preprocessor.transform(test_features)

    # get model 
    model = get_classifier(model_type)

    scores = cross_val_score(model, train_features, train_labels.values.ravel(), cv=5, scoring='accuracy')

    mean_score = scores.mean()
    # Print the mean accuracy score
    print("Mean accuracy:", mean_score)

    # Train the model on the entire training dataset
    model.fit(train_features, train_labels.values.ravel())

    # Make predictions on the test features
    test_predictions = model.predict(test_features)
    output_df = pd.DataFrame({'id': test_ids,'status_group': test_predictions})
    output_df.to_csv(output_file, index=False)

    return mean_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("train_input_file", help="Path to the training input file.")
    parser.add_argument("train_labels_file", help="Path to the training labels file")
    parser.add_argument("test_input_file", help="Path to the test input file")
    parser.add_argument("numerical_preprocessing", help="Either StandardScalar or None")
    parser.add_argument("categorical_preprocessing", help="Either OneHotEncoder, OrdinalEncoder, TargetEncoder")
    parser.add_argument("model_type", help="Either LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, MLPClassifier")
    parser.add_argument("test_prediction_output_file", help="Path to the test prediction output file")

    args = parser.parse_args()

    run(args.categorical_preprocessing, 
        args.numerical_preprocessing,
        args.model_type,
        args.train_input_file, 
        args.train_labels_file,
        args.test_input_file,
        args.test_prediction_output_file)
    
