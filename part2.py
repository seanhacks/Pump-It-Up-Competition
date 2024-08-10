import optuna
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler 
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from part1 import read_data, CustomTransformer
import argparse

def get_cat_preprocesser(model : str):
    if model.lower() == "mlpclassifier":
        return OneHotEncoder(sparse_output=False)
    elif model.lower() == "randomforestclassifier" or model.lower() == "histgradientboostingclassifier":
        return OrdinalEncoder()

def objective_mlp(trial, X, y):
    activation = trial.suggest_categorical('activation', ['identity', 'logistic', 'tanh', 'relu'])
    init_lr = trial.suggest_categorical('learning_rate_init', [1e-1, 1e-2, 1e-3])
    lr_change = trial.suggest_categorical('learning_rate', ["invscaling", "adaptive"])
    alpha = trial.suggest_float('alpha', 1e-5, 1e-2)
    mo = trial.suggest_float('momentum', 0.4, 1.0)

    # Instantiate MLPClassifier with hyperparameters
    model = MLPClassifier(learning_rate_init=init_lr,
                          learning_rate=lr_change,
                          momentum=mo,
                          activation=activation,
                          alpha=alpha,
                          random_state=42)
    
    # Perform cross-validation
    return cross_val_score(model, X, y, cv=5).mean()

def objective_hist(trial, X, y):
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 20, 100)
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 10, 100)
    
    # Instantiate HistGradientBoostingClassifier with hyperparameters
    model = HistGradientBoostingClassifier(learning_rate=learning_rate,
                                           min_samples_leaf=min_samples_leaf,
                                           max_leaf_nodes=max_leaf_nodes,
                                           random_state=42)
    
    # Perform cross-validation
    return cross_val_score(model, X, y, cv=5).mean()

def objective_forest(trial, X, y):
    n_estimators = trial.suggest_int('n_estimators', 50, 250)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 5)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None, 0.2, 0.5, 0.3])

    # Instantiate RandomForestClassifier with hyperparameters
    model = RandomForestClassifier(n_estimators=n_estimators,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   criterion=criterion,
                                   random_state=42)
    
    # Perform cross-validation
    return cross_val_score(model, X, y, cv=5).mean()

if __name__ == "__main__":
    TRAIN_VALUES = "../data/training_values.csv"
    TRAIN_LABELS = "../data/training_labels.csv"
    TEST_LABELS  = "../data/test_values.csv"

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("model_type", help="Either RandomForestClassifier, HistGradientBoostingClassifier, MLPClassifier")
    args = parser.parse_args()

    train_features, train_labels, test_features = read_data(
        TRAIN_VALUES, 
        TRAIN_LABELS, 
        TEST_LABELS
    )
    train_labels = train_labels.values.ravel()

    transformer = CustomTransformer()
    train_features = transformer.fit_transform(train_features)

    col_transformer = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), selector(dtype_include=['float64', 'int64'])),
            ('cat', get_cat_preprocesser(args.model_type), selector(dtype_include=['object']))
        ],
        verbose=True
    )

    train_features = col_transformer.fit_transform(train_features, train_labels)
    
    study = optuna.create_study(direction='maximize')

    if args.model_type.lower() == "mlpclassifier":
        study.optimize(lambda trial : objective_mlp(trial, train_features, train_labels), n_trials=50)
    elif args.model_type.lower() == "randomforestclassifier":
        study.optimize(lambda trial : objective_forest(trial, train_features, train_labels), n_trials=50)
    elif args.model_type.lower() == "histgradientboostingclassifier":
        study.optimize(lambda trial : objective_hist(trial, train_features, train_labels), n_trials=100)

    trial = study.best_trial
    print(f"Score: {trial.value}")
    for key, value in trial.params.items():
        print(f"{key}: {value}")