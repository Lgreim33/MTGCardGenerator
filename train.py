from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split, PredefinedSplit, GridSearchCV,RandomizedSearchCV
from sklearn.metrics import log_loss, make_scorer
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import joblib

def train(imbalanced_mode=True):
    # Load data
    datasets = {
        'color_id': (pd.read_pickle("SplitData/Color_ID/color_id_train.pkl"),
                     pd.read_pickle("SplitData/Color_ID/color_id_train_target.pkl")),
        'mana': (pd.read_pickle("SplitData/Mana/mana_train.pkl"),
                 pd.read_pickle("SplitData/Mana/mana_train_target.pkl")),
        'type': (pd.read_pickle("SplitData/Type/type_train.pkl"),
                 pd.read_pickle("SplitData/Type/type_train_target.pkl")),
        'supertype': (pd.read_pickle("SplitData/Supertype/supertype_train.pkl"),
                      pd.read_pickle("SplitData/Supertype/supertype_train_target.pkl")),
        'subtype': (pd.read_pickle("SplitData/Subtype/subtype_train.pkl"),
                    pd.read_pickle("SplitData/Subtype/subtype_train_target.pkl")),
        'keyword': (pd.read_pickle("SplitData/Keywords/keyword_train.pkl"),
                    pd.read_pickle("SplitData/Keywords/keyword_train_target.pkl")),
        'power': (pd.read_pickle("SplitData/Power/power_train.pkl"),
                  pd.read_pickle("SplitData/Power/power_train_target.pkl")),
        'toughness': (pd.read_pickle("SplitData/Toughness/toughness_train.pkl"),
                      pd.read_pickle("SplitData/Toughness/toughness_train_target.pkl"))
    }

    # Simplified hyperparameter search space
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_lambda': [1.0],
        'reg_alpha': [0.1]
    }

    models = {}

    for name, (X, y) in datasets.items():
        print(f"\nTraining model for {name}...")

        # Verify the target variable
        unique_classes = np.unique(y)
        num_classes = len(unique_classes)
        print(f"Number of classes in {name}: {num_classes}")

        # Ensure classes are sequential integers starting from 0
        if unique_classes[0] != 0 or unique_classes[-1] != num_classes - 1:
            raise ValueError(f"Classes in {name} are not sequential from 0 to {num_classes - 1}. Found: {unique_classes}")

        # Oversample rare classes to ensure at least 5 examples per class
        class_counts = pd.Series(y).value_counts()
        min_examples = 5  # Ensure at least 5 examples per class
        rare_classes = class_counts[class_counts < min_examples].index
        for cls in rare_classes:
            rare_samples = X[y == cls]
            y_rare = y[y == cls]
            # Duplicate until we have at least min_examples
            current_count = class_counts.get(cls, 0)
            while current_count < min_examples:
                X = pd.concat([X, rare_samples], axis=0)
                y = np.concatenate([y, y_rare])
                current_count += len(y_rare)

        # Split into train and validation sets with stratification
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )


        # Verify that all classes are present in both sets
        train_classes = set(np.unique(y_train))
        val_classes = set(np.unique(y_val))
        expected_classes = set(unique_classes)
        if train_classes != expected_classes or val_classes != expected_classes:
            raise ValueError(f"Class mismatch after split - y_train: {train_classes}, y_val: {val_classes}, expected: {expected_classes}")

        # Create a PredefinedSplit for GridSearchCV, because our data contains so many rare cases, even with oversampling
        split_index = [-1] * len(y_train) + [0] * len(y_val)
        X_combined = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
        y_combined = np.concatenate([y_train, y_val])
        ps = PredefinedSplit(test_fold=split_index)

        # Handle imbalanced data for training set
        sample_weights = None
        if imbalanced_mode:
            sample_weights = compute_sample_weight(class_weight="balanced", y=y_combined)

        # Initialize model
        if num_classes == 2:
            base_model = XGBClassifier(
                enable_categorical=True,
                eval_metric='logloss'
            )
        else:
            base_model = XGBClassifier(
                enable_categorical=True,
                eval_metric='mlogloss',
                num_class=num_classes
            )

        # Use GridSearchCV with PredefinedSplit
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=20, 
            scoring='neg_log_loss',
            cv=ps,
            verbose=1,
            n_jobs=1,
            random_state=42
        )

        random_search.fit(
            X_combined,
            y_combined,
            sample_weight=sample_weights
        )

        # Use the best parameters
        best_params = random_search.best_params_
        print(f"Best parameters for {name}_model: {best_params}")
        print(f"Best validation log loss (negative): {random_search.best_score_}")

        # Train the final model on the full dataset with the best parameters
        if num_classes == 2:
            model = XGBClassifier(
                enable_categorical=True,
                eval_metric='logloss',
                **best_params
            )
        else:
            model = XGBClassifier(
                enable_categorical=True,
                eval_metric='mlogloss',
                num_class=num_classes,
                **best_params
            )

        full_sample_weights = compute_sample_weight(class_weight="balanced", y=y_train) if imbalanced_mode else None
        model.fit(X_train, y_train, sample_weight=full_sample_weights)

        # Verify the number of classes the model learned
        print(f"Number of classes in model: {model.n_classes_}")

        # Save the model
        joblib.dump(model, f"Models/{name}_model.pkl")
        models[name] = model

    return models

# Train the models
models = train(imbalanced_mode=True)