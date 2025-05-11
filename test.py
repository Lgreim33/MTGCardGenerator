import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, f1_score, cohen_kappa_score



def evaluate_models():


    # Model and data paths
    model_info = {
        'Color ID': {
            'model_path': 'Models/color_id_model.pkl',
            'X_test_path': 'SplitData/Color_ID/color_id_test.pkl',
            'y_test_path': 'SplitData/Color_ID/color_id_test_target.pkl',
        },
        'Mana Cost': {
            'model_path': 'Models/mana_model.pkl',
            'X_test_path': 'SplitData/Mana/mana_test.pkl',
            'y_test_path': 'SplitData/Mana/mana_test_target.pkl',
        },
        'Type': {
            'model_path': 'Models/type_model.pkl',
            'X_test_path': 'SplitData/Type/type_test.pkl',
            'y_test_path': 'SplitData/Type/type_test_target.pkl',
        },
        'Subtype': {
            'model_path': 'Models/subtype_model.pkl',
            'X_test_path': 'SplitData/Subtype/subtype_test.pkl',
            'y_test_path': 'SplitData/Subtype/subtype_test_target.pkl',
        },
        'Supertype': {
            'model_path': 'Models/supertype_model.pkl',
            'X_test_path': 'SplitData/Supertype/supertype_test.pkl',
            'y_test_path': 'SplitData/Supertype/supertype_test_target.pkl',
        },
        'Keyword': {
            'model_path': 'Models/keyword_model.pkl',
            'X_test_path': 'SplitData/Keywords/keyword_test.pkl',
            'y_test_path': 'SplitData/Keywords/keyword_test_target.pkl',
        },
        'Power': {
            'model_path': 'Models/power_model.pkl',
            'X_test_path': 'SplitData/Power/power_test.pkl',
            'y_test_path': 'SplitData/Power/power_test_target.pkl',
        },
        'Toughness': {
            'model_path': 'Models/toughness_model.pkl',
            'X_test_path': 'SplitData/Toughness/toughness_test.pkl',
            'y_test_path': 'SplitData/Toughness/toughness_test_target.pkl',
        },
    }

    # Score values to graph for each model
    f1_scores = {}
    losses = {}
    kappas = {}
    num_classes = {}

    # Evaluate each model against the test data
    for name, paths in model_info.items():
        print(f"Evaluating {name}...")
        model = joblib.load(paths['model_path'])
        X_test = pd.read_pickle(paths['X_test_path'])
        y_test = pd.read_pickle(paths['y_test_path'])

        # get the actual prediction along with the probabilities
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        # Calculate metrics
        cohen_kappa = cohen_kappa_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred,average="macro")
        loss = log_loss(y_test, y_prob, labels=model.classes_)

        # Store metric 
        f1_scores[name] = f1
        kappas[name] = cohen_kappa
        losses[name] = loss
        num_classes[name] = len(model.classes_)

        print(f"  F1 Score: {f1:.3f}")
        print(f"  Cohen Kappa Score: {cohen_kappa:.3f}")
        print(f"  Log Loss: {loss:.3f}\n")

    # Plotting F1 Score
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 4, 1)
    plt.bar(f1_scores.keys(), f1_scores.values(), color='skyblue')
    plt.xticks(rotation=45)
    plt.ylabel('F1 Score')
    plt.title('Model F1 Score Comparison')

    # Plotting Cohen Kappa Score
    plt.subplot(1, 4, 2)
    plt.bar(kappas.keys(), kappas.values(), color='lightgreen')
    plt.xticks(rotation=45)
    plt.ylabel('Kappa Score')
    plt.title('Model Cohen Kappa Score Comparison')

    # Plotting log loss
    plt.subplot(1, 4, 3)
    plt.bar(losses.keys(), losses.values(), color='salmon')
    plt.xticks(rotation=45)
    plt.ylabel('Log Loss')
    plt.title('Model Log Loss Comparison')

    # number of classes
    plt.subplot(1, 4, 4)
    plt.bar(num_classes.keys(), num_classes.values(), color='orchid')
    plt.xticks(rotation=45)
    plt.ylabel('Number of Classes')
    plt.title('Number of Possible Features per Model')

    plt.tight_layout()
    plt.show()


evaluate_models()