from sklearn.model_selection import cross_validate
from sklearn.utils import shuffle
import numpy as np

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def average_fft(fft, num_bins):
    num_samples_per_bin = int(len(fft) / num_bins)
    fft = fft[: num_samples_per_bin * num_bins]
    averaged_fft = np.average(fft.reshape(-1, num_samples_per_bin), axis=1)
    return averaged_fft

def truncate_fft(fft, truncating_point):
    return fft[: truncating_point]

def preprocess_fft(fft, num_bins, truncating_point):
    truncated_fft = np.abs(truncate_fft(fft, truncating_point))
    averaged_fft = average_fft(truncated_fft, num_bins)

    return averaged_fft

def remove_nan_values_from_dataset(inputs, integer_outputs):
    input_contains_nan_counter = 0
    severe_nan_counter = 0
    between_nan_counter = 0
    mild_nan_counter = 0
    indices_to_delete = []

    for input_index in range(inputs.shape[0]):

        # check if this input example contains a nan value
        if np.isnan(inputs[input_index]).any():
            input_contains_nan_counter += 1

            if integer_outputs[input_index] == 2:
                severe_nan_counter += 1
            elif integer_outputs[input_index] == 1:
                mild_nan_counter += 1
            else:
                between_nan_counter += 1

            indices_to_delete.append(input_index)


    print(f'Total number of input examples containing a nan value: {input_contains_nan_counter}')
    print(f'Number of input examples containing a nan value from the "severe" class: {severe_nan_counter}')
    print(f'Number of input examples containing a nan value from the "between" class: {between_nan_counter}')
    print(f'Number of input examples containing a nan value from the "mild" class: {mild_nan_counter}\n')

    inputs = np.delete(inputs, indices_to_delete, 0)
    integer_outputs = np.delete(integer_outputs, indices_to_delete, 0)
    print(f'Inputs still contain nan values: {np.isnan(inputs).any()}\n')
    print(f'Shape of input data (without nan values): {inputs.shape}')
    print(f'Shape of output data (without nan values): {integer_outputs.shape}\n')

    return inputs, integer_outputs

def print_class_balance(inputs, integer_outputs):
    severe_counter = 0
    mild_counter = 0
    between_counter = 0

    for input_index in range(inputs.shape[0]):

        if integer_outputs[input_index] == 2:
            severe_counter += 1
        elif integer_outputs[input_index] == 1:
            mild_counter += 1
        else:
            between_counter += 1

    print(f'Number of input examples from the "severe" class: {severe_counter}')
    print(f'Number of input examples from the "between" class: {between_counter}')
    print(f'Number of input examples from the "mild" class: {mild_counter}')

def cross_validation(model, _X, _y, _cv=5):
    _scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_weighted']

    shuffled_X, shuffled_y = shuffle(_X, _y)

    results = cross_validate(estimator=model,
                            X=shuffled_X,
                            y=shuffled_y,
                            cv=_cv,
                            scoring=_scoring,
                            return_train_score=False,
                            n_jobs=8,
                            verbose=2)

    return {"Accuracy": results['test_accuracy'].mean(),
            "Macro Precision": results['test_precision_macro'].mean(),
            "Macro Recall": results['test_recall_macro'].mean(),
            "Macro F1 Score": results['test_f1_macro'].mean(),
            "Weighted F1 Score": results['test_f1_weighted'].mean(),
            }

def get_model(model_type):
    if model_type == 'decision_tree':
        model = DecisionTreeClassifier()

    elif model_type == 'random_forest':
        num_estimators = 10
        model = RandomForestClassifier(n_estimators=num_estimators)

    elif model_type == 'SVM':
        kernel = 'poly'
        C = 1
        model = SVC(C=C, kernel=kernel)

    elif model_type == 'MLP':
        # hidden_layer_sizes = (10, 10, 10, 10)
        # hidden_layer_sizes = (100, 100, 100, 100)
        hidden_layer_sizes = (500, 500, 500, 500)
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes)

    else:
        raise ValueError("Invalid model type chosen, choose betweeen ['random_forest', 'decision_tree', 'SVM', 'MLP']")

    return model