import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


def generate_polinomial_data(coeffs, fromX, toX, n_samples, noise, random_sate=None, filepath=None):
    np.random.seed(random_sate)
    X = np.random.uniform(fromX, toX, n_samples)
    y = np.polyval(coeffs[::-1], X) + noise * np.random.randn(n_samples)
    if filepath:
        df = pd.DataFrame({'x': X, 'y': y})
        df.to_csv(filepath, index=False, header=False)
    return X.reshape(-1,1), y


coefficients = [100, 1, 0.2]
X, y = generate_polinomial_data(coefficients,
                                fromX=-5, toX=7, n_samples=500, noise=1, random_sate=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def load_data(file_path):
    df = pd.read_csv(file_path, sep=',', header=None, names=['x', 'y'])
    X = df['x'].values.reshape(-1, 1)
    y = df['y'].values
    return X, y


def demonstrate_test_split(X_train, X_test, y_train, y_test):
    plt.scatter(X_train, y_train, label="Train", alpha=0.5)
    plt.scatter(X_test, y_test, label="Test", alpha=0.5)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Train-Test Split")
    plt.legend()
    plt.show()


def create_polynomial_model(degree=1):
    name = "Polinomial_" + str(degree)
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    return name, model


def create_train_and_eval_poly_model(X_train, X_test, y_train, y_test, degree=15):
    name, model=create_polynomial_model(degree)
    model.fit(X_train, y_train)
    coefficients_on_train_set=model.named_steps['linearregression'].coef_
    y_pred = model.predict(X_test)
    mse_on_test_set = mean_squared_error(y_test, y_pred)
    return name, model, mse_on_test_set, coefficients_on_train_set, y_pred


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return model, mse


def find_min_mse(X_train, X_test, y_train, y_test):
    msemin = np.inf
    best_pred = None
    coeffs=None
    for d in range(1, 20):
        name, model, mse_on_test_set, coeffs_on_test_set, y_pred=create_train_and_eval_poly_model(X_train, X_test, y_train,
                                                                                            y_test, degree=d)
        print(d, mse_on_test_set)
        if mse_on_test_set < msemin:
            msemin = mse_on_test_set
            dmin=d
            best_pred=y_pred
            coeffs=coeffs_on_test_set
    print("min:", dmin, msemin)
    print("coeffs: ", coeffs)


def print_coeffs(text, model):
    if 'linear_regression' in model.named_steps.keys():
        linreg = 'linear_regression'
    else:
        linreg = 'linearregression'
    coeffs = np.concatenate(([model.named_steps[linreg].intercept_], model.named_steps[linreg].coef_[1:]))
    coeffs_str = ' '.join(np.format_float_positional(coeff, precision=4) for coeff in coeffs)
    print(text + coeffs_str)


def cross_validate(X, y, n_splits=5, from_degree=1, to_degree=10):
    degrees = range(from_degree, to_degree+1)
    kf = KFold(n_splits=n_splits)
    results = {}
    best_model = None
    best_degree = None
    best_mse = np.inf
    np.set_printoptions(precision=4)
    for degree in degrees:
        name, model = create_polynomial_model(degree)
        mse_sum = 0
        for train_idx, val_idx in kf.split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            model, mse = train_and_evaluate_model(model, X_train, y_train, X_val, y_val)
            print_coeffs("Coefficients: ", model)
            mse_sum += mse
        avg_mse = mse_sum / n_splits
        results[degree] = avg_mse
        print(f"for degree: {degree}, MSE: {avg_mse}")
        # fit for the whole dataset
        # model, mse = train_and_evaluate_model(model, X, y, X_val, y_val)1
        model.fit(X, y)
        print_coeffs("Final Coefficients: ", model)
        if avg_mse < best_mse:
            best_mse = avg_mse
            best_degree = degree
            best_model = model
    print(f"Best model: degree={best_degree}, MSE={best_mse}")
    print_coeffs("Coefficients for best model: ", best_model)
    return best_model


def plot_data_and_prediction(X, y, model, title=None):
    plt.scatter(X, y, color='blue', label='Data Points')
    X_pred = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model.predict(X_pred)
    plt.plot(X_pred, y_pred, color='red', label='Model Prediction')
    plt.title(title)
    plt.legend()

