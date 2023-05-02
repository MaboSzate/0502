import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
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
    return name, model, mse_on_test_set, coefficients_on_train_set


def find_min_mse(X_train, X_test, y_train, y_test):
    msemin = 100
    for d in range(1, 20):
        name, model, mse_on_test_set, coeffs_on_test_set=create_train_and_eval_poly_model(X_train, X_test, y_train,
                                                                                            y_test, degree=d)
        print(d, mse_on_test_set)
        if mse_on_test_set < msemin:
            msemin = mse_on_test_set
            dmin=d
    print("min:", dmin, msemin)




