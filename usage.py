import main as m

coefficients = [100, 1, 0.2]
X, y = m.generate_polinomial_data(coefficients,
                                fromX=-5, toX=7, n_samples=500, noise=1, random_sate=42)

X_train, X_test, y_train, y_test = m.train_test_split(X, y, test_size=0.2, random_state=42)

m.find_min_mse(X_train, X_test, y_train, y_test)

m.demonstrate_test_split(X_train, X_test, y_train, y_test)