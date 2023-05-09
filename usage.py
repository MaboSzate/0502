import numpy as np
import pandas as pd
import numpy as np
import main as m
import matplotlib.pyplot as plt

X,y=m.load_data("data_competition2_train.csv")
model=m.cross_validate(X,y)
m.plot_data_and_prediction(X,y,model)
plt.show()
#X_train, X_test, y_train, y_test = m.train_test_split(X, y, test_size=0.2, random_state=42)

#m.find_min_mse(X_train.reshape(-1,1), X_test.reshape(-1,1), y_train.reshape(-1,1), y_test.reshape(-1,1))

#m.demonstrate_test_split(X_train, X_test, y_train, y_test)
