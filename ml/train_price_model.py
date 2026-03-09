import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# example dataset
data = pd.read_csv("mandi_prices.csv")

X = data[['day']]
y = data['price']

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "price_model.pkl")