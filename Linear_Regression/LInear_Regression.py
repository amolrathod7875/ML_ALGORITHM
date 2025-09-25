import numpy as np 
import pandas as pd
from sklearn import datasets

class LinearRegression:
    def __init__(self,lr: int = 0.01,n_iters:int = 1000) -> None:           ##"Learning Rate : lr"
        self.lr = lr
        self.n_iters = n_iters
        self.weight = None 
        self.bias = None


    def fit(self,X,y):
        num_samples,num_features = X.shape
        self.weight = np.random.rand(num_features)
        self.bias = 0

        for i in range(self.n_iters):
            y_pred = np.dot(X,self.weight) + self.bias              #y = mx + b
            diff = y_pred - y 
            dw = (1 / num_samples) * np.dot(X.T, diff)
            db = (1/ num_samples) * np.sum(diff)

            self.weight = self.weight - self.lr * dw
            self.bias = self.bias - self.lr * db

        return self
    
    def predict(self,X):
        return np.dot(X, self.weight) + self.bias
    


if __name__== "__main__":
    X,y = datasets.make_regression(n_samples=500,n_features=1,noise=25,random_state=4)
    model = LinearRegression(lr=0.01,n_iters=1000)
    model.fit(X,y)

    print(f"Weight : {model.weight}")
    print(f"Bias : {model.bias}")



