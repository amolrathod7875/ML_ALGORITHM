

***

### 🧠 A Simple Linear Regression Model from Scratch

This repository contains a simple yet powerful implementation of a linear regression model built entirely from scratch using Python's fundamental libraries. The goal is to provide a clear, educational example of how linear regression works under the hood, powered by the **gradient descent** algorithm.

---

### 🚀 Getting Started

To run this code, you'll need the following libraries. If you don't have them, you can install them with a single command:

`pip install numpy pandas scikit-learn joblib`

* **NumPy**: The backbone for all mathematical and numerical operations.
* **Pandas**: Used for data handling.
* **Scikit-learn**: We use this to generate a sample dataset and split it, saving us from a lot of manual work!
* **Joblib**: For saving and loading our trained model.

---

### How It Works

The core of this project is the `LinearRegression` class, which implements the following key methods:

* **`__init__(self, lr, n_iters)`**: This is where the magic begins! We initialize our model with a **learning rate (`lr`)** and the number of **iterations (`n_iters`)**. The learning rate controls the size of the steps our model takes to learn, while the iterations determine how many times we repeat the learning process.

* **`fit(self, X, y)`**: This is the training method. It's here that the model learns the relationships in your data. It starts with random weights and a bias and then iteratively adjusts them using **gradient descent**. This process minimizes the difference between the model's predictions and the actual data points, making the model more accurate with each step.

* **`predict(self, X)`**: Once the model is trained, this method uses the learned **weights** and **bias** to make predictions on new data. The prediction is calculated using the simple linear equation: $y_{pred} = X \cdot \text{weights} + \text{bias}$.

---

### 💻 Running the Code

The script is ready to run as is! Just execute the file, and it will:

1.  **Generate a sample dataset** 📊 with 500 data points.
2.  **Split the data** into training and testing sets.
3.  **Train the `LinearRegression` model** using the training data.
4.  **Make predictions** on the unseen test data.
5.  **Evaluate the model's performance** by calculating and printing the **R-squared score**. A score closer to 1.0 means the model is a great fit for the data!
6.  **Print the final `weight` and `bias`** values that the model learned.

