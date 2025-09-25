📈 Custom Linear Regression from Scratch
This project implements a simple Linear Regression model from scratch using only NumPy ⚡ for the math. It’s trained with Gradient Descent and evaluated using the R² Score 🔥.

✨ Features
📝 Custom Linear Regression implementation (no pre-built sklearn models)

⚡ Gradient Descent optimization

🔮 Make predictions on new data

📊 Evaluate with R² Score

🧪 Uses synthetic regression dataset (make_regression)

⚙️ Installation
Clone the repository and install dependencies:

bash
git clone https://github.com/amolrathod7875/ML_ALGORITHM.git
cd https://github.com/amolrathod7875/ML_ALGORITHM.git
pip install numpy pandas scikit-learn joblib
📂 Project Structure
text
├── linear_regression.py    # 📜 Main implementation file
├── README.md               # 📖 Documentation
🛠 How It Works
🔧 Initialize → Random weights + bias (0).

📚 Train (fit) → Iteratively update weights & bias with gradient descent.

🔮 Predict (predict) → Use trained values to make predictions.

📊 Evaluate → Measure performance using R² score.

▶️ Usage
Run the script:

bash
python linear_regression.py
Sample output:

text
r2_score : 0.8647329382856575
Weight : [34.812]
Bias : -0.582
💻 Example in Code
python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from linear_regression import LinearRegression

# 🔨 Generate dataset
X, y = datasets.make_regression(n_samples=500, n_features=1, noise=25, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 🤖 Train model
model = LinearRegression(lr=0.01, n_iters=1000)
model.fit(X_train, y_train)

# 🔮 Predictions
predictions = model.predict(X_test)

# 📊 Evaluation
print("R2 Score:", r2_score(y_test, predictions))
print("Weight:", model.weight)
print("Bias:", model.bias)
📦 Requirements
🐍 Python 3.8+

🔢 NumPy

🐼 Pandas

📚 scikit-learn

💾 joblib

Install via:

bash
pip install -r requirements.txt
🚀 Future Improvements
📊 Add MSE and MAE evaluation

🔁 Save & Load model with joblib

🏋️‍♂️ Add regularization (Ridge, Lasso)

🎨 Plot regression line with Matplotlib