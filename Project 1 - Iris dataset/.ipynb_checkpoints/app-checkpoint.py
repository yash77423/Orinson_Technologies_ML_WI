import streamlit as st
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Load the Iris dataset and preprocess
iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_scaled, y)

# Streamlit app UI
st.title('Iris Flower Classifier')
sepal_length = st.slider('Sepal Length', 4.0, 8.0, 5.0)
sepal_width = st.slider('Sepal Width', 2.0, 5.0, 3.5)
petal_length = st.slider('Petal Length', 1.0, 7.0, 1.4)
petal_width = st.slider('Petal Width', 0.1, 3.0, 0.2)

# Prediction
user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
user_input_scaled = scaler.transform(user_input)
prediction = model.predict(user_input_scaled)
species = iris.target_names[prediction]

# Display result
st.write(f"Predicted species: {species[0]}")
