# 🌿 Tea Production Prediction Model

Welcome to the Tea Production Prediction Model! This project aims to predict tea production based on weather data using a machine learning approach. The model is built using Python and leverages the power of the RandomForestRegressor from the scikit-learn library. 🌦️🍃

## 📁 Project Structure

- **Data Loading**: 📥 The script begins by loading weather data from three CSV files using pandas.
- **Data Filtering**: 🗂️ Filters the data to include only records related to the crop 'Sugarcane'.
- **Data Concatenation**: 🧩 Combines the filtered records into a single dataset.
- **Data Preprocessing**: 🧹 Drops unnecessary columns from the dataset.
- **Feature and Target Separation**: 🏷️ Separates features (X) and the target variable (Y).
- **One-Hot Encoding**: 🔄 Encodes the categorical feature 'Season'.
- **Model Training**: 🏋️ Trains a RandomForestRegressor model using the processed data.
- **Model Evaluation**: 📊 Evaluates the model's performance.
- **Model Saving**: 💾 Saves the trained model using joblib.
- **Making Predictions**: 🔮 Shows how to make predictions with the trained model.

## 🚀 Getting Started

### Prerequisites

Make sure you have the following Python packages installed:

```bash
pip install pandas scikit-learn joblib numpy
```

### Running the Script

1. **Load the data**: The script loads weather data from `weather.csv`, `weather2.csv`, and `weather3.csv`.

    ```python
    exp_data = pd.read_csv('weather.csv')
    exp_data2 = pd.read_csv('weather2.csv')
    exp_data3 = pd.read_csv('weather3.csv')
    ```

2. **Filter and concatenate data**: Filter records for 'Sugarcane' and combine them.

    ```python
    tea_records = exp_data[exp_data['Crop'] == 'Sugarcane']
    tea_records2 = exp_data2[exp_data2['Crop'] == 'Sugarcane']
    tea_records3 = exp_data3[exp_data3['Crop'] == 'Sugarcane']
    combined_tea_records = pd.concat([tea_records, tea_records2, tea_records3], ignore_index=True)
    ```

3. **Preprocess data**: Drop unnecessary columns.

    ```python
    fields_to_drop = ['State_Name', 'District_Name', 'Crop']
    combined_tea_records.drop(columns=fields_to_drop, inplace=True)
    ```

4. **Separate features and target**: Split the data into features (X) and target (Y).

    ```python
    Y = combined_tea_records['Production']
    X = combined_tea_records.drop("Production", axis=1)
    ```

5. **One-Hot Encoding**: Encode the 'Season' column.

    ```python
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer

    categorical_features = ["Season"]
    one_hot = OneHotEncoder()
    transformed = ColumnTransformer([("one_hot", one_hot, categorical_features)], remainder="passthrough")

    transformed_x = transformed.fit_transform(X)
    ```

6. **Train the model**: Train a RandomForestRegressor model.

    ```python
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    np.random.seed(42)
    X_train, X_test, Y_train, Y_test = train_test_split(transformed_x, Y, test_size=0.2)

    model = RandomForestRegressor()
    model.fit(X_train, Y_train)
    ```

7. **Evaluate the model**: Check the model's performance.

    ```python
    model.score(X_test, Y_test)
    ```

8. **Save the model**: Save the trained model for future use.

    ```python
    import joblib

    joblib.dump(model, 'model_tea_weather.pkl')
    ```

9. **Make predictions**: Use the trained model to make predictions.

    ```python
    input_data = (2010, 1, 750, 30, 1500, 80, 90)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = model.predict(input_data_reshaped)
    print(prediction)
    ```

## 📊 Model Details

The model uses `RandomForestRegressor` which is an ensemble learning method for regression tasks. It combines multiple decision trees to improve prediction accuracy and control over-fitting.

### Why RandomForestRegressor? 🌲

- **Ensemble Learning**: Improves prediction accuracy and reduces over-fitting.
- **Bootstrap Aggregation (Bagging)**: Uses different subsets of training data, increasing robustness.
- **Feature Importance**: Provides insights into the most influential features for predictions.

## 🤖 Alternatives

Consider these alternative models for similar tasks:

- **Linear Regression**: For linear relationships.
- **Decision Tree Regressor**: For non-linear relationships.
- **Gradient Boosting Regressor**: High accuracy, robust to overfitting.
- **Support Vector Regressor (SVR)**: Effective in high-dimensional spaces.
- **Neural Networks (MLP Regressor)**: For complex, non-linear relationships.
- **K-Nearest Neighbors Regressor (KNN)**: Simple and intuitive.
- **XGBoost**: Optimized, efficient, and scalable.

## 📝 Summary

- **Data Loading**: 📥 Reading data from CSV files.
- **Data Filtering**: 🗂️ Filtering records for 'Sugarcane'.
- **Data Concatenation**: 🧩 Combining records.
- **Data Preprocessing**: 🧹 Dropping unnecessary columns, encoding features.
- **Model Training**: 🏋️ Training a RandomForestRegressor.
- **Model Evaluation**: 📊 Evaluating performance.
- **Model Saving**: 💾 Saving the model.
- **Making Predictions**: 🔮 Using the model for predictions.

Enjoy predicting tea production with this robust model! 🌿🍃

---

Feel free to customize this README further based on your specific needs!
