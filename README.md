# California Housing Price Prediction

This project focuses on predicting housing prices in California using various machine learning models. The analysis covers the entire machine learning workflow: data loading, exploratory data analysis (EDA), target selection, geographic clustering, feature engineering, preprocessing pipelines, model training, evaluation, and model serialization.

---

## Dataset

The model is trained on the California Housing dataset. The dataset includes features such as:
- `longitude` and `latitude` (geographic coordinates)
- `housing_median_age` (median age of houses in the block)
- `total_rooms` and `total_bedrooms`
- `population` and `households`
- `median_income` (median income of households in the block)
- `ocean_proximity` (categorical location category)
- **Target Variable**: `median_house_value` (median house value for households)

---

## Machine Learning Pipeline

The project is structured in the following steps:

1. **Step 1 - Load Data**: Loads the dataset from a local path (`/content/housing.csv`) or falls back to fetching it from `scikit-learn`'s datasets (`fetch_california_housing`).
2. **Step 2 - Exploratory Data Analysis (EDA)**: Analyzes distributions, correlations, and identifies relationships between features.
3. **Step 3 - Choose Target**: Targets `median_house_value` for prediction.
4. **Step 4 - Spatial Region Clustering**: Uses coordinate features (`latitude` and `longitude`) to group properties into geographic region clusters.
5. **Step 5 - Preprocessing Pipelines**:
   - **Numeric Features**: Imputed using `SimpleImputer(strategy='median')` and scaled using `StandardScaler()`.
   - **Categorical Features**: Encoded using `OneHotEncoder(handle_unknown='ignore')`.
   - Combined using `ColumnTransformer`.
6. **Step 6 - Train/Test Split**: Splits the data into training and test sets.
7. **Step 7 - Linear Models**: Trains and evaluates `LinearRegression` and `Ridge` regression.
8. **Step 8 - Random Forest**: Trains a `RandomForestRegressor`.
9. **Step 10 - Results Summary**: Compares the metrics of all trained models.
10. **Step 11 - Save Best Model**: Saves the best-performing model for deployment.

---

## Model Evaluation Results

The models were evaluated using Root Mean Squared Error (RMSE) and the Coefficient of Determination ($R^2$ score) on the test set:

| Model | Test RMSE | Test $R^2$ Score |
|---|---|---|
| **Random Forest Regressor** | **~50,047.8** | **~0.809** |
| **Linear Regression** | ~70,384.4 | ~0.622 |
| **Ridge Regression** | ~70,387.2 | ~0.622 |

### Key Findings
- The **Random Forest Regressor** significantly outperformed the linear models, achieving an $R^2$ score of **80.9%** and reducing the root mean squared error to approximately **$50,048**.
- Linear models (OLS and Ridge) showed similar performance, capturing about **62.2%** of the variance.

---

## Requirements

To run the Jupyter notebook, install the following dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## How to Run

1. Clone this repository.
2. Open `california_housing_price_prediction.ipynb` in Jupyter Notebook or Google Colab.
3. Run the cells sequentially to execute the pipeline, visualize the data, train the models, and inspect the results.
