# RealEstate AI — House Price Prediction System

## 1. Project Overview

RealEstate AI is an end-to-end machine learning application that predicts residential property prices based on property characteristics.

The project combines:

* Data cleaning
* Exploratory Data Analysis (EDA)
* Feature engineering
* Machine learning
* Model evaluation
* Model improvement
* Model serialization
* FastAPI backend
* Pydantic validation
* HTML/CSS/JavaScript frontend
* Frontend–backend integration
* Automated API testing

The final system allows a user to enter property information through a web interface and receive a predicted property price.

---

# 2. Project Objective

The main objective is to build a complete machine learning application that can:

1. Process a large real-estate dataset.
2. Clean and analyze the data.
3. Engineer useful features.
4. Train a house-price prediction model.
5. Evaluate the model.
6. Improve the baseline model.
7. Save the trained model.
8. Expose the model through a REST API.
9. Connect the API to a web frontend.
10. Return real-time property price predictions.

---

# 3. Technology Stack

## Machine Learning

* Python
* NumPy
* Pandas
* Scikit-learn
* Joblib

## Data Analysis

* Jupyter Notebook
* Matplotlib
* Seaborn

## Backend

* FastAPI
* Pydantic
* Uvicorn

## Frontend

* HTML
* CSS
* JavaScript

## Testing

* Pytest
* FastAPI TestClient
* HTTPX

---

# 4. Dataset

The project uses a real-estate dataset containing approximately 2.2 million property records.

Dataset file:

```text
data/realtor-data.zip.csv
```

Original dataset shape:

```text
2,226,382 rows
12 columns
```

The main columns are:

| Column         | Description         |
| -------------- | ------------------- |
| brokered_by    | Broker identifier   |
| status         | Property status     |
| price          | Property price      |
| bed            | Number of bedrooms  |
| bath           | Number of bathrooms |
| acre_lot       | Lot size in acres   |
| street         | Street identifier   |
| city           | Property city       |
| state          | Property state      |
| zip_code       | ZIP code            |
| house_size     | Property size       |
| prev_sold_date | Previous sale date  |

---

# 5. Data Cleaning

A copy of the original dataset was created before cleaning:

```python
df_clean = df.copy()
```

## 5.1 Missing Target Values

Rows without a property price were removed because `price` is the target variable.

```python
df_clean = df_clean.dropna(subset=["price"])
```

## 5.2 Invalid Prices

Properties with zero or negative prices were removed.

```python
df_clean = df_clean[df_clean["price"] > 0]
```

## 5.3 Remaining Missing Values

All remaining missing rows were removed for the initial model:

```python
df_clean = df_clean.dropna()
```

Final cleaned dataset:

```text
Rows: 1,084,909
Columns: 12
Missing values: 0
Duplicate rows: 0
```

## 5.4 Logical Validation

The following conditions were checked:

* Bedrooms must be positive.
* Bathrooms must be positive.
* Lot size cannot be negative.
* House size must be positive.
* Price must be positive.

No invalid values remained for these checks.

---

# 6. Outlier Analysis

Outliers were investigated using the Interquartile Range (IQR) method.

For example, the price boundaries were calculated using:

```text
Q1 = 240,000
Q3 = 600,000
IQR = 360,000
Upper Bound = 1,140,000
```

There were many properties above this threshold.

However, inspection showed that many high-priced properties represented legitimate luxury properties.

Therefore, the project did not automatically remove all statistical outliers.

This is important because blindly removing high-priced properties could remove genuine real-estate examples and reduce the model's ability to represent the real market.

---

# 7. Exploratory Data Analysis

EDA was performed to understand the structure and relationships within the dataset.

The following analyses were performed:

## 7.1 Price Distribution

A histogram was used to examine the distribution of property prices.

The dataset showed a strongly right-skewed price distribution.

## 7.2 Feature Distributions

Distributions were analyzed for:

* Bedrooms
* Bathrooms
* Lot size
* House size

## 7.3 Feature vs Price Relationships

Scatterplots were created for:

* Bedrooms vs price
* Bathrooms vs price
* House size vs price

These visualizations helped identify relationships between property characteristics and price.

## 7.4 Correlation Analysis

Correlation was calculated between important numerical variables.

For example:

```text
bed ↔ bath       ≈ 0.612
bed ↔ house_size ≈ 0.190
bath ↔ house_size ≈ 0.224
```

Bedrooms and bathrooms showed a relatively strong positive relationship.

---

# 8. Feature Engineering

The initial model used:

```text
bed
bath
acre_lot
house_size
```

The previous sale date was converted into a proper datetime:

```python
df_clean["prev_sold_date"] = pd.to_datetime(
    df_clean["prev_sold_date"]
)
```

The year was extracted:

```python
df_clean["prev_sold_year"] = (
    df_clean["prev_sold_date"].dt.year
)
```

The final numerical features were:

```text
bed
bath
acre_lot
house_size
prev_sold_year
```

Categorical features were later added:

```text
status
state
```

---

# 9. Train/Test Split

The dataset was divided into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

The split used:

```text
80% training
20% testing
```

`random_state=42` was used to make the split reproducible.

---

# 10. Baseline Model

The first machine learning model was Linear Regression.

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

The model learns relationships between the input features and property price.

Conceptually:

```text
Property Features
       ↓
Linear Regression
       ↓
Predicted Price
```

---

# 11. Model Evaluation

The baseline model was evaluated using three metrics.

## Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted prices.

Baseline result:

```text
MAE ≈ $341,916
```

## Root Mean Squared Error (RMSE)

RMSE gives greater importance to large prediction errors.

Baseline result:

```text
RMSE ≈ $941,328
```

## R² Score

R² measures how much variation in the target is explained by the model.

Baseline result:

```text
R² ≈ 0.199
```

This means the baseline model explained approximately 19.9% of the variation in property prices.

---

# 12. Model Improvement

Several approaches were investigated.

## 12.1 Log Transformation

Log transformations were tested on:

* acre_lot
* house_size
* price

The resulting model performed extremely poorly after inverse transformation.

Therefore, this experiment was discarded.

This demonstrates an important machine-learning principle:

> A technique should be evaluated using actual model performance rather than assumed to be beneficial.

---

# 13. Improved Linear Regression

Additional categorical information was added:

```text
status
state
```

Because machine learning models cannot directly use text categories, One-Hot Encoding was applied.

```python
OneHotEncoder(handle_unknown="ignore")
```

A `ColumnTransformer` was used to process numerical and categorical features.

The complete preprocessing and model were combined into a Pipeline.

Conceptually:

```text
Numerical Features
       ↓
Passthrough
       │
       ├────→ ColumnTransformer
       │
Categorical Features
       ↓
One-Hot Encoding
       │
       ↓
Linear Regression
```

---

# 14. Improved Model Results

The improved model achieved:

| Metric | Basic Linear Regression | Improved Linear Regression |
| ------ | ----------------------: | -------------------------: |
| MAE    |              341,915.61 |                 314,266.79 |
| RMSE   |              941,328.01 |                 909,373.22 |
| R²     |                   0.199 |                      0.252 |

The improved model became the project's final selected model.

The improvement included:

* Lower MAE
* Lower RMSE
* Higher R²

The final R² was approximately:

```text
0.252
```

---

# 15. Why the Entire Pipeline Was Saved

The final model was saved using Joblib:

```python
joblib.dump(
    improved_model,
    "../models/real_estate_price_model.pkl"
)
```

The entire Pipeline was saved rather than only the Linear Regression model.

This is important because the pipeline contains:

* Numerical feature handling
* One-Hot Encoding
* Categorical preprocessing
* Linear Regression

Therefore, the same preprocessing used during training is automatically applied during prediction.

---

# 16. Model Verification

The saved model was loaded again:

```python
loaded_model = joblib.load(
    "../models/real_estate_price_model.pkl"
)
```

A sample prediction was then generated to verify that the saved model worked correctly.

---

# 17. FastAPI Backend

FastAPI was selected to expose the machine learning model through an API.

The main API file is:

```text
api/main.py
```

The API provides:

```text
GET  /
GET  /health
POST /predict
```

---

# 18. Pydantic Validation

The API uses a Pydantic schema:

```text
api/schemas.py
```

The input model validates:

```text
bed
bath
acre_lot
house_size
prev_sold_year
status
state
```

For example:

```text
bed > 0
bath > 0
acre_lot >= 0
house_size > 0
1800 <= prev_sold_year <= 2100
```

Invalid data produces a:

```text
422 Unprocessable Entity
```

response.

---

# 19. Prediction API

The prediction endpoint is:

```text
POST /predict
```

Example request:

```json
{
    "bed": 3,
    "bath": 2,
    "acre_lot": 0.25,
    "house_size": 1800,
    "prev_sold_year": 2020,
    "status": "for_sale",
    "state": "California"
}
```

The backend:

1. Receives the JSON request.
2. Validates it with Pydantic.
3. Converts the validated data into a dictionary.
4. Creates a Pandas DataFrame.
5. Passes the DataFrame to the saved ML pipeline.
6. Generates a prediction.
7. Returns the prediction as JSON.

Example response:

```json
{
    "predicted_price": 425000.0,
    "currency": "USD"
}
```

---

# 20. Health Check

The API provides:

```text
GET /health
```

Example response:

```json
{
    "api_status": "running",
    "model_status": "loaded"
}
```

This endpoint provides a simple way to verify that the API and model are available.

---

# 21. Frontend

The frontend is located in:

```text
frontend/
├── index.html
├── style.css
└── script.js
```

## HTML

`index.html` provides:

* Navigation
* Hero section
* Prediction form
* Property input fields
* Prediction result section
* Project information
* Technology information
* Footer

## CSS

`style.css` provides:

* Responsive layout
* Modern UI
* Cards
* Buttons
* Forms
* Animations and transitions
* Mobile responsiveness

## JavaScript

`script.js` handles:

* Form submission
* Input collection
* Data conversion
* API requests
* Response handling
* Error handling
* Price formatting
* DOM updates

---

# 22. Frontend–FastAPI Connection

The frontend communicates with FastAPI using JavaScript `fetch()`.

The main request is:

```javascript
const response = await fetch("/predict", {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify(propertyData)
});
```

The connection works as follows:

```text
HTML Form
    ↓
JavaScript
    ↓
fetch()
    ↓
POST /predict
    ↓
FastAPI
    ↓
Pydantic Validation
    ↓
ML Pipeline
    ↓
Prediction
    ↓
JSON Response
    ↓
JavaScript
    ↓
HTML Result
```

---

# 23. JSON Data Flow

Frontend creates a JavaScript object:

```javascript
const propertyData = {
    bed: 3,
    bath: 2,
    acre_lot: 0.25,
    house_size: 1800,
    prev_sold_year: 2020,
    status: "for_sale",
    state: "California"
};
```

`JSON.stringify()` converts it into JSON.

FastAPI receives the JSON request.

After prediction, FastAPI returns JSON.

JavaScript uses:

```javascript
const result = await response.json();
```

to convert the response back into a JavaScript object.

The predicted value can then be displayed using:

```javascript
predictedPrice.textContent = formattedPrice;
```

---

# 24. CORS

During development, CORS was configured to allow the frontend to communicate with the FastAPI backend.

Allowed local origins include:

```text
http://127.0.0.1:5500
http://localhost:5500
```

Later, when frontend and backend are served from the same FastAPI application, the frontend can use:

```text
/predict
```

instead of a separate backend URL.

---

# 25. Serving the Frontend Through FastAPI

The final setup allows FastAPI to serve the frontend directly.

The root route returns:

```text
frontend/index.html
```

Static frontend resources are served through:

```text
/static
```

Therefore, the complete application can run through one server.

Start the application with:

```bash
python -m uvicorn api.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

---

# 26. Automated Testing

Automated tests are stored in:

```text
tests/test_api.py
```

The project tests:

1. Home endpoint
2. Health endpoint
3. Valid prediction
4. Invalid prediction

Tests are executed using:

```bash
python -m pytest -v
```

Expected successful test structure:

```text
test_home PASSED
test_health_check PASSED
test_prediction PASSED
test_invalid_prediction PASSED
```

---

# 27. Error Handling

The application handles errors at multiple levels.

## Frontend

JavaScript uses:

```text
try
catch
finally
```

to handle:

* API errors
* Network errors
* Invalid responses

## Backend

FastAPI returns:

```text
422
```

for validation errors.

Unexpected prediction errors are handled with:

```text
500 Internal Server Error
```

while the actual error is logged on the backend.

---

# 28. Debugging

## Browser Console

Open:

```text
F12 → Console
```

Useful for identifying JavaScript errors.

## Network Tab

Open:

```text
F12 → Network → predict
```

Check:

```text
Request URL
Request Method
Status Code
Payload
Response
```

## FastAPI Logs

The terminal running Uvicorn shows:

* Server startup
* Requests
* Model errors
* Prediction errors

## Automated Tests

Run:

```bash
python -m pytest -v
```

to quickly detect API regressions.

---

# 29. Final Project Structure

```text
RealEstate AI/
│
├── api/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   └── schemas.py
│
├── data/
│   └── realtor-data.zip.csv
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── models/
│   └── real_estate_price_model.pkl
│
├── notebooks/
│   └── real_estate_ml.ipynb
│
├── tests/
│   └── test_api.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 30. Complete System Architecture

```text
                    REAL ESTATE AI
                         │
                         ▼
                    Raw Dataset
                         │
                         ▼
                  Data Cleaning
                         │
                         ▼
                        EDA
                         │
                         ▼
                Feature Engineering
                         │
                         ▼
                  Train/Test Split
                         │
                         ▼
                 Linear Regression
                         │
                         ▼
                  Model Evaluation
                         │
                         ▼
                 Model Improvement
                         │
                         ▼
                 Final ML Pipeline
                         │
                         ▼
              real_estate_price_model.pkl
                         │
                         ▼
                    FastAPI API
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
          Pydantic              Prediction
          Validation             Endpoint
              │                     │
              └──────────┬──────────┘
                         ▼
                    JSON Response
                         │
                         ▼
               HTML/CSS/JavaScript
                         │
                         ▼
                  User Interface
                         │
                         ▼
                Predicted House Price
```

---

# 31. Key Machine Learning Concepts Used

The project implemented the following concepts:

| Concept                | Purpose                                            |
| ---------------------- | -------------------------------------------------- |
| Data Cleaning          | Remove unusable records                            |
| Missing Value Handling | Create a complete training dataset                 |
| Outlier Analysis       | Understand extreme observations                    |
| EDA                    | Understand dataset patterns                        |
| Correlation            | Analyze numerical relationships                    |
| Feature Engineering    | Create useful model inputs                         |
| Train/Test Split       | Separate training and evaluation data              |
| Linear Regression      | Predict property prices                            |
| One-Hot Encoding       | Convert categorical data into numerical features   |
| ColumnTransformer      | Apply different preprocessing to different columns |
| Pipeline               | Combine preprocessing and model                    |
| MAE                    | Measure average absolute error                     |
| RMSE                   | Penalize large errors                              |
| R²                     | Measure explained variance                         |
| Model Serialization    | Save the trained model                             |
| FastAPI                | Expose the model through an API                    |
| Pydantic               | Validate API input                                 |
| JSON                   | Transfer data between frontend and backend         |
| Pytest                 | Automatically test the API                         |

---

# 32. Final Model

The final selected model is an improved Linear Regression pipeline containing:

### Numerical features

```text
bed
bath
acre_lot
house_size
prev_sold_year
```

### Categorical features

```text
status
state
```

### Preprocessing

```text
Numerical → passthrough
Categorical → One-Hot Encoding
```

### Model

```text
Linear Regression
```

The final model achieved approximately:

```text
MAE  = 314,266.79
RMSE = 909,373.22
R²   = 0.252
```

These metrics describe performance on the project's held-out test set and should not be interpreted as a guarantee of prediction accuracy for every future property.

---

# 33. How to Run the Project

Activate the virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

Start FastAPI:

```powershell
python -m uvicorn api.main:app --reload
```

Open the application:

```text
http://127.0.0.1:8000
```

Open API documentation:

```text
http://127.0.0.1:8000/docs
```

Run automated tests:

```powershell
python -m pytest -v
```

---

# 34. Project Learning Outcome

This project demonstrates the complete lifecycle of a machine learning application:

```text
Raw Data
   ↓
Clean Data
   ↓
Explore Data
   ↓
Engineer Features
   ↓
Train Model
   ↓
Evaluate Model
   ↓
Improve Model
   ↓
Save Model
   ↓
Build API
   ↓
Validate Requests
   ↓
Connect Frontend
   ↓
Serve Predictions
   ↓
Test Application
```

The project therefore goes beyond simply training a machine learning model. It demonstrates how a trained model can be converted into a usable software application.

---

# 35. Future Improvements

Possible future improvements include:

* More advanced regression models such as Random Forest, Gradient Boosting, XGBoost, or HistGradientBoosting.
* Hyperparameter tuning.
* Better treatment of highly skewed numerical features.
* Controlled handling of high-cardinality city information.
* Cross-validation.
* Feature importance analysis.
* Prediction confidence or uncertainty estimates.
* Database integration.
* User authentication.
* Cloud deployment.
* Docker containerization.
* CI/CD.
* Production logging and monitoring.
* Model versioning.
* Automated model retraining.
* More comprehensive API tests.

These improvements are separate from the current completed implementation.
  