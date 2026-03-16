####
---
### Project Summary: Walmart Store Sales Forecasting
---
####

This notebook details the process of forecasting weekly sales for Walmart stores and departments, leveraging a comprehensive dataset that includes historical sales, store information, and various economic indicators. The project involved extensive data preprocessing, feature engineering, exploratory data analysis (EDA), and the development of both machine learning (XGBoost) and time series (SARIMAX) models.

### **1. Data Overview**

The dataset comprises:
- **`stores.csv`**: Anonymized store information (Type and Size).
- **`train.csv`**: Historical training data (2010-02-05 to 2012-11-01) with `Store`, `Dept`, `Date`, `Weekly_Sales`, and `IsHoliday`.
- **`test.csv`**: Identical to `train.csv` but `Weekly_Sales` are withheld for prediction.
- **`features.csv`**: Additional data including `Temperature`, `Fuel_Price`, `MarkDown1-5`, `CPI`, `Unemployment`, and `IsHoliday`.

Key challenge: Modeling promotional markdown events preceding major holidays (Super Bowl, Labor Day, Thanksgiving, Christmas), which are weighted five times higher in evaluation.

### **2. Data Preprocessing & Feature Engineering**

- **Master Merge**: `train` and `test` datasets were concatenated, then merged with `stores` and `features` based on `Store`, `Date`, and `IsHoliday`.
- **Missing Values**: Markdown-related columns (`MarkDown1-5`), `CPI`, and `Unemployment` were initially filled with `0`.
- **Date Conversion**: The `Date` column was converted to datetime objects.
- **Temporal Features**: `Day`, `Month`, `Year`, and `WeekOfYear` were extracted from the `Date`.
- **Days to Holiday**: A `Days_To_Holiday` feature was created to count days until the next major holiday (Christmas dates were used as proxies).
- **Lag Features**: `Sales_Last_Year` was created by shifting `Weekly_Sales` by 52 weeks to capture strong seasonal patterns. Initial missing values were filled with `0`.
- **Categorical Encoding**: The `Type` column (A, B, C) was numerically mapped (1, 2, 3) and `IsHoliday` was converted to integer (0 or 1).
- **Additional Features**:
    - `IsMajorHoliday`: Binary flag for approximate major holiday weeks.
    - `Promo_Intensity`: Sum of all MarkDown values.
    - `Size_Bin`: Categorized `Size` into 'Small', 'Medium', 'Large'.
    - `Size_x_Holiday`: Interaction term between `Size` and `IsHoliday`.
    - `IsWeekend`: Binary flag if the day is a weekend.

### **3. Exploratory Data Analysis (EDA) Insights**

- **Weekly Sales Distribution**: Highly skewed, with a long tail indicating high sales volumes during peak periods, especially holidays. Minimum sales could be negative (returns).
- **Seasonal Patterns**: Strong seasonal trends observed; sales peak in Q4 (November–December) due to holidays and promotions, with lows in Q1.
- **Holiday Impact**: Average sales during holiday weeks (`$17035`) are significantly higher (approx. 20-50%) than non-holiday weeks (`$15901`), confirming their importance.
- **Promotional Impact**: `Promo_Intensity` shows a moderate correlation (r~0.2–0.4) with sales, particularly noticeable in data post-2011.
- **Store Type and Size**: Larger stores (Type A) exhibit significantly higher average sales compared to Type C stores. Store `Size` is a key predictor (r~0.25).
- **Economic Factors**: `Unemployment` negatively impacts sales (r~-0.1), while `Fuel_Price` has a minimal direct effect. `CPI` and `Fuel_Price` show gradual changes over time, contrasting with the more dramatic fluctuations in sales.
- **Top Performers**: Departments 92 and 95 in Store 14 (and others) consistently generate the highest revenue.
- **Time Series Analysis Pre-requisite**: For classical time series models, Store 14, Department 92 was identified as a promising candidate due to its consistent data and significant holiday weeks.

### **4. XGBoost Model Performance**

- **Baseline Model**: An initial XGBoost model was trained and evaluated.
    - Mean Absolute Error (MAE): `3891.06`
    - R-squared Score: `0.9111`
    
- **Hyperparameter Tuning (GridSearchCV)**:
    - Best parameters found: `learning_rate`: 0.1, `max_depth`: 9, `n_estimators`: 500.
    - **Tuned Model Performance**:
        - Mean Absolute Error (MAE): `1592.77`
        - R-squared Score: `0.9806`
    - **Comparison with Baseline**:
        - MAE Improvement: `2674.64`
        - R2 Improvement: `0.0836`

The tuned XGBoost model demonstrated a significant improvement, explaining over 98% of the variance in weekly sales, and its predictions aligned much more closely with actual sales values.

- **Feature Importance (XGBoost)**:
    The feature importance analysis highlighted the most influential factors in predicting weekly sales, with `Sales_Last_Year` likely being a top predictor due to its strong seasonal signal, followed by other temporal and store-specific features.

### **5. SARIMAX Analysis (for Store 14, Department 92)**

- **Stationarity Check**: The `Weekly_Sales` series for Store 14, Dept 92 was found to be non-stationary (ADF p-value > 0.05). First-order differencing (`shift(2)`) was applied to achieve stationarity.
- **SARIMAX Model**: A SARIMAX model was initially fitted with `order=(1,1,1)` and `seasonal_order=(1,0,1,52)` using selected exogenous variables.
    - Validation WMAE (original scale): `21532.16`
    - Regular MAE (original scale): `22241.95`

- **Hyperparameter Tuning (`auto_arima`)**:
    - `auto_arima` was used to find optimal `(p,d,q)(P,D,Q,s)` orders, incorporating exogenous variables.
    - Best SARIMAX order found: `(0, 1, 2)`
    - Best seasonal order found: `(0, 1, 0, 52)`

- **Retuned SARIMAX Model Performance**:
    - The model was refitted with the optimal parameters from `auto_arima`.
    - Retuned Validation WMAE (original scale): `17248.17`
    - Retuned Regular MAE (original scale): `18791.82`
    - **WMAE Improvement**: `4283.98` (compared to the original SARIMAX).

### **6. Conclusion**

The project successfully developed and evaluated models for Walmart sales forecasting. The XGBoost model performed exceptionally well across the entire dataset after hyperparameter tuning, achieving a high R-squared and significantly reducing MAE. The SARIMAX model, applied to a specific store-department combination, also showed strong performance, especially after `auto_arima` tuning, demonstrating the effectiveness of time series methods for individual series forecasting. The combination of strong feature engineering and robust modeling techniques proved effective in capturing the complex patterns of retail sales, including seasonal and holiday-driven fluctuations.
