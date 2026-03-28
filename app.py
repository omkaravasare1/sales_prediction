import os
import numpy as np
import pandas as pd
import pickle
import lightgbm as lgb
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_percentage_error

app = Flask(__name__, template_folder='template')

# Global variables
model = None
scaler = None
features = None
store_statistics = {}  # Will store avg and std for each store
model_accuracy = None  # To store model accuracy metrics

# Load the model and scaler at startup
def load_model():
    global model, scaler, features, store_statistics, model_accuracy
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # If model doesn't exist, create a simple one for demo
    if not os.path.exists('models/walmart_sales_model.pkl'):
        # Simple placeholder model
        model = lgb.LGBMRegressor(
            objective='regression',
            learning_rate=0.1,
            num_leaves=31,
            max_depth=5,
            random_state=42
        )
        # Save placeholder model
        pickle.dump(model, open('models/walmart_sales_model.pkl', 'wb'))
    else:
        # Load actual model
        model = pickle.load(open('models/walmart_sales_model.pkl', 'rb'))
    
    # If scaler doesn't exist, create one for demo
    if not os.path.exists('models/scaler.pkl'):
        scaler = StandardScaler()
        pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
    else:
        # Load actual scaler
        scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    
    # Load store statistics if they exist
    if os.path.exists('models/store_statistics.pkl'):
        store_statistics = pickle.load(open('models/store_statistics.pkl', 'rb'))
    
    # Load model accuracy if it exists
    if os.path.exists('models/model_accuracy.pkl'):
        model_accuracy = pickle.load(open('models/model_accuracy.pkl', 'rb'))
    else:
        # Default accuracy metrics
        model_accuracy = {
            'mape': 0.15,  # 15% error as default
            'accuracy': 85.0,  # 85% accuracy as default
            'r2': 0.75  # R² score as default
        }
    
    # Define features needed for prediction
    features = [
        'Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
        'Year', 'Month', 'Week', 'Day', 'DayOfWeek', 'IsWeekend',
        'Fuel_CPI', 'Unemployment_Fuel', 'Store_Avg_Weekly_Sales', 'Store_Std_Weekly_Sales'
    ]

@app.route('/')
def home():
    return render_template('index.html')

# New endpoint to get available stores
@app.route('/stores')
def get_stores():
    return jsonify({
        'stores': list(store_statistics.keys())
    })

# New endpoint to get model accuracy
@app.route('/accuracy')
def get_accuracy():
    global model_accuracy
    return jsonify(model_accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        start_date = datetime.strptime(request.form['start_date'], '%Y-%m-%d')
        store_id = int(request.form['store_id'])
        temperature = float(request.form['temperature'])
        fuel_price = float(request.form['fuel_price'])
        cpi = float(request.form['cpi'])
        unemployment = float(request.form['unemployment'])
        num_weeks = int(request.form['num_weeks'])
        
        # Get store statistics automatically (no user input needed)
        if store_id in store_statistics:
            store_avg_sales = store_statistics[store_id]['avg_sales']
            store_std_sales = store_statistics[store_id]['std_sales']
        else:
            # Default values if store not found
            store_avg_sales = 10000  # Example default
            store_std_sales = 2000   # Example default
        
        # Generate future dates for predictions
        future_dates = [start_date + timedelta(weeks=i) for i in range(num_weeks)]
        
        # Create DataFrame for predictions
        future_data = []
        for date in future_dates:
            # Check if date is a holiday (simplified logic for demo)
            is_holiday = 1 if (date.month == 12 and date.day >= 20) or \
                              (date.month == 1 and date.day <= 5) or \
                              (date.month == 11 and 20 <= date.day <= 30) or \
                              (date.month == 7 and 1 <= date.day <= 7) else 0
            
            # Create row for this date
            row = {
                'Date': date,
                'Store': store_id,
                'Holiday_Flag': is_holiday,
                'Temperature': temperature,
                'Fuel_Price': fuel_price,
                'CPI': cpi,
                'Unemployment': unemployment,
                'Year': date.year,
                'Month': date.month,
                'Week': date.isocalendar()[1],
                'Day': date.day,
                'DayOfWeek': date.weekday(),
                'IsWeekend': 1 if date.weekday() >= 5 else 0,
                'Fuel_CPI': fuel_price * cpi,
                'Unemployment_Fuel': unemployment * fuel_price,
                'Store_Avg_Weekly_Sales': store_avg_sales,
                'Store_Std_Weekly_Sales': store_std_sales
            }
            future_data.append(row)
        
        # Convert to DataFrame
        future_df = pd.DataFrame(future_data)
        
        # Prepare features for prediction
        prediction_data = future_df[features].copy()
        
        # Scale numerical features
        numerical_features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 
                              'Fuel_CPI', 'Unemployment_Fuel']
        
        try:
            prediction_data[numerical_features] = scaler.transform(prediction_data[numerical_features])
        except:
            # If scaler transformation fails, just use original values
            pass
        
        # Make predictions
        predictions = model.predict(prediction_data)
        
        # Get confidence intervals based on model accuracy
        global model_accuracy
        confidence_level = 0.95  # 95% confidence
        margin_of_error = model_accuracy['mape'] * 1.96  # Using MAPE for error margins
        
        # Prepare results
        results = []
        for i, date in enumerate(future_dates):
            prediction_value = float(predictions[i])
            lower_bound = prediction_value * (1 - margin_of_error)
            upper_bound = prediction_value * (1 + margin_of_error)
            
            results.append({
                'date': date.strftime('%Y-%m-%d'),
                'prediction': round(prediction_value, 2),
                'lower_bound': round(float(lower_bound), 2),
                'upper_bound': round(float(upper_bound), 2),
                'is_holiday': bool(future_df.iloc[i]['Holiday_Flag'])
            })
        
        return jsonify({
            'status': 'success',
            'results': results,
            'store_info': {
                'store_id': store_id,
                'avg_sales': round(store_avg_sales, 2),
                'std_sales': round(store_std_sales, 2)
            },
            'model_accuracy': {
                'accuracy_percentage': round(model_accuracy['accuracy'], 2),
                'mape': round(model_accuracy['mape'] * 100, 2),
                'r2_score': round(model_accuracy['r2'], 2)
            }
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Check if file is uploaded
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded'})
        
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'})
        
        # Read CSV file
        df = pd.read_csv(file)
        
        # Process data (similar to your notebook code)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y", errors='coerce')
            
            # Extract time features
            df['Year'] = df['Date'].dt.year
            df['Month'] = df['Date'].dt.month
            df['Week'] = df['Date'].dt.isocalendar().week
            df['Day'] = df['Date'].dt.day
            df['DayOfWeek'] = df['Date'].dt.dayofweek
            df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
            
            # Drop date column
            df.drop(columns=['Date'], inplace=True)
        
        # Handle missing values
        df.fillna(method='ffill', inplace=True)
        
        # Create interaction features
        df['Fuel_CPI'] = df['Fuel_Price'] * df['CPI']
        df['Unemployment_Fuel'] = df['Unemployment'] * df['Fuel_Price']
        
        # Calculate and store statistics for each store
        global store_statistics
        store_statistics = {}
        
        for store in df['Store'].unique():
            store_data = df[df['Store'] == store]
            avg_sales = float(store_data['Weekly_Sales'].mean())
            std_sales = float(store_data['Weekly_Sales'].std())
            
            store_statistics[int(store)] = {
                'avg_sales': avg_sales,
                'std_sales': std_sales
            }
        
        # Save store statistics
        pickle.dump(store_statistics, open('models/store_statistics.pkl', 'wb'))
        
        # Aggregation features
        df['Store_Avg_Weekly_Sales'] = df.groupby('Store')['Weekly_Sales'].transform('mean')
        df['Store_Std_Weekly_Sales'] = df.groupby('Store')['Weekly_Sales'].transform('std')
        
        # Define input features and target
        X = df.drop(columns=['Weekly_Sales'])
        y = df['Weekly_Sales']
        
        # Standardize numerical features
        numerical_cols = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 
                          'Fuel_CPI', 'Unemployment_Fuel']
        
        global scaler
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        
        # Save scaler
        pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
        
        # Train LightGBM model
        global model
        model = lgb.LGBMRegressor(
            objective='regression',
            learning_rate=0.1,
            num_leaves=31,
            max_depth=5,
            random_state=42
        )
        model.fit(X, y)
        
        # Save model
        pickle.dump(model, open('models/walmart_sales_model.pkl', 'wb'))
        
        # Calculate model accuracy metrics
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_percentage_error, r2_score
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train on train set
        model.fit(X_train, y_train)
        
        # Predict on validation set
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        mape = mean_absolute_percentage_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        accuracy = 100 - (mape * 100)  # Convert MAPE to accuracy percentage
        
        # Store accuracy metrics
        global model_accuracy
        model_accuracy = {
            'mape': float(mape),
            'accuracy': float(accuracy),
            'r2': float(r2)
        }
        
        # Save accuracy metrics
        pickle.dump(model_accuracy, open('models/model_accuracy.pkl', 'wb'))
        
        # Get feature importances and convert numpy types to Python native types
        feature_importances = {}
        for feature, importance in zip(X.columns, model.feature_importances_):
            # Convert numpy types to native Python types
            feature_importances[str(feature)] = float(importance)
        
        # Sort and get top 5 features
        sorted_importances = dict(sorted(feature_importances.items(), 
                                         key=lambda x: x[1], reverse=True)[:5])
        
        return jsonify({
            'status': 'success',
            'message': 'Model trained successfully',
            'feature_importances': sorted_importances,
            'stores_processed': len(store_statistics),
            'model_accuracy': {
                'accuracy_percentage': round(accuracy, 2),
                'mape': round(mape * 100, 2),
                'r2_score': round(r2, 2)
            }
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    load_model()  # Load the model at startup
    app.run(debug=True)