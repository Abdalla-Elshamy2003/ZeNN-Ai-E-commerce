import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import os
warnings.filterwarnings('ignore')

class DemandForecastingModels:
    def __init__(self):
        self.xgb_model = None
        self.lstm_model = None
        self.scaler = None
        self.model_info = {}
        
    def load_and_prepare_data(self, file_path=None):
        """Load and prepare data"""
        try:
            if file_path and os.path.exists(file_path):
                # Try different Excel reading methods
                try:
                    df = pd.read_excel(file_path)
                    print(f"✅ Data loaded successfully from {file_path}")
                except Exception as e:
                    print(f"⚠️ Error reading Excel file: {e}")
                    print("🔄 Trying alternative reading methods...")
                    try:
                        df = pd.read_excel(file_path, engine='openpyxl')
                        print(f"✅ Data loaded successfully using openpyxl engine")
                    except:
                        try:
                            df = pd.read_excel(file_path, engine='xlrd')
                            print(f"✅ Data loaded successfully using xlrd engine")
                        except:
                            raise Exception("Could not read Excel file with any engine")
            else:
                print("⚠️ File not found or path not provided. Creating sample data...")
                # Create sample data
                dates = pd.date_range(start='2019-12-01', end='2021-12-31', freq='D')
                sample_data = []
                for date in dates:
                    for hour in range(9, 22):  # 9 AM to 9 PM
                        demand = np.random.normal(6, 2.5)  # Random demand around 6 kg
                        sample_data.append({
                            'DOB': date,
                            'DAY_NAME': date.strftime('%A'),
                            'Hour': hour,
                            'Demand in Kgs': max(0, demand)
                        })
                df = pd.DataFrame(sample_data)
                print("✅ Sample data created successfully")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
        
        # Display data info
        print(f"\n📊 Dataset Info:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        # Data preprocessing
        df['DOB'] = pd.to_datetime(df['DOB'])
        df['timestamp'] = df['DOB'] + pd.to_timedelta(df['Hour'], unit='h')
        df['Year'] = df['DOB'].dt.year
        df['Month'] = df['DOB'].dt.month
        df['Quarter'] = df['DOB'].dt.quarter
        df['weekday'] = df['DOB'].dt.day_name()
        df['AM_PM'] = df['Hour'].apply(lambda x: 'AM' if x < 12 else 'PM')
        df['Hour_12'] = df['Hour'].apply(lambda x: 12 if x == 12 or x == 0 else x % 12)
        df['Hour_AMPM'] = df['Hour_12'].astype(str) + ' ' + df['AM_PM']
        df['Quarter_Year'] = 'Q' + df['Quarter'].astype(str) + ' ' + df['Year'].astype(str)
        
        print(f"\n📅 Date range: {df['DOB'].min()} to {df['DOB'].max()}")
        print(f"📈 Demand statistics:")
        print(df['Demand in Kgs'].describe())
        
        return df
    
    def prepare_xgb_data(self, df):
        """Prepare data for XGBoost model"""
        print("\n🔄 Preparing XGBoost data...")
        df_daily = df.set_index('timestamp').resample('D')['Demand in Kgs'].sum()
        df_xgb = df_daily.to_frame()
        
        # Feature engineering
        df_xgb['dayofweek'] = df_xgb.index.dayofweek
        df_xgb['is_weekend'] = df_xgb['dayofweek'].isin([5, 6]).astype(int)
        df_xgb['month'] = df_xgb.index.month
        df_xgb['quarter'] = df_xgb.index.quarter
        df_xgb['lag_1'] = df_xgb['Demand in Kgs'].shift(1)
        df_xgb['lag_2'] = df_xgb['Demand in Kgs'].shift(2)
        df_xgb['lag_7'] = df_xgb['Demand in Kgs'].shift(7)
        df_xgb['rolling_mean_3'] = df_xgb['Demand in Kgs'].shift(1).rolling(window=3).mean()
        df_xgb['rolling_mean_7'] = df_xgb['Demand in Kgs'].shift(1).rolling(window=7).mean()
        df_xgb['rolling_std_7'] = df_xgb['Demand in Kgs'].shift(1).rolling(window=7).std()
        
        df_xgb.dropna(inplace=True)
        print(f"✅ XGBoost data prepared: {df_xgb.shape}")
        
        return df_xgb
    
    def train_xgb_model(self, df_xgb):
        """Train XGBoost model"""
        print("\n🚀 Training XGBoost model...")
        
        feature_columns = ['dayofweek', 'is_weekend', 'month', 'quarter', 
                          'lag_1', 'lag_2', 'lag_7', 'rolling_mean_3', 
                          'rolling_mean_7', 'rolling_std_7']
        
        X = df_xgb[feature_columns]
        y = df_xgb['Demand in Kgs']
        
        # Split data (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")
        
        # Train model
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        self.xgb_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = self.xgb_model.predict(X_train)
        y_pred_test = self.xgb_model.predict(X_test)
        
        # Calculate metrics
        train_metrics = self.calculate_metrics(y_train, y_pred_train)
        test_metrics = self.calculate_metrics(y_test, y_pred_test)
        
        self.model_info['xgb'] = {
            'feature_columns': feature_columns,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': dict(zip(feature_columns, self.xgb_model.feature_importances_))
        }
        
        print("✅ XGBoost model trained successfully!")
        print(f"📊 Training Metrics - MAE: {train_metrics['MAE']:.2f}, RMSE: {train_metrics['RMSE']:.2f}, MAPE: {train_metrics['MAPE']:.2f}%")
        print(f"📊 Test Metrics - MAE: {test_metrics['MAE']:.2f}, RMSE: {test_metrics['RMSE']:.2f}, MAPE: {test_metrics['MAPE']:.2f}%")
        
        # Feature importance
        print(f"\n🏆 Top 5 Feature Importances:")
        feature_imp = sorted(self.model_info['xgb']['feature_importance'].items(), 
                           key=lambda x: x[1], reverse=True)
        for feature, importance in feature_imp[:5]:
            print(f"  {feature}: {importance:.3f}")
        
        return y_test, y_pred_test
    
    def prepare_lstm_data(self, df, window_size=24):
        """Prepare data for LSTM model"""
        print(f"\n🔄 Preparing LSTM data with window size: {window_size}...")
        
        df_hourly = df.set_index('timestamp').sort_index()['Demand in Kgs']
        
        # Initialize and fit scaler
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(df_hourly.values.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - window_size):
            X.append(scaled_data[i:i+window_size])
            y.append(scaled_data[i+window_size])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        print(f"✅ LSTM data prepared: X shape {X.shape}, y shape {y.shape}")
        
        return X, y, window_size
    
    def train_lstm_model(self, X, y, window_size):
        """Train LSTM model"""
        print(f"\n🚀 Training LSTM model...")
        
        # Split data (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")
        
        # Build LSTM model
        self.lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(window_size, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        
        self.lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        print("🏗️ LSTM Model Architecture:")
        self.lstm_model.summary()
        
        # Early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train model
        print("\n🎯 Starting LSTM training...")
        history = self.lstm_model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Make predictions
        y_pred_train = self.lstm_model.predict(X_train, verbose=0)
        y_pred_test = self.lstm_model.predict(X_test, verbose=0)
        
        # Inverse transform
        y_train_inv = self.scaler.inverse_transform(y_train)
        y_test_inv = self.scaler.inverse_transform(y_test)
        y_pred_train_inv = self.scaler.inverse_transform(y_pred_train)
        y_pred_test_inv = self.scaler.inverse_transform(y_pred_test)
        
        # Calculate metrics
        train_metrics = self.calculate_metrics(y_train_inv, y_pred_train_inv)
        test_metrics = self.calculate_metrics(y_test_inv, y_pred_test_inv)
        
        self.model_info['lstm'] = {
            'window_size': window_size,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'history': history.history
        }
        
        print("✅ LSTM model trained successfully!")
        print(f"📊 Training Metrics - MAE: {train_metrics['MAE']:.2f}, RMSE: {train_metrics['RMSE']:.2f}, MAPE: {train_metrics['MAPE']:.2f}%")
        print(f"📊 Test Metrics - MAE: {test_metrics['MAE']:.2f}, RMSE: {test_metrics['RMSE']:.2f}, MAPE: {test_metrics['MAPE']:.2f}%")
        
        return y_test_inv, y_pred_test_inv
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Handle division by zero in MAPE calculation
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = 0
        
        return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    
    def save_models(self, models_dir='models'):
        """Save all trained models and components"""
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        print(f"\n💾 Saving models to {models_dir}/...")
        
        # Save XGBoost model
        if self.xgb_model is not None:
            joblib.dump(self.xgb_model, f'{models_dir}/xgb_model.pkl')
            print("✅ XGBoost model saved")
        
        # Save LSTM model
        if self.lstm_model is not None:
            self.lstm_model.save(f'{models_dir}/lstm_model.h5')
            print("✅ LSTM model saved")
        
        # Save scaler
        if self.scaler is not None:
            joblib.dump(self.scaler, f'{models_dir}/scaler.pkl')
            print("✅ Scaler saved")
        
        # Save model info
        with open(f'{models_dir}/model_info.pkl', 'wb') as f:
            pickle.dump(self.model_info, f)
        print("✅ Model info saved")
        
        print("🎉 All models and components saved successfully!")
    
    def generate_sample_forecasts(self, df):
        """Generate sample forecasts for testing"""
        print("\n🔮 Generating sample forecasts...")
        
        # XGBoost forecast (next 7 days)
        if self.xgb_model is not None:
            print("\n📈 XGBoost Forecasting (Next 7 days):")
            df_xgb = self.prepare_xgb_data(df)
            last_row = df_xgb.iloc[-1].copy()
            
            xgb_forecasts = []
            for i in range(7):
                # Prepare features for next day
                features = last_row[self.model_info['xgb']['feature_columns']].values.reshape(1, -1)
                forecast = self.xgb_model.predict(features)[0]
                xgb_forecasts.append(forecast)
                
                # Update features for next iteration
                last_row['lag_2'] = last_row['lag_1']
                last_row['lag_1'] = forecast
            
            for i, forecast in enumerate(xgb_forecasts, 1):
                print(f"  Day {i}: {forecast:.2f} kg")
        
        # LSTM forecast (next 24 hours)
        if self.lstm_model is not None and self.scaler is not None:
            print("\n🧠 LSTM Forecasting (Next 24 hours):")
            df_hourly = df.set_index('timestamp').sort_index()['Demand in Kgs']
            window_size = self.model_info['lstm']['window_size']
            
            # Get last window_size values
            last_sequence = df_hourly.values[-window_size:].reshape(-1, 1)
            last_sequence_scaled = self.scaler.transform(last_sequence)
            
            lstm_forecasts = []
            current_sequence = last_sequence_scaled.copy()
            
            for i in range(24):
                # Reshape for prediction
                X_pred = current_sequence.reshape(1, window_size, 1)
                forecast_scaled = self.lstm_model.predict(X_pred, verbose=0)[0]
                forecast = self.scaler.inverse_transform(forecast_scaled.reshape(-1, 1))[0, 0]
                lstm_forecasts.append(forecast)
                
                # Update sequence
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = forecast_scaled
            
            # Display first 12 hours
            for i, forecast in enumerate(lstm_forecasts[:12], 1):
                print(f"  Hour {i}: {forecast:.2f} kg")
            print(f"  ... (showing first 12 hours)")

def main():
    """Main training function"""
    print("🚀 Starting Demand Forecasting Model Training...")
    print("=" * 60)
    
    # Initialize model trainer
    trainer = DemandForecastingModels()
    
    # Load and prepare data
    print("📁 Looking for Sample_data.xlsx...")
    df = trainer.load_and_prepare_data("Sample_data.xlsx")
    if df is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Train XGBoost model
    print("\n" + "=" * 60)
    print("📊 TRAINING XGBOOST MODEL")
    print("=" * 60)
    
    try:
        df_xgb = trainer.prepare_xgb_data(df)
        xgb_y_test, xgb_y_pred = trainer.train_xgb_model(df_xgb)
    except Exception as e:
        print(f"❌ Error training XGBoost model: {e}")
        return
    
    # Train LSTM model
    print("\n" + "=" * 60)
    print("🧠 TRAINING LSTM MODEL")
    print("=" * 60)
    
    try:
        lstm_X, lstm_y, window_size = trainer.prepare_lstm_data(df)
        lstm_y_test, lstm_y_pred = trainer.train_lstm_model(lstm_X, lstm_y, window_size)
    except Exception as e:
        print(f"❌ Error training LSTM model: {e}")
        # Continue even if LSTM fails
    
    # Save all models
    print("\n" + "=" * 60)
    print("💾 SAVING MODELS")
    print("=" * 60)
    
    trainer.save_models()
    
    # Generate sample forecasts
    print("\n" + "=" * 60)
    print("🔮 SAMPLE FORECASTS")
    print("=" * 60)
    
    trainer.generate_sample_forecasts(df)
    
    print("\n" + "🎉" * 20)
    print("SUCCESS! Training completed successfully!")
    print("🎉" * 20)
    
    # Print summary
    print("\n📋 TRAINING SUMMARY:")
    print("-" * 40)
    if 'xgb' in trainer.model_info:
        xgb_metrics = trainer.model_info['xgb']['test_metrics']
        print(f"✅ XGBoost - MAE: {xgb_metrics['MAE']:.2f}, RMSE: {xgb_metrics['RMSE']:.2f}, MAPE: {xgb_metrics['MAPE']:.2f}%")
    
    if 'lstm' in trainer.model_info:
        lstm_metrics = trainer.model_info['lstm']['test_metrics']
        print(f"✅ LSTM - MAE: {lstm_metrics['MAE']:.2f}, RMSE: {lstm_metrics['RMSE']:.2f}, MAPE: {lstm_metrics['MAPE']:.2f}%")
    
    print("\n📁 Files saved in 'models/' directory:")
    print("  - xgb_model.pkl (XGBoost model)")
    print("  - lstm_model.h5 (LSTM model)")  
    print("  - scaler.pkl (Data scaler)")
    print("  - model_info.pkl (Model information)")
    
    print("\n✅ Models are ready for deployment!")

if __name__ == "__main__":
    main()