import logging

import yfinance as yf
from datetime import datetime, timedelta

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import pandas as pd
import warnings


warnings.filterwarnings('ignore')


class ShortTermPredictor:
    def __init__(self, symbol, minutes):
        self.symbol = symbol
        self.minutes = minutes
        self.pipeline = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            ))
        ])
        self.feature_columns = None
        self.HIGH_CONFIDENCE_THRESHOLD = 0.65

    def prepare_features(self, df):
        try:
            # Create a DataFrame with single-level columns
            data = pd.DataFrame()

            # Handle both multi-index and single-index DataFrames
            if isinstance(df.columns, pd.MultiIndex):
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    data[col] = df[(col, self.symbol)]
            else:
                data = df.copy()

            # Calculate returns
            data['returns'] = data['Close'].pct_change()

            # Calculate technical features
            data['vol'] = data['returns'].rolling(window=5).std()
            data['momentum'] = data['Close'] - data['Close'].shift(5)

            # Moving averages
            data['SMA5'] = data['Close'].rolling(window=5).mean()
            data['SMA10'] = data['Close'].rolling(window=10).mean()
            data['MA_crossover'] = data['SMA5'] - data['SMA10']

            # Price volatility
            data['high_low_range'] = data['High'] - data['Low']
            data['close_open_range'] = data['Close'] - data['Open']

            # Volume ratio
            volume_ma5 = data['Volume'].rolling(window=5).mean()
            data['volume_ratio'] = data['Volume'].div(volume_ma5).clip(upper=10)

            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            data['bollinger_mid'] = data['Close'].rolling(window=20).mean()
            data['bollinger_std'] = data['Close'].rolling(window=20).std()
            data['bollinger_upper'] = data['bollinger_mid'] + (data['bollinger_std'] * 2)
            data['bollinger_lower'] = data['bollinger_mid'] - (data['bollinger_std'] * 2)

            # MACD
            short_ema = data['Close'].ewm(span=12, adjust=False).mean()
            long_ema = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = short_ema - long_ema
            data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

            # Target (future price movement)
            data['target'] = (data['Close'].shift(-self.minutes) > data['Close']).astype(int)

            # Add lagged features
            for lag in range(1, 4):
                data[f'returns_lag{lag}'] = data['returns'].shift(lag)
                data[f'vol_lag{lag}'] = data['vol'].shift(lag)
                data[f'RSI_lag{lag}'] = data['RSI'].shift(lag)

            # Drop NaN values
            data = data.dropna()

            self.feature_columns = [
                'returns', 'vol', 'momentum', 'MA_crossover', 'high_low_range',
                'close_open_range', 'volume_ratio', 'RSI', 'bollinger_upper',
                'bollinger_lower', 'MACD', 'MACD_signal',
                'returns_lag1', 'returns_lag2', 'returns_lag3',
                'vol_lag1', 'vol_lag2', 'vol_lag3',
                'RSI_lag1', 'RSI_lag2', 'RSI_lag3'
            ]

            # Check if we have enough data
            if len(data) < 30:
                raise ValueError("Insufficient data points for reliable prediction")

            # Extract only the features we need
            features = data[self.feature_columns]
            target = data['target']

            logging.info(f"Successfully prepared features. Shape: {features.shape}")
            return features, target

        except Exception as e:
            logging.error(f"Error in feature preparation: {str(e)}")
            raise

    def train_model(self):
        try:
            # Download historical data with more history
            end = datetime.now()
            start = end - timedelta(days=1)  # Increased from 1 to 7 days
            df = yf.download(self.symbol, start=start, end=end, interval=f'{self.minutes}m')
            logging.info(f"Downloaded {len(df)} rows of data for {self.symbol}")

            if df.empty:
                raise ValueError("No data downloaded")

            X, y = self.prepare_features(df)

            if len(X) < 50:  # Ensure minimum training data
                raise ValueError("Insufficient training data")

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False  # Time-series aware split
            )

            # Train the pipeline
            self.pipeline.fit(X_train, y_train)

            # Evaluate
            y_pred = self.pipeline.predict(X_test)
            logging.info("\nModel Performance:")
            logging.info(classification_report(y_test, y_pred))

            # Verify model isn't biased
            pred_proba = self.pipeline.predict_proba(X_test)
            avg_prob = pred_proba.mean(axis=0)
            logging.info(f"Average prediction probabilities: {avg_prob}")

            if abs(avg_prob[0] - avg_prob[1]) > 0.2:  # Check for severe bias
                logging.warning("Model shows significant bias in predictions")

        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise

    def predict_next_movement(self):
        try:
            # Download more historical data for feature calculation
            end = datetime.now()
            start = end - timedelta(hours=24)  # Get 24 hours of data
            recent_data = yf.download(self.symbol, start=start, end=end, interval=f'{self.minutes}m')

            if recent_data.empty:
                raise ValueError("No recent data downloaded")

            X, _ = self.prepare_features(recent_data)

            if len(X) == 0:
                raise ValueError("No valid features generated from recent data")

            latest_features = X.iloc[-1:].values

            # Get prediction and probabilities using the pipeline
            probabilities = self.pipeline.predict_proba(latest_features)[0]
            prediction = 1 if probabilities[1] >= 0.5 else 0

            # Add prediction validation
            if abs(probabilities[0] - probabilities[1]) < 0.1:
                logging.warning("Low prediction confidence - probabilities near 0.5")

            confidence_level = "High" if max(probabilities) >= self.HIGH_CONFIDENCE_THRESHOLD else "Low"

            logging.info(f"Prediction: {'Up' if prediction == 1 else 'Down'}, "
                         f"Probabilities: {probabilities}, Confidence: {confidence_level}")

            return prediction, probabilities, recent_data['Close'].iloc[-1], datetime.now(), confidence_level

        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise