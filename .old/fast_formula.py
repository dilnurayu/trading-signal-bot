import logging

import yfinance as yf
from datetime import datetime, timedelta

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import pandas as pd
import warnings


warnings.filterwarnings('ignore')


class ShortTermPredictor:
    def __init__(self, symbol, minutes):
        self.symbol = symbol
        self.minutes = minutes
        self.pipeline = Pipeline([
            ('smote', SMOTE(random_state=42, k_neighbors=3)),  # Reduced k_neighbors
            ('scaler', RobustScaler()),  # Changed to RobustScaler
            ('classifier', GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=4,
                min_samples_leaf=2,
                subsample=0.8,
                random_state=42
            ))
        ])
        self.feature_columns = None
        self.HIGH_CONFIDENCE_THRESHOLD = 0.70  # Increased threshold
        self.LOW_CONFIDENCE_THRESHOLD = 0.55

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

            # Moving averages with different windows
            data['SMA5'] = data['Close'].rolling(window=5).mean()
            data['SMA10'] = data['Close'].rolling(window=10).mean()
            data['SMA20'] = data['Close'].rolling(window=20).mean()
            data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
            data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
            data['MA_crossover'] = data['SMA5'] - data['SMA10']
            data['EMA_crossover'] = data['EMA12'] - data['EMA26']

            # Price volatility
            data['high_low_range'] = data['High'] - data['Low']
            data['close_open_range'] = data['Close'] - data['Open']
            data['ATR'] = data['high_low_range'].rolling(window=14).mean()

            # Volume ratio with different windows
            volume_ma5 = data['Volume'].rolling(window=5).mean()
            volume_ma10 = data['Volume'].rolling(window=10).mean()
            data['volume_ratio'] = data['Volume'].div(volume_ma5).clip(upper=10)
            data['volume_ratio_10'] = data['Volume'].div(volume_ma10).clip(upper=10)

            # RSI with different windows
            for window in [7, 14, 21]:
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                data[f'RSI_{window}'] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            data['bollinger_mid'] = data['Close'].rolling(window=20).mean()
            data['bollinger_std'] = data['Close'].rolling(window=20).std()
            data['bollinger_upper'] = data['bollinger_mid'] + (data['bollinger_std'] * 2)
            data['bollinger_lower'] = data['bollinger_mid'] - (data['bollinger_std'] * 2)
            data['bollinger_width'] = data['bollinger_upper'] - data['bollinger_lower']

            # MACD
            data['MACD'] = data['EMA12'] - data['EMA26']
            data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['MACD_hist'] = data['MACD'] - data['MACD_signal']

            # Stochastic Oscillator
            data['%K'] = (data['Close'] - data['Low'].rolling(window=14).min()) / (
                        data['High'].rolling(window=14).max() - data['Low'].rolling(window=14).min()) * 100
            data['%D'] = data['%K'].rolling(window=3).mean()

            # Rate of Change (ROC)
            data['ROC'] = data['Close'].pct_change(periods=12) * 100

            # On-Balance Volume (OBV)
            data['OBV'] = (data['Volume'] * (~data['Close'].diff().le(0) * 2 - 1)).cumsum()

            # Target (future price movement)
            data['target'] = (data['Close'].shift(-self.minutes) > data['Close']).astype(int)

            # Add lagged features with different lags
            for lag in range(1, 6):
                data[f'returns_lag{lag}'] = data['returns'].shift(lag)
                data[f'vol_lag{lag}'] = data['vol'].shift(lag)
                data[f'RSI_14_lag{lag}'] = data['RSI_14'].shift(lag)
                data[f'MACD_lag{lag}'] = data['MACD'].shift(lag)
                data[f'volume_ratio_lag{lag}'] = data['volume_ratio'].shift(lag)

            # Drop NaN values
            data = data.dropna()

            self.feature_columns = [
                'returns', 'vol', 'momentum', 'MA_crossover', 'EMA_crossover',
                'high_low_range', 'close_open_range', 'ATR', 'volume_ratio',
                'volume_ratio_10', 'RSI_7', 'RSI_14', 'RSI_21', 'bollinger_upper',
                'bollinger_lower', 'bollinger_width', 'MACD', 'MACD_signal',
                'MACD_hist', '%K', '%D', 'ROC', 'OBV',
                'returns_lag1', 'returns_lag2', 'returns_lag3', 'returns_lag4', 'returns_lag5',
                'vol_lag1', 'vol_lag2', 'vol_lag3', 'vol_lag4', 'vol_lag5',
                'RSI_14_lag1', 'RSI_14_lag2', 'RSI_14_lag3', 'RSI_14_lag4', 'RSI_14_lag5',
                'MACD_lag1', 'MACD_lag2', 'MACD_lag3', 'MACD_lag4', 'MACD_lag5',
                'volume_ratio_lag1', 'volume_ratio_lag2', 'volume_ratio_lag3', 'volume_ratio_lag4', 'volume_ratio_lag5'
            ]

            # Check if we have enough data
            if len(data) < 50:
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
            start = end - timedelta(days=15)  # Increased to 7 days
            df = yf.download(self.symbol, start=start, end=end, interval=f'{self.minutes}m')
            logging.info(f"Downloaded {len(df)} rows of data for {self.symbol}")

            if df.empty:
                raise ValueError("No data downloaded")

            X, y = self.prepare_features(df)

            if len(X) < 100:  # Increased minimum training data
                raise ValueError("Insufficient training data")

            # Split the data using TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=2)
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                # Train the pipeline
                self.pipeline.fit(X_train, y_train)

                # Evaluate
                y_pred = self.pipeline.predict(X_test)
                logging.info("\nModel Performance:")
                logging.info(classification_report(y_test, y_pred))

                # Cross-validation for a more robust evaluation
                cv_scores = cross_val_score(self.pipeline, X, y, cv=tscv, scoring='f1')
                logging.info(f"Cross-validation F1 scores: {cv_scores}")
                logging.info(f"Average cross-validation F1 score: {cv_scores.mean()}")

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
            start = end - timedelta(hours=48)  # Increased to 48 hours
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

            # Add prediction validation with refined confidence levels
            confidence_level = "Undefined"
            if max(probabilities) >= self.HIGH_CONFIDENCE_THRESHOLD:
                confidence_level = "High"
            elif max(probabilities) >= self.LOW_CONFIDENCE_THRESHOLD:
                confidence_level = "Medium"
            elif abs(probabilities[0] - probabilities[1]) < 0.1:
                confidence_level = "Very Low"
            else:
                confidence_level = "Low"

            logging.info(f"Prediction: {'Up' if prediction == 1 else 'Down'}, "
                         f"Probabilities: {probabilities}, Confidence: {confidence_level}")

            return prediction, probabilities, recent_data['Close'].iloc[-1], datetime.now(), confidence_level

        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise