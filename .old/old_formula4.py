import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import QuantileTransformer
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')

class ShortTermPredictor:
    def __init__(self, symbol, minutes):
        """Initialize the ShortTermPredictor with a stock symbol and prediction interval.

        Args:
            symbol (str): Stock ticker symbol (e.g., 'AAPL').
            minutes (int): Time interval in minutes for predictions (e.g., 5).
        """
        self.symbol = symbol
        self.minutes = minutes
        self.pipeline = Pipeline([
            ('under', RandomUnderSampler(random_state=42)),
            ('scaler', QuantileTransformer(output_distribution='normal', random_state=42)),
            ('classifier', XGBClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=5,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            ))
        ])
        self.feature_columns = None
        self.HIGH_CONFIDENCE_THRESHOLD = 0.70
        self.LOW_CONFIDENCE_THRESHOLD = 0.55

    def prepare_features(self, df):
        """Prepare features and target variable from raw stock data.

        Args:
            df (pd.DataFrame): Raw OHLCV data from yfinance.

        Returns:
            tuple: (features DataFrame, target Series)
        """
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
        """Train the model using historical data."""
        try:
            # Download historical data with more history (30 days)
            end = datetime.now()
            start = end - timedelta(days=30)
            df = yf.download(self.symbol, start=start, end=end, interval=f'{self.minutes}m')
            logging.info(f"Downloaded {len(df)} rows of data for {self.symbol}")

            if df.empty:
                raise ValueError("No data downloaded")

            X, y = self.prepare_features(df)

            if len(X) < 100:
                raise ValueError("Insufficient training data")

            # Split the data using TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                # Train the pipeline
                self.pipeline.fit(X_train, y_train)

                # Evaluate
                y_pred = self.pipeline.predict(X_test)
                logging.info("\nModel Performance:")
                logging.info(classification_report(y_test, y_pred))

            # Log feature importances for potential refinement
            feature_importances = self.pipeline.named_steps['classifier'].feature_importances_
            importance_dict = dict(zip(self.feature_columns, feature_importances))
            sorted_importances = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            logging.info("Feature Importances:")
            for feature, importance in sorted_importances:
                logging.info(f"{feature}: {importance:.4f}")

        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise

    def predict_next_movement(self):
        """Predict the next price movement based on recent data.

        Returns:
            tuple: (prediction, probabilities, current_price, timestamp, confidence_level)
        """
        try:
            # Download more historical data for feature calculation (48 hours)
            end = datetime.now()
            start = end - timedelta(hours=48)
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


