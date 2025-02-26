import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import QuantileTransformer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')

# Try importing TA-Lib for faster technical indicator computations
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.info("TA-Lib not available, using slower vectorized computations.")

class ShortTermPredictor:
    def __init__(self, symbol, minutes):
        """
        Initialize the ShortTermPredictor with a stock symbol and prediction interval.
        """
        self.symbol = symbol
        self.minutes = minutes
        self.pipeline = Pipeline([
            ('under', RandomUnderSampler(random_state=42)),
            ('scaler', QuantileTransformer(output_distribution='normal', random_state=42)),
            ('classifier', XGBClassifier(
                n_estimators=100,         # Reduced estimators
                learning_rate=0.05,
                max_depth=3,              # Reduced depth
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1,                # Use all CPU cores
                tree_method='hist'        # Fast histogram-based algorithm
            ))
        ])
        self.feature_columns = None
        self.HIGH_CONFIDENCE_THRESHOLD = 0.70
        self.LOW_CONFIDENCE_THRESHOLD = 0.55

    def prepare_features(self, df):
        """
        Prepare features and target variable from raw stock data.
        """
        try:
            # Handle multi-index DataFrame if necessary
            if isinstance(df.columns, pd.MultiIndex):
                data = pd.DataFrame({col: df[(col, self.symbol)] for col in ['Open', 'High', 'Low', 'Close', 'Volume']})
            else:
                data = df.copy()

            # Calculate basic returns and volatility
            data['returns'] = data['Close'].pct_change()
            data['vol'] = data['returns'].rolling(window=5).std()
            data['momentum'] = data['Close'] - data['Close'].shift(5)

            # Compute technical indicators using TA-Lib if available
            if TALIB_AVAILABLE:
                data['SMA5'] = talib.SMA(data['Close'], timeperiod=5)
                data['SMA10'] = talib.SMA(data['Close'], timeperiod=10)
                data['EMA12'] = talib.EMA(data['Close'], timeperiod=12)
                data['EMA26'] = talib.EMA(data['Close'], timeperiod=26)
                data['MACD'], data['MACD_signal'], _ = talib.MACD(
                    data['Close'], fastperiod=12, slowperiod=26, signalperiod=9
                )
                data['RSI_14'] = talib.RSI(data['Close'], timeperiod=14)
            else:
                data['SMA5'] = data['Close'].rolling(window=5).mean()
                data['SMA10'] = data['Close'].rolling(window=10).mean()
                data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
                data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
                data['MACD'] = data['EMA12'] - data['EMA26']
                data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
                delta = data['Close'].diff()
                gain = delta.clip(lower=0).rolling(window=14).mean()
                loss = (-delta.clip(upper=0)).rolling(window=14).mean()
                rs = gain / loss
                data['RSI_14'] = 100 - (100 / (1 + rs))

            # Additional indicators and features
            data['MA_crossover'] = data['SMA5'] - data['SMA10']
            data['high_low_range'] = data['High'] - data['Low']
            data['ATR'] = data['high_low_range'].rolling(window=14).mean()
            volume_ma5 = data['Volume'].rolling(window=5).mean()
            data['volume_ratio'] = (data['Volume'] / volume_ma5).clip(upper=10)

            # Define the target (future price movement)
            data['target'] = (data['Close'].shift(-self.minutes) > data['Close']).astype(int)

            # Add lagged features (lags 1 to 3)
            for lag in range(1, 4):
                data[f'returns_lag{lag}'] = data['returns'].shift(lag)
                data[f'vol_lag{lag}'] = data['vol'].shift(lag)
                data[f'RSI_14_lag{lag}'] = data['RSI_14'].shift(lag)
                data[f'MACD_lag{lag}'] = data['MACD'].shift(lag)
                data[f'volume_ratio_lag{lag}'] = data['volume_ratio'].shift(lag)

            # Drop rows with any NaN values (from rolling calculations)
            data.dropna(inplace=True)

            self.feature_columns = [
                'returns', 'vol', 'momentum', 'MA_crossover',
                'high_low_range', 'ATR', 'volume_ratio',
                'RSI_14', 'MACD', 'MACD_signal',
                'returns_lag1', 'returns_lag2', 'returns_lag3',
                'vol_lag1', 'vol_lag2', 'vol_lag3',
                'RSI_14_lag1', 'RSI_14_lag2', 'RSI_14_lag3',
                'MACD_lag1', 'MACD_lag2', 'MACD_lag3',
                'volume_ratio_lag1', 'volume_ratio_lag2', 'volume_ratio_lag3'
            ]

            if len(data) < 50:
                raise ValueError("Insufficient data points for reliable prediction")

            features = data[self.feature_columns]
            target = data['target']

            logging.info(f"Features prepared: {features.shape}")
            return features, target

        except Exception as e:
            logging.error(f"Error in feature preparation: {str(e)}")
            raise

    def train_model(self):
        """
        Train the model using historical data.
        """
        try:
            # Download historical data (30 days)
            end = datetime.now()
            start = end - timedelta(days=1)
            df = yf.download(self.symbol, start=start, end=end, interval=f'{self.minutes}m')
            logging.info(f"Downloaded {len(df)} rows for {self.symbol}")

            if df.empty:
                raise ValueError("No data downloaded")

            X, y = self.prepare_features(df)

            if len(X) < 100:
                raise ValueError("Insufficient training data")

            # Single train-test split (80/20)
            split_idx = int(0.8 * len(X))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            self.pipeline.fit(X_train, y_train)
            y_pred = self.pipeline.predict(X_test)

            logging.info("\nModel Performance:")
            logging.info(classification_report(y_test, y_pred))

            # Log feature importances
            feature_importances = self.pipeline.named_steps['classifier'].feature_importances_
            importance_dict = dict(zip(self.feature_columns, feature_importances))
            for feature, importance in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
                logging.info(f"{feature}: {importance:.4f}")

        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise

    def predict_next_movement(self):
        """
        Predict the next price movement based on recent data.
        Returns:
            tuple: (prediction, probabilities, current_price, timestamp, confidence_level)
        """
        try:
            end = datetime.now()
            start = end - timedelta(hours=24)
            recent_data = yf.download(self.symbol, start=start, end=end, interval=f'{self.minutes}m')

            if recent_data.empty:
                raise ValueError("No recent data downloaded")

            X, _ = self.prepare_features(recent_data)
            if len(X) == 0:
                raise ValueError("No valid features from recent data")

            latest_features = X.iloc[-1:].values
            probabilities = self.pipeline.predict_proba(latest_features)[0]
            prediction = int(probabilities[1] >= 0.5)

            max_prob = max(probabilities)
            if max_prob >= self.HIGH_CONFIDENCE_THRESHOLD:
                confidence_level = "High"
            elif max_prob >= self.LOW_CONFIDENCE_THRESHOLD:
                confidence_level = "Medium"
            elif abs(probabilities[0] - probabilities[1]) < 0.1:
                confidence_level = "Very Low"
            else:
                confidence_level = "Low"

            logging.info(f"Prediction: {'Up' if prediction==1 else 'Down'} | Probabilities: {probabilities} | Confidence: {confidence_level}")
            return prediction, probabilities, recent_data['Close'].iloc[-1], datetime.now(), confidence_level

        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise
