import logging
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


class ShortTermPredictor:
    def __init__(self, symbol, minutes):
        self.symbol = symbol
        self.minutes = minutes

        # Create base models for ensemble
        gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=4,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42
        )

        mlp = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            learning_rate='adaptive',
            max_iter=1000,
            random_state=42
        )

        # Create a voting classifier ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('gb', gb),
                ('rf', rf),
                ('mlp', mlp)
            ],
            voting='soft',
            weights=[2, 1, 1]
        )

        # Create the full pipeline
        self.pipeline = Pipeline([
            ('smote', SMOTETomek(random_state=42)),  # Better handling of imbalance
            ('scaler', RobustScaler()),
            ('classifier', ensemble)
        ])

        self.feature_columns = None
        self.selected_features = None
        self.HIGH_CONFIDENCE_THRESHOLD = 0.75
        self.LOW_CONFIDENCE_THRESHOLD = 0.60
        self.feature_importance = None
        self.market_regime = "unknown"  # Track market regime

    def _detect_market_regime(self, df):
        """Detect if market is in trending or mean-reverting regime"""
        returns = df['Close'].pct_change().dropna()

        # Calculate Hurst exponent to identify market regime
        def hurst_exponent(series):
            lags = range(2, min(20, len(series) // 4))
            tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
            m = np.polyfit(np.log(lags), np.log(tau), 1)
            return m[0]

        h = hurst_exponent(returns.values)

        if h > 0.6:
            self.market_regime = "trending"
        elif h < 0.4:
            self.market_regime = "mean_reverting"
        else:
            self.market_regime = "random_walk"

        logging.info(f"Detected market regime: {self.market_regime} (Hurst: {h:.3f})")
        return self.market_regime

    def prepare_features(self, df, is_training=True):
        try:
            # Create a DataFrame with single-level columns
            data = pd.DataFrame()

            # Handle both multi-index and single-index DataFrames
            if isinstance(df.columns, pd.MultiIndex):
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    data[col] = df[(col, self.symbol)]
            else:
                data = df.copy()

            # Detect market regime
            regime = self._detect_market_regime(data)

            # Calculate returns
            data['returns'] = data['Close'].pct_change()
            data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))

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

            # Normalized ATR (volatility relative to price)
            data['norm_ATR'] = data['ATR'] / data['Close']

            # Volume ratio with different windows
            volume_ma5 = data['Volume'].rolling(window=5).mean()
            volume_ma10 = data['Volume'].rolling(window=10).mean()
            data['volume_ratio'] = data['Volume'].div(volume_ma5).clip(upper=10)
            data['volume_ratio_10'] = data['Volume'].div(volume_ma10).clip(upper=10)

            # Volume weighted price (VWAP)
            data['VWAP'] = ((data['Close'] + data['High'] + data['Low']) / 3 * data['Volume']).cumsum() / data[
                'Volume'].cumsum()

            # Price distance from VWAP
            data['price_to_VWAP'] = (data['Close'] - data['VWAP']) / data['VWAP']

            # RSI with different windows
            for window in [7, 14, 21]:
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                data[f'RSI_{window}'] = 100 - (100 / (1 + rs))

            # RSI divergence
            data['RSI_divergence'] = data['RSI_14'] - data['RSI_14'].shift(5)

            # Bollinger Bands
            data['bollinger_mid'] = data['Close'].rolling(window=20).mean()
            data['bollinger_std'] = data['Close'].rolling(window=20).std()
            data['bollinger_upper'] = data['bollinger_mid'] + (data['bollinger_std'] * 2)
            data['bollinger_lower'] = data['bollinger_mid'] - (data['bollinger_std'] * 2)
            data['bollinger_width'] = data['bollinger_upper'] - data['bollinger_lower']

            # Bollinger Band position (where price is within the bands)
            data['bb_position'] = (data['Close'] - data['bollinger_lower']) / (
                        data['bollinger_upper'] - data['bollinger_lower'])

            # MACD
            data['MACD'] = data['EMA12'] - data['EMA26']
            data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['MACD_hist'] = data['MACD'] - data['MACD_signal']

            # MACD histogram changes
            data['MACD_hist_change'] = data['MACD_hist'] - data['MACD_hist'].shift(1)

            # Stochastic Oscillator
            data['%K'] = (data['Close'] - data['Low'].rolling(window=14).min()) / (
                    data['High'].rolling(window=14).max() - data['Low'].rolling(window=14).min()) * 100
            data['%D'] = data['%K'].rolling(window=3).mean()

            # Stochastic RSI
            stoch_rsi_k = (data['RSI_14'] - data['RSI_14'].rolling(window=14).min()) / (
                    data['RSI_14'].rolling(window=14).max() - data['RSI_14'].rolling(window=14).min())
            data['Stoch_RSI'] = 100 * stoch_rsi_k

            # Rate of Change (ROC)
            data['ROC'] = data['Close'].pct_change(periods=12) * 100

            # On-Balance Volume (OBV)
            data['OBV'] = (data['Volume'] * (~data['Close'].diff().le(0) * 2 - 1)).cumsum()

            # Chaikin Money Flow (CMF)
            mfm = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
            mfv = mfm * data['Volume']
            data['CMF'] = mfv.rolling(window=20).sum() / data['Volume'].rolling(window=20).sum()

            # Average Directional Index (ADX)
            plus_dm = data['High'].diff()
            minus_dm = data['Low'].diff(-1).abs()
            plus_dm[plus_dm < 0] = 0
            plus_dm[plus_dm < minus_dm] = 0
            minus_dm[minus_dm < 0] = 0
            minus_dm[minus_dm < plus_dm] = 0

            tr1 = (data['High'] - data['Low']).abs()
            tr2 = (data['High'] - data['Close'].shift(1)).abs()
            tr3 = (data['Low'] - data['Close'].shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            plus_di = 100 * (plus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean())
            dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
            data['ADX'] = dx.rolling(window=14).mean()

            # Awesome Oscillator
            data['AO'] = (data['High'] + data['Low']).rolling(window=5).mean() / 2 - (
                        data['High'] + data['Low']).rolling(window=34).mean() / 2

            # Add lagged features with different lags
            for lag in range(1, 6):
                data[f'returns_lag{lag}'] = data['returns'].shift(lag)
                data[f'vol_lag{lag}'] = data['vol'].shift(lag)
                data[f'RSI_14_lag{lag}'] = data['RSI_14'].shift(lag)
                data[f'MACD_lag{lag}'] = data['MACD'].shift(lag)
                data[f'volume_ratio_lag{lag}'] = data['volume_ratio'].shift(lag)
                data[f'bb_position_lag{lag}'] = data['bb_position'].shift(lag)

            # Statistical features
            for window in [5, 10, 20]:
                # Skewness of returns
                data[f'returns_skew_{window}'] = data['returns'].rolling(window=window).skew()
                # Kurtosis of returns
                data[f'returns_kurt_{window}'] = data['returns'].rolling(window=window).kurt()

            # Price acceleration
            data['price_acceleration'] = data['returns'].diff()

            # Add day of week and hour of day as cyclical features
            # Convert index to datetime if needed
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)

            # Add time-based features
            data['hour_sin'] = np.sin(2 * np.pi * data.index.hour / 24)
            data['hour_cos'] = np.cos(2 * np.pi * data.index.hour / 24)
            data['day_sin'] = np.sin(2 * np.pi * data.index.dayofweek / 7)
            data['day_cos'] = np.cos(2 * np.pi * data.index.dayofweek / 7)

            # Regime-specific features
            data['regime'] = self.market_regime
            data['regime_numeric'] = 0
            if self.market_regime == "trending":
                data['regime_numeric'] = 1
            elif self.market_regime == "mean_reverting":
                data['regime_numeric'] = -1

            # Target (future price movement)
            # Different target calculation based on regime
            if self.market_regime == "trending":
                # In trending markets, we might want to be more directional
                data['target'] = (data['Close'].shift(-self.minutes) > data['Close'] * 1.001).astype(int)
            elif self.market_regime == "mean_reverting":
                # In mean-reverting, we look for bounces off extremes
                z_score = (data['Close'] - data['Close'].rolling(window=20).mean()) / data['Close'].rolling(
                    window=20).std()
                data['target'] = ((z_score < -1.5) & (data['Close'].shift(-self.minutes) > data['Close'])).astype(int)
                data.loc[z_score > 1.5, 'target'] = (data['Close'].shift(-self.minutes) < data['Close']).astype(int)
            else:
                # Default target
                data['target'] = (data['Close'].shift(-self.minutes) > data['Close']).astype(int)

            # Define all potential features
            all_features = [
                'returns', 'log_returns', 'vol', 'momentum', 'MA_crossover', 'EMA_crossover',
                'high_low_range', 'close_open_range', 'ATR', 'norm_ATR', 'volume_ratio',
                'volume_ratio_10', 'RSI_7', 'RSI_14', 'RSI_21', 'RSI_divergence',
                'bollinger_upper', 'bollinger_lower', 'bollinger_width', 'bb_position',
                'MACD', 'MACD_signal', 'MACD_hist', 'MACD_hist_change',
                '%K', '%D', 'Stoch_RSI', 'ROC', 'OBV', 'CMF', 'ADX', 'AO',
                'price_to_VWAP', 'price_acceleration',
                'returns_skew_5', 'returns_skew_10', 'returns_skew_20',
                'returns_kurt_5', 'returns_kurt_10', 'returns_kurt_20',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'regime_numeric',
                'returns_lag1', 'returns_lag2', 'returns_lag3', 'returns_lag4', 'returns_lag5',
                'vol_lag1', 'vol_lag2', 'vol_lag3', 'vol_lag4', 'vol_lag5',
                'RSI_14_lag1', 'RSI_14_lag2', 'RSI_14_lag3', 'RSI_14_lag4', 'RSI_14_lag5',
                'MACD_lag1', 'MACD_lag2', 'MACD_lag3', 'MACD_lag4', 'MACD_lag5',
                'volume_ratio_lag1', 'volume_ratio_lag2', 'volume_ratio_lag3', 'volume_ratio_lag4', 'volume_ratio_lag5',
                'bb_position_lag1', 'bb_position_lag2', 'bb_position_lag3', 'bb_position_lag4', 'bb_position_lag5'
            ]

            # Drop NaN values
            data = data.dropna()

            # Check if we have enough data
            if len(data) < 50:
                raise ValueError("Insufficient data points for reliable prediction")

            # If we're training, set the feature columns
            if is_training:
                self.feature_columns = all_features.copy()

                # Extract only the features we need
                features = data[self.feature_columns].copy()
                target = data['target'].copy()

                # Remove highly correlated features during training
                corr_matrix = features.corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
                features = features.drop(to_drop, axis=1)

                # Remove features with too many missing values or zero variance
                features = features.loc[:, features.isnull().mean() < 0.3]
                variances = features.var()
                zero_var_cols = variances[variances <= 0.0000001].index.tolist()
                features = features.drop(zero_var_cols, axis=1)

                # Update feature columns list for future reference
                self.selected_features = features.columns.tolist()

                logging.info(f"Training with {len(self.selected_features)} features after cleaning")
            else:
                # If predicting, use the previously determined features
                if self.selected_features is None:
                    raise ValueError("Model has not been trained yet. Call train_model() first.")

                # Extract only the features we selected during training
                features = data[self.selected_features].copy()
                target = data['target'].copy() if 'target' in data.columns else None

            logging.info(f"Successfully prepared features. Shape: {features.shape}")
            return features, target

        except Exception as e:
            logging.error(f"Error in feature preparation: {str(e)}")
            raise

    def train_model(self):
        try:
            # Download historical data with more history for robust training
            end = datetime.now()
            start = end - timedelta(days=15)  # Increased to 30 days for more training data
            df = yf.download(self.symbol, start=start, end=end, interval=f'{self.minutes}m', auto_adjust=False)
            logging.info(f"Downloaded {len(df)} rows of data for {self.symbol}")

            if df.empty:
                raise ValueError("No data downloaded")

            X, y = self.prepare_features(df, is_training=True)

            if len(X) < 100:
                raise ValueError("Insufficient training data")

            # Use TimeSeriesSplit with more folds for better validation
            tscv = TimeSeriesSplit(n_splits=2)

            # Hyperparameter tuning for ensemble weights
            param_grid = {
                'classifier__weights': [[2, 1, 1], [1, 2, 1], [1, 1, 2], [1, 1, 1]]
            }

            # Find optimal parameters using grid search
            grid_search = GridSearchCV(
                self.pipeline,
                param_grid,
                cv=tscv,
                scoring='f1',
                n_jobs=-1
            )

            grid_search.fit(X, y)
            logging.info(f"Best parameters: {grid_search.best_params_}")

            # Update pipeline with best params
            self.pipeline = grid_search.best_estimator_

            # Check class balance
            logging.info(f"Class balance: {np.bincount(y)}")

            # Store feature importance if using a model that supports it
            ensemble_model = self.pipeline.named_steps['classifier']

            if hasattr(ensemble_model, 'estimators_'):
                for i, estimator in enumerate(ensemble_model.estimators_):
                    if hasattr(estimator, 'feature_importances_'):
                        model_name = ensemble_model.estimators[i][0]
                        importances = estimator.feature_importances_
                        self.feature_importance = pd.DataFrame({
                            'feature': self.selected_features,
                            f'importance_{model_name}': importances
                        }).sort_values(by=f'importance_{model_name}', ascending=False)
                        logging.info(f"\nTop 10 important features for {model_name}:")
                        logging.info(self.feature_importance.head(10))

            # Final evaluation on the latest data (most relevant for prediction)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            # Train on this split
            self.pipeline.fit(X_train, y_train)

            # Evaluate
            y_pred = self.pipeline.predict(X_test)

            logging.info("\nModel Performance on Latest Data:")
            performance_report = classification_report(y_test, y_pred)
            logging.info(performance_report)

            # Check prediction distributions
            pred_proba = self.pipeline.predict_proba(X_test)
            avg_prob = pred_proba.mean(axis=0)
            logging.info(f"Average prediction probabilities: {avg_prob}")

            if abs(avg_prob[0] - avg_prob[1]) > 0.3:
                logging.warning("Model shows significant bias in predictions")

            return performance_report

        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise

    def predict_next_movement(self):
        try:
            # Download more historical data for feature calculation
            end = datetime.now()
            start = end - timedelta(hours=24)  # Increased to 72 hours
            recent_data = yf.download(self.symbol, start=start, end=end, interval=f'{self.minutes}m', auto_adjust=False)

            if recent_data.empty:
                raise ValueError("No recent data downloaded")

            X, _ = self.prepare_features(recent_data, is_training=False)

            if len(X) == 0:
                raise ValueError("No valid features generated from recent data")

            latest_features = X.iloc[-1:].values

            # Get prediction and probabilities
            probabilities = self.pipeline.predict_proba(latest_features)[0]
            prediction = 1 if probabilities[1] >= 0.5 else 0

            # Get confidence level
            confidence_level = "Undefined"
            if max(probabilities) >= self.HIGH_CONFIDENCE_THRESHOLD:
                confidence_level = "High"
            elif max(probabilities) >= self.LOW_CONFIDENCE_THRESHOLD:
                confidence_level = "Medium"
            elif abs(probabilities[0] - probabilities[1]) < 0.1:
                confidence_level = "Very Low"
            else:
                confidence_level = "Low"

            # Enhanced prediction report
            # prediction_report = {
            #     'prediction': 'Up' if prediction == 1 else 'Down',
            #     'up_probability': float(probabilities[1]),
            #     'down_probability': float(probabilities[0]),
            #     'confidence': confidence_level,
            #     'current_price': float(recent_data['Close'].iloc[-1]),
            #     'market_regime': self.market_regime,
            #     'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            #     'symbol': self.symbol,
            #     'minutes_ahead': self.minutes
            # }

            # Add key indicators that influenced the prediction
            # if hasattr(self, 'feature_importance') and self.feature_importance is not None:
            #     top_features = self.feature_importance.head(5)['feature'].tolist()
            #     feature_values = {feature: float(X.iloc[-1][feature]) for feature in top_features if
            #                       feature in X.columns}
            #     prediction_report['key_indicators'] = feature_values
            #
            # logging.info(f"Prediction report: {prediction_report}")

            return (prediction, probabilities, recent_data['Close'].iloc[
                -1], datetime.now(), confidence_level,
                    # prediction_report
                    )

        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise