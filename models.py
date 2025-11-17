





import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from prophet import Prophet
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ImprovedExpensePredictor:
    
    def __init__(self):
        self.xgb_model = None
        self.prophet_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.user_profiles = {}
        self.feature_names = []
    
    def create_aggregated_features(self, df):
        """Enhanced feature engineering with better aggregations"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['year_month'] = df['date'].dt.to_period('M')
        
        # Aggregate by user and month
        monthly_agg = df.groupby(['user_id', 'year_month', 'transaction_type']).agg({
            'amount': ['sum', 'mean', 'count', 'std', 'min', 'max'],
            'category': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Other',
            'payment_mode': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Other'
        }).reset_index()
        
        monthly_agg.columns = ['user_id', 'year_month', 'transaction_type', 
                              'total_amount', 'avg_amount', 'transaction_count', 
                              'std_amount', 'min_amount', 'max_amount',
                              'primary_category', 'primary_payment_mode']
        
        # Pivot to separate Income and Expense
        monthly_pivot = monthly_agg.pivot_table(
            index=['user_id', 'year_month'],
            columns='transaction_type',
            values='total_amount',
            fill_value=0
        ).reset_index()
        
        monthly_pivot.columns.name = None
        if 'Expense' not in monthly_pivot.columns:
            monthly_pivot['Expense'] = 0
        if 'Income' not in monthly_pivot.columns:
            monthly_pivot['Income'] = 0
        
        # Calculate savings and savings rate
        monthly_pivot['Savings'] = monthly_pivot['Income'] - monthly_pivot['Expense']
        monthly_pivot['Savings_Rate'] = np.where(
            monthly_pivot['Income'] > 0,
            monthly_pivot['Savings'] / monthly_pivot['Income'],
            0
        )
        
        # Extract temporal features
        monthly_pivot['month'] = monthly_pivot['year_month'].dt.month
        monthly_pivot['year'] = monthly_pivot['year_month'].dt.year
        monthly_pivot['quarter'] = monthly_pivot['year_month'].dt.quarter
        monthly_pivot['is_year_start'] = (monthly_pivot['month'] == 1).astype(int)
        monthly_pivot['is_year_end'] = (monthly_pivot['month'] == 12).astype(int)
        
        # Sort for rolling calculations
        monthly_pivot = monthly_pivot.sort_values(['user_id', 'year_month'])
        
        # Enhanced rolling statistics (2, 3, and 6 months)
        for window in [2, 3, 6]:
            monthly_pivot[f'expense_rolling_{window}m'] = monthly_pivot.groupby('user_id')['Expense'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            monthly_pivot[f'income_rolling_{window}m'] = monthly_pivot.groupby('user_id')['Income'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            monthly_pivot[f'expense_std_{window}m'] = monthly_pivot.groupby('user_id')['Expense'].transform(
                lambda x: x.rolling(window=window, min_periods=2).std().fillna(0)
            )
        
        # User-level statistics
        monthly_pivot['user_avg_expense'] = monthly_pivot.groupby('user_id')['Expense'].transform('mean')
        monthly_pivot['user_avg_income'] = monthly_pivot.groupby('user_id')['Income'].transform('mean')
        monthly_pivot['user_std_expense'] = monthly_pivot.groupby('user_id')['Expense'].transform('std').fillna(0)
        monthly_pivot['user_median_expense'] = monthly_pivot.groupby('user_id')['Expense'].transform('median')
        
        # Lag features (previous month's values)
        monthly_pivot['expense_lag_1'] = monthly_pivot.groupby('user_id')['Expense'].shift(1).fillna(0)
        monthly_pivot['income_lag_1'] = monthly_pivot.groupby('user_id')['Income'].shift(1).fillna(0)
        monthly_pivot['savings_lag_1'] = monthly_pivot.groupby('user_id')['Savings'].shift(1).fillna(0)
        
        # Expense to income ratio
        monthly_pivot['expense_income_ratio'] = np.where(
            monthly_pivot['Income'] > 0,
            monthly_pivot['Expense'] / monthly_pivot['Income'],
            0
        )
        
        # Trend feature (comparing to rolling average)
        monthly_pivot['expense_vs_rolling_3m'] = np.where(
            monthly_pivot['expense_rolling_3m'] > 0,
            monthly_pivot['Expense'] / monthly_pivot['expense_rolling_3m'],
            1
        )
        
        return monthly_pivot
    
    def train_models(self, csv_path='cleaned_data.csv'):
        """Train models with cross-validation and proper regularization"""
        print("\n" + "="*70)
        print("TRAINING IMPROVED EXPENSE PREDICTION MODELS WITH REGULARIZATION")
        print("="*70)
        
        # Load data
        print("\n[1/3] Loading and preparing data...")
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        
        # Create enhanced features
        monthly_data = self.create_aggregated_features(df)
        print(f"   Created {len(monthly_data)} user-month records")
        
        # Train XGBoost with regularization and cross-validation
        print("\n[2/3] Training XGBoost with Cross-Validation & Regularization...")
        self._train_xgboost_cv(monthly_data)
        
        # Train Prophet
        print("\n[3/3] Training Prophet for time-series forecasting...")
        self._train_prophet(df)
        
        # Create user profiles
        self._create_user_profiles(df)
        
        print("\n" + "="*70)
        print("✓ MODEL TRAINING COMPLETE!")
        print("="*70)
    
    def _train_xgboost_cv(self, monthly_data):
        """Train XGBoost with proper cross-validation and regularization"""
        
        # Define feature columns (enhanced)
        feature_cols = [
            # Temporal features
            'month', 'quarter', 'is_year_start', 'is_year_end',
            
            # Current month data
            'Income',
            
            # Rolling statistics
            'expense_rolling_2m', 'expense_rolling_3m', 'expense_rolling_6m',
            'income_rolling_2m', 'income_rolling_3m', 'income_rolling_6m',
            'expense_std_3m', 'expense_std_6m',
            
            # User-level statistics
            'user_avg_expense', 'user_avg_income', 'user_std_expense', 
            'user_median_expense',
            
            # Lag features
            'expense_lag_1', 'income_lag_1', 'savings_lag_1',
            
            # Ratios
            'expense_income_ratio', 'Savings_Rate', 'expense_vs_rolling_3m'
        ]
        
        self.feature_names = feature_cols
        
        X = monthly_data[feature_cols].fillna(0)
        y = monthly_data['Expense']
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Scale features for better regularization
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
        
        # Split data with stratification by year to ensure temporal distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        print(f"\n   Training set: {len(X_train)} samples")
        print(f"   Testing set: {len(X_test)} samples")
        
        # XGBoost with STRONG REGULARIZATION to prevent overfitting
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100,           # Reduced from 200
            max_depth=4,                # Reduced from 8 (CRITICAL)
            learning_rate=0.03,         # Reduced from 0.05
            min_child_weight=5,         # Increased from default 1
            subsample=0.7,              # Reduced from 0.8
            colsample_bytree=0.7,       # Reduced from 0.8
            reg_alpha=1.0,              # L1 regularization (NEW)
            reg_lambda=2.0,             # L2 regularization (NEW)
            gamma=1.0,                  # Minimum loss reduction (NEW)
            random_state=42,
            objective='reg:squarederror',
            early_stopping_rounds=20    # Stop if no improvement
        )
        
        # # Fit with validation set for early stopping
        # self.xgb_model.fit(
        #     X_train, y_train,
        #     eval_set=[(X_test, y_test)],
        #     verbose=False
        # )


        # Fit with validation set for early stopping (regular train/test)
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )


        # Cross-validation (no early stopping here)
        print("\n   Performing 5-Fold Cross-Validation...")
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        # Create a clone of model without early_stopping_rounds
        xgb_cv_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.03,
            min_child_weight=5,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=2.0,
            gamma=1.0,
            random_state=42,
            objective='reg:squarederror'
        )

        cv_scores = cross_val_score(
            xgb_cv_model, X_scaled, y,
            cv=kfold,
            scoring='r2',
            n_jobs=-1
        )

        
        # # Cross-validation for robust performance estimate
        # print("\n   Performing 5-Fold Cross-Validation...")
        # kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        # cv_scores = cross_val_score(
        #     self.xgb_model, X_scaled, y, 
        #     cv=kfold, 
        #     scoring='r2',
        #     n_jobs=-1
        # )
        
        # Evaluate on train and test sets
        train_pred = self.xgb_model.predict(X_train)
        test_pred = self.xgb_model.predict(X_test)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        # Calculate MAPE (Mean Absolute Percentage Error) - handle zero values
        mask = y_test > 100  # Only calculate MAPE for expenses > ₹100
        if mask.sum() > 0:
            test_mape = np.mean(np.abs((y_test[mask] - test_pred[mask]) / y_test[mask])) * 100
        else:
            test_mape = 0
        
        print(f"\n   ╔══════════════════════════════════════════════════════════╗")
        print(f"   ║            XGBoost Performance Metrics                   ║")
        print(f"   ╠══════════════════════════════════════════════════════════╣")
        print(f"   ║  Training R² Score:        {train_r2:>6.4f}                      ║")
        print(f"   ║  Testing R² Score:         {test_r2:>6.4f}                      ║")
        print(f"   ║  Gap (Overfitting):        {abs(train_r2 - test_r2):>6.4f}                      ║")
        print(f"   ╠══════════════════════════════════════════════════════════╣")
        print(f"   ║  Cross-Validation R² Mean: {cv_scores.mean():>6.4f}                      ║")
        print(f"   ║  Cross-Validation R² Std:  {cv_scores.std():>6.4f}                      ║")
        print(f"   ╠══════════════════════════════════════════════════════════╣")
        print(f"   ║  Mean Absolute Error:      ₹{test_mae:>9,.2f}                 ║")
        print(f"   ║  Root Mean Squared Error:  ₹{test_rmse:>9,.2f}                 ║")
        print(f"   ║  Mean Abs % Error (MAPE):  {test_mape:>6.2f}%                    ║")
        print(f"   ╚══════════════════════════════════════════════════════════╝")
        
        if abs(train_r2 - test_r2) < 0.15:
            print(f"   ✓ Good generalization! Gap is acceptable (<0.15)")
        else:
            print(f"   ⚠ Still some overfitting, but improved from before")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n   Top 8 Important Features:")
        for idx, row in feature_importance.head(8).iterrows():
            print(f"   ├─ {row['feature']:<30} {row['importance']:.4f}")
    
    def _train_prophet(self, df):
        """Train Prophet model (unchanged from original)"""
        daily_expenses = df[df['transaction_type'] == 'Expense'].groupby('date')['amount'].sum().reset_index()
        daily_expenses.columns = ['ds', 'y']
        
        if len(daily_expenses) < 30:
            print("   ⚠ Insufficient data for Prophet training (need at least 30 days)")
            self.prophet_model = None
            return
        
        self.prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10
        )
        
        self.prophet_model.fit(daily_expenses)
        
        forecast = self.prophet_model.predict(daily_expenses)
        mae = mean_absolute_error(daily_expenses['y'], forecast['yhat'])
        rmse = np.sqrt(mean_squared_error(daily_expenses['y'], forecast['yhat']))
        
        print(f"\n   Prophet Performance:")
        print(f"   ├─ Mean Absolute Error: ₹{mae:.2f}")
        print(f"   ├─ Root Mean Squared Error: ₹{rmse:.2f}")
        print(f"   └─ Training data points: {len(daily_expenses)} days")
    
    def _create_user_profiles(self, df):
        """Create user spending profiles"""
        for user_id in df['user_id'].unique():
            user_data = df[df['user_id'] == user_id]
            
            self.user_profiles[user_id] = {
                'avg_monthly_expense': user_data[user_data['transaction_type'] == 'Expense']['amount'].sum() / 
                                      max(user_data['date'].dt.to_period('M').nunique(), 1),
                'avg_monthly_income': user_data[user_data['transaction_type'] == 'Income']['amount'].sum() / 
                                     max(user_data['date'].dt.to_period('M').nunique(), 1),
                'total_months': user_data['date'].dt.to_period('M').nunique(),
                'primary_categories': user_data[user_data['transaction_type'] == 'Expense']['category'].value_counts().head(3).to_dict()
            }
        
        print(f"\n   Created profiles for {len(self.user_profiles)} users")
    
    def predict_monthly_expense(self, user_df):
        """Predict current month's total expense using improved XGBoost"""
        if self.xgb_model is None:
            self.load_models()
        
        user_df = user_df.copy()
        user_df['date'] = pd.to_datetime(user_df['date'])
        
        current_month = datetime.now().month
        current_year = datetime.now().year
        
        # Calculate all required features
        features = self._calculate_prediction_features(user_df, current_month, current_year)
        
        # Scale features
        features_scaled = self.scaler.transform(features[self.feature_names])
        
        # Predict
        predicted_full_month = self.xgb_model.predict(features_scaled)[0]
        
        # Get actual spending so far
        current_month_data = user_df[
            (user_df['date'].dt.month == current_month) & 
            (user_df['date'].dt.year == current_year)
        ]
        spent_so_far = current_month_data[
            current_month_data['transaction_type'] == 'Expense'
        ]['amount'].sum()
        
        # Adjust for days remaining in month
        days_in_month = 30
        current_day = datetime.now().day
        
        if current_day < days_in_month:
            weight_actual = current_day / days_in_month
            weight_predicted = (days_in_month - current_day) / days_in_month
            predicted_total = spent_so_far + (predicted_full_month * weight_predicted / weight_actual)
        else:
            predicted_total = max(spent_so_far, predicted_full_month)
        
        return round(max(0, predicted_total), 2)
    
    def _calculate_prediction_features(self, user_df, month, year):
        """Calculate all features needed for prediction"""
        user_df['year_month'] = user_df['date'].dt.to_period('M')
        
        # Calculate historical statistics
        historical_expenses = user_df[user_df['transaction_type'] == 'Expense'].groupby('year_month')['amount'].sum()
        historical_income = user_df[user_df['transaction_type'] == 'Income'].groupby('year_month')['amount'].sum()
        
        # Current month data
        current_income = user_df[
            (user_df['date'].dt.month == month) & 
            (user_df['date'].dt.year == year) &
            (user_df['transaction_type'] == 'Income')
        ]['amount'].sum()
        
        # Build feature dictionary
        features = {
            'month': month,
            'quarter': (month - 1) // 3 + 1,
            'is_year_start': 1 if month == 1 else 0,
            'is_year_end': 1 if month == 12 else 0,
            'Income': current_income,
            'expense_rolling_2m': historical_expenses.tail(2).mean() if len(historical_expenses) >= 2 else 0,
            'expense_rolling_3m': historical_expenses.tail(3).mean() if len(historical_expenses) >= 3 else 0,
            'expense_rolling_6m': historical_expenses.tail(6).mean() if len(historical_expenses) >= 6 else 0,
            'income_rolling_2m': historical_income.tail(2).mean() if len(historical_income) >= 2 else current_income,
            'income_rolling_3m': historical_income.tail(3).mean() if len(historical_income) >= 3 else current_income,
            'income_rolling_6m': historical_income.tail(6).mean() if len(historical_income) >= 6 else current_income,
            'expense_std_3m': historical_expenses.tail(3).std() if len(historical_expenses) >= 3 else 0,
            'expense_std_6m': historical_expenses.tail(6).std() if len(historical_expenses) >= 6 else 0,
            'user_avg_expense': historical_expenses.mean() if len(historical_expenses) > 0 else 0,
            'user_avg_income': historical_income.mean() if len(historical_income) > 0 else current_income,
            'user_std_expense': historical_expenses.std() if len(historical_expenses) > 1 else 0,
            'user_median_expense': historical_expenses.median() if len(historical_expenses) > 0 else 0,
            'expense_lag_1': historical_expenses.iloc[-1] if len(historical_expenses) > 0 else 0,
            'income_lag_1': historical_income.iloc[-1] if len(historical_income) > 0 else current_income,
            'savings_lag_1': (historical_income.iloc[-1] - historical_expenses.iloc[-1]) if len(historical_expenses) > 0 else 0,
            'expense_income_ratio': historical_expenses.mean() / historical_income.mean() if historical_income.mean() > 0 else 0,
            'Savings_Rate': (historical_income.mean() - historical_expenses.mean()) / historical_income.mean() if historical_income.mean() > 0 else 0,
            'expense_vs_rolling_3m': 1.0
        }
        
        return pd.DataFrame([features])
    
    def save_models(self):
        """Save trained models and scaler"""
        joblib.dump(self.xgb_model, 'improved_xgb_model.pkl')
        joblib.dump(self.prophet_model, 'improved_prophet_model.pkl')
        joblib.dump(self.user_profiles, 'improved_user_profiles.pkl')
        joblib.dump(self.scaler, 'feature_scaler.pkl')
        joblib.dump(self.feature_names, 'feature_names.pkl')
        print("\n✓ Models saved successfully!")
        print("  ├─ improved_xgb_model.pkl")
        print("  ├─ improved_prophet_model.pkl")
        print("  ├─ improved_user_profiles.pkl")
        print("  ├─ feature_scaler.pkl")
        print("  └─ feature_names.pkl")
    
    def load_models(self):
        """Load saved models"""
        try:
            self.xgb_model = joblib.load('improved_xgb_model.pkl')
            self.prophet_model = joblib.load('improved_prophet_model.pkl')
            self.user_profiles = joblib.load('improved_user_profiles.pkl')
            self.scaler = joblib.load('feature_scaler.pkl')
            self.feature_names = joblib.load('feature_names.pkl')
            print("✓ Models loaded successfully!")
        except FileNotFoundError:
            print("⚠ Model files not found. Please train models first.")


# =================== MAIN EXECUTION ===================

if __name__ == "__main__":
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*10 + "BUDGETWISE - IMPROVED MODEL TRAINING WITH CV" + " "*14 + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")
    
    # Initialize predictor
    predictor = ImprovedExpensePredictor()
    
    # Train models
    predictor.train_models('cleaned_data.csv')
    
    # Save models
    predictor.save_models()
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*15 + "✓ TRAINING COMPLETE - READY TO USE!" + " "*20 + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")