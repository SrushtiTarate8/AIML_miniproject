# import pandas as pd
# import numpy as np
# import re
# from datetime import datetime
# from imblearn.over_sampling import SMOTE
# from sklearn.preprocessing import LabelEncoder
# import warnings
# warnings.filterwarnings('ignore')

# class DataPreprocessor:
    
#     def __init__(self, file_path):
#         """Initialize with dataset path"""
#         print(f"Loading dataset from {file_path}...")
#         self.df = pd.read_csv(file_path)
#         print(f"Loaded {len(self.df)} records with {self.df.shape[1]} columns")
        
#         # City mapping dictionary (shortform -> full form)
#         self.city_mapping = {
#             'DEL': 'Delhi',
#             'BAN': 'Bangalore',
#             'BANGALORE': 'Bangalore',
#             'MUM': 'Mumbai',
#             'MUMBAI': 'Mumbai',
#             'KOL': 'Kolkata',
#             'KOLKATA': 'Kolkata',
#             'CHE': 'Chennai',
#             'CHENNAI': 'Chennai',
#             'HYD': 'Hyderabad',
#             'HYDERABAD': 'Hyderabad',
#             'PUN': 'Pune',
#             'PUNE': 'Pune',
#             'AHM': 'Ahmedabad',
#             'AHMEDABAD': 'Ahmedabad',
#             'LUC': 'Lucknow',
#             'LUCKNOW': 'Lucknow',
#             'JAI': 'Jaipur',
#             'JAIPUR': 'Jaipur',
#             'SUR': 'Surat',
#             'SURAT': 'Surat'
#         }
    
#     def clean_data(self):
#         """Main cleaning pipeline"""
#         print("\n" + "="*60)
#         print("STARTING DATA CLEANING PIPELINE")
#         print("="*60)
        
#         initial_count = len(self.df)
        
#         # Step 1: Remove exact duplicates based on transaction_id
#         print("\n[1/11] Removing duplicate transaction IDs...")
#         before = len(self.df)
#         self.df = self.df.drop_duplicates(subset=['transaction_id'], keep='first')
#         print(f"   Removed {before - len(self.df)} duplicate records")
        
#         # Step 2: Clean dates (multiple formats)
#         print("\n[2/11] Cleaning and parsing dates...")
#         self.df['date'] = self.df['date'].apply(self.parse_date)
#         invalid_dates = self.df['date'].isna().sum()
#         print(f"   Found {invalid_dates} invalid dates")
        
#         # Step 3: Clean amounts (remove currency symbols, handle outliers)
#         print("\n[3/11] Cleaning amount column...")
#         self.df['amount'] = self.df['amount'].apply(self.clean_amount)
#         invalid_amounts = self.df['amount'].isna().sum()
#         print(f"   Found {invalid_amounts} invalid amounts")
        
#         # Step 4: Standardize transaction types
#         print("\n[4/11] Standardizing transaction types...")
#         self.df['transaction_type'] = self.df['transaction_type'].str.strip().str.capitalize()
#         print(f"   Transaction types: {self.df['transaction_type'].value_counts().to_dict()}")
        
#         # Step 5: Standardize categories (fix typos)
#         print("\n[5/11] Standardizing categories...")
#         before_unique = self.df['category'].nunique()
#         self.df['category'] = self.df['category'].apply(self.standardize_category)
#         after_unique = self.df['category'].nunique()
#         print(f"   Reduced categories from {before_unique} to {after_unique} unique values")
        
#         # Step 6: Standardize payment modes
#         print("\n[6/11] Standardizing payment modes...")
#         before_unique = self.df['payment_mode'].nunique()
#         self.df['payment_mode'] = self.df['payment_mode'].apply(self.standardize_payment)
#         after_unique = self.df['payment_mode'].nunique()
#         print(f"   Reduced payment modes from {before_unique} to {after_unique} unique values")
        
#         # Step 7: Standardize locations (CRITICAL - handle shortforms and full forms)
#         print("\n[7/11] Standardizing location names...")
#         before_unique = self.df['location'].nunique()
#         self.df['location'] = self.df['location'].apply(self.standardize_location)
#         after_unique = self.df['location'].nunique()
#         print(f"   Reduced locations from {before_unique} to {after_unique} unique cities")
        
#         # Step 8: Clean notes column (remove gibberish)
#         print("\n[8/11] Cleaning notes column...")
#         self.df['notes'] = self.df['notes'].apply(self.clean_notes)
        
#         # Step 9: Handle missing values
#         print("\n[9/11] Handling missing values...")
#         missing_before = self.df.isnull().sum().sum()
#         self._handle_missing_values()
#         missing_after = self.df.isnull().sum().sum()
#         print(f"   Reduced missing values from {missing_before} to {missing_after}")
        
#         # Step 10: Remove outliers and invalid records
#         print("\n[10/11] Removing outliers and invalid records...")
#         before = len(self.df)
#         self._remove_outliers()
#         print(f"   Removed {before - len(self.df)} outlier records")
        
#         # Step 11: Sort by date and reset index
#         print("\n[11/11] Sorting by date and resetting index...")
#         self.df = self.df.sort_values(['user_id', 'date']).reset_index(drop=True)
        
#         print("\n" + "="*60)
#         print(f"CLEANING COMPLETE!")
#         print(f"Final dataset: {len(self.df)} records ({initial_count - len(self.df)} removed)")
#         print("="*60)
        
#         return self.df
    
#     def parse_date(self, date_str):
#         """Parse multiple date formats"""
#         if pd.isna(date_str) or str(date_str).strip() == '':
#             return None
        
#         date_str = str(date_str).strip()
        
#         # List of date formats found in the dataset
#         date_formats = [
#             '%Y-%m-%d',       # 2024-08-13
#             '%m/%d/%Y',       # 08/05/2022
#             '%d-%m-%y',       # 31-12-23
#             '%d-%m-%Y',       # 13-03-2024
#             '%m/%d/%y',       # 04/11/24
#             '%d/%m/%Y',       # 17/07/2024
#             '%Y/%m/%d',       # 2024/08/13
#             '%d-%m-%y',       # 21-09-24
#             '%m-%d-%Y',       # 04-11-2024
#         ]
        
#         for fmt in date_formats:
#             try:
#                 parsed_date = pd.to_datetime(date_str, format=fmt)
#                 # Validate date is reasonable (between 2020-2025)
#                 if parsed_date.year >= 2020 and parsed_date.year <= 2025:
#                     return parsed_date
#             except:
#                 continue
        
#         # Try pandas auto-parsing as last resort
#         try:
#             parsed_date = pd.to_datetime(date_str)
#             if parsed_date.year >= 2020 and parsed_date.year <= 2025:
#                 return parsed_date
#         except:
#             pass
        
#         return None
    
#     def clean_amount(self, amount):
#         """Remove currency symbols and convert to float"""
#         if pd.isna(amount):
#             return None
        
#         # Convert to string
#         amount_str = str(amount).strip()
        
#         # Remove all currency symbols and extra characters
#         amount_str = re.sub(r'[₹$Rs\.,\s]', '', amount_str)
        
#         # Remove any remaining non-numeric characters except decimal point
#         amount_str = re.sub(r'[^\d.]', '', amount_str)
        
#         try:
#             value = float(amount_str)
#             # Return None for clearly invalid values
#             if value <= 0 or value > 10000000:  # 1 crore max
#                 return None
#             return value
#         except:
#             return None
    
#     def standardize_category(self, category):
#         """Fix typos and standardize categories"""
#         if pd.isna(category) or str(category).strip() == '':
#             return 'Other'
        
#         category = str(category).lower().strip()
        
#         # Remove extra spaces
#         category = re.sub(r'\s+', ' ', category)
        
#         # Category mappings (typos -> correct form)
#         mappings = {
#             'food': ['fod', 'foods', 'foood'],
#             'rent': ['rentt', 'rnt'],
#             'education': ['educaton', 'eduction', 'education'],
#             'utilities': ['utility', 'utilties', 'utlities', 'utilities'],
#             'health': ['healthcare', 'helth', 'health'],
#             'entertainment': ['entertaiment', 'entertainment'],
#             'travel': ['travle', 'travel', 'travelling'],
#             'freelance': ['freelance', 'freelancing'],
#             'salary': ['salary', 'salry'],
#             'shopping': ['shopping', 'shop'],
#             'transportation': ['transport', 'transportation'],
#             'others': ['other', 'others', 'misc', 'miscellaneous']
#         }
        
#         # Check for exact matches first
#         for standard, variations in mappings.items():
#             if category in variations:
#                 return standard.capitalize()
        
#         # Check for partial matches
#         for standard, variations in mappings.items():
#             for variation in variations:
#                 if variation in category or category in variation:
#                     return standard.capitalize()
        
#         # If no match found, capitalize first letter
#         return category.capitalize()
    
#     def standardize_payment(self, mode):
#         """Standardize payment modes"""
#         if pd.isna(mode) or str(mode).strip() in ['N/A', 'n/a', '']:
#             return 'Other'
        
#         mode = str(mode).lower().strip()
        
#         # Payment mode mappings
#         if mode in ['upi', 'upi']:
#             return 'UPI'
#         elif mode in ['card', 'crd', 'credit card', 'debit card']:
#             return 'Card'
#         elif mode in ['cash', 'csh']:
#             return 'Cash'
#         elif mode in ['bank transfer', 'banktransfer', 'bank', 'transfer']:
#             return 'Bank Transfer'
#         elif mode in ['wallet', 'e-wallet', 'digital wallet']:
#             return 'Wallet'
#         elif mode in ['cheque', 'check']:
#             return 'Cheque'
#         else:
#             return 'Other'
    
#     def standardize_location(self, location):
#         """Standardize location names - handle both shortforms and full forms"""
#         if pd.isna(location) or str(location).strip() == '':
#             return 'Unknown'
        
#         # Convert to uppercase and strip
#         location = str(location).upper().strip()
        
#         # Remove extra spaces
#         location = re.sub(r'\s+', ' ', location)
        
#         # Check if it's in our city mapping (handles shortforms)
#         if location in self.city_mapping:
#             return self.city_mapping[location]
        
#         # Check for partial matches (handles typos in full names)
#         for shortform, fullname in self.city_mapping.items():
#             if fullname.upper() in location or location in fullname.upper():
#                 return fullname
        
#         # If not found, return capitalized version
#         return location.title()
    
#     def clean_notes(self, note):
#         """Clean notes column - remove gibberish"""
#         if pd.isna(note) or str(note).strip() in ['N/A', 'n/a', 'xyz123', 'asdfgh', 'test', '']:
#             return ''
        
#         note = str(note).strip()
        
#         # Remove notes that are just random characters
#         if len(note) < 3:
#             return ''
        
#         # Check if note is mostly gibberish (more than 50% non-alphabetic)
#         alpha_ratio = sum(c.isalpha() or c.isspace() for c in note) / len(note)
#         if alpha_ratio < 0.5:
#             return ''
        
#         return note
    
#     def _handle_missing_values(self):
#         """Handle missing values appropriately for each column"""
#         # Critical columns - drop rows if missing
#         critical_cols = ['date', 'amount', 'transaction_type']
#         self.df = self.df.dropna(subset=critical_cols)
        
#         # Fill categorical columns with appropriate defaults
#         self.df['category'].fillna('Other', inplace=True)
#         self.df['payment_mode'].fillna('Other', inplace=True)
#         self.df['location'].fillna('Unknown', inplace=True)
#         self.df['notes'].fillna('', inplace=True)
        
#         # User_id should not be missing
#         self.df = self.df.dropna(subset=['user_id'])
    
#     def _remove_outliers(self):
#         """Remove outliers and clearly invalid records"""
#         # Remove negative amounts
#         self.df = self.df[self.df['amount'] > 0]
        
#         # Remove extreme outliers (amounts > 1 million or suspiciously high like 999999999)
#         self.df = self.df[self.df['amount'] < 1000000]
        
#         # Remove dates that are clearly invalid
#         self.df = self.df.dropna(subset=['date'])
        
#         # Remove invalid transaction types
#         valid_types = ['Income', 'Expense']
#         self.df = self.df[self.df['transaction_type'].isin(valid_types)]
    
#     def handle_class_imbalance(self):
#         """Handle imbalanced transaction_type using SMOTE"""
#         print("\n" + "="*60)
#         print("HANDLING CLASS IMBALANCE (SMOTE)")
#         print("="*60)
        
#         print("\nClass distribution BEFORE SMOTE:")
#         print(self.df['transaction_type'].value_counts())
#         print(f"Ratio - Expense: {(self.df['transaction_type'] == 'Expense').sum() / len(self.df) * 100:.1f}%")
#         print(f"Ratio - Income: {(self.df['transaction_type'] == 'Income').sum() / len(self.df) * 100:.1f}%")
        
#         # Prepare features for SMOTE
#         # Encode categorical variables
#         label_encoders = {}
#         categorical_cols = ['category', 'payment_mode', 'location']
        
#         df_encoded = self.df.copy()
        
#         for col in categorical_cols:
#             le = LabelEncoder()
#             df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].astype(str))
#             label_encoders[col] = le
        
#         # Add time-based features
#         df_encoded['month'] = pd.to_datetime(df_encoded['date']).dt.month
#         df_encoded['day_of_week'] = pd.to_datetime(df_encoded['date']).dt.dayofweek
#         df_encoded['year'] = pd.to_datetime(df_encoded['date']).dt.year
        
#         # Features for SMOTE
#         feature_cols = [
#             'amount', 'month', 'day_of_week', 'year',
#             'category_encoded', 'payment_mode_encoded', 'location_encoded'
#         ]
        
#         X = df_encoded[feature_cols]
#         y = df_encoded['transaction_type']
        
#         # Encode target variable
#         le_target = LabelEncoder()
#         y_encoded = le_target.fit_transform(y)
        
#         # Apply SMOTE
#         smote = SMOTE(random_state=42, k_neighbors=5)
#         X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
        
#         # Decode target back
#         y_resampled = le_target.inverse_transform(y_resampled)
        
#         # Create new dataframe with synthetic samples
#         df_resampled = pd.DataFrame(X_resampled, columns=feature_cols)
#         df_resampled['transaction_type'] = y_resampled
        
#         # Decode categorical variables back
#         for col in categorical_cols:
#             df_resampled[col] = label_encoders[col].inverse_transform(
#                 df_resampled[f'{col}_encoded'].astype(int)
#             )
        
#         # Add back other columns (use mode for categorical, mean for numerical)
#         df_resampled['user_id'] = np.random.choice(
#             self.df['user_id'].unique(), 
#             size=len(df_resampled)
#         )
#         df_resampled['transaction_id'] = [f'T{i:05d}' for i in range(len(df_resampled))]
        
#         # Reconstruct dates from year, month
#         df_resampled['date'] = pd.to_datetime(
#             df_resampled[['year', 'month']].assign(day=15)
#         )
        
#         # Add notes (empty for synthetic data)
#         df_resampled['notes'] = ''
        
#         # Select final columns in original order
#         final_cols = ['transaction_id', 'user_id', 'date', 'transaction_type', 
#                      'category', 'amount', 'payment_mode', 'location', 'notes']
        
#         self.df = df_resampled[final_cols]
        
#         print("\nClass distribution AFTER SMOTE:")
#         print(self.df['transaction_type'].value_counts())
#         print(f"Ratio - Expense: {(self.df['transaction_type'] == 'Expense').sum() / len(self.df) * 100:.1f}%")
#         print(f"Ratio - Income: {(self.df['transaction_type'] == 'Income').sum() / len(self.df) * 100:.1f}%")
#         print(f"\nTotal records after SMOTE: {len(self.df)}")
#         print("="*60)
        
#         return self.df
    
#     def generate_summary_report(self):
#         """Generate a summary report of the cleaned data"""
#         print("\n" + "="*60)
#         print("DATA SUMMARY REPORT")
#         print("="*60)
        
#         print(f"\nTotal Records: {len(self.df)}")
#         print(f"Date Range: {self.df['date'].min()} to {self.df['date'].max()}")
#         print(f"Unique Users: {self.df['user_id'].nunique()}")
        
#         print("\n--- Transaction Type Distribution ---")
#         print(self.df['transaction_type'].value_counts())
        
#         print("\n--- Category Distribution (Top 10) ---")
#         print(self.df['category'].value_counts().head(10))
        
#         print("\n--- Payment Mode Distribution ---")
#         print(self.df['payment_mode'].value_counts())
        
#         print("\n--- Location Distribution (Top 10) ---")
#         print(self.df['location'].value_counts().head(10))
        
#         print("\n--- Amount Statistics ---")
#         print(f"Min: ₹{self.df['amount'].min():.2f}")
#         print(f"Max: ₹{self.df['amount'].max():.2f}")
#         print(f"Mean: ₹{self.df['amount'].mean():.2f}")
#         print(f"Median: ₹{self.df['amount'].median():.2f}")
        
#         print("\n--- Missing Values ---")
#         missing = self.df.isnull().sum()
#         if missing.sum() > 0:
#             print(missing[missing > 0])
#         else:
#             print("No missing values!")
        
#         print("="*60)
    
#     def save_cleaned_data(self, output_path='cleaned_data.csv'):
#         """Save cleaned data to CSV"""
#         self.df.to_csv(output_path, index=False)
#         print(f"\n✓ Cleaned data saved to: {output_path}")
#         print(f"  File size: {len(self.df)} rows × {len(self.df.columns)} columns")


# # =================== MAIN EXECUTION ===================

# if __name__ == "__main__":
#     print("\n" + "█"*60)
#     print("█" + " "*58 + "█")
#     print("█" + " "*15 + "BUDGETWISE DATA PREPROCESSING" + " "*15 + "█")
#     print("█" + " "*58 + "█")
#     print("█"*60 + "\n")
    
#     # Initialize preprocessor
#     preprocessor = DataPreprocessor('budgetwise_finance_dataset.csv')
    
#     # Step 1: Clean the data
#     cleaned_df = preprocessor.clean_data()
    
#     # Step 2: Handle class imbalance with SMOTE
#     balanced_df = preprocessor.handle_class_imbalance()
    
#     # Step 3: Generate summary report
#     preprocessor.generate_summary_report()
    
#     # Step 4: Save cleaned data
#     preprocessor.save_cleaned_data('cleaned_data.csv')
    
#     print("\n✓ Preprocessing complete! You can now use 'cleaned_data.csv' for model training.")
#     print("\n" + "█"*60 + "\n")








import pandas as pd
import numpy as np
import re
from datetime import datetime
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    
    def __init__(self, file_path):
        """Initialize with dataset path"""
        print(f"Loading dataset from {file_path}...")
        self.df = pd.read_csv(file_path)
        print(f"Loaded {len(self.df)} records with {self.df.shape[1]} columns")
        
        # City mapping dictionary (shortform -> full form)
        self.city_mapping = {
            'DEL': 'Delhi',
            'BAN': 'Bangalore',
            'BANGALORE': 'Bangalore',
            'MUM': 'Mumbai',
            'MUMBAI': 'Mumbai',
            'KOL': 'Kolkata',
            'KOLKATA': 'Kolkata',
            'CHE': 'Chennai',
            'CHENNAI': 'Chennai',
            'HYD': 'Hyderabad',
            'HYDERABAD': 'Hyderabad',
            'PUN': 'Pune',
            'PUNE': 'Pune',
            'AHM': 'Ahmedabad',
            'AHMEDABAD': 'Ahmedabad',
            'LUC': 'Lucknow',
            'LUCKNOW': 'Lucknow',
            'JAI': 'Jaipur',
            'JAIPUR': 'Jaipur',
            'SUR': 'Surat',
            'SURAT': 'Surat'
        }
    
    def clean_data(self):
        """Main cleaning pipeline"""
        print("\n" + "="*60)
        print("STARTING DATA CLEANING PIPELINE")
        print("="*60)
        
        initial_count = len(self.df)
        
        # Step 1: Remove exact duplicates based on transaction_id
        print("\n[1/11] Removing duplicate transaction IDs...")
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=['transaction_id'], keep='first')
        print(f"   Removed {before - len(self.df)} duplicate records")
        
        # Step 2: Clean dates (multiple formats)
        print("\n[2/11] Cleaning and parsing dates...")
        self.df['date'] = self.df['date'].apply(self.parse_date)
        invalid_dates = self.df['date'].isna().sum()
        print(f"   Found {invalid_dates} invalid dates")
        
        # Step 3: Clean amounts (remove currency symbols, handle outliers)
        print("\n[3/11] Cleaning amount column...")
        self.df['amount'] = self.df['amount'].apply(self.clean_amount)
        invalid_amounts = self.df['amount'].isna().sum()
        print(f"   Found {invalid_amounts} invalid amounts")
        
        # Step 4: Standardize transaction types
        print("\n[4/11] Standardizing transaction types...")
        self.df['transaction_type'] = self.df['transaction_type'].str.strip().str.capitalize()
        print(f"   Transaction types: {self.df['transaction_type'].value_counts().to_dict()}")
        
        # Step 5: Standardize categories (fix typos)
        print("\n[5/11] Standardizing categories...")
        before_unique = self.df['category'].nunique()
        self.df['category'] = self.df['category'].apply(self.standardize_category)
        after_unique = self.df['category'].nunique()
        print(f"   Reduced categories from {before_unique} to {after_unique} unique values")
        
        # Step 6: Standardize payment modes
        print("\n[6/11] Standardizing payment modes...")
        before_unique = self.df['payment_mode'].nunique()
        self.df['payment_mode'] = self.df['payment_mode'].apply(self.standardize_payment)
        after_unique = self.df['payment_mode'].nunique()
        print(f"   Reduced payment modes from {before_unique} to {after_unique} unique values")
        
        # Step 7: Standardize locations (CRITICAL - handle shortforms and full forms)
        print("\n[7/11] Standardizing location names...")
        before_unique = self.df['location'].nunique()
        self.df['location'] = self.df['location'].apply(self.standardize_location)
        after_unique = self.df['location'].nunique()
        print(f"   Reduced locations from {before_unique} to {after_unique} unique cities")
        
        # Step 8: Clean notes column (remove gibberish)
        print("\n[8/11] Cleaning notes column...")
        self.df['notes'] = self.df['notes'].apply(self.clean_notes)
        
        # Step 9: Handle missing values
        print("\n[9/11] Handling missing values...")
        missing_before = self.df.isnull().sum().sum()
        self._handle_missing_values()
        missing_after = self.df.isnull().sum().sum()
        print(f"   Reduced missing values from {missing_before} to {missing_after}")
        
        # Step 10: Remove outliers and invalid records
        print("\n[10/11] Removing outliers and invalid records...")
        before = len(self.df)
        self._remove_outliers()
        print(f"   Removed {before - len(self.df)} outlier records")
        
        # Step 11: Sort by date and reset index
        print("\n[11/11] Sorting by date and resetting index...")
        self.df = self.df.sort_values(['user_id', 'date']).reset_index(drop=True)
        
        print("\n" + "="*60)
        print(f"CLEANING COMPLETE!")
        print(f"Final dataset: {len(self.df)} records ({initial_count - len(self.df)} removed)")
        print("="*60)
        
        return self.df
    
    def parse_date(self, date_str):
        """Parse multiple date formats"""
        if pd.isna(date_str) or str(date_str).strip() == '':
            return None
        
        date_str = str(date_str).strip()
        
        # List of date formats found in the dataset
        date_formats = [
            '%Y-%m-%d',       # 2024-08-13
            '%m/%d/%Y',       # 08/05/2022
            '%d-%m-%y',       # 31-12-23
            '%d-%m-%Y',       # 13-03-2024
            '%m/%d/%y',       # 04/11/24
            '%d/%m/%Y',       # 17/07/2024
            '%Y/%m/%d',       # 2024/08/13
            '%d-%m-%y',       # 21-09-24
            '%m-%d-%Y',       # 04-11-2024
        ]
        
        for fmt in date_formats:
            try:
                parsed_date = pd.to_datetime(date_str, format=fmt)
                # Validate date is reasonable (between 2020-2025)
                if parsed_date.year >= 2020 and parsed_date.year <= 2025:
                    return parsed_date
            except:
                continue
        
        # Try pandas auto-parsing as last resort
        try:
            parsed_date = pd.to_datetime(date_str)
            if parsed_date.year >= 2020 and parsed_date.year <= 2025:
                return parsed_date
        except:
            pass
        
        return None
    
    def clean_amount(self, amount):
        """Remove currency symbols and convert to float"""
        if pd.isna(amount):
            return None
        
        # Convert to string
        amount_str = str(amount).strip()
        
        # Remove all currency symbols and extra characters
        amount_str = re.sub(r'[₹$Rs\.,\s]', '', amount_str)
        
        # Remove any remaining non-numeric characters except decimal point
        amount_str = re.sub(r'[^\d.]', '', amount_str)
        
        try:
            value = float(amount_str)
            # Return None for clearly invalid values
            if value <= 0 or value > 10000000:  # 1 crore max
                return None
            return value
        except:
            return None
    
    def standardize_category(self, category):
        """Fix typos and standardize categories"""
        if pd.isna(category) or str(category).strip() == '':
            return 'Other'
        
        category = str(category).lower().strip()
        
        # Remove extra spaces
        category = re.sub(r'\s+', ' ', category)
        
        # Category mappings (typos -> correct form)
        mappings = {
            'food': ['fod', 'foods', 'foood'],
            'rent': ['rentt', 'rnt'],
            'education': ['educaton', 'eduction', 'education'],
            'utilities': ['utility', 'utilties', 'utlities', 'utilities'],
            'health': ['healthcare', 'helth', 'health'],
            'entertainment': ['entertaiment', 'entertainment'],
            'travel': ['travle', 'travel', 'travelling'],
            'freelance': ['freelance', 'freelancing'],
            'salary': ['salary', 'salry'],
            'shopping': ['shopping', 'shop'],
            'transportation': ['transport', 'transportation'],
            'others': ['other', 'others', 'misc', 'miscellaneous']
        }
        
        # Check for exact matches first
        for standard, variations in mappings.items():
            if category in variations:
                return standard.capitalize()
        
        # Check for partial matches
        for standard, variations in mappings.items():
            for variation in variations:
                if variation in category or category in variation:
                    return standard.capitalize()
        
        # If no match found, capitalize first letter
        return category.capitalize()
    
    def standardize_payment(self, mode):
        """Standardize payment modes"""
        if pd.isna(mode) or str(mode).strip() in ['N/A', 'n/a', '']:
            return 'Other'
        
        mode = str(mode).lower().strip()
        
        # Payment mode mappings
        if mode in ['upi', 'upi']:
            return 'UPI'
        elif mode in ['card', 'crd', 'credit card', 'debit card']:
            return 'Card'
        elif mode in ['cash', 'csh']:
            return 'Cash'
        elif mode in ['bank transfer', 'banktransfer', 'bank', 'transfer']:
            return 'Bank Transfer'
        elif mode in ['wallet', 'e-wallet', 'digital wallet']:
            return 'Wallet'
        elif mode in ['cheque', 'check']:
            return 'Cheque'
        else:
            return 'Other'
    
    def standardize_location(self, location):
        """Standardize location names - handle both shortforms and full forms"""
        if pd.isna(location) or str(location).strip() == '':
            return 'Unknown'
        
        # Convert to uppercase and strip
        location = str(location).upper().strip()
        
        # Remove extra spaces
        location = re.sub(r'\s+', ' ', location)
        
        # Check if it's in our city mapping (handles shortforms)
        if location in self.city_mapping:
            return self.city_mapping[location]
        
        # Check for partial matches (handles typos in full names)
        for shortform, fullname in self.city_mapping.items():
            if fullname.upper() in location or location in fullname.upper():
                return fullname
        
        # If not found, return capitalized version
        return location.title()
    
    def clean_notes(self, note):
        """Clean notes column - remove gibberish"""
        if pd.isna(note) or str(note).strip() in ['N/A', 'n/a', 'xyz123', 'asdfgh', 'test', '']:
            return ''
        
        note = str(note).strip()
        
        # Remove notes that are just random characters
        if len(note) < 3:
            return ''
        
        # Check if note is mostly gibberish (more than 50% non-alphabetic)
        alpha_ratio = sum(c.isalpha() or c.isspace() for c in note) / len(note)
        if alpha_ratio < 0.5:
            return ''
        
        return note
    
    def _handle_missing_values(self):
        """Handle missing values appropriately for each column"""
        # Critical columns - drop rows if missing
        critical_cols = ['date', 'amount', 'transaction_type']
        self.df = self.df.dropna(subset=critical_cols)
        
        # Fill categorical columns with appropriate defaults
        self.df['category'].fillna('Other', inplace=True)
        self.df['payment_mode'].fillna('Other', inplace=True)
        self.df['location'].fillna('Unknown', inplace=True)
        self.df['notes'].fillna('', inplace=True)
        
        # User_id should not be missing
        self.df = self.df.dropna(subset=['user_id'])
    
    def _remove_outliers(self):
        """Remove outliers and clearly invalid records"""
        # Remove negative amounts
        self.df = self.df[self.df['amount'] > 0]
        
        # Remove extreme outliers (amounts > 1 million or suspiciously high like 999999999)
        self.df = self.df[self.df['amount'] < 1000000]
        
        # Remove dates that are clearly invalid
        self.df = self.df.dropna(subset=['date'])
        
        # Remove invalid transaction types
        valid_types = ['Income', 'Expense']
        self.df = self.df[self.df['transaction_type'].isin(valid_types)]
    
    def handle_class_imbalance(self, balance_method='smote'):
        """Handle imbalanced transaction_type using SMOTE or None"""
        print("\n" + "="*60)
        print("HANDLING CLASS IMBALANCE")
        print("="*60)
        
        print("\nClass distribution BEFORE balancing:")
        print(self.df['transaction_type'].value_counts())
        expense_pct = (self.df['transaction_type'] == 'Expense').sum() / len(self.df) * 100
        income_pct = (self.df['transaction_type'] == 'Income').sum() / len(self.df) * 100
        print(f"Ratio - Expense: {expense_pct:.1f}%")
        print(f"Ratio - Income: {income_pct:.1f}%")
        
        if balance_method == 'none':
            print("\nSkipping SMOTE - keeping original imbalanced data for better real-world predictions")
            print("="*60)
            return self.df
        
        # Prepare features for SMOTE
        # Encode categorical variables
        label_encoders = {}
        categorical_cols = ['category', 'payment_mode', 'location']
        
        df_encoded = self.df.copy()
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
        
        # Add time-based features
        df_encoded['month'] = pd.to_datetime(df_encoded['date']).dt.month
        df_encoded['day_of_week'] = pd.to_datetime(df_encoded['date']).dt.dayofweek
        df_encoded['year'] = pd.to_datetime(df_encoded['date']).dt.year
        
        # Features for SMOTE
        feature_cols = [
            'amount', 'month', 'day_of_week', 'year',
            'category_encoded', 'payment_mode_encoded', 'location_encoded'
        ]
        
        X = df_encoded[feature_cols]
        y = df_encoded['transaction_type']
        
        # Encode target variable
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)
        
        # Apply SMOTE
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
        
        # Decode target back
        y_resampled = le_target.inverse_transform(y_resampled)
        
        # Create new dataframe with synthetic samples
        df_resampled = pd.DataFrame(X_resampled, columns=feature_cols)
        df_resampled['transaction_type'] = y_resampled
        
        # Decode categorical variables back
        for col in categorical_cols:
            df_resampled[col] = label_encoders[col].inverse_transform(
                df_resampled[f'{col}_encoded'].astype(int)
            )
        
        # Add back other columns (use mode for categorical, mean for numerical)
        df_resampled['user_id'] = np.random.choice(
            self.df['user_id'].unique(), 
            size=len(df_resampled)
        )
        df_resampled['transaction_id'] = [f'T{i:05d}' for i in range(len(df_resampled))]
        
        # Reconstruct dates from year, month
        df_resampled['date'] = pd.to_datetime(
            df_resampled[['year', 'month']].assign(day=15)
        )
        
        # Add notes (empty for synthetic data)
        df_resampled['notes'] = ''
        
        # Select final columns in original order
        final_cols = ['transaction_id', 'user_id', 'date', 'transaction_type', 
                     'category', 'amount', 'payment_mode', 'location', 'notes']
        
        self.df = df_resampled[final_cols]
        
        print("\nClass distribution AFTER SMOTE:")
        print(self.df['transaction_type'].value_counts())
        expense_pct = (self.df['transaction_type'] == 'Expense').sum() / len(self.df) * 100
        income_pct = (self.df['transaction_type'] == 'Income').sum() / len(self.df) * 100
        print(f"Ratio - Expense: {expense_pct:.1f}%")
        print(f"Ratio - Income: {income_pct:.1f}%")
        print(f"\nTotal records after SMOTE: {len(self.df)}")
        print("="*60)
        
        return self.df
    
    def generate_summary_report(self):
        """Generate a summary report of the cleaned data"""
        print("\n" + "="*60)
        print("DATA SUMMARY REPORT")
        print("="*60)
        
        print(f"\nTotal Records: {len(self.df)}")
        print(f"Date Range: {self.df['date'].min()} to {self.df['date'].max()}")
        print(f"Unique Users: {self.df['user_id'].nunique()}")
        
        print("\n--- Transaction Type Distribution ---")
        print(self.df['transaction_type'].value_counts())
        
        print("\n--- Category Distribution (Top 10) ---")
        print(self.df['category'].value_counts().head(10))
        
        print("\n--- Payment Mode Distribution ---")
        print(self.df['payment_mode'].value_counts())
        
        print("\n--- Location Distribution (Top 10) ---")
        print(self.df['location'].value_counts().head(10))
        
        print("\n--- Amount Statistics ---")
        print(f"Min: ₹{self.df['amount'].min():.2f}")
        print(f"Max: ₹{self.df['amount'].max():.2f}")
        print(f"Mean: ₹{self.df['amount'].mean():.2f}")
        print(f"Median: ₹{self.df['amount'].median():.2f}")
        
        print("\n--- Missing Values ---")
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("No missing values!")
        
        print("="*60)
    
    def save_cleaned_data(self, output_path='cleaned_data.csv'):
        """Save cleaned data to CSV"""
        self.df.to_csv(output_path, index=False)
        print(f"\n✓ Cleaned data saved to: {output_path}")
        print(f"  File size: {len(self.df)} rows × {len(self.df.columns)} columns")


# =================== MAIN EXECUTION ===================

if __name__ == "__main__":
    print("\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█" + " "*15 + "BUDGETWISE DATA PREPROCESSING" + " "*15 + "█")
    print("█" + " "*58 + "█")
    print("█"*60 + "\n")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor('budgetwise_finance_dataset.csv')
    
    # Step 1: Clean the data
    cleaned_df = preprocessor.clean_data()
    
    # Step 2: Handle class imbalance - SET TO 'none' FOR BETTER PREDICTIONS
    # Use 'smote' for balanced training data, or 'none' to keep real-world imbalance
    balanced_df = preprocessor.handle_class_imbalance(balance_method='none')
    
    # Step 3: Generate summary report
    preprocessor.generate_summary_report()
    
    # Step 4: Save cleaned data
    preprocessor.save_cleaned_data('cleaned_data.csv')
    
    print("\n✓ Preprocessing complete! You can now use 'cleaned_data.csv' for model training.")
    print("\n" + "█"*60 + "\n")