






import cv2
import pytesseract
import re
import os
from PIL import Image
from datetime import datetime
import numpy as np

# Tesseract installation path (update based on your OS)
# Windows: r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Mac: /usr/local/bin/tesseract or /opt/homebrew/bin/tesseract
# Linux: /usr/bin/tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class BillOCRProcessor:
    def __init__(self):
        """Initialize the Bill OCR Processor with patterns and keywords"""
        
        # Regex patterns for extracting information
        self.patterns = {
            # Date patterns (DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD, etc.)
            "date": [
                r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b',  # DD/MM/YYYY or DD-MM-YYYY
                r'\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b',    # YYYY-MM-DD
                r'\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{2,4})\b',  # DD Mon YYYY
            ],
            
            # Amount patterns (₹, Rs, Rs., INR, with commas and decimals)
            "amount": [
                r'(?:Total|Amount|Grand Total|Net Amount|Payable|Bill Amount|Sum)[\s:]*(?:Rs\.?|₹|INR)?\s*([\d,]+\.?\d*)',
                r'(?:Rs\.?|₹|INR)\s*([\d,]+\.?\d*)',
                r'\b([\d,]+\.\d{2})\b',  # Decimal amounts
            ],
            
            # Payment mode patterns
            "payment": [
                r'\b(UPI|GPAY|Google Pay|PhonePe|Paytm|BHIM)\b',
                r'\b(Cash|CASH)\b',
                r'\b(Credit Card|CREDIT|Visa|Mastercard|Amex)\b',
                r'\b(Debit Card|DEBIT)\b',
                r'\b(NEFT|RTGS|IMPS|Bank Transfer|Net Banking)\b',
            ],
        }
        
        # Category keywords mapping
        self.category_keywords = {
            'Food & Dining': [
                'restaurant', 'cafe', 'food', 'dining', 'meal', 'lunch', 'dinner', 
                'breakfast', 'swiggy', 'zomato', 'dominos', 'pizza', 'burger', 
                'mcdonald', 'kfc', 'subway', 'starbucks', 'biryani', 'hotel'
            ],
            'Groceries': [
                'grocery', 'supermarket', 'mart', 'store', 'reliance', 'dmart', 
                'big bazaar', 'vegetables', 'fruits', 'more', 'spencers', 'kirana',
                'provisions'
            ],
            'Transportation': [
                'uber', 'ola', 'taxi', 'cab', 'metro', 'bus', 'fuel', 'petrol', 
                'diesel', 'parking', 'toll', 'rapido', 'auto', 'transport'
            ],
            'Shopping': [
                'shopping', 'mall', 'fashion', 'clothing', 'shoes', 'amazon', 
                'flipkart', 'myntra', 'apparel', 'garment', 'footwear', 'lifestyle'
            ],
            'Entertainment': [
                'movie', 'cinema', 'pvr', 'inox', 'theatre', 'concert', 'event', 
                'game', 'entertainment', 'ticket', 'show', 'multiplex'
            ],
            'Healthcare': [
                'hospital', 'clinic', 'pharmacy', 'medical', 'doctor', 'medicine', 
                'apollo', 'medplus', 'health', 'diagnostic', 'lab', 'test'
            ],
            'Utilities': [
                'electricity', 'water', 'internet', 'mobile', 'recharge', 'airtel', 
                'jio', 'vi', 'utility', 'bill', 'broadband', 'wifi'
            ],
            'Education': [
                'school', 'college', 'university', 'course', 'books', 'tuition', 
                'education', 'academy', 'institute', 'learning', 'coaching'
            ],
            'Travel': [
                'hotel', 'flight', 'train', 'booking', 'makemytrip', 'goibibo', 
                'travel', 'irctc', 'oyo', 'resort', 'accommodation', 'ticket'
            ],
        }
        
        # Payment mode keyword mapping
        self.payment_keywords = {
            'UPI': ['upi', 'gpay', 'phonepe', 'paytm', 'bhim', 'google pay', 'phone pe'],
            'Credit Card': ['credit card', 'credit', 'visa', 'mastercard', 'amex', 'american express'],
            'Debit Card': ['debit card', 'debit', 'atm card'],
            'Cash': ['cash', 'cash payment'],
            'Bank Transfer': ['neft', 'rtgs', 'imps', 'bank transfer', 'netbanking', 'net banking'],
        }
    
    def preprocess_image(self, image_path):
        """
        Advanced image preprocessing for better OCR accuracy
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple preprocessing techniques for better accuracy
        
        # 1. Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # 2. Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(denoised)
        
        # 3. Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 4. Morphological operations to remove noise
        kernel = np.ones((1, 1), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return morph
    
    def extract_text(self, image_path):
        """
        Extract text from image using Tesseract OCR
        """
        try:
            # Try multiple preprocessing approaches
            processed_img = self.preprocess_image(image_path)
            
            # Use multiple PSM modes for better accuracy
            configs = [
                "--psm 6",  # Assume a single uniform block of text
                "--psm 4",  # Assume a single column of text of variable sizes
                "--psm 3",  # Fully automatic page segmentation
            ]
            
            texts = []
            for config in configs:
                text = pytesseract.image_to_string(processed_img, lang='eng', config=config)
                texts.append(text)
            
            # Combine all extracted texts
            full_text = "\n".join(texts)
            
            return full_text
            
        except Exception as e:
            print(f"OCR extraction error: {e}")
            return ""
    
    def extract_date(self, text):
        """Extract date from text"""
        for pattern in self.patterns['date']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    # Try to parse different date formats
                    groups = match.groups()
                    
                    # DD/MM/YYYY or DD-MM-YYYY
                    if len(groups) == 3 and groups[0].isdigit():
                        day, month, year = groups
                        if len(year) == 2:
                            year = '20' + year
                        
                        try:
                            date_obj = datetime(int(year), int(month), int(day))
                            return date_obj.strftime('%Y-%m-%d')
                        except:
                            pass
                    
                    # DD Mon YYYY
                    if len(groups) == 3 and not groups[1].isdigit():
                        day, month_str, year = groups
                        month_map = {
                            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                        }
                        month = month_map.get(month_str[:3].lower())
                        if month:
                            if len(year) == 2:
                                year = '20' + year
                            try:
                                date_obj = datetime(int(year), month, int(day))
                                return date_obj.strftime('%Y-%m-%d')
                            except:
                                pass
                
                except Exception as e:
                    print(f"Date parsing error: {e}")
                    continue
        
        # Default to today if no date found
        return datetime.now().strftime('%Y-%m-%d')
    
    def extract_amount(self, text):
        """Extract amount from text"""
        amounts = []
        
        for pattern in self.patterns['amount']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                amount_str = match.group(1)
                # Remove commas and convert to float
                try:
                    amount = float(amount_str.replace(',', ''))
                    amounts.append(amount)
                except:
                    pass
        
        # Return the maximum amount found (usually the total)
        return max(amounts) if amounts else 0.0
    
    def extract_payment_mode(self, text):
        """Extract payment mode from text"""
        text_lower = text.lower()
        
        # Check for payment mode keywords
        for payment_mode, keywords in self.payment_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return payment_mode
        
        return 'Other'
    
    def extract_category(self, text):
        """Extract category based on merchant name and keywords"""
        text_lower = text.lower()
        
        # Score each category
        category_scores = {}
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                category_scores[category] = score
        
        # Return category with highest score
        if category_scores:
            return max(category_scores, key=category_scores.get)
        
        return 'Other'
    
    def extract_location(self, text):
        """Extract merchant/location name"""
        lines = text.split('\n')
        
        # Usually merchant name is in first few lines
        for line in lines[:5]:
            line = line.strip()
            # Look for lines with significant text (potential merchant names)
            if len(line) > 3 and not line.isdigit():
                # Remove special characters
                clean_line = re.sub(r'[^\w\s]', '', line)
                if len(clean_line) > 3:
                    return clean_line[:200]  # Limit length
        
        return ''
    
    def extract_notes(self, text):
        """Extract additional notes like items purchased"""
        # Look for item lists or descriptions
        lines = text.split('\n')
        items = []
        
        for line in lines:
            line = line.strip()
            # Look for lines with items (usually have prices after them)
            if re.search(r'\d+\.?\d*$', line):
                items.append(line)
        
        if items:
            return '; '.join(items[:5])  # Limit to first 5 items
        
        return ''
    
    def parse_bill(self, text):
        """Parse all information from OCR text"""
        extracted_data = {
            'date': self.extract_date(text),
            'transaction_type': 'Expense',  # Most bills are expenses
            'category': self.extract_category(text),
            'amount': self.extract_amount(text),
            'payment_mode': self.extract_payment_mode(text),
            'location': self.extract_location(text),
            'notes': self.extract_notes(text)
        }
        
        return extracted_data
    
    def validate_and_clean(self, data):
        """Validate and clean extracted data"""
        cleaned = {}
        
        # Validate date
        try:
            datetime.strptime(data['date'], '%Y-%m-%d')
            cleaned['date'] = data['date']
        except:
            cleaned['date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Validate transaction type
        trans_type = data.get('transaction_type', 'Expense')
        cleaned['transaction_type'] = trans_type if trans_type in ['Expense', 'Income'] else 'Expense'
        
        # Validate category
        valid_categories = [
            'Food & Dining', 'Groceries', 'Transportation', 'Shopping',
            'Entertainment', 'Healthcare', 'Utilities', 'Education', 
            'Travel', 'Other', 'Salary', 'Investment', 'Gift'
        ]
        category = data.get('category', 'Other')
        cleaned['category'] = category if category in valid_categories else 'Other'
        
        # Validate amount
        try:
            amount = float(data.get('amount', 0))
            cleaned['amount'] = round(max(0, amount), 2)
        except:
            cleaned['amount'] = 0.0
        
        # Validate payment mode
        valid_payments = ['UPI', 'Cash', 'Credit Card', 'Debit Card', 'Bank Transfer', 'Other']
        payment = data.get('payment_mode', 'Other')
        cleaned['payment_mode'] = payment if payment in valid_payments else 'Other'
        
        # Clean text fields
        cleaned['location'] = str(data.get('location', ''))[:200]
        cleaned['notes'] = str(data.get('notes', ''))[:500]
        
        return cleaned
    
    def run_pipeline(self, image_path):
        """
        Run the complete OCR pipeline
        
        Args:
            image_path: Path to the bill image
            
        Returns:
            dict: Processing result with extracted data
        """
        try:
            # Step 1: Extract text using OCR
            print("📄 Extracting text from image...")
            text = self.extract_text(image_path)
            
            if not text or len(text.strip()) < 10:
                return {
                    'success': False,
                    'error': 'Could not extract sufficient text from image. Please ensure the image is clear and contains a bill/receipt.',
                    'raw_text': text
                }
            
            print("✓ Text extracted successfully")
            
            # Step 2: Parse bill information
            print("🔍 Parsing bill information...")
            parsed_data = self.parse_bill(text)
            print("✓ Bill information parsed")
            
            # Step 3: Validate and clean data
            print("✅ Validating data...")
            cleaned_data = self.validate_and_clean(parsed_data)
            print("✓ Data validated")
            
            return {
                'success': True,
                'data': cleaned_data,
                'raw_text': text,
                'confidence': 'medium'
            }
            
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': f'Failed to process bill: {str(e)}',
                'raw_text': ''
            }
    
    def process_uploaded_file(self, file_path):
        """
        Process an uploaded file and return transaction data
        
        Args:
            file_path: Path to uploaded image file
            
        Returns:
            dict: Processed transaction data
        """
        if not os.path.exists(file_path):
            return {
                'success': False,
                'error': 'File not found'
            }
        
        # Run the OCR pipeline
        result = self.run_pipeline(file_path)
        
        # Clean up uploaded file after processing (optional)
        # Uncomment if you want to delete the file after processing
        # try:
        #     os.remove(file_path)
        # except:
        #     pass
        
        return result


def process_bill_image(file_path):
    """
    Simple function to process a bill image and return transaction data
    
    Args:
        file_path (str): Path to the uploaded bill image
        
    Returns:
        dict: Processed transaction data ready for database insertion
    """
    ocr_processor = BillOCRProcessor()
    return ocr_processor.process_uploaded_file(file_path)


# Test function for development
if __name__ == "__main__":
    # Test the OCR pipeline
    test_image = "image.png"  # Replace with actual test image
    
    if os.path.exists(test_image):
        print("=" * 60)
        print("🧪 Testing Bill OCR Pipeline")
        print("=" * 60)
        
        result = process_bill_image(test_image)
        
        if result['success']:
            print("\n✅ SUCCESS!")
            print("\n📊 Extracted Data:")
            print("-" * 60)
            for key, value in result['data'].items():
                print(f"{key:20}: {value}")
            print("-" * 60)
            
            if 'raw_text' in result:
                print("\n📝 Raw OCR Text:")
                print("-" * 60)
                print(result['raw_text'][:500])  # First 500 chars
                print("-" * 60)
        else:
            print(f"\n❌ FAILED: {result['error']}")
    else:
        print(f"❌ Test image '{test_image}' not found!")
        print("Please place a test bill image and update the path.")