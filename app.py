
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file
import bcrypt
from database import Database
from models import ImprovedExpensePredictor
from report_generator import PDFReportGenerator
import pandas as pd
from datetime import datetime
import os
import tempfile
from recommendation_engine import RecommendationEngine




#new imports for ocr


from werkzeug.utils import secure_filename
from bill_ocr_processor import BillOCRProcessor
import uuid


#initalisation
app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production-2024'



#api key

GEMINI_API_KEY = "AIzaSyCrfULEpZOgk5PxjO5-lk_pFN_2Xgc77CE"  # Get from https://makersuite.google.com/app/apikey
recommendation_engine = RecommendationEngine(GEMINI_API_KEY)


#for ocr
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'tiff'}

# Create upload directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize OCR processor
ocr_processor = BillOCRProcessor()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#database
db = Database()
predictor = ImprovedExpensePredictor()

# Load models at startup
try:
    predictor.load_models()
    print("✓ ML Models loaded successfully!")
except Exception as e:
    print(f"⚠ Warning: Could not load models - {e}")
    print("  Please run 'python models.py' first to train and save models.")

# =================== AUTH ROUTES ===================

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('home'))
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        conn = db.get_connection()
        if not conn:
            return "Database connection failed", 500
            
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
                (username, email, hashed)
            )
            conn.commit()
            return redirect(url_for('login'))
        except Exception as e:
            print(f"Signup error: {e}")
            return "Username or email already exists", 400
        finally:
            cursor.close()
            conn.close()
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = db.get_connection()
        if not conn:
            return "Database connection failed", 500
            
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
            session['user_id'] = user['user_id']
            session['username'] = user['username']
            return redirect(url_for('home'))
        else:
            return "Invalid credentials", 401
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# =================== HOME & TRANSACTION ROUTES: home page ===================

@app.route('/home')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('home.html', username=session['username'])

@app.route('/add_transaction', methods=['POST'])
def add_transaction():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    data = request.form
    user_id = session['user_id']
    
    conn = db.get_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
        
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO transactions 
            (user_id, date, transaction_type, category, amount, payment_mode, location, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            user_id, data['date'], data['transaction_type'], 
            data['category'], data['amount'], data['payment_mode'],
            data.get('location', ''), data.get('notes', '')
        ))
        conn.commit()
    except Exception as e:
        print(f"Add transaction error: {e}")
        return jsonify({'error': 'Failed to add transaction'}), 500
    finally:
        cursor.close()
        conn.close()
    
    return redirect(url_for('home', success='true'))




#ocr functionalities

@app.route('/upload-bill', methods=['POST'])
def upload_bill():
    """Handle bill image upload and OCR processing"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    # Check if file is present
    if 'bill_image' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    
    file = request.files['bill_image']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    # Validate file type
    if not allowed_file(file.filename):
        return jsonify({
            'success': False, 
            'error': 'Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF, WEBP, BMP, TIFF)'
        }), 400
    
    try:
        # Generate unique filename
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        unique_filename = f"{session['user_id']}_{timestamp}_{unique_id}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save file
        file.save(filepath)
        print(f"📁 File saved: {filepath}")
        
        # Process with OCR
        print("🔄 Starting OCR processing...")
        result = ocr_processor.process_uploaded_file(filepath)
        
        # Keep the file for debugging (optional - remove in production)
        # Uncomment the following to delete file after processing:
        # try:
        #     os.remove(filepath)
        #     print(f"🗑️ Cleaned up: {filepath}")
        # except Exception as e:
        #     print(f"⚠️ Cleanup failed: {e}")
        
        if result['success']:
            print("✅ OCR processing successful!")
            return jsonify({
                'success': True,
                'data': result['data'],
                'message': 'Bill processed successfully! Please review and confirm the extracted data.',
                'confidence': result.get('confidence', 'medium')
            })
        else:
            print(f"❌ OCR processing failed: {result.get('error')}")
            return jsonify({
                'success': False,
                'error': result.get('error', 'Failed to process bill. Please try again or enter manually.')
            }), 400
            
    except Exception as e:
        print(f"❌ Upload error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Failed to process upload: {str(e)}'
        }), 500

@app.route('/ocr-test')
def ocr_test():
    """Test route to check if OCR is working"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    return render_template('ocr_test.html')

# OCR status check
@app.route('/api/ocr-status')
def ocr_status():
    """Check if Tesseract OCR is properly installed"""
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        return jsonify({
            'status': 'ok',
            'tesseract_installed': True,
            'version': str(version)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'tesseract_installed': False,
            'error': str(e),
            'message': 'Please install Tesseract OCR. Visit: https://github.com/tesseract-ocr/tesseract'
        }), 500



# =================== PREDICTION ROUTES: Home Page ===================

@app.route('/predict/expense')
def predict_expense():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    user_id = session['user_id']
    
    conn = db.get_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
        
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM transactions WHERE user_id = %s ORDER BY date", (user_id,))
    transactions = cursor.fetchall()
    cursor.close()
    conn.close()
    
    if not transactions:
        return jsonify({
            'error': 'No transaction history found. Add some transactions first!'
        }), 400
    
    try:
        user_df = pd.DataFrame(transactions)
        user_df['date'] = pd.to_datetime(user_df['date'])
        
        expense_df = user_df[user_df['transaction_type'] == 'Expense']
        
        if len(expense_df) < 5:
            return jsonify({
                'error': 'Need at least 5 expense transactions for prediction. Please add more data.'
            }), 400
        
        unique_months = user_df['date'].dt.to_period('M').nunique()
        
        if unique_months < 2:
            avg_monthly = expense_df.groupby(expense_df['date'].dt.to_period('M'))['amount'].sum().mean()
            predicted = round(avg_monthly, 2)
            message = "⚠️ Limited historical data. Using simple average. Add more months for better predictions."
        else:
            try:
                if predictor.xgb_model is not None:
                    predicted = predictor.predict_monthly_expense(user_df)
                    message = None
                else:
                    monthly_expenses = expense_df.groupby(expense_df['date'].dt.to_period('M'))['amount'].sum()
                    predicted = round(monthly_expenses.mean(), 2)
                    message = "Using statistical prediction. Train ML models for better accuracy."
            except Exception as e:
                print(f"ML prediction failed: {e}")
                monthly_expenses = expense_df.groupby(expense_df['date'].dt.to_period('M'))['amount'].sum()
                predicted = round(monthly_expenses.mean(), 2)
                message = "ML prediction unavailable. Using statistical average."
        
        conn = db.get_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions (user_id, prediction_type, predicted_amount, month_year)
                VALUES (%s, %s, %s, %s)
            """, (user_id, 'monthly_expense', predicted, datetime.now().strftime('%B %Y')))
            conn.commit()
            cursor.close()
            conn.close()
        
        response = {'predicted_expense': float(predicted)}
        if message:
            response['message'] = message
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/predict/savings')
def predict_savings():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    user_id = session['user_id']
    
    conn = db.get_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
        
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM transactions WHERE user_id = %s ORDER BY date", (user_id,))
    transactions = cursor.fetchall()
    cursor.close()
    conn.close()
    
    if not transactions:
        return jsonify({
            'error': 'No transaction history found. Add some transactions first!'
        }), 400
    
    try:
        user_df = pd.DataFrame(transactions)
        user_df['date'] = pd.to_datetime(user_df['date'])
        
        income_df = user_df[user_df['transaction_type'] == 'Income']
        expense_df = user_df[user_df['transaction_type'] == 'Expense']
        
        if len(income_df) == 0:
            return jsonify({
                'error': 'No income transactions found. Add income data first!'
            }), 400
        
        monthly_income = income_df.groupby(income_df['date'].dt.to_period('M'))['amount'].sum()
        avg_income = round(monthly_income.mean(), 2)
        
        if len(expense_df) > 0:
            monthly_expense = expense_df.groupby(expense_df['date'].dt.to_period('M'))['amount'].sum()
            avg_expense = round(monthly_expense.mean(), 2)
        else:
            avg_expense = 0
        
        try:
            if predictor.xgb_model is not None and len(expense_df) >= 5:
                predicted_expense = predictor.predict_monthly_expense(user_df)
            else:
                predicted_expense = avg_expense
        except:
            predicted_expense = avg_expense
        
        predicted_savings = round(avg_income - predicted_expense, 2)
        
        conn = db.get_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions (user_id, prediction_type, predicted_amount, month_year)
                VALUES (%s, %s, %s, %s)
            """, (user_id, 'monthly_savings', predicted_savings, datetime.now().strftime('%B %Y')))
            conn.commit()
            cursor.close()
            conn.close()
        
        return jsonify({
            'predicted_savings': float(predicted_savings),
            'predicted_income': float(avg_income),
            'predicted_expense': float(predicted_expense)
        })
        
    except Exception as e:
        print(f"Savings prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Savings prediction failed: {str(e)}'
        }), 500

@app.route('/predict/forecast')
def predict_forecast():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    user_id = session['user_id']
    
    conn = db.get_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
        
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM transactions WHERE user_id = %s ORDER BY date", (user_id,))
    transactions = cursor.fetchall()
    cursor.close()
    conn.close()
    
    if not transactions:
        return jsonify({
            'error': 'No transaction history found. Add some transactions first!'
        }), 400
    
    try:
        user_df = pd.DataFrame(transactions)
        user_df['date'] = pd.to_datetime(user_df['date'])
        
        income_df = user_df[user_df['transaction_type'] == 'Income']
        expense_df = user_df[user_df['transaction_type'] == 'Expense']
        
        monthly_income = income_df.groupby(income_df['date'].dt.to_period('M'))['amount'].sum()
        monthly_expense = expense_df.groupby(expense_df['date'].dt.to_period('M'))['amount'].sum()
        
        avg_income = round(monthly_income.mean(), 2) if len(monthly_income) > 0 else 0
        avg_expense = round(monthly_expense.mean(), 2) if len(monthly_expense) > 0 else 0
        
        forecast = []
        current_date = datetime.now()
        
        for i in range(1, 4):
            future_month = current_date.month + i
            future_year = current_date.year
            
            while future_month > 12:
                future_month -= 12
                future_year += 1
            
            month_name = datetime(future_year, future_month, 1).strftime('%B %Y')
            
            import random
            variation = 1 + (random.random() * 0.1 - 0.05)
            
            predicted_income = round(avg_income * variation, 2)
            predicted_expense = round(avg_expense * variation, 2)
            predicted_savings = round(predicted_income - predicted_expense, 2)
            
            forecast.append({
                'month': month_name,
                'predicted_expense': predicted_expense,
                'predicted_income': predicted_income,
                'predicted_savings': predicted_savings
            })
        
        return jsonify({'forecast': forecast})
        
    except Exception as e:
        print(f"Forecast error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Forecast failed: {str(e)}'
        }), 500
    


# =================== BUDGET GOALS ROUTES : Dashboard ===================

@app.route('/api/budget-goals', methods=['GET'])
def get_budget_goals():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    user_id = session['user_id']
    
    conn = db.get_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("""
        SELECT * FROM budget_goals 
        WHERE user_id = %s 
        ORDER BY created_at DESC
    """, (user_id,))
    goals = cursor.fetchall()
    
    for goal in goals:
        month_year = goal['month_year']
        year, month = map(int, month_year.split('-'))
        
        cursor.execute("""
            SELECT COALESCE(SUM(amount), 0) as spent
            FROM transactions
            WHERE user_id = %s 
            AND category = %s
            AND transaction_type = 'Expense'
            AND YEAR(date) = %s
            AND MONTH(date) = %s
        """, (user_id, goal['category'], year, month))
        
        result = cursor.fetchone()
        goal['spent'] = float(result['spent'])
        goal['budget_limit'] = float(goal['budget_limit'])
    
    cursor.close()
    conn.close()
    
    return jsonify({'goals': goals})

@app.route('/api/budget-goals', methods=['POST'])
def create_budget_goal():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    user_id = session['user_id']
    data = request.json
    
    conn = db.get_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO budget_goals (user_id, category, budget_limit, month_year)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE budget_limit = %s
        """, (
            user_id,
            data['category'],
            data['budget_limit'],
            data['month_year'],
            data['budget_limit']
        ))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error creating budget goal: {e}")
        cursor.close()
        conn.close()
        return jsonify({'error': str(e)}), 500

@app.route('/api/budget-goals/<int:goal_id>', methods=['DELETE'])
def delete_budget_goal(goal_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    user_id = session['user_id']
    
    conn = db.get_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            DELETE FROM budget_goals 
            WHERE goal_id = %s AND user_id = %s
        """, (goal_id, user_id))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting budget goal: {e}")
        cursor.close()
        conn.close()
        return jsonify({'error': str(e)}), 500




# =================== SAVINGS GOALS ROUTES ===================
# Place these routes in app.py after your Budget Goals routes

@app.route('/api/savings-goals', methods=['GET'])
def get_savings_goals():
    """Get all savings goals for the current user"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    user_id = session['user_id']
    
    conn = db.get_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    cursor = conn.cursor(dictionary=True)
    
    try:
        # Get all savings goals with their contributions
        cursor.execute("""
            SELECT 
                sg.goal_id,
                sg.goal_name,
                sg.target_amount,
                sg.current_amount,
                sg.deadline,
                sg.created_at,
                sg.icon,
                COALESCE(
                    (SELECT SUM(amount) 
                     FROM savings_contributions 
                     WHERE goal_id = sg.goal_id), 0
                ) as total_contributed
            FROM savings_goals sg
            WHERE sg.user_id = %s
            ORDER BY sg.created_at DESC
        """, (user_id,))
        
        goals = cursor.fetchall()
        
        # Calculate progress for each goal
        for goal in goals:
            goal['target_amount'] = float(goal['target_amount'])
            goal['current_amount'] = float(goal['current_amount'])
            goal['total_contributed'] = float(goal['total_contributed'])
            
            # Calculate days remaining
            if goal['deadline']:
                days_remaining = (goal['deadline'] - datetime.now().date()).days
                goal['days_remaining'] = max(0, days_remaining)
            else:
                goal['days_remaining'] = None
            
            # Calculate progress percentage
            if goal['target_amount'] > 0:
                goal['progress_percentage'] = min(
                    (goal['current_amount'] / goal['target_amount']) * 100, 
                    100
                )
            else:
                goal['progress_percentage'] = 0
            
            # Remaining amount
            goal['remaining_amount'] = max(0, goal['target_amount'] - goal['current_amount'])
        
        cursor.close()
        conn.close()
        
        return jsonify({'goals': goals})
        
    except Exception as e:
        print(f"Error getting savings goals: {e}")
        import traceback
        traceback.print_exc()
        cursor.close()
        conn.close()
        return jsonify({'error': str(e)}), 500


@app.route('/api/savings-goals', methods=['POST'])
def create_savings_goal():
    """Create a new savings goal"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    user_id = session['user_id']
    data = request.json
    
    conn = db.get_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO savings_goals 
            (user_id, goal_name, target_amount, current_amount, deadline, icon)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            user_id,
            data['goal_name'],
            data['target_amount'],
            0,  # Initial current_amount is 0
            data.get('deadline'),
            data.get('icon', '🎯')
        ))
        
        conn.commit()
        goal_id = cursor.lastrowid
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'goal_id': goal_id
        })
        
    except Exception as e:
        print(f"Error creating savings goal: {e}")
        cursor.close()
        conn.close()
        return jsonify({'error': str(e)}), 500


@app.route('/api/savings-goals/<int:goal_id>', methods=['DELETE'])
def delete_savings_goal(goal_id):
    """Delete a savings goal"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    user_id = session['user_id']
    
    conn = db.get_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    cursor = conn.cursor()
    
    try:
        # Delete contributions first (foreign key constraint)
        cursor.execute("""
            DELETE FROM savings_contributions 
            WHERE goal_id = %s
        """, (goal_id,))
        
        # Delete the goal
        cursor.execute("""
            DELETE FROM savings_goals 
            WHERE goal_id = %s AND user_id = %s
        """, (goal_id, user_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"Error deleting savings goal: {e}")
        cursor.close()
        conn.close()
        return jsonify({'error': str(e)}), 500


@app.route('/api/savings-goals/<int:goal_id>/contribute', methods=['POST'])
def add_savings_contribution(goal_id):
    """Add money to a savings goal"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    user_id = session['user_id']
    data = request.json
    
    conn = db.get_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    cursor = conn.cursor(dictionary=True)
    
    try:
        amount = float(data['amount'])
        
        if amount <= 0:
            return jsonify({'error': 'Amount must be greater than 0'}), 400
        
        # Verify goal belongs to user
        cursor.execute("""
            SELECT goal_id, current_amount, target_amount 
            FROM savings_goals 
            WHERE goal_id = %s AND user_id = %s
        """, (goal_id, user_id))
        
        goal = cursor.fetchone()
        
        if not goal:
            cursor.close()
            conn.close()
            return jsonify({'error': 'Savings goal not found'}), 404
        
        # Add contribution record
        cursor.execute("""
            INSERT INTO savings_contributions 
            (goal_id, amount, contribution_date, notes)
            VALUES (%s, %s, %s, %s)
        """, (
            goal_id,
            amount,
            datetime.now(),
            data.get('notes', '')
        ))
        
        # Update current amount in savings_goals
        new_amount = float(goal['current_amount']) + amount
        cursor.execute("""
            UPDATE savings_goals 
            SET current_amount = %s
            WHERE goal_id = %s
        """, (new_amount, goal_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'new_amount': new_amount
        })
        
    except Exception as e:
        print(f"Error adding contribution: {e}")
        import traceback
        traceback.print_exc()
        cursor.close()
        conn.close()
        return jsonify({'error': str(e)}), 500


@app.route('/api/savings-goals/<int:goal_id>/contributions', methods=['GET'])
def get_savings_contributions(goal_id):
    """Get all contributions for a specific savings goal"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    user_id = session['user_id']
    
    conn = db.get_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    cursor = conn.cursor(dictionary=True)
    
    try:
        # Verify goal belongs to user
        cursor.execute("""
            SELECT goal_id FROM savings_goals 
            WHERE goal_id = %s AND user_id = %s
        """, (goal_id, user_id))
        
        if not cursor.fetchone():
            cursor.close()
            conn.close()
            return jsonify({'error': 'Savings goal not found'}), 404
        
        # Get contributions
        cursor.execute("""
            SELECT 
                contribution_id,
                amount,
                contribution_date,
                notes
            FROM savings_contributions
            WHERE goal_id = %s
            ORDER BY contribution_date DESC
        """, (goal_id,))
        
        contributions = cursor.fetchall()
        
        for contrib in contributions:
            contrib['amount'] = float(contrib['amount'])
        
        cursor.close()
        conn.close()
        
        return jsonify({'contributions': contributions})
        
    except Exception as e:
        print(f"Error getting contributions: {e}")
        cursor.close()
        conn.close()
        return jsonify({'error': str(e)}), 500




# =================== AI Recommendations: Dashboards ===================


@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    """Generate personalized financial recommendations for the user"""
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    user_id = session['user_id']
    
    conn = db.get_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    cursor = conn.cursor(dictionary=True)
    
    try:
        # Get user transactions
        cursor.execute("""
            SELECT * FROM transactions 
            WHERE user_id = %s 
            ORDER BY date DESC
        """, (user_id,))
        transactions = cursor.fetchall()
        
        # Get savings goals
        cursor.execute("""
            SELECT goal_id, goal_name, target_amount, current_amount, deadline, icon
            FROM savings_goals 
            WHERE user_id = %s
        """, (user_id,))
        savings_goals = cursor.fetchall()
        
        # Get budget goals with spent amounts
        cursor.execute("""
            SELECT * FROM budget_goals 
            WHERE user_id = %s 
            ORDER BY created_at DESC
        """, (user_id,))
        budget_goals = cursor.fetchall()
        
        # Calculate spent for each budget goal
        for goal in budget_goals:
            month_year = goal['month_year']
            year, month = map(int, month_year.split('-'))
            
            cursor.execute("""
                SELECT COALESCE(SUM(amount), 0) as spent
                FROM transactions
                WHERE user_id = %s 
                AND category = %s
                AND transaction_type = 'Expense'
                AND YEAR(date) = %s
                AND MONTH(date) = %s
            """, (user_id, goal['category'], year, month))
            
            result = cursor.fetchone()
            goal['spent'] = float(result['spent'])
            goal['budget_limit'] = float(goal['budget_limit'])
        
        cursor.close()
        conn.close()
        
        # Check if user has any data
        if not transactions:
            return jsonify({
                'recommendations': [
                    "🎯 Start Tracking: Add your first transaction to begin receiving personalized recommendations!",
                    "💰 Set a Savings Goal: Create a savings goal to visualize your financial targets.",
                    "📊 Create a Budget: Set monthly budget limits for different spending categories.",
                    "💡 Log Income: Don't forget to track your income sources for accurate savings calculations.",
                    "📈 Build Habits: Consistent tracking leads to better financial insights and smarter decisions."
                ]
            })
        
        # Generate profile and recommendations
        profile = recommendation_engine.analyze_user_profile(
            transactions, 
            savings_goals, 
            budget_goals
        )
        
        recommendations = recommendation_engine.generate_recommendations(
            profile,
            savings_goals,
            budget_goals
        )
        
        return jsonify({
            'recommendations': recommendations,
            'profile': {
                'savings_rate': round(profile['savings_rate'], 1),
                'monthly_income': round(profile['monthly_avg_income'], 2),
                'monthly_expense': round(profile['monthly_avg_expense'], 2)
            }
        })
        
    except Exception as e:
        print(f"Error generating recommendations: {e}")
        import traceback
        traceback.print_exc()
        
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        
        # Return fallback recommendations on error
        return jsonify({
            'recommendations': [
                "💰 Review Your Spending: Take time to analyze your transaction history and identify patterns.",
                "📊 Set Monthly Budgets: Create budget limits for your top spending categories.",
                "🎯 Define Savings Goals: Set specific financial targets to stay motivated.",
                "💡 Track Regularly: Make it a habit to log transactions weekly for better insights.",
                "📈 Monitor Progress: Check your dashboard monthly to see how you're doing."
            ]
        }), 200



# =================== PDF REPORT GENERATION: Dashboard ===================

@app.route('/generate-report')
def generate_report():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    user_id = session['user_id']
    username = session['username']
    
    conn = db.get_connection()
    if not conn:
        return jsonify({'error': 'Database connection failed'}), 500
    
    cursor = conn.cursor(dictionary=True)
    
    # Get transactions
    cursor.execute("SELECT * FROM transactions WHERE user_id = %s ORDER BY date DESC", (user_id,))
    transactions = cursor.fetchall()
    
    # Get budget goals
    cursor.execute("""
        SELECT * FROM budget_goals 
        WHERE user_id = %s 
        ORDER BY created_at DESC
    """, (user_id,))
    goals = cursor.fetchall()
    
    # Calculate spent for each goal
    for goal in goals:
        month_year = goal['month_year']
        year, month = map(int, month_year.split('-'))
        
        cursor.execute("""
            SELECT COALESCE(SUM(amount), 0) as spent
            FROM transactions
            WHERE user_id = %s 
            AND category = %s
            AND transaction_type = 'Expense'
            AND YEAR(date) = %s
            AND MONTH(date) = %s
        """, (user_id, goal['category'], year, month))
        
        result = cursor.fetchone()
        goal['spent'] = float(result['spent'])
        goal['budget_limit'] = float(goal['budget_limit'])
    
    cursor.close()
    conn.close()
    
    # Calculate stats
    df = pd.DataFrame(transactions) if transactions else pd.DataFrame()
    
    stats = {
        'total_income': 0,
        'total_expense': 0,
        'savings_rate': 0,
        'transaction_count': len(transactions)
    }
    
    if not df.empty:
        stats['total_income'] = float(df[df['transaction_type'] == 'Income']['amount'].sum())
        stats['total_expense'] = float(df[df['transaction_type'] == 'Expense']['amount'].sum())
        if stats['total_income'] > 0:
            stats['savings_rate'] = ((stats['total_income'] - stats['total_expense']) / stats['total_income']) * 100
    
    # Generate PDF
    try:
        user_data = {'username': username}
        
        # Create temporary file
        temp_dir = tempfile.gettempdir()
        filename = os.path.join(temp_dir, f"BudgetWise_Report_{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        
        generator = PDFReportGenerator(user_data, transactions, stats, goals)
        generator.generate_report(filename)
        
        return send_file(
            filename,
            as_attachment=True,
            download_name=f"BudgetWise_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mimetype='application/pdf'
        )
    except Exception as e:
        print(f"PDF generation error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to generate report: {str(e)}'}), 500

# =================== DASHBOARD ROUTE ===================

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    
    conn = db.get_connection()
    if not conn:
        return "Database connection failed", 500
        
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("SELECT * FROM transactions WHERE user_id = %s ORDER BY date DESC", (user_id,))
    transactions = cursor.fetchall()
    
    cursor.execute("""
        SELECT * FROM predictions 
        WHERE user_id = %s 
        ORDER BY created_at DESC 
        LIMIT 10
    """, (user_id,))
    predictions = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    df = pd.DataFrame(transactions) if transactions else pd.DataFrame()
    
    stats = {
        'total_income': 0,
        'total_expense': 0,
        'savings_rate': 0,
        'transaction_count': len(transactions)
    }
    
    if not df.empty:
        stats['total_income'] = float(df[df['transaction_type'] == 'Income']['amount'].sum())
        stats['total_expense'] = float(df[df['transaction_type'] == 'Expense']['amount'].sum())
        if stats['total_income'] > 0:
            stats['savings_rate'] = ((stats['total_income'] - stats['total_expense']) / stats['total_income']) * 100
    
    return render_template('dashboard.html', 
                         transactions=transactions,
                         predictions=predictions,
                         stats=stats)

# =================== API HEALTH CHECK ===================

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': predictor.xgb_model is not None,
        'database_connected': db.get_connection() is not None
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting BudgetWise Application")
    print("="*60)
    print("\n📊 Server running at: http://127.0.0.1:5000")
    print("📝 Press CTRL+C to stop\n")
    app.run(debug=True, host='0.0.0.0', port=5000)