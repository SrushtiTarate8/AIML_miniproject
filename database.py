# import mysql.connector
# from mysql.connector import Error

# class Database:
#     def __init__(self):
#         self.host = "localhost"
#         self.user = "root"
#         self.password = "Feyoni@1819"  # Change this
#         self.database = "budgetwise"
    
#     def get_connection(self):
#         try:
#             connection = mysql.connector.connect(
#                 host=self.host,
#                 user=self.user,
#                 password=self.password,
#                 database=self.database
#             )
#             return connection
#         except Error as e:
#             print(f"Error: {e}")
#             return None
        

# # if __name__ == "__main__":
# #     db = Database()
# #     conn = db.get_connection()
# #     if conn:
# #         print("✅ Database connection successful!")
# #         conn.close()
# #     else:
# #         print("❌ Failed to connect to the database.")

import mysql.connector
from mysql.connector import Error

class Database:
    def __init__(self):
        self.host = "localhost"
        self.user = "root"
        self.password = "smt@050308"  # Change this
        self.database = "budgetwise"
    
    def get_connection(self):
        try:
            connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            return connection
        except Error as e:
            print(f"Error connecting to database: {e}")
            return None
    
    def initialize_database(self):
        """Create database and tables if they don't exist"""
        try:
            # Connect without database to create it
            connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password
            )
            cursor = connection.cursor()
            
            # Create database
            cursor.execute("CREATE DATABASE IF NOT EXISTS budgetwise")
            cursor.execute("USE budgetwise")
            
            # Create users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INT PRIMARY KEY AUTO_INCREMENT,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create transactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    transaction_id INT PRIMARY KEY AUTO_INCREMENT,
                    user_id INT NOT NULL,
                    date DATE NOT NULL,
                    transaction_type ENUM('Income', 'Expense') NOT NULL,
                    category VARCHAR(50) NOT NULL,
                    amount DECIMAL(10,2) NOT NULL,
                    payment_mode VARCHAR(30),
                    location VARCHAR(50),
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                    INDEX idx_user_date (user_id, date),
                    INDEX idx_transaction_type (transaction_type),
                    INDEX idx_category (category)
                )
            """)
            
            # Create predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id INT PRIMARY KEY AUTO_INCREMENT,
                    user_id INT NOT NULL,
                    prediction_type VARCHAR(50) NOT NULL,
                    predicted_amount DECIMAL(10,2) NOT NULL,
                    month_year VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                    INDEX idx_user_type (user_id, prediction_type)
                )
            """)
            

            # Create savings_goals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS savings_goals (
                    goal_id INT PRIMARY KEY AUTO_INCREMENT,
                    user_id INT NOT NULL,
                    goal_name VARCHAR(255) NOT NULL,
                    target_amount DECIMAL(15,2) NOT NULL,
                    current_amount DECIMAL(15,2) DEFAULT 0,
                    deadline DATE,
                    icon VARCHAR(10) DEFAULT '🎯',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                    INDEX idx_user_goals (user_id)
                )
            """)


            # Create savings_contributions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS savings_contributions (
                    contribution_id INT PRIMARY KEY AUTO_INCREMENT,
                    goal_id INT NOT NULL,
                    amount DECIMAL(15,2) NOT NULL,
                    contribution_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    notes VARCHAR(500),
                    FOREIGN KEY (goal_id) REFERENCES savings_goals(goal_id) ON DELETE CASCADE,
                    INDEX idx_goal_contributions (goal_id)
                )
            """)


            # Create budget_goals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS budget_goals (
                    goal_id INT PRIMARY KEY AUTO_INCREMENT,
                    user_id INT NOT NULL,
                    category VARCHAR(50) NOT NULL,
                    budget_limit DECIMAL(10,2) NOT NULL,
                    month_year VARCHAR(20) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                    UNIQUE KEY unique_user_category_month (user_id, category, month_year),
                    INDEX idx_user_month (user_id, month_year)
                )
            """)
            
            connection.commit()
            cursor.close()
            connection.close()
            
            print("✓ Database and tables initialized successfully!")
            return True
            
        except Error as e:
            print(f"Error initializing database: {e}")
            return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("DATABASE INITIALIZATION")
    print("="*60 + "\n")
    
    db = Database()
    if db.initialize_database():
        print("\n✓ All tables created successfully!")
        print("\nTables created:")
        print("  1. users")
        print("  2. transactions")
        print("  3. predictions")
        print("  4. budget_goals")
        print("  5. savings_goals")
        print("  6. savings_contributions")
    else:
        print("\n✗ Database initialization failed!")
    
    print("\n" + "="*60 + "\n")




