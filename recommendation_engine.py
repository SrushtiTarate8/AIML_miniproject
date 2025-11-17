# import google.generativeai as genai
# from datetime import datetime
# import json

# class RecommendationEngine:
#     def __init__(self, api_key):
#         """Initialize Gemini Flash for recommendations"""
#         genai.configure(api_key=api_key)
#         self.model = genai.GenerativeModel('gemini-1.5-flash')
    
#     def analyze_user_profile(self, transactions, savings_goals=None, budget_goals=None):
#         """
#         Analyze user's financial data and generate insights
        
#         Args:
#             transactions: List of user transactions
#             savings_goals: List of savings goals (optional)
#             budget_goals: List of budget goals (optional)
            
#         Returns:
#             dict with insights
#         """
#         if not transactions:
#             return {
#                 'total_income': 0,
#                 'total_expense': 0,
#                 'savings_rate': 0,
#                 'transaction_count': 0,
#                 'top_categories': {},
#                 'monthly_avg_expense': 0,
#                 'monthly_avg_income': 0
#             }
        
#         # Calculate basic metrics
#         total_income = sum(float(t['amount']) for t in transactions if t['transaction_type'] == 'Income')
#         total_expense = sum(float(t['amount']) for t in transactions if t['transaction_type'] == 'Expense')
        
#         savings_rate = ((total_income - total_expense) / total_income * 100) if total_income > 0 else 0
        
#         # Category breakdown
#         category_expenses = {}
#         for t in transactions:
#             if t['transaction_type'] == 'Expense':
#                 category = t['category']
#                 category_expenses[category] = category_expenses.get(category, 0) + float(t['amount'])
        
#         # Sort categories by spending
#         top_categories = dict(sorted(category_expenses.items(), key=lambda x: x[1], reverse=True)[:5])
        
#         # Calculate monthly averages
#         dates = [t['date'] for t in transactions if 'date' in t]
#         if dates:
#             unique_months = len(set((d.year, d.month) if hasattr(d, 'year') else (datetime.fromisoformat(str(d)).year, datetime.fromisoformat(str(d)).month) for d in dates))
#             unique_months = max(1, unique_months)
#         else:
#             unique_months = 1
        
#         monthly_avg_expense = total_expense / unique_months
#         monthly_avg_income = total_income / unique_months
        
#         return {
#             'total_income': total_income,
#             'total_expense': total_expense,
#             'savings_rate': savings_rate,
#             'transaction_count': len(transactions),
#             'top_categories': top_categories,
#             'monthly_avg_expense': monthly_avg_expense,
#             'monthly_avg_income': monthly_avg_income,
#             'unique_months': unique_months
#         }
    
#     def generate_recommendations(self, user_profile, savings_goals=None, budget_goals=None):
#         """
#         Generate personalized recommendations using Gemini
        
#         Args:
#             user_profile: Dict with user's financial metrics
#             savings_goals: List of savings goals
#             budget_goals: List of budget goals
            
#         Returns:
#             List of recommendations
#         """
#         try:
#             # Prepare category breakdown
#             category_breakdown = ""
#             if user_profile['top_categories']:
#                 category_breakdown = "\n".join([
#                     f"  - {cat}: ₹{amount:,.0f} ({amount/user_profile['total_expense']*100:.1f}%)"
#                     for cat, amount in user_profile['top_categories'].items()
#                 ])
            
#             # Prepare savings goals info
#             savings_info = ""
#             if savings_goals and len(savings_goals) > 0:
#                 savings_info = f"\nActive Savings Goals: {len(savings_goals)}"
#                 for goal in savings_goals[:3]:  # Show top 3
#                     progress = (goal.get('current_amount', 0) / goal.get('target_amount', 1)) * 100
#                     savings_info += f"\n  - {goal['goal_name']}: {progress:.0f}% complete (₹{goal.get('current_amount', 0):,.0f} / ₹{goal.get('target_amount', 0):,.0f})"
            
#             # Prepare budget goals info
#             budget_info = ""
#             if budget_goals and len(budget_goals) > 0:
#                 budget_info = f"\nActive Budget Goals: {len(budget_goals)}"
#                 for goal in budget_goals[:3]:
#                     spent_pct = (goal.get('spent', 0) / goal.get('budget_limit', 1)) * 100
#                     status = "⚠️ Over" if spent_pct > 100 else "✓ On track"
#                     budget_info += f"\n  - {goal['category']}: {spent_pct:.0f}% spent ({status})"
            
#             # Create prompt for Gemini
#             prompt = f"""You are a personal finance advisor. Analyze this user's financial profile and provide 5 personalized, actionable recommendations.

# USER FINANCIAL PROFILE:
# ━━━━━━━━━━━━━━━━━━━━
# Income & Expenses:
#   • Monthly Income: ₹{user_profile['monthly_avg_income']:,.0f}
#   • Monthly Expenses: ₹{user_profile['monthly_avg_expense']:,.0f}
#   • Savings Rate: {user_profile['savings_rate']:.1f}%
#   • Total Transactions: {user_profile['transaction_count']}

# Top Spending Categories:
# {category_breakdown}
# {savings_info}
# {budget_info}

# INSTRUCTIONS:
# 1. Provide exactly 5 recommendations
# 2. Each recommendation must be:
#    - Specific and actionable
#    - Include exact numbers (amounts, percentages)
#    - Focus on one clear action
#    - Be encouraging and positive
# 3. Cover different areas: spending reduction, savings increase, budgeting, goal achievement
# 4. Use Indian Rupee (₹) for all amounts
# 5. Be concise - max 2 sentences per recommendation

# Format each recommendation as:
# [Icon] [Title]: [Specific action with numbers]

# Example:
# 💰 Increase Emergency Savings: Build an emergency fund of ₹90,000 (6 months expenses). Start by saving ₹5,000 monthly from reducing entertainment expenses.

# Now provide 5 recommendations:"""

#             # Generate recommendations
#             response = self.model.generate_content(prompt)
#             recommendations_text = response.text.strip()
            
#             # Parse recommendations into list
#             recommendations = []
#             lines = recommendations_text.split('\n')
            
#             for line in lines:
#                 line = line.strip()
#                 if line and (line[0].isdigit() or any(emoji in line for emoji in ['💰', '🎯', '📊', '💡', '✨', '🏆', '📈', '💳', '🛡️', '🎁'])):
#                     # Remove numbering if present
#                     if line[0].isdigit() and '.' in line[:3]:
#                         line = line.split('.', 1)[1].strip()
                    
#                     recommendations.append(line)
            
#             # Ensure we have at least 3 recommendations
#             if len(recommendations) < 3:
#                 recommendations = [
#                     f"💰 Boost Your Savings: Your savings rate is {user_profile['savings_rate']:.1f}%. Try to increase it to 20% by saving ₹{(user_profile['monthly_avg_income'] * 0.20 - (user_profile['monthly_avg_income'] - user_profile['monthly_avg_expense'])):.0f} more per month.",
#                     f"📊 Track Top Expenses: Your highest spending category is {list(user_profile['top_categories'].keys())[0] if user_profile['top_categories'] else 'Unknown'} at ₹{list(user_profile['top_categories'].values())[0] if user_profile['top_categories'] else 0:,.0f}. Set a monthly limit to control this expense.",
#                     "🎯 Set Savings Goals: Create specific savings goals (emergency fund, vacation, gadgets) to stay motivated and track progress visually."
#                 ]
            
#             return recommendations[:5]  # Return max 5
            
#         except Exception as e:
#             print(f"Error generating recommendations: {e}")
#             import traceback
#             traceback.print_exc()
            
#             # Fallback recommendations
#             return [
#                 f"💰 Improve Savings Rate: Your current savings rate is {user_profile['savings_rate']:.1f}%. Aim for 20% by reducing discretionary spending.",
#                 f"📊 Review Top Expenses: You spent ₹{user_profile['total_expense']:,.0f} total. Focus on your top spending categories to find savings opportunities.",
#                 "🎯 Set Financial Goals: Create specific savings goals to give your money purpose and stay motivated.",
#                 "📈 Track Monthly Progress: Review your spending weekly to catch overspending early and adjust habits.",
#                 "💡 Build Emergency Fund: Aim for 6 months of expenses (₹{:.0f}) in a separate savings account for financial security.".format(user_profile['monthly_avg_expense'] * 6)
#             ]


# # Simple function for easy import
# def get_recommendations(api_key, transactions, savings_goals=None, budget_goals=None):
#     """
#     Simple function to get recommendations
    
#     Args:
#         api_key: Gemini API key
#         transactions: List of transactions
#         savings_goals: Optional savings goals
#         budget_goals: Optional budget goals
    
#     Returns:
#         List of recommendation strings
#     """
#     engine = RecommendationEngine(api_key)
#     profile = engine.analyze_user_profile(transactions, savings_goals, budget_goals)
#     recommendations = engine.generate_recommendations(profile, savings_goals, budget_goals)
#     return recommendations




import google.generativeai as genai
from datetime import datetime, timedelta
import json
from collections import defaultdict

class RecommendationEngine:
    def __init__(self, api_key):
        """Initialize Gemini Flash for recommendations"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def analyze_user_profile(self, transactions, savings_goals=None, budget_goals=None):
        """
        Advanced analysis of user's financial data with deep insights
        
        Args:
            transactions: List of user transactions
            savings_goals: List of savings goals (optional)
            budget_goals: List of budget goals (optional)
            
        Returns:
            dict with comprehensive insights
        """
        if not transactions:
            return {
                'total_income': 0,
                'total_expense': 0,
                'savings_rate': 0,
                'transaction_count': 0,
                'top_categories': {},
                'monthly_avg_expense': 0,
                'monthly_avg_income': 0,
                'spending_trend': 'stable',
                'high_spending_days': [],
                'payment_mode_distribution': {},
                'category_trends': {},
                'financial_health_score': 0
            }
        
        # Calculate basic metrics
        total_income = sum(float(t['amount']) for t in transactions if t['transaction_type'] == 'Income')
        total_expense = sum(float(t['amount']) for t in transactions if t['transaction_type'] == 'Expense')
        
        savings_rate = ((total_income - total_expense) / total_income * 100) if total_income > 0 else 0
        
        # Category breakdown with detailed analysis
        category_expenses = {}
        category_transaction_count = {}
        for t in transactions:
            if t['transaction_type'] == 'Expense':
                category = t['category']
                category_expenses[category] = category_expenses.get(category, 0) + float(t['amount'])
                category_transaction_count[category] = category_transaction_count.get(category, 0) + 1
        
        # Sort categories by spending
        top_categories = dict(sorted(category_expenses.items(), key=lambda x: x[1], reverse=True)[:5])
        
        # Calculate monthly averages and trends
        dates = [t['date'] for t in transactions if 'date' in t]
        monthly_data = defaultdict(lambda: {'income': 0, 'expense': 0})
        
        if dates:
            for t in transactions:
                date = t['date']
                if hasattr(date, 'year'):
                    month_key = f"{date.year}-{date.month:02d}"
                else:
                    dt = datetime.fromisoformat(str(date))
                    month_key = f"{dt.year}-{dt.month:02d}"
                
                if t['transaction_type'] == 'Income':
                    monthly_data[month_key]['income'] += float(t['amount'])
                else:
                    monthly_data[month_key]['expense'] += float(t['amount'])
            
            unique_months = len(monthly_data)
        else:
            unique_months = 1
        
        monthly_avg_expense = total_expense / unique_months
        monthly_avg_income = total_income / unique_months
        
        # Analyze spending trend (increasing/decreasing/stable)
        spending_trend = self._analyze_spending_trend(monthly_data)
        
        # Identify high spending days (weekends vs weekdays)
        high_spending_days = self._analyze_spending_patterns(transactions)
        
        # Payment mode distribution
        payment_mode_distribution = {}
        for t in transactions:
            if t['transaction_type'] == 'Expense':
                mode = t.get('payment_mode', 'Other')
                payment_mode_distribution[mode] = payment_mode_distribution.get(mode, 0) + float(t['amount'])
        
        # Category trends (growing/shrinking)
        category_trends = self._analyze_category_trends(transactions, monthly_data)
        
        # Calculate financial health score (0-100)
        financial_health_score = self._calculate_health_score(
            savings_rate, 
            len(savings_goals) if savings_goals else 0,
            len(budget_goals) if budget_goals else 0,
            category_expenses,
            total_income,
            monthly_data
        )
        
        # Identify spending anomalies
        spending_anomalies = self._detect_spending_anomalies(transactions, monthly_avg_expense)
        
        # Calculate category averages and frequency
        category_insights = {}
        for cat, amount in category_expenses.items():
            category_insights[cat] = {
                'total': amount,
                'avg_per_transaction': amount / category_transaction_count[cat],
                'frequency': category_transaction_count[cat],
                'percentage': (amount / total_expense * 100) if total_expense > 0 else 0
            }
        
        return {
            'total_income': total_income,
            'total_expense': total_expense,
            'savings_rate': savings_rate,
            'transaction_count': len(transactions),
            'top_categories': top_categories,
            'monthly_avg_expense': monthly_avg_expense,
            'monthly_avg_income': monthly_avg_income,
            'unique_months': unique_months,
            'spending_trend': spending_trend,
            'high_spending_days': high_spending_days,
            'payment_mode_distribution': payment_mode_distribution,
            'category_trends': category_trends,
            'financial_health_score': financial_health_score,
            'spending_anomalies': spending_anomalies,
            'category_insights': category_insights,
            'monthly_data': dict(monthly_data)
        }
    
    def _analyze_spending_trend(self, monthly_data):
        """Analyze if spending is increasing, decreasing, or stable"""
        if len(monthly_data) < 2:
            return 'stable'
        
        months = sorted(monthly_data.keys())
        expenses = [monthly_data[m]['expense'] for m in months]
        
        # Calculate trend using recent 3 months vs previous 3 months
        if len(expenses) >= 6:
            recent_avg = sum(expenses[-3:]) / 3
            previous_avg = sum(expenses[-6:-3]) / 3
            change = ((recent_avg - previous_avg) / previous_avg * 100) if previous_avg > 0 else 0
            
            if change > 10:
                return 'increasing'
            elif change < -10:
                return 'decreasing'
        
        return 'stable'
    
    def _analyze_spending_patterns(self, transactions):
        """Identify high spending days and patterns"""
        day_spending = defaultdict(float)
        
        for t in transactions:
            if t['transaction_type'] == 'Expense':
                date = t['date']
                if hasattr(date, 'weekday'):
                    day_name = date.strftime('%A')
                else:
                    dt = datetime.fromisoformat(str(date))
                    day_name = dt.strftime('%A')
                
                day_spending[day_name] += float(t['amount'])
        
        if not day_spending:
            return []
        
        # Find days with above-average spending
        avg_spending = sum(day_spending.values()) / len(day_spending)
        high_days = [day for day, amount in day_spending.items() if amount > avg_spending * 1.2]
        
        return high_days
    
    def _analyze_category_trends(self, transactions, monthly_data):
        """Analyze which categories are growing or shrinking"""
        if len(monthly_data) < 2:
            return {}
        
        category_monthly = defaultdict(lambda: defaultdict(float))
        
        for t in transactions:
            if t['transaction_type'] == 'Expense':
                date = t['date']
                if hasattr(date, 'year'):
                    month_key = f"{date.year}-{date.month:02d}"
                else:
                    dt = datetime.fromisoformat(str(date))
                    month_key = f"{dt.year}-{dt.month:02d}"
                
                category_monthly[t['category']][month_key] += float(t['amount'])
        
        trends = {}
        for category, months in category_monthly.items():
            if len(months) >= 2:
                sorted_months = sorted(months.keys())
                recent = months[sorted_months[-1]]
                previous = months[sorted_months[-2]] if len(sorted_months) > 1 else recent
                
                change = ((recent - previous) / previous * 100) if previous > 0 else 0
                
                if change > 15:
                    trends[category] = 'increasing'
                elif change < -15:
                    trends[category] = 'decreasing'
                else:
                    trends[category] = 'stable'
        
        return trends
    
    def _calculate_health_score(self, savings_rate, num_savings_goals, num_budget_goals, 
                                category_expenses, total_income, monthly_data):
        """Calculate financial health score (0-100)"""
        score = 0
        
        # Savings rate (40 points max)
        if savings_rate >= 30:
            score += 40
        elif savings_rate >= 20:
            score += 30
        elif savings_rate >= 10:
            score += 20
        else:
            score += max(0, savings_rate)
        
        # Having savings goals (15 points max)
        score += min(15, num_savings_goals * 5)
        
        # Having budget goals (15 points max)
        score += min(15, num_budget_goals * 5)
        
        # Expense diversification (15 points max) - not putting all eggs in one basket
        if category_expenses:
            max_category_pct = max(category_expenses.values()) / sum(category_expenses.values()) * 100
            if max_category_pct < 30:
                score += 15
            elif max_category_pct < 40:
                score += 10
            elif max_category_pct < 50:
                score += 5
        
        # Consistent income (15 points max)
        if len(monthly_data) >= 2:
            incomes = [data['income'] for data in monthly_data.values()]
            avg_income = sum(incomes) / len(incomes)
            if avg_income > 0:
                variance = sum((i - avg_income) ** 2 for i in incomes) / len(incomes)
                cv = (variance ** 0.5) / avg_income  # Coefficient of variation
                
                if cv < 0.1:  # Very consistent
                    score += 15
                elif cv < 0.2:
                    score += 10
                elif cv < 0.3:
                    score += 5
        
        return min(100, score)
    
    def _detect_spending_anomalies(self, transactions, avg_expense):
        """Detect unusual spending patterns"""
        anomalies = []
        
        for t in transactions:
            if t['transaction_type'] == 'Expense':
                amount = float(t['amount'])
                # Flag transactions 3x higher than average
                if amount > avg_expense * 3:
                    anomalies.append({
                        'category': t['category'],
                        'amount': amount,
                        'date': str(t['date'])
                    })
        
        return anomalies[:5]  # Return top 5 anomalies
    
    def generate_recommendations(self, user_profile, savings_goals=None, budget_goals=None):
        """
        Generate advanced personalized recommendations using Gemini with comprehensive analysis
        
        Args:
            user_profile: Dict with user's financial metrics
            savings_goals: List of savings goals
            budget_goals: List of budget goals
            
        Returns:
            List of recommendations
        """
        try:
            # Prepare detailed category breakdown
            category_breakdown = ""
            if user_profile['top_categories']:
                category_breakdown = "\n".join([
                    f"  - {cat}: ₹{amount:,.0f} ({amount/user_profile['total_expense']*100:.1f}%)"
                    for cat, amount in user_profile['top_categories'].items()
                ])
            
            # Prepare category insights
            category_insights_text = ""
            if 'category_insights' in user_profile:
                top_3_cats = list(user_profile['category_insights'].items())[:3]
                category_insights_text = "\n".join([
                    f"  - {cat}: {info['frequency']} transactions, avg ₹{info['avg_per_transaction']:,.0f} per transaction"
                    for cat, info in top_3_cats
                ])
            
            # Prepare savings goals info with progress
            savings_info = ""
            if savings_goals and len(savings_goals) > 0:
                savings_info = f"\nSavings Goals ({len(savings_goals)} active):"
                for goal in savings_goals[:3]:
                    progress = (goal.get('current_amount', 0) / goal.get('target_amount', 1)) * 100
                    remaining = goal.get('target_amount', 0) - goal.get('current_amount', 0)
                    savings_info += f"\n  - {goal['goal_name']}: {progress:.0f}% (₹{remaining:,.0f} remaining)"
            
            # Prepare budget goals info with warnings
            budget_info = ""
            budget_warnings = []
            if budget_goals and len(budget_goals) > 0:
                budget_info = f"\nBudget Goals ({len(budget_goals)} active):"
                for goal in budget_goals[:3]:
                    spent_pct = (goal.get('spent', 0) / goal.get('budget_limit', 1)) * 100
                    if spent_pct > 100:
                        status = "⚠️ OVER BUDGET"
                        budget_warnings.append(goal['category'])
                    elif spent_pct > 80:
                        status = "⚠️ Almost full"
                    else:
                        status = "✓ On track"
                    budget_info += f"\n  - {goal['category']}: {spent_pct:.0f}% ({status})"
            
            # Prepare spending trend analysis
            trend_info = f"\nSpending Trend: {user_profile['spending_trend'].upper()}"
            if user_profile['spending_trend'] == 'increasing':
                trend_info += " ⚠️ (Your expenses are rising)"
            elif user_profile['spending_trend'] == 'decreasing':
                trend_info += " ✓ (Good job reducing expenses!)"
            
            # High spending days
            high_days_info = ""
            if user_profile['high_spending_days']:
                high_days_info = f"\nHigh Spending Days: {', '.join(user_profile['high_spending_days'])}"
            
            # Financial health score
            health_score = user_profile.get('financial_health_score', 0)
            health_status = "Excellent" if health_score >= 80 else "Good" if health_score >= 60 else "Fair" if health_score >= 40 else "Needs Improvement"
            health_info = f"\nFinancial Health Score: {health_score}/100 ({health_status})"
            
            # Spending anomalies
            anomalies_info = ""
            if user_profile.get('spending_anomalies'):
                anomalies_info = f"\nUnusual Large Expenses Detected: {len(user_profile['spending_anomalies'])} transactions"
            
            # Category trends
            growing_categories = [cat for cat, trend in user_profile.get('category_trends', {}).items() if trend == 'increasing']
            if growing_categories:
                anomalies_info += f"\nGrowing Expenses: {', '.join(growing_categories[:3])}"
            
            # Create comprehensive prompt for Gemini
            prompt = f"""You are an expert personal finance advisor with deep knowledge of Indian financial practices. Analyze this comprehensive financial profile and provide 5 highly personalized, actionable recommendations.

USER FINANCIAL PROFILE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Financial Overview:
  • Monthly Income: ₹{user_profile['monthly_avg_income']:,.0f}
  • Monthly Expenses: ₹{user_profile['monthly_avg_expense']:,.0f}
  • Savings Rate: {user_profile['savings_rate']:.1f}%
  • Total Transactions: {user_profile['transaction_count']} ({user_profile['unique_months']} months)
{health_info}

💳 Top Spending Categories:
{category_breakdown}

📈 Behavioral Insights:
{category_insights_text}
{trend_info}
{high_days_info}
{anomalies_info}
{savings_info}
{budget_info}

CRITICAL ANALYSIS POINTS:
{"- ⚠️ URGENT: Over budget in " + ', '.join(budget_warnings) if budget_warnings else ""}
{"- ⚠️ Savings rate below recommended 20%" if user_profile['savings_rate'] < 20 else ""}
{"- ✓ Excellent savings rate!" if user_profile['savings_rate'] >= 30 else ""}
{"- ⚠️ High concentration in " + list(user_profile['top_categories'].keys())[0] + f" ({list(user_profile['top_categories'].values())[0]/user_profile['total_expense']*100:.0f}%)" if user_profile['top_categories'] and list(user_profile['top_categories'].values())[0]/user_profile['total_expense'] > 0.4 else ""}

INSTRUCTIONS FOR RECOMMENDATIONS:
1. Provide EXACTLY 5 recommendations based on the data above
2. Each must be:
   ✓ Hyper-specific with exact rupee amounts and percentages
   ✓ Directly address the user's actual patterns (not generic advice)
   ✓ Actionable with clear next steps
   ✓ Encouraging and positive tone
   ✓ Consider Indian financial context (UPI, savings culture, etc.)
3. Prioritize recommendations based on:
   - Budget overruns (highest priority)
   - Growing expense categories
   - Low savings rate
   - Savings goals progress
   - Spending patterns and anomalies
4. Include variety: spending cuts, savings strategies, behavior changes, goal optimization
5. Use exact numbers from the profile - be data-driven!

FORMAT (MUST FOLLOW):
[Emoji] [Bold Title]: [Specific action with exact numbers]. [Why it matters or benefit].

EXAMPLE:
💰 Cut Entertainment by 25%: Reduce your ₹8,500 monthly entertainment spend to ₹6,375 (save ₹2,125/month). Redirect these savings to your Home savings goal, reaching it 4 months faster.

NOW PROVIDE 5 PERSONALIZED RECOMMENDATIONS:"""

            # Generate recommendations
            response = self.model.generate_content(prompt)
            recommendations_text = response.text.strip()
            
            # Parse recommendations into list with intelligent filtering
            recommendations = []
            lines = recommendations_text.split('\n')
            
            for line in lines:
                line = line.strip()
                
                # Check if line contains recommendation (has emoji or starts with number)
                if line and (
                    any(emoji in line for emoji in ['💰', '🎯', '📊', '💡', '✨', '🏆', '📈', '💳', '🛡️', '🎁', '🔴', '🟡', '🟢', '⚠️', '✅', '🚀', '💪', '🎨', '🏠', '🚗', '✈️', '💍']) 
                    or (line[0].isdigit() and '.' in line[:3])
                ):
                    # Remove numbering if present
                    if line[0].isdigit() and '.' in line[:3]:
                        line = line.split('.', 1)[1].strip()
                    
                    # Remove markdown bold markers
                    line = line.replace('**', '')
                    
                    # Only add if it's substantial (more than 20 characters)
                    if len(line) > 20:
                        recommendations.append(line)
            
            # Smart fallback if AI response is inadequate
            if len(recommendations) < 3:
                recommendations = self._generate_smart_fallback_recommendations(
                    user_profile, savings_goals, budget_goals, budget_warnings
                )
            
            return recommendations[:5]  # Return exactly 5
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            import traceback
            traceback.print_exc()
            
            # Intelligent fallback recommendations
            return self._generate_smart_fallback_recommendations(
                user_profile, savings_goals, budget_goals, []
            )
    
    def _generate_smart_fallback_recommendations(self, profile, savings_goals, budget_goals, budget_warnings):
        """Generate intelligent fallback recommendations based on profile analysis"""
        recommendations = []
        
        # 1. Savings rate recommendation
        if profile['savings_rate'] < 20:
            target_savings = profile['monthly_avg_income'] * 0.20
            current_savings = profile['monthly_avg_income'] - profile['monthly_avg_expense']
            additional_needed = target_savings - current_savings
            recommendations.append(
                f"💰 Boost Savings to 20%: Increase your savings by ₹{additional_needed:,.0f}/month to reach the recommended 20% savings rate. Your current rate of {profile['savings_rate']:.1f}% needs improvement for financial security."
            )
        else:
            recommendations.append(
                f"✅ Excellent Savings Rate: You're saving {profile['savings_rate']:.1f}% - well above the 20% target! Consider investing surplus in mutual funds or fixed deposits for wealth building."
            )
        
        # 2. Top category recommendation
        if profile['top_categories']:
            top_cat = list(profile['top_categories'].keys())[0]
            top_amount = list(profile['top_categories'].values())[0]
            top_pct = (top_amount / profile['total_expense'] * 100) if profile['total_expense'] > 0 else 0
            
            if top_pct > 30:
                reduction = top_amount * 0.15
                recommendations.append(
                    f"📊 Reduce {top_cat} Spending: Your {top_cat} expenses are {top_pct:.0f}% of total spending (₹{top_amount:,.0f}). Cut by 15% (₹{reduction:,.0f}) to balance your budget better."
                )
            else:
                recommendations.append(
                    f"✓ Balanced {top_cat} Spending: Your top category ({top_cat}) is {top_pct:.0f}% of expenses - well distributed. Maintain this balance while optimizing other areas."
                )
        
        # 3. Spending trend recommendation
        if profile['spending_trend'] == 'increasing':
            recommendations.append(
                "⚠️ Rising Expenses Detected: Your spending has increased recently. Review last month's transactions to identify and eliminate unnecessary expenses before they become habits."
            )
        elif profile['spending_trend'] == 'decreasing':
            recommendations.append(
                "🎉 Great Progress on Expenses: Your spending is decreasing! Keep this momentum by setting a lower expense target for next month and celebrating small wins."
            )
        
        # 4. Budget goals recommendation
        if budget_warnings:
            recommendations.append(
                f"🔴 Budget Alert: You've exceeded limits in {', '.join(budget_warnings)}. Review these categories immediately and set stricter alerts or use cash-only for these expenses."
            )
        elif budget_goals and len(budget_goals) > 0:
            recommendations.append(
                f"✓ Budget Discipline: You're managing {len(budget_goals)} budgets well! Add 2-3 more categories to strengthen financial control and track seasonal variations."
            )
        else:
            recommendations.append(
                "🎯 Set Budget Goals: Create monthly budget limits for your top 3 spending categories. This simple step can reduce expenses by 10-15% through increased awareness."
            )
        
        # 5. Savings goals recommendation
        if savings_goals and len(savings_goals) > 0:
            incomplete = [g for g in savings_goals if (g.get('current_amount', 0) / g.get('target_amount', 1) * 100) < 100]
            if incomplete:
                goal = incomplete[0]
                remaining = goal.get('target_amount', 0) - goal.get('current_amount', 0)
                monthly_needed = remaining / 6  # Assume 6 months
                recommendations.append(
                    f"🎯 Accelerate '{goal['goal_name']}': You need ₹{remaining:,.0f} more. Save ₹{monthly_needed:,.0f}/month to complete this goal in 6 months. Consider automating monthly transfers."
                )
        else:
            emergency_fund = profile['monthly_avg_expense'] * 6
            recommendations.append(
                f"🛡️ Build Emergency Fund: Create a savings goal for ₹{emergency_fund:,.0f} (6 months expenses). Start with ₹{emergency_fund/12:,.0f}/month to build financial safety in 1 year."
            )
        
        # Ensure we have 5 recommendations
        while len(recommendations) < 5:
            generic = [
                "💳 Optimize Payment Methods: Use credit cards with cashback (2-5%) for regular expenses and pay full balance monthly to avoid interest charges.",
                f"📈 Review High Spending Days: You spend more on {profile['high_spending_days'][0] if profile['high_spending_days'] else 'weekends'}. Plan activities in advance to reduce impulsive purchases.",
                "💡 Track Daily Expenses: Use your app daily instead of weekly. Real-time tracking reduces spending by 12% on average through increased awareness.",
                "🏆 Financial Health Improvement: Your health score is {profile.get('financial_health_score', 50)}/100. Focus on diversifying expenses and building consistent savings habits.",
                "🚀 Automate Savings: Set up automatic transfers of 20% of income to savings account on salary day. 'Pay yourself first' strategy ensures consistent wealth building."
            ]
            recommendations.append(generic[len(recommendations) - 5])
        
        return recommendations[:5]


# Simple function for easy import
def get_recommendations(api_key, transactions, savings_goals=None, budget_goals=None):
    """
    Simple function to get recommendations
    
    Args:
        api_key: Gemini API key
        transactions: List of transactions
        savings_goals: Optional savings goals
        budget_goals: Optional budget goals
    
    Returns:
        List of recommendation strings
    """
    engine = RecommendationEngine(api_key)
    profile = engine.analyze_user_profile(transactions, savings_goals, budget_goals)
    recommendations = engine.generate_recommendations(profile, savings_goals, budget_goals)
    return recommendations