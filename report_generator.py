from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
from datetime import datetime
import pandas as pd
import os
import tempfile

class PDFReportGenerator:
    def __init__(self, user_data, transactions, stats, budget_goals=None):
        self.user_data = user_data
        self.transactions = transactions
        self.stats = stats
        self.budget_goals = budget_goals or []
        self.styles = getSampleStyleSheet()
        self.temp_dir = tempfile.mkdtemp()
        
        # Custom styles
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#4f46e5'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#1e293b'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        self.normal_style = ParagraphStyle(
            'CustomNormal',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#475569')
        )

    def create_header_footer(self, canvas, doc):
        """Add header and footer to each page"""
        canvas.saveState()
        
        # Header
        canvas.setFillColor(colors.HexColor('#4f46e5'))
        canvas.rect(0, letter[1] - 50, letter[0], 50, fill=1, stroke=0)
        canvas.setFillColor(colors.white)
        canvas.setFont('Helvetica-Bold', 16)
        canvas.drawString(50, letter[1] - 30, "BudgetWise Financial Report")
        
        # Footer
        canvas.setFillColor(colors.HexColor('#94a3b8'))
        canvas.setFont('Helvetica', 9)
        canvas.drawString(50, 30, f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
        canvas.drawRightString(letter[0] - 50, 30, f"Page {doc.page}")
        
        canvas.restoreState()

    def create_chart_image(self, chart_type, data):
        """Create matplotlib charts and return as image"""
        fig, ax = plt.subplots(figsize=(6, 4))
        
        if chart_type == 'income_expense_pie':
            if data['income'] > 0 or data['expense'] > 0:
                sizes = [data['income'], data['expense']]
                labels = ['Income', 'Expense']
                colors_list = ['#10b981', '#ef4444']
                explode = (0.05, 0.05)
                
                ax.pie(sizes, labels=labels, colors=colors_list, autopct='%1.1f%%',
                       startangle=90, explode=explode, textprops={'fontsize': 10, 'weight': 'bold'})
                ax.set_title('Income vs Expense Distribution', fontsize=12, weight='bold', pad=20)
        
        elif chart_type == 'category_bar':
            if data:
                categories = list(data.keys())[:8]  # Top 8 categories
                amounts = [data[cat] for cat in categories]
                
                bars = ax.barh(categories, amounts, color='#6366f1')
                ax.set_xlabel('Amount (₹)', fontsize=10, weight='bold')
                ax.set_title('Expense by Category', fontsize=12, weight='bold', pad=20)
                ax.grid(axis='x', alpha=0.3)
                
                # Add value labels on bars
                for i, (bar, amount) in enumerate(zip(bars, amounts)):
                    ax.text(amount, i, f' ₹{amount:,.0f}', 
                           va='center', fontsize=8, weight='bold')
        
        elif chart_type == 'monthly_trend':
            if data:
                months = list(data.keys())
                income_data = [data[m]['income'] for m in months]
                expense_data = [data[m]['expense'] for m in months]
                
                ax.plot(months, income_data, marker='o', linewidth=2, 
                       label='Income', color='#10b981', markersize=6)
                ax.plot(months, expense_data, marker='s', linewidth=2, 
                       label='Expense', color='#ef4444', markersize=6)
                
                ax.set_xlabel('Month', fontsize=10, weight='bold')
                ax.set_ylabel('Amount (₹)', fontsize=10, weight='bold')
                ax.set_title('Monthly Trend Analysis', fontsize=12, weight='bold', pad=20)
                ax.legend(loc='best', fontsize=9)
                ax.grid(alpha=0.3)
                plt.xticks(rotation=45, ha='right', fontsize=8)
        
        elif chart_type == 'budget_goals':
            if data:
                categories = [g['category'] for g in data[:6]]  # Top 6 goals
                spent = [g['spent'] for g in data[:6]]
                remaining = [g['budget_limit'] - g['spent'] for g in data[:6]]
                
                x = range(len(categories))
                width = 0.35
                
                ax.bar(x, spent, width, label='Spent', color='#ef4444')
                ax.bar([i + width for i in x], remaining, width, 
                      label='Remaining', color='#10b981')
                
                ax.set_xlabel('Category', fontsize=10, weight='bold')
                ax.set_ylabel('Amount (₹)', fontsize=10, weight='bold')
                ax.set_title('Budget Goals Progress', fontsize=12, weight='bold', pad=20)
                ax.set_xticks([i + width/2 for i in x])
                ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
                ax.legend(loc='best', fontsize=9)
                ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close(fig)
        
        return img_buffer

    def generate_report(self, filename):
        """Generate the complete PDF report"""
        doc = SimpleDocTemplate(filename, pagesize=letter,
                               topMargin=70, bottomMargin=50)
        story = []
        
        # Title
        title = Paragraph(f"Financial Report - {self.user_data['username']}", self.title_style)
        story.append(title)
        story.append(Spacer(1, 0.2*inch))
        
        # Report period
        period_text = f"<i>Report Period: {datetime.now().strftime('%B %Y')}</i>"
        story.append(Paragraph(period_text, self.normal_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.heading_style))
        
        summary_data = [
            ['Metric', 'Value'],
            ['Total Income', f"₹{self.stats['total_income']:,.2f}"],
            ['Total Expenses', f"₹{self.stats['total_expense']:,.2f}"],
            ['Net Savings', f"₹{self.stats['total_income'] - self.stats['total_expense']:,.2f}"],
            ['Savings Rate', f"{self.stats['savings_rate']:.1f}%"],
            ['Total Transactions', str(self.stats['transaction_count'])]
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4f46e5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 0.4*inch))
        
        # Income vs Expense Chart
        story.append(Paragraph("Income vs Expense Overview", self.heading_style))
        chart_data = {
            'income': self.stats['total_income'],
            'expense': self.stats['total_expense']
        }
        img_buffer = self.create_chart_image('income_expense_pie', chart_data)
        img = Image(img_buffer, width=5*inch, height=3.5*inch)
        story.append(img)
        story.append(Spacer(1, 0.3*inch))
        
        # Category Analysis
        if self.transactions:
            df = pd.DataFrame(self.transactions)
            expense_df = df[df['transaction_type'] == 'Expense']
            
            if not expense_df.empty:
                category_data = expense_df.groupby('category')['amount'].sum().to_dict()
                
                story.append(Paragraph("Expense Analysis by Category", self.heading_style))
                img_buffer = self.create_chart_image('category_bar', category_data)
                img = Image(img_buffer, width=5.5*inch, height=3.5*inch)
                story.append(img)
                story.append(Spacer(1, 0.3*inch))
        
        # Page break before next section
        story.append(PageBreak())
        
        # Monthly Trend
        if self.transactions:
            df = pd.DataFrame(self.transactions)
            df['date'] = pd.to_datetime(df['date'])
            df['month_year'] = df['date'].dt.strftime('%b %Y')
            
            monthly_data = {}
            for month in df['month_year'].unique():
                month_df = df[df['month_year'] == month]
                monthly_data[month] = {
                    'income': float(month_df[month_df['transaction_type'] == 'Income']['amount'].sum()),
                    'expense': float(month_df[month_df['transaction_type'] == 'Expense']['amount'].sum())
                }
            
            # Sort by date
            sorted_months = sorted(monthly_data.keys(), 
                                 key=lambda x: datetime.strptime(x, '%b %Y'))
            monthly_data = {k: monthly_data[k] for k in sorted_months}
            
            if len(monthly_data) > 1:
                story.append(Paragraph("Monthly Trend Analysis", self.heading_style))
                img_buffer = self.create_chart_image('monthly_trend', monthly_data)
                img = Image(img_buffer, width=6*inch, height=3.5*inch)
                story.append(img)
                story.append(Spacer(1, 0.3*inch))
        
        # Budget Goals Section
        if self.budget_goals:
            story.append(Paragraph("Budget Goals Progress", self.heading_style))
            
            img_buffer = self.create_chart_image('budget_goals', self.budget_goals)
            img = Image(img_buffer, width=6*inch, height=3.5*inch)
            story.append(img)
            story.append(Spacer(1, 0.3*inch))
            
            # Budget goals table
            goals_table_data = [['Category', 'Budget Limit', 'Spent', 'Remaining', 'Status']]
            
            for goal in self.budget_goals[:10]:  # Top 10 goals
                spent = goal['spent']
                limit = goal['budget_limit']
                remaining = limit - spent
                percentage = (spent / limit * 100) if limit > 0 else 0
                
                if percentage >= 90:
                    status = '🔴 Over Budget' if percentage > 100 else '🔴 Critical'
                elif percentage >= 70:
                    status = '🟡 Warning'
                else:
                    status = '🟢 On Track'
                
                goals_table_data.append([
                    goal['category'],
                    f"₹{limit:,.0f}",
                    f"₹{spent:,.0f}",
                    f"₹{remaining:,.0f}",
                    status
                ])
            
            goals_table = Table(goals_table_data, colWidths=[1.3*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.3*inch])
            goals_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4f46e5')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
                ('TOPPADDING', (0, 1), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ]))
            
            story.append(goals_table)
            story.append(Spacer(1, 0.3*inch))
        
        # Page break before transactions
        story.append(PageBreak())
        
        # Recent Transactions
        story.append(Paragraph("Recent Transactions", self.heading_style))
        
        if self.transactions:
            trans_data = [['Date', 'Type', 'Category', 'Amount', 'Payment Mode']]
            
            for trans in self.transactions[:20]:  # Last 20 transactions
                trans_date = trans['date'].strftime('%d %b %Y') if hasattr(trans['date'], 'strftime') else str(trans['date'])
                trans_data.append([
                    trans_date,
                    trans['transaction_type'],
                    trans['category'],
                    f"₹{trans['amount']:,.2f}",
                    trans['payment_mode']
                ])
            
            trans_table = Table(trans_data, colWidths=[1.2*inch, 1*inch, 1.5*inch, 1.3*inch, 1.2*inch])
            trans_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4f46e5')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
                ('TOPPADDING', (0, 1), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
            ]))
            
            story.append(trans_table)
        else:
            story.append(Paragraph("<i>No transactions found.</i>", self.normal_style))
        
        # Build PDF
        doc.build(story, onFirstPage=self.create_header_footer, 
                 onLaterPages=self.create_header_footer)
        
        return filename