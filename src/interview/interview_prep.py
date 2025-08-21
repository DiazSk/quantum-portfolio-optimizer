"""
Interview Preparation System
Epic 8: Comprehensive technical interview prep for FAANG career advancement
"""

import json
import random
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InterviewQuestion:
    """Interview question data structure"""
    id: str
    category: str
    difficulty: str  # Easy, Medium, Hard
    question: str
    solution: str
    hints: List[str]
    tags: List[str]
    companies: List[str]
    time_limit: int  # minutes
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class PracticeSession:
    """Practice session tracking"""
    session_id: str
    questions_attempted: List[str]
    scores: Dict[str, int]  # question_id -> score (0-100)
    start_time: datetime
    end_time: Optional[datetime]
    total_score: float
    category_focus: str
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['start_time'] = data['start_time'].isoformat()
        if data['end_time']:
            data['end_time'] = data['end_time'].isoformat()
        return data

class QuestionBank:
    """Question bank management"""
    
    def __init__(self):
        self.questions = self._initialize_question_bank()
        logger.info(f"Question bank initialized with {len(self.questions)} questions")
    
    def _initialize_question_bank(self) -> Dict[str, InterviewQuestion]:
        """Initialize with comprehensive question bank"""
        questions = {}
        
        # Python/Data Science Questions
        python_questions = [
            {
                "id": "py_001",
                "category": "Python",
                "difficulty": "Medium",
                "question": "Implement a function to find the maximum profit from buying and selling stocks. You can only hold one stock at a time.",
                "solution": """
def max_profit(prices):
    if len(prices) < 2:
        return 0
    
    min_price = prices[0]
    max_profit = 0
    
    for price in prices[1:]:
        max_profit = max(max_profit, price - min_price)
        min_price = min(min_price, price)
    
    return max_profit

# Time: O(n), Space: O(1)
""",
                "hints": ["Track minimum price seen so far", "Calculate profit at each step"],
                "tags": ["dynamic_programming", "array", "greedy"],
                "companies": ["Google", "Facebook", "Amazon"],
                "time_limit": 20
            },
            {
                "id": "py_002", 
                "category": "Python",
                "difficulty": "Hard",
                "question": "Design a data structure for a portfolio optimization system that supports: add_stock(symbol, shares), remove_stock(symbol), get_portfolio_value(), and rebalance(target_weights).",
                "solution": """
class Portfolio:
    def __init__(self):
        self.holdings = {}  # symbol -> shares
        self.prices = {}    # symbol -> current_price
        
    def add_stock(self, symbol: str, shares: int):
        self.holdings[symbol] = self.holdings.get(symbol, 0) + shares
        # In real implementation, fetch current price
        self.prices[symbol] = self._get_current_price(symbol)
    
    def remove_stock(self, symbol: str, shares: int = None):
        if symbol not in self.holdings:
            return False
        
        if shares is None:
            del self.holdings[symbol]
            del self.prices[symbol]
        else:
            self.holdings[symbol] -= shares
            if self.holdings[symbol] <= 0:
                del self.holdings[symbol]
                del self.prices[symbol]
        return True
    
    def get_portfolio_value(self) -> float:
        return sum(shares * self.prices[symbol] 
                  for symbol, shares in self.holdings.items())
    
    def rebalance(self, target_weights: Dict[str, float]):
        total_value = self.get_portfolio_value()
        
        for symbol, target_weight in target_weights.items():
            target_value = total_value * target_weight
            current_price = self.prices.get(symbol, 0)
            
            if current_price > 0:
                target_shares = int(target_value / current_price)
                current_shares = self.holdings.get(symbol, 0)
                
                if target_shares > current_shares:
                    self.add_stock(symbol, target_shares - current_shares)
                elif target_shares < current_shares:
                    self.remove_stock(symbol, current_shares - target_shares)
    
    def _get_current_price(self, symbol: str) -> float:
        # Mock implementation
        return random.uniform(50, 200)
""",
                "hints": ["Use dictionaries for O(1) lookups", "Handle edge cases for stock removal", "Calculate target shares based on weights"],
                "tags": ["system_design", "data_structures", "finance"],
                "companies": ["Goldman Sachs", "Morgan Stanley", "Two Sigma"],
                "time_limit": 45
            }
        ]
        
        # SQL Questions
        sql_questions = [
            {
                "id": "sql_001",
                "category": "SQL",
                "difficulty": "Medium", 
                "question": "Write a SQL query to find the top 3 performing stocks by return percentage in the last 30 days.",
                "solution": """
WITH stock_performance AS (
    SELECT 
        symbol,
        (MAX(price) - MIN(price)) / MIN(price) * 100 as return_pct
    FROM stock_prices 
    WHERE date >= DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY)
    GROUP BY symbol
)
SELECT 
    symbol,
    ROUND(return_pct, 2) as return_percentage
FROM stock_performance
ORDER BY return_pct DESC
LIMIT 3;
""",
                "hints": ["Use window functions or GROUP BY", "Calculate percentage return formula", "Filter by date range"],
                "tags": ["window_functions", "aggregation", "date_functions"],
                "companies": ["Meta", "Netflix", "Uber"],
                "time_limit": 25
            },
            {
                "id": "sql_002",
                "category": "SQL", 
                "difficulty": "Hard",
                "question": "Write a query to calculate the rolling 30-day Sharpe ratio for each stock. Assume risk-free rate of 2%.",
                "solution": """
WITH daily_returns AS (
    SELECT 
        symbol,
        date,
        price,
        (price - LAG(price) OVER (PARTITION BY symbol ORDER BY date)) / LAG(price) OVER (PARTITION BY symbol ORDER BY date) AS daily_return
    FROM stock_prices
),
rolling_stats AS (
    SELECT 
        symbol,
        date,
        AVG(daily_return) OVER (
            PARTITION BY symbol 
            ORDER BY date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS avg_return_30d,
        STDDEV(daily_return) OVER (
            PARTITION BY symbol 
            ORDER BY date 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW  
        ) AS stddev_30d
    FROM daily_returns
    WHERE daily_return IS NOT NULL
)
SELECT 
    symbol,
    date,
    CASE 
        WHEN stddev_30d > 0 THEN 
            (avg_return_30d * 252 - 0.02) / (stddev_30d * SQRT(252))
        ELSE NULL 
    END AS sharpe_ratio_30d
FROM rolling_stats
WHERE date >= (SELECT MIN(date) + INTERVAL 30 DAY FROM stock_prices);
""",
                "hints": ["Use LAG for returns calculation", "STDDEV and AVG with window functions", "Annualize returns and volatility"],
                "tags": ["window_functions", "financial_metrics", "complex_aggregation"], 
                "companies": ["Citadel", "Jane Street", "DE Shaw"],
                "time_limit": 40
            }
        ]
        
        # System Design Questions
        system_questions = [
            {
                "id": "sys_001",
                "category": "System Design",
                "difficulty": "Hard",
                "question": "Design a real-time portfolio monitoring system that can handle 10,000 concurrent users and process 1M price updates per second.",
                "solution": """
High-Level Architecture:

1. Data Ingestion Layer:
   - Apache Kafka for real-time market data streams
   - Multiple partitions for parallel processing
   - Schema registry for data validation

2. Processing Layer:
   - Apache Flink/Storm for stream processing
   - Redis for caching latest prices
   - Calculate portfolio metrics in real-time

3. Storage Layer:
   - PostgreSQL for user portfolios & transactions
   - InfluxDB for time-series market data
   - MongoDB for user preferences & alerts

4. API Layer:
   - FastAPI/GraphQL for client APIs
   - WebSocket connections for real-time updates
   - Rate limiting & authentication

5. Frontend:
   - React with WebSocket subscriptions
   - Chart.js for visualization
   - Progressive Web App (PWA)

6. Infrastructure:
   - Kubernetes for orchestration
   - HAProxy for load balancing
   - Prometheus + Grafana for monitoring

Key Considerations:
- Horizontal scaling with microservices
- Event sourcing for audit trails
- Circuit breakers for fault tolerance
- Database sharding by user_id
- CDN for static assets
""",
                "hints": ["Think about data flow", "Consider scaling bottlenecks", "Design for fault tolerance"],
                "tags": ["system_design", "microservices", "real_time", "scalability"],
                "companies": ["Google", "Amazon", "Microsoft"],
                "time_limit": 60
            }
        ]
        
        # Machine Learning Questions
        ml_questions = [
            {
                "id": "ml_001",
                "category": "Machine Learning",
                "difficulty": "Medium",
                "question": "Implement a simple linear regression model from scratch to predict stock prices. Include gradient descent optimization.",
                "solution": """
import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.costs = []
    
    def fit(self, X, y):
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute cost (MSE)
            cost = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)
            self.costs.append(cost)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)  # R-squared

# Example usage for stock prediction:
# Features: [volume, previous_price, moving_average]
# Target: next_day_price
""",
                "hints": ["Start with cost function (MSE)", "Derive gradients analytically", "Don't forget bias term"],
                "tags": ["machine_learning", "optimization", "linear_algebra"],
                "companies": ["Tesla", "Nvidia", "OpenAI"],
                "time_limit": 35
            }
        ]
        
        # Combine all questions
        all_questions = python_questions + sql_questions + system_questions + ml_questions
        
        for q_data in all_questions:
            question = InterviewQuestion(**q_data)
            questions[question.id] = question
            
        return questions
    
    def get_questions_by_category(self, category: str) -> List[InterviewQuestion]:
        """Get all questions for a specific category"""
        return [q for q in self.questions.values() if q.category == category]
    
    def get_questions_by_difficulty(self, difficulty: str) -> List[InterviewQuestion]:
        """Get all questions for a specific difficulty"""
        return [q for q in self.questions.values() if q.difficulty == difficulty]
    
    def get_questions_by_company(self, company: str) -> List[InterviewQuestion]:
        """Get questions commonly asked by a company"""
        return [q for q in self.questions.values() if company in q.companies]
    
    def get_random_question(self, **filters) -> Optional[InterviewQuestion]:
        """Get a random question matching filters"""
        filtered_questions = list(self.questions.values())
        
        if 'category' in filters:
            filtered_questions = [q for q in filtered_questions if q.category == filters['category']]
        
        if 'difficulty' in filters:
            filtered_questions = [q for q in filtered_questions if q.difficulty == filters['difficulty']]
        
        if 'company' in filters:
            filtered_questions = [q for q in filtered_questions if filters['company'] in q.companies]
        
        return random.choice(filtered_questions) if filtered_questions else None

class InterviewPrepSystem:
    """Main interview preparation system"""
    
    def __init__(self, db_path: str = "interview_prep.db"):
        self.db_path = db_path
        self.question_bank = QuestionBank()
        self._init_database()
        self.current_session = None
        
        logger.info("Interview Preparation System initialized")
    
    def _init_database(self):
        """Initialize SQLite database for tracking progress"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS practice_sessions (
            session_id TEXT PRIMARY KEY,
            category_focus TEXT,
            start_time TEXT,
            end_time TEXT,
            total_score REAL,
            questions_attempted INTEGER
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS question_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            question_id TEXT,
            score INTEGER,
            time_taken INTEGER,
            attempt_time TEXT,
            FOREIGN KEY (session_id) REFERENCES practice_sessions (session_id)
        )
        """)
        
        conn.commit()
        conn.close()
    
    def start_practice_session(self, category_focus: str = "All") -> str:
        """Start a new practice session"""
        session_id = f"SESSION_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = PracticeSession(
            session_id=session_id,
            questions_attempted=[],
            scores={},
            start_time=datetime.now(),
            end_time=None,
            total_score=0.0,
            category_focus=category_focus
        )
        
        logger.info(f"Started practice session: {session_id}")
        return session_id
    
    def get_next_question(self, **filters) -> Optional[InterviewQuestion]:
        """Get next question for current session"""
        if not self.current_session:
            raise ValueError("No active practice session")
        
        # Apply session category filter
        if self.current_session.category_focus != "All":
            filters['category'] = self.current_session.category_focus
        
        # Avoid already attempted questions
        attempted = set(self.current_session.questions_attempted)
        available_questions = [q for q in self.question_bank.questions.values() 
                             if q.id not in attempted]
        
        if not available_questions:
            return None
        
        # Apply filters
        for key, value in filters.items():
            if key == 'category':
                available_questions = [q for q in available_questions if q.category == value]
            elif key == 'difficulty':
                available_questions = [q for q in available_questions if q.difficulty == value]
            elif key == 'company':
                available_questions = [q for q in available_questions if value in q.companies]
        
        return random.choice(available_questions) if available_questions else None
    
    def submit_answer(self, question_id: str, score: int, time_taken: int):
        """Submit answer for a question"""
        if not self.current_session:
            raise ValueError("No active practice session")
        
        self.current_session.questions_attempted.append(question_id)
        self.current_session.scores[question_id] = score
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO question_attempts 
        (session_id, question_id, score, time_taken, attempt_time)
        VALUES (?, ?, ?, ?, ?)
        """, (
            self.current_session.session_id,
            question_id,
            score,
            time_taken,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Answer submitted for {question_id}: {score}/100")
    
    def end_practice_session(self) -> Dict[str, Any]:
        """End current practice session and return summary"""
        if not self.current_session:
            raise ValueError("No active practice session")
        
        self.current_session.end_time = datetime.now()
        
        # Calculate total score
        if self.current_session.scores:
            self.current_session.total_score = sum(self.current_session.scores.values()) / len(self.current_session.scores)
        
        # Store session in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO practice_sessions 
        (session_id, category_focus, start_time, end_time, total_score, questions_attempted)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            self.current_session.session_id,
            self.current_session.category_focus,
            self.current_session.start_time.isoformat(),
            self.current_session.end_time.isoformat(),
            self.current_session.total_score,
            len(self.current_session.questions_attempted)
        ))
        
        conn.commit()
        conn.close()
        
        # Generate summary
        summary = {
            'session_id': self.current_session.session_id,
            'duration': (self.current_session.end_time - self.current_session.start_time).total_seconds() / 60,
            'questions_attempted': len(self.current_session.questions_attempted),
            'average_score': self.current_session.total_score,
            'category_focus': self.current_session.category_focus,
            'score_breakdown': self.current_session.scores
        }
        
        logger.info(f"Session ended: {summary}")
        self.current_session = None
        
        return summary
    
    def get_progress_analytics(self) -> Dict[str, Any]:
        """Get comprehensive progress analytics"""
        conn = sqlite3.connect(self.db_path)
        
        # Session statistics
        sessions_df = pd.read_sql_query("""
        SELECT * FROM practice_sessions 
        WHERE end_time IS NOT NULL
        ORDER BY start_time DESC
        """, conn)
        
        # Question attempt statistics - simplified approach
        attempts_df = pd.read_sql_query("""
        SELECT qa.*, ps.category_focus
        FROM question_attempts qa
        JOIN practice_sessions ps ON qa.session_id = ps.session_id
        """, conn)
        
        conn.close()
        
        # Add question categories manually
        if not attempts_df.empty:
            attempts_df['question_category'] = attempts_df['question_id'].map(
                lambda qid: self.question_bank.questions[qid].category if qid in self.question_bank.questions else 'Unknown'
            )
        
        analytics = {
            'total_sessions': len(sessions_df),
            'total_questions_attempted': len(attempts_df),
            'average_session_score': sessions_df['total_score'].mean() if not sessions_df.empty else 0,
            'score_trend': sessions_df['total_score'].tolist()[-10:],  # Last 10 sessions
            'category_performance': {},
            'difficulty_performance': {},
            'recent_activity': sessions_df.head(5).to_dict('records') if not sessions_df.empty else []
        }
        
        # Category performance
        if not attempts_df.empty and 'question_category' in attempts_df.columns:
            category_stats = attempts_df.groupby('question_category')['score'].agg(['mean', 'count']).round(2)
            analytics['category_performance'] = category_stats.to_dict('index')
        
        return analytics
    
    def get_personalized_recommendations(self) -> List[str]:
        """Get personalized study recommendations"""
        analytics = self.get_progress_analytics()
        recommendations = []
        
        # Analyze weak areas
        if analytics['category_performance']:
            weak_categories = [cat for cat, stats in analytics['category_performance'].items() 
                             if stats['mean'] < 70]
            
            if weak_categories:
                recommendations.append(f"Focus on improving: {', '.join(weak_categories)}")
        
        # Session frequency
        if analytics['total_sessions'] < 5:
            recommendations.append("Complete more practice sessions to build consistency")
        
        # Score trends
        if len(analytics['score_trend']) >= 3:
            recent_avg = sum(analytics['score_trend'][-3:]) / 3
            if recent_avg < 75:
                recommendations.append("Practice easier questions to build confidence")
            elif recent_avg > 85:
                recommendations.append("Challenge yourself with harder difficulty questions")
        
        # Company-specific prep
        top_companies = ['Google', 'Amazon', 'Microsoft', 'Meta', 'Apple']
        recommendations.append(f"Practice company-specific questions for: {random.choice(top_companies)}")
        
        return recommendations
    
    def generate_study_plan(self, target_company: str, weeks: int = 4) -> Dict[str, Any]:
        """Generate a structured study plan"""
        plan = {
            'target_company': target_company,
            'duration_weeks': weeks,
            'weekly_schedule': {},
            'question_distribution': {},
            'milestones': []
        }
        
        # Company-specific focus
        company_questions = self.question_bank.get_questions_by_company(target_company)
        
        categories = list(set(q.category for q in company_questions))
        
        # Weekly breakdown
        for week in range(1, weeks + 1):
            if week <= 2:
                difficulty_focus = "Easy to Medium"
                daily_questions = 2
            else:
                difficulty_focus = "Medium to Hard"
                daily_questions = 3
            
            plan['weekly_schedule'][f'Week {week}'] = {
                'focus_categories': categories[:2] if week % 2 == 1 else categories[2:],
                'difficulty_focus': difficulty_focus,
                'daily_questions': daily_questions,
                'practice_sessions': 3
            }
        
        # Milestones
        plan['milestones'] = [
            f"Week 1: Complete 10 {categories[0]} questions",
            f"Week 2: Achieve 80% average in {categories[1]} category",
            f"Week 3: Complete 5 Hard difficulty questions",
            f"Week 4: Mock interview simulation with company-specific questions"
        ]
        
        return plan

class MockInterviewSimulator:
    """Mock interview simulation system"""
    
    def __init__(self, prep_system: InterviewPrepSystem):
        self.prep_system = prep_system
        self.current_interview = None
    
    def start_mock_interview(self, company: str, role: str = "Software Engineer") -> Dict[str, Any]:
        """Start a mock interview simulation"""
        interview_id = f"MOCK_{company}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Select questions based on company and role
        company_questions = self.prep_system.question_bank.get_questions_by_company(company)
        
        # Interview structure: 
        # 1 Easy (warm-up), 2 Medium (core), 1 Hard (challenge)
        easy_q = [q for q in company_questions if q.difficulty == "Easy"]
        medium_q = [q for q in company_questions if q.difficulty == "Medium"] 
        hard_q = [q for q in company_questions if q.difficulty == "Hard"]
        
        interview_questions = []
        if easy_q:
            interview_questions.append(random.choice(easy_q))
        if medium_q:
            interview_questions.extend(random.sample(medium_q, min(2, len(medium_q))))
        if hard_q:
            interview_questions.append(random.choice(hard_q))
        
        self.current_interview = {
            'interview_id': interview_id,
            'company': company,
            'role': role,
            'questions': interview_questions,
            'current_question': 0,
            'start_time': datetime.now(),
            'scores': {},
            'total_time': 0
        }
        
        return {
            'interview_id': interview_id,
            'total_questions': len(interview_questions),
            'estimated_duration': sum(q.time_limit for q in interview_questions),
            'first_question': interview_questions[0] if interview_questions else None
        }
    
    def get_current_question(self) -> Optional[InterviewQuestion]:
        """Get current interview question"""
        if not self.current_interview or self.current_interview['current_question'] >= len(self.current_interview['questions']):
            return None
        
        return self.current_interview['questions'][self.current_interview['current_question']]
    
    def submit_interview_answer(self, score: int, time_taken: int) -> bool:
        """Submit answer for current question"""
        if not self.current_interview:
            return False
        
        current_q = self.get_current_question()
        if current_q:
            self.current_interview['scores'][current_q.id] = score
            self.current_interview['total_time'] += time_taken
            self.current_interview['current_question'] += 1
            return True
        
        return False
    
    def finish_mock_interview(self) -> Dict[str, Any]:
        """Finish mock interview and get results"""
        if not self.current_interview:
            return {}
        
        end_time = datetime.now()
        duration = (end_time - self.current_interview['start_time']).total_seconds() / 60
        
        scores = list(self.current_interview['scores'].values())
        average_score = sum(scores) / len(scores) if scores else 0
        
        # Performance analysis
        performance_level = "Excellent" if average_score >= 90 else \
                          "Good" if average_score >= 75 else \
                          "Needs Improvement"
        
        feedback = self._generate_interview_feedback(average_score, self.current_interview)
        
        results = {
            'interview_id': self.current_interview['interview_id'],
            'company': self.current_interview['company'],
            'duration_minutes': round(duration, 1),
            'questions_completed': len(scores),
            'average_score': round(average_score, 1),
            'performance_level': performance_level,
            'detailed_scores': self.current_interview['scores'],
            'feedback': feedback,
            'next_steps': self._get_improvement_suggestions(average_score)
        }
        
        self.current_interview = None
        return results
    
    def _generate_interview_feedback(self, average_score: float, interview_data: Dict) -> List[str]:
        """Generate detailed interview feedback"""
        feedback = []
        
        if average_score >= 85:
            feedback.append("Strong performance! You demonstrated solid technical skills.")
        elif average_score >= 70:
            feedback.append("Good foundation, with room for improvement in implementation details.")
        else:
            feedback.append("Focus on fundamental concepts and practice more coding problems.")
        
        # Time management feedback
        total_time = interview_data['total_time']
        expected_time = sum(q.time_limit for q in interview_data['questions'])
        
        if total_time > expected_time * 1.2:
            feedback.append("Work on time management - practice solving problems faster.")
        elif total_time < expected_time * 0.8:
            feedback.append("Consider taking more time to explain your thought process.")
        
        return feedback
    
    def _get_improvement_suggestions(self, average_score: float) -> List[str]:
        """Get specific improvement suggestions"""
        suggestions = []
        
        if average_score < 70:
            suggestions.extend([
                "Practice fundamental algorithms and data structures",
                "Complete at least 50 easy problems before attempting medium",
                "Focus on understanding problem patterns"
            ])
        elif average_score < 85:
            suggestions.extend([
                "Practice medium to hard problems daily",
                "Work on optimizing time and space complexity",
                "Practice explaining solutions clearly"
            ])
        else:
            suggestions.extend([
                "Focus on system design questions",
                "Practice behavioral interview questions",
                "Consider mock interviews with peers"
            ])
        
        return suggestions

# Example usage and demo
if __name__ == "__main__":
    def demo_interview_prep_system():
        print("🎯 Interview Preparation System Demo")
        print("Epic 8: FAANG Career Advancement")
        print("=" * 60)
        
        # Initialize system
        prep_system = InterviewPrepSystem()
        
        # Start practice session
        print("\n📚 Starting Practice Session...")
        session_id = prep_system.start_practice_session("Python")
        print(f"   Session ID: {session_id}")
        
        # Get some questions
        print("\n❓ Getting Practice Questions...")
        for i in range(3):
            question = prep_system.get_next_question()
            if question:
                print(f"   Question {i+1}: {question.difficulty} - {question.question[:80]}...")
                # Simulate answering (random score for demo)
                score = random.randint(60, 95)
                time_taken = random.randint(15, 45)
                prep_system.submit_answer(question.id, score, time_taken)
        
        # End session
        summary = prep_system.end_practice_session()
        print(f"\n📊 Session Summary:")
        print(f"   Duration: {summary['duration']:.1f} minutes")
        print(f"   Questions: {summary['questions_attempted']}")
        print(f"   Average Score: {summary['average_score']:.1f}%")
        
        # Generate study plan
        print("\n📋 Generating Study Plan...")
        study_plan = prep_system.generate_study_plan("Google", weeks=4)
        print(f"   Target: {study_plan['target_company']}")
        print(f"   Duration: {study_plan['duration_weeks']} weeks")
        print(f"   Milestones: {len(study_plan['milestones'])}")
        
        # Mock interview simulation
        print("\n🎭 Mock Interview Simulation...")
        simulator = MockInterviewSimulator(prep_system)
        mock_interview = simulator.start_mock_interview("Google")
        print(f"   Interview ID: {mock_interview['interview_id']}")
        print(f"   Total Questions: {mock_interview['total_questions']}")
        
        # Simulate interview completion
        for _ in range(mock_interview['total_questions']):
            if simulator.get_current_question():
                score = random.randint(70, 100)
                time_taken = random.randint(20, 40)
                simulator.submit_interview_answer(score, time_taken)
        
        results = simulator.finish_mock_interview()
        print(f"   Final Score: {results['average_score']}%")
        print(f"   Performance: {results['performance_level']}")
        
        # Get recommendations
        print("\n💡 Personalized Recommendations:")
        recommendations = prep_system.get_personalized_recommendations()
        for rec in recommendations:
            print(f"   • {rec}")
        
        print("\n🚀 Epic 8: Interview Preparation System - OPERATIONAL")
        print("   • Comprehensive question bank loaded")
        print("   • Practice session tracking active")
        print("   • Mock interview simulation ready")
        print("   • Progress analytics and recommendations enabled")
    
    # Run demo
    demo_interview_prep_system()
