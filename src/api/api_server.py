"""
FastAPI Server for Real-time Portfolio Optimization
Provides REST API and WebSocket endpoints for portfolio management
"""

from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import asyncio
import json
import pandas as pd
import numpy as np
from enum import Enum

# Import our modules (these would be in src/ directory)
# from src.portfolio_optimizer import MLPortfolioOptimizer
# from src.alternative_data_collector import AlternativeDataCollector

app = FastAPI(
    title="Quantum Portfolio Optimizer API",
    description="ML-powered portfolio optimization with alternative data",
    version="1.0.0"
)

# Enable CORS for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Data Models ====================

class OptimizationMethod(str, Enum):
    max_sharpe = "max_sharpe"
    min_volatility = "min_volatility"
    hrp = "hrp"
    risk_parity = "risk_parity"
    max_diversification = "max_diversification"

class PortfolioRequest(BaseModel):
    tickers: List[str] = Field(..., min_items=2, max_items=50)
    optimization_method: OptimizationMethod = OptimizationMethod.max_sharpe
    use_ml_predictions: bool = True
    use_alternative_data: bool = True
    initial_capital: float = Field(100000, gt=0)
    risk_free_rate: float = Field(0.04, ge=0, le=0.2)

class PortfolioResponse(BaseModel):
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    risk_metrics: Dict[str, float]
    regime: str
    ml_confidence: float
    timestamp: datetime

class BacktestRequest(BaseModel):
    tickers: List[str]
    start_date: str
    end_date: str
    rebalance_frequency: str = "monthly"
    initial_capital: float = 100000

class RiskAlert(BaseModel):
    level: str  # "info", "warning", "critical"
    message: str
    metric: str
    value: float
    threshold: float
    timestamp: datetime

# ==================== In-Memory Storage ====================

# Store active portfolios and their performance
active_portfolios = {}
portfolio_performance = {}
risk_alerts = []

# WebSocket connections for real-time updates
active_connections = []

# ==================== WebSocket Manager ====================

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        await websocket.send_json({
            "type": "connection",
            "message": "Connected to Portfolio Optimizer",
            "timestamp": datetime.now().isoformat()
        })

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# ==================== API Endpoints ====================

@app.get("/")
async def root():
    return {
        "name": "Quantum Portfolio Optimizer API",
        "status": "operational",
        "endpoints": {
            "optimize": "/api/optimize",
            "backtest": "/api/backtest",
            "risk_metrics": "/api/risk/{portfolio_id}",
            "websocket": "/ws"
        }
    }

@app.post("/api/optimize", response_model=PortfolioResponse)
async def optimize_portfolio(request: PortfolioRequest, background_tasks: BackgroundTasks):
    """
    Optimize portfolio using ML predictions and alternative data
    """
    try:
        # In production, these would use actual implementations
        # optimizer = MLPortfolioOptimizer(request.tickers)
        
        # For demo, return mock optimized portfolio
        weights = {}
        remaining = 1.0
        for i, ticker in enumerate(request.tickers):
            if i == len(request.tickers) - 1:
                weights[ticker] = round(remaining, 4)
            else:
                weight = np.random.uniform(0.05, remaining / (len(request.tickers) - i))
                weights[ticker] = round(weight, 4)
                remaining -= weight
        
        # Normalize weights
        total = sum(weights.values())
        weights = {k: round(v/total, 4) for k, v in weights.items()}
        
        # Calculate mock performance metrics
        expected_return = np.random.uniform(0.08, 0.25)
        volatility = np.random.uniform(0.10, 0.20)
        sharpe_ratio = (expected_return - request.risk_free_rate) / volatility
        
        portfolio_id = f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        response = PortfolioResponse(
            weights=weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            risk_metrics={
                "var_95": -np.random.uniform(0.02, 0.05),
                "cvar_95": -np.random.uniform(0.03, 0.07),
                "max_drawdown": -np.random.uniform(0.05, 0.15),
                "sortino_ratio": np.random.uniform(1.5, 3.0),
                "calmar_ratio": np.random.uniform(1.0, 2.5)
            },
            regime="neutral",
            ml_confidence=np.random.uniform(0.7, 0.95),
            timestamp=datetime.now()
        )
        
        # Store portfolio
        active_portfolios[portfolio_id] = response.dict()
        
        # Schedule background monitoring
        background_tasks.add_task(monitor_portfolio_risk, portfolio_id)
        
        # Broadcast update to WebSocket clients
        await manager.broadcast({
            "type": "portfolio_optimized",
            "portfolio_id": portfolio_id,
            "data": response.dict()
        })
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/backtest")
async def backtest_strategy(request: BacktestRequest):
    """
    Backtest portfolio strategy over historical period
    """
    try:
        # Generate mock backtest results
        days = pd.date_range(start=request.start_date, end=request.end_date, freq='D')
        
        # Simulate returns
        daily_returns = np.random.normal(0.0008, 0.012, len(days))
        cumulative_returns = (1 + daily_returns).cumprod()
        portfolio_values = request.initial_capital * cumulative_returns
        
        # Calculate metrics
        total_return = (portfolio_values[-1] / request.initial_capital - 1)
        annual_return = (1 + total_return) ** (252 / len(days)) - 1
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility
        
        # Find max drawdown
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            "summary": {
                "total_return": round(total_return, 4),
                "annual_return": round(annual_return, 4),
                "volatility": round(volatility, 4),
                "sharpe_ratio": round(sharpe, 2),
                "max_drawdown": round(max_drawdown, 4),
                "win_rate": round(np.mean(daily_returns > 0), 2)
            },
            "timeline": {
                "dates": [d.isoformat() for d in days[::30]],  # Sample every 30 days
                "values": portfolio_values[::30].tolist(),
                "returns": (daily_returns[::30] * 100).tolist()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/risk/{portfolio_id}")
async def get_risk_metrics(portfolio_id: str):
    """
    Get real-time risk metrics for a portfolio
    """
    if portfolio_id not in active_portfolios:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    portfolio = active_portfolios[portfolio_id]
    
    # Check for risk alerts
    current_alerts = check_risk_thresholds(portfolio)
    
    return {
        "portfolio_id": portfolio_id,
        "risk_metrics": portfolio["risk_metrics"],
        "alerts": current_alerts,
        "last_updated": datetime.now().isoformat()
    }

@app.get("/api/alternative-data/{ticker}")
async def get_alternative_data(ticker: str):
    """
    Get alternative data signals for a specific ticker
    """
    # Mock alternative data
    return {
        "ticker": ticker,
        "sentiment": {
            "reddit": np.random.uniform(-0.5, 0.5),
            "twitter": np.random.uniform(-0.5, 0.5),
            "news": np.random.uniform(-0.5, 0.5)
        },
        "google_trends": {
            "interest": np.random.randint(30, 100),
            "momentum": np.random.uniform(-0.2, 0.3)
        },
        "satellite_data": {
            "activity_index": np.random.uniform(0.3, 0.9),
            "trend": np.random.uniform(-0.1, 0.1)
        },
        "composite_score": np.random.uniform(0.3, 0.8),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/market-regime")
async def get_market_regime():
    """
    Get current market regime detection
    """
    regimes = ["bull_market", "bear_market", "high_volatility", "neutral"]
    current_regime = np.random.choice(regimes)
    
    return {
        "regime": current_regime,
        "confidence": np.random.uniform(0.6, 0.95),
        "indicators": {
            "vix_level": np.random.uniform(12, 35),
            "market_breadth": np.random.uniform(0.3, 0.8),
            "momentum": np.random.uniform(-0.2, 0.2),
            "correlation": np.random.uniform(0.3, 0.8)
        },
        "recommendation": get_regime_recommendation(current_regime),
        "timestamp": datetime.now().isoformat()
    }

# ==================== WebSocket Endpoint ====================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket for real-time portfolio updates
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # Send periodic updates
            await asyncio.sleep(5)  # Update every 5 seconds
            
            # Mock real-time data
            update = {
                "type": "market_update",
                "data": {
                    "spy_price": 450 + np.random.uniform(-5, 5),
                    "vix": 15 + np.random.uniform(-2, 2),
                    "market_sentiment": np.random.uniform(-1, 1),
                    "active_portfolios": len(active_portfolios)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            await websocket.send_json(update)
            
    except Exception as e:
        manager.disconnect(websocket)

# ==================== Helper Functions ====================

def check_risk_thresholds(portfolio: dict) -> List[dict]:
    """Check if any risk metrics exceed thresholds"""
    alerts = []
    
    risk_metrics = portfolio["risk_metrics"]
    
    # Check VaR threshold
    if risk_metrics["var_95"] < -0.05:
        alerts.append({
            "level": "warning",
            "message": "Value at Risk exceeds 5% threshold",
            "metric": "var_95",
            "value": risk_metrics["var_95"],
            "threshold": -0.05
        })
    
    # Check max drawdown
    if risk_metrics["max_drawdown"] < -0.10:
        alerts.append({
            "level": "critical",
            "message": "Maximum drawdown exceeds 10%",
            "metric": "max_drawdown",
            "value": risk_metrics["max_drawdown"],
            "threshold": -0.10
        })
    
    # Check Sharpe ratio
    if portfolio["sharpe_ratio"] < 1.0:
        alerts.append({
            "level": "info",
            "message": "Sharpe ratio below target",
            "metric": "sharpe_ratio",
            "value": portfolio["sharpe_ratio"],
            "threshold": 1.0
        })
    
    return alerts

def get_regime_recommendation(regime: str) -> str:
    """Get portfolio recommendation based on market regime"""
    recommendations = {
        "bull_market": "Increase equity allocation, consider growth stocks",
        "bear_market": "Increase defensive positions, consider bonds and gold",
        "high_volatility": "Reduce position sizes, increase cash allocation",
        "neutral": "Maintain balanced allocation, rebalance regularly"
    }
    return recommendations.get(regime, "Monitor closely")

async def monitor_portfolio_risk(portfolio_id: str):
    """Background task to monitor portfolio risk"""
    # This would run continuously in production
    await asyncio.sleep(60)  # Check every minute
    
    if portfolio_id in active_portfolios:
        alerts = check_risk_thresholds(active_portfolios[portfolio_id])
        if alerts:
            await manager.broadcast({
                "type": "risk_alert",
                "portfolio_id": portfolio_id,
                "alerts": alerts,
                "timestamp": datetime.now().isoformat()
            })

# ==================== Startup/Shutdown ====================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("🚀 Quantum Portfolio Optimizer API Started")
    print("📊 WebSocket available at /ws")
    print("📈 API documentation at /docs")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("👋 Shutting down Portfolio Optimizer API")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)