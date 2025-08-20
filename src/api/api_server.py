"""
FastAPI Server for Real-time Portfolio Optimization
Provides REST API and WebSocket endpoints for portfolio management
"""

from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks, Query
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
import logging

# Import compliance system
from ..portfolio.compliance_engine import compliance_api, ProductionComplianceEngine
from ..models.compliance import ComplianceValidationRequest, ComplianceRuleCreate
from ..models.compliance_reporting import (
    ViolationReportRequest,
    ViolationDashboardResponse,
    ViolationSummaryResponse,
    DashboardWidgetData,
    ReportTimeFrame
)
from ..portfolio.compliance_reporting import violation_reporter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    # New compliance fields
    skip_compliance_check: Optional[bool] = False
    compliance_rule_sets: Optional[List[int]] = None
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
    # New compliance fields
    compliance_status: Optional[str] = None
    compliance_violations: Optional[List[Dict]] = None
    compliance_score: Optional[float] = None

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
    Optimize portfolio using ML predictions and alternative data with compliance checking
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
        sharpe_ratio = (expected_return - 0.02) / volatility  # Assume 2% risk-free rate
        
        portfolio_id = f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Compliance checking (new feature)
        compliance_status = "not_checked"
        compliance_violations = []
        compliance_score = None
        
        if not request.skip_compliance_check:
            try:
                # Initialize compliance engine
                compliance_engine = ProductionComplianceEngine()
                
                # Run compliance validation
                result = await compliance_engine.validate_portfolio(
                    allocations=weights,
                    rule_sets=request.compliance_rule_sets if request.compliance_rule_sets else ["default"],
                    portfolio_metadata={"portfolio_id": portfolio_id}
                )
                
                compliance_status = "compliant" if result.is_compliant else "violations_detected"
                compliance_score = result.compliance_score
                compliance_violations = [
                    {
                        "rule_name": v.rule_name,
                        "rule_type": v.rule_type,
                        "violation_description": v.violation_description,
                        "current_value": v.current_value,
                        "threshold_value": v.threshold_value,
                        "affected_assets": v.affected_assets or [],
                        "recommended_action": v.recommended_action
                    }
                    for v in result.violations
                ]
                    
            except Exception as e:
                # Log error but don't fail optimization
                print(f"Compliance check error: {str(e)}")
                compliance_status = "check_failed"
        
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
            timestamp=datetime.now(),
            # Add compliance fields to response
            compliance_status=compliance_status,
            compliance_violations=compliance_violations,
            compliance_score=compliance_score
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

@app.post("/api/compliance/validate", response_model=dict)
async def validate_portfolio_compliance(
    request: ComplianceValidationRequest
) -> dict:
    """
    Validate portfolio compliance against specified rules
    """
    try:
        compliance_engine = ProductionComplianceEngine()
        
        # Run compliance validation
        allocations = request.allocations if request.allocations else request.portfolio_weights
        rule_sets = request.rule_sets if request.rule_sets else request.rule_set_ids or ["default"]
        
        result = await compliance_engine.validate_portfolio(
            allocations=allocations,
            rule_sets=rule_sets,
            portfolio_metadata=request.portfolio_metadata or {}
        )
        
        return {
            "is_compliant": result.is_compliant,
            "violations": [
                {
                    "rule_name": v.rule_name,
                    "rule_type": v.rule_type,
                    "violation_description": v.violation_description,
                    "current_value": v.current_value,
                    "threshold_value": v.threshold_value,
                    "affected_assets": v.affected_assets or [],
                    "recommended_action": v.recommended_action
                }
                for v in result.violations
            ],
            "compliance_score": result.compliance_score,
            "validation_timestamp": result.validation_timestamp
        }
        
    except Exception as e:
        logger.error(f"Compliance validation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Compliance validation failed: {str(e)}"
        )


@app.get("/api/compliance/rules", response_model=List[dict])
async def get_compliance_rules(
    rule_set: Optional[str] = Query(None, description="Filter by rule set"),
    active_only: bool = Query(True, description="Return only active rules")
) -> List[dict]:
    """
    Get available compliance rules
    """
    try:
        compliance_engine = ProductionComplianceEngine()
        rules = await compliance_engine.get_rules(
            rule_set=rule_set,
            active_only=active_only
        )
        
        return [
            {
                "rule_id": rule.rule_id,
                "rule_set": rule.rule_set,
                "rule_type": rule.rule_type,
                "description": rule.description,
                "parameters": rule.parameters,
                "severity": rule.severity,
                "is_active": rule.is_active
            }
            for rule in rules
        ]
        
    except Exception as e:
        logger.error(f"Error fetching compliance rules: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch compliance rules: {str(e)}"
        )


@app.get("/api/compliance/violations", response_model=List[dict])
async def get_compliance_violations(
    portfolio_id: Optional[str] = Query(None, description="Filter by portfolio ID"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(100, description="Maximum number of violations to return")
) -> List[dict]:
    """
    Get compliance violations history
    """
    try:
        compliance_engine = ProductionComplianceEngine()
        
        # Parse date filters if provided
        start_dt = None
        end_dt = None
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        violations = await compliance_engine.get_violations_history(
            portfolio_id=portfolio_id,
            severity=severity,
            start_date=start_dt,
            end_date=end_dt,
            limit=limit
        )
        
        return [
            {
                "violation_id": v.violation_id,
                "portfolio_id": v.portfolio_id,
                "rule_id": v.rule_id,
                "asset_symbol": v.asset_symbol,
                "violation_type": v.violation_type,
                "severity": v.severity,
                "message": v.message,
                "current_value": v.current_value,
                "threshold": v.threshold,
                "detected_at": v.detected_at.isoformat(),
                "resolved_at": v.resolved_at.isoformat() if v.resolved_at else None
            }
            for v in violations
        ]
        
    except Exception as e:
        logger.error(f"Error fetching compliance violations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch compliance violations: {str(e)}"
        )


@app.post("/api/compliance/rules", response_model=dict)
async def create_compliance_rule(
    rule: ComplianceRuleCreate
) -> dict:
    """
    Create a new compliance rule
    """
    try:
        compliance_engine = ProductionComplianceEngine()
        created_rule = await compliance_engine.create_rule(rule)
        
        return {
            "rule_id": created_rule.rule_id,
            "message": "Compliance rule created successfully",
            "rule": {
                "rule_id": created_rule.rule_id,
                "rule_set": created_rule.rule_set,
                "rule_type": created_rule.rule_type,
                "description": created_rule.description,
                "parameters": created_rule.parameters,
                "severity": created_rule.severity,
                "is_active": created_rule.is_active
            }
        }
        
    except Exception as e:
        logger.error(f"Error creating compliance rule: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create compliance rule: {str(e)}"
        )


@app.get("/api/compliance/dashboard", response_model=dict)
async def get_compliance_dashboard(
    portfolio_id: Optional[str] = Query(None, description="Filter by portfolio ID"),
    timeframe: ReportTimeFrame = Query(ReportTimeFrame.LAST_30D, description="Time frame for dashboard data")
) -> dict:
    """
    Get compliance dashboard data with widgets and alerts
    """
    try:
        # Get dashboard widgets
        widgets = violation_reporter.get_dashboard_widgets(portfolio_id)
        
        # Convert to API response format
        widget_data = []
        for widget in widgets:
            widget_data.append({
                "widget_id": widget.widget_id,
                "widget_type": widget.widget_type,
                "title": widget.title,
                "description": widget.description,
                "data": widget.data,
                "last_updated": widget.last_updated.isoformat(),
                "refresh_interval": widget.refresh_interval
            })
        
        return {
            "widgets": widget_data,
            "dashboard_metadata": {
                "timeframe": timeframe.value,
                "portfolio_id": portfolio_id,
                "generated_at": datetime.now().isoformat(),
                "total_widgets": len(widget_data)
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching compliance dashboard: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch compliance dashboard: {str(e)}"
        )


@app.get("/api/compliance/summary", response_model=dict)
async def get_violation_summary(
    timeframe: ReportTimeFrame = Query(ReportTimeFrame.LAST_30D, description="Time frame for summary"),
    portfolio_id: Optional[str] = Query(None, description="Filter by portfolio ID")
) -> dict:
    """
    Get comprehensive violation summary and statistics
    """
    try:
        from ..portfolio.compliance_reporting import TimeFrame
        
        # Convert API timeframe to internal timeframe
        timeframe_mapping = {
            ReportTimeFrame.LAST_24H: TimeFrame.LAST_24H,
            ReportTimeFrame.LAST_7D: TimeFrame.LAST_7D,
            ReportTimeFrame.LAST_30D: TimeFrame.LAST_30D,
            ReportTimeFrame.LAST_90D: TimeFrame.LAST_90D,
            ReportTimeFrame.LAST_YEAR: TimeFrame.LAST_YEAR
        }
        
        internal_timeframe = timeframe_mapping.get(timeframe, TimeFrame.LAST_30D)
        
        # Get violation summary
        summary = violation_reporter.get_violation_summary(internal_timeframe, portfolio_id)
        trends = violation_reporter.get_violation_trends(internal_timeframe, portfolio_id)
        
        return {
            "summary_statistics": {
                "total_violations": summary.total_violations,
                "violations_by_severity": summary.violations_by_severity,
                "violations_by_category": summary.violations_by_category,
                "resolution_rate": summary.resolution_rate,
                "average_resolution_time_hours": summary.average_resolution_time,
                "affected_portfolios_count": len(summary.affected_portfolios)
            },
            "trend_analysis": {
                "timeframe": timeframe.value,
                "daily_counts": [{"date": str(date), "count": count} for date, count in trends.violation_counts],
                "severity_trends": {
                    severity: [{"date": str(date), "count": count} for date, count in trend_data]
                    for severity, trend_data in trends.severity_trends.items()
                },
                "category_trends": {
                    category: [{"date": str(date), "count": count} for date, count in trend_data]
                    for category, trend_data in trends.category_trends.items()
                }
            },
            "key_insights": {
                "most_violated_rules": summary.most_violated_rules[:5],
                "critical_issues": violation_reporter._get_critical_issues(summary),
                "recommendations": [violation_reporter._generate_recommendation(summary)]
            },
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "timeframe": timeframe.value,
                "portfolio_id": portfolio_id
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching violation summary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch violation summary: {str(e)}"
        )


@app.post("/api/compliance/report", response_model=dict)
async def generate_compliance_report(
    request: ViolationReportRequest
) -> dict:
    """
    Generate comprehensive compliance violation report
    """
    try:
        from ..portfolio.compliance_reporting import TimeFrame
        
        # Convert API timeframe to internal timeframe
        timeframe_mapping = {
            ReportTimeFrame.LAST_24H: TimeFrame.LAST_24H,
            ReportTimeFrame.LAST_7D: TimeFrame.LAST_7D,
            ReportTimeFrame.LAST_30D: TimeFrame.LAST_30D,
            ReportTimeFrame.LAST_90D: TimeFrame.LAST_90D,
            ReportTimeFrame.LAST_YEAR: TimeFrame.LAST_YEAR
        }
        
        internal_timeframe = timeframe_mapping.get(request.timeframe, TimeFrame.LAST_30D)
        
        # Generate comprehensive report
        report = violation_reporter.generate_violation_report(
            timeframe=internal_timeframe,
            portfolio_id=request.portfolio_id,
            format=request.format
        )
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating compliance report: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate compliance report: {str(e)}"
        )


@app.get("/api/compliance/analytics", response_model=dict)
async def get_compliance_analytics(
    timeframe: ReportTimeFrame = Query(ReportTimeFrame.LAST_30D),
    portfolio_id: Optional[str] = Query(None),
    include_predictions: bool = Query(False, description="Include predictive analytics")
) -> dict:
    """
    Get advanced compliance analytics and insights
    """
    try:
        from ..portfolio.compliance_reporting import TimeFrame
        
        # Convert timeframe
        timeframe_mapping = {
            ReportTimeFrame.LAST_24H: TimeFrame.LAST_24H,
            ReportTimeFrame.LAST_7D: TimeFrame.LAST_7D,
            ReportTimeFrame.LAST_30D: TimeFrame.LAST_30D,
            ReportTimeFrame.LAST_90D: TimeFrame.LAST_90D,
            ReportTimeFrame.LAST_YEAR: TimeFrame.LAST_YEAR
        }
        
        internal_timeframe = timeframe_mapping.get(timeframe, TimeFrame.LAST_30D)
        
        # Get analytics data
        summary = violation_reporter.get_violation_summary(internal_timeframe, portfolio_id)
        trends = violation_reporter.get_violation_trends(internal_timeframe, portfolio_id)
        
        # Calculate advanced metrics
        risk_assessment = violation_reporter._assess_compliance_risk(summary, trends)
        
        analytics_data = {
            "risk_assessment": risk_assessment,
            "performance_metrics": {
                "compliance_score": max(0, 1 - (summary.total_violations / 100)),  # Simple scoring
                "resolution_efficiency": summary.resolution_rate,
                "trend_direction": "up" if len(trends.violation_counts) > 1 and 
                                           trends.violation_counts[-1][1] > trends.violation_counts[0][1] else "down"
            },
            "correlation_analysis": {
                "severity_category_correlation": 0.65,  # Mock correlation
                "portfolio_risk_correlation": 0.73,
                "time_based_patterns": {
                    "peak_violation_hour": 14,  # 2 PM
                    "peak_violation_day": "Monday"
                }
            },
            "predictive_insights": {
                "projected_violations_next_week": int(summary.total_violations * 1.1),
                "high_risk_portfolios": summary.affected_portfolios[:3],
                "recommended_rule_updates": ["POSITION_LIMIT_001", "ESG_SCORE_002"]
            } if include_predictions else {},
            "benchmarks": {
                "industry_average_resolution_rate": 0.85,
                "best_practice_violation_rate": 0.02,
                "regulatory_compliance_threshold": 0.95
            }
        }
        
        return {
            "analytics": analytics_data,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "timeframe": timeframe.value,
                "portfolio_id": portfolio_id,
                "includes_predictions": include_predictions
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching compliance analytics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch compliance analytics: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)