# Performance Testing Suite - Enterprise-Grade Load Testing
# Designed for FAANG-level performance validation

import asyncio
import time
import statistics
import json
import logging
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LoadTestConfig:
    """Configuration for load testing scenarios."""
    base_url: str = "http://localhost:8000"
    concurrent_users: int = 100
    duration_seconds: int = 300  # 5 minutes
    ramp_up_seconds: int = 60
    endpoints: List[str] = None
    request_timeout: int = 30
    
    def __post_init__(self):
        if self.endpoints is None:
            self.endpoints = [
                "/api/portfolio/optimize",
                "/api/data/market-data",
                "/api/risk/calculate",
                "/api/models/predict",
                "/health"
            ]

@dataclass
class RequestResult:
    """Results from a single request."""
    endpoint: str
    response_time: float
    status_code: int
    success: bool
    timestamp: datetime
    user_id: int
    error_message: str = None

@dataclass
class LoadTestResults:
    """Aggregated results from load testing."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    error_rate: float
    throughput_mbps: float
    concurrent_users: int
    test_duration: float
    endpoint_metrics: Dict[str, Dict[str, float]]

class PortfolioLoadTester:
    """
    Comprehensive load testing suite for portfolio optimizer.
    
    Features:
    - Concurrent user simulation
    - Multiple endpoint testing
    - Real-time metrics collection
    - Performance regression detection
    - Detailed reporting and visualization
    """
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.results: List[RequestResult] = []
        self.session = requests.Session()
        
        # Configure session for performance
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=config.concurrent_users,
            pool_maxsize=config.concurrent_users * 2,
            max_retries=0
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
    
    async def simulate_user_session(self, user_id: int, duration: float) -> List[RequestResult]:
        """
        Simulate a single user's session with realistic behavior.
        
        Args:
            user_id: Unique identifier for the user
            duration: How long the user session should last
        
        Returns:
            List of request results from this user session
        """
        user_results = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Randomly select endpoint based on realistic usage patterns
            endpoint_weights = {
                "/health": 0.1,
                "/api/data/market-data": 0.3,
                "/api/portfolio/optimize": 0.2,
                "/api/risk/calculate": 0.2,
                "/api/models/predict": 0.2
            }
            
            endpoint = np.random.choice(
                list(endpoint_weights.keys()),
                p=list(endpoint_weights.values())
            )
            
            result = await self._make_request(endpoint, user_id)
            user_results.append(result)
            
            # Realistic user think time (1-5 seconds)
            think_time = np.random.exponential(2.0)
            await asyncio.sleep(min(think_time, 5.0))
        
        return user_results
    
    async def _make_request(self, endpoint: str, user_id: int) -> RequestResult:
        """Make a single HTTP request and record metrics."""
        url = f"{self.config.base_url}{endpoint}"
        start_time = time.time()
        timestamp = datetime.utcnow()
        
        try:
            # Prepare request data based on endpoint
            payload = self._get_request_payload(endpoint)
            
            if endpoint == "/health":
                response = self.session.get(url, timeout=self.config.request_timeout)
            elif "optimize" in endpoint:
                response = self.session.post(url, json=payload, timeout=self.config.request_timeout)
            else:
                response = self.session.get(url, timeout=self.config.request_timeout)
            
            response_time = time.time() - start_time
            
            return RequestResult(
                endpoint=endpoint,
                response_time=response_time,
                status_code=response.status_code,
                success=response.status_code < 400,
                timestamp=timestamp,
                user_id=user_id
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return RequestResult(
                endpoint=endpoint,
                response_time=response_time,
                status_code=0,
                success=False,
                timestamp=timestamp,
                user_id=user_id,
                error_message=str(e)
            )
    
    def _get_request_payload(self, endpoint: str) -> Dict[str, Any]:
        """Generate realistic request payloads for different endpoints."""
        if "optimize" in endpoint:
            return {
                "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
                "constraints": {
                    "max_weight": 0.3,
                    "min_weight": 0.05,
                    "risk_tolerance": "moderate"
                },
                "optimization_method": "quantum_inspired"
            }
        elif "risk" in endpoint:
            return {
                "portfolio_id": f"portfolio_{np.random.randint(1, 1000)}",
                "risk_metrics": ["var", "sharpe", "drawdown"]
            }
        elif "predict" in endpoint:
            return {
                "symbols": ["AAPL", "GOOGL"],
                "prediction_horizon": 30
            }
        else:
            return {}
    
    async def run_load_test(self) -> LoadTestResults:
        """
        Execute the complete load test scenario.
        
        Returns:
            Comprehensive test results and metrics
        """
        logger.info(f"Starting load test with {self.config.concurrent_users} users for {self.config.duration_seconds}s")
        
        start_time = time.time()
        
        # Create user tasks with staggered start times (ramp-up)
        tasks = []
        ramp_up_delay = self.config.ramp_up_seconds / self.config.concurrent_users
        
        for user_id in range(self.config.concurrent_users):
            # Calculate when this user should start
            user_start_delay = user_id * ramp_up_delay
            user_duration = self.config.duration_seconds - user_start_delay
            
            if user_duration > 0:
                task = asyncio.create_task(
                    self._delayed_user_session(user_id, user_start_delay, user_duration)
                )
                tasks.append(task)
        
        # Wait for all user sessions to complete
        all_results = await asyncio.gather(*tasks)
        
        # Flatten results from all users
        for user_results in all_results:
            self.results.extend(user_results)
        
        total_duration = time.time() - start_time
        
        # Calculate comprehensive metrics
        return self._calculate_metrics(total_duration)
    
    async def _delayed_user_session(self, user_id: int, delay: float, duration: float) -> List[RequestResult]:
        """Start a user session after a delay (for ramp-up)."""
        await asyncio.sleep(delay)
        return await self.simulate_user_session(user_id, duration)
    
    def _calculate_metrics(self, total_duration: float) -> LoadTestResults:
        """Calculate comprehensive performance metrics from results."""
        if not self.results:
            raise ValueError("No results to analyze")
        
        # Basic metrics
        total_requests = len(self.results)
        successful_requests = sum(1 for r in self.results if r.success)
        failed_requests = total_requests - successful_requests
        
        # Response time metrics
        response_times = [r.response_time for r in self.results if r.success]
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            median_response_time = statistics.median(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
        else:
            avg_response_time = median_response_time = p95_response_time = p99_response_time = 0
        
        # Throughput metrics
        requests_per_second = total_requests / total_duration if total_duration > 0 else 0
        error_rate = (failed_requests / total_requests) * 100 if total_requests > 0 else 0
        
        # Estimate throughput in Mbps (rough approximation)
        avg_response_size_kb = 5  # Assume 5KB average response
        throughput_mbps = (successful_requests * avg_response_size_kb * 8) / (total_duration * 1000) if total_duration > 0 else 0
        
        # Per-endpoint metrics
        endpoint_metrics = self._calculate_endpoint_metrics()
        
        return LoadTestResults(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time=avg_response_time,
            median_response_time=median_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            throughput_mbps=throughput_mbps,
            concurrent_users=self.config.concurrent_users,
            test_duration=total_duration,
            endpoint_metrics=endpoint_metrics
        )
    
    def _calculate_endpoint_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate metrics broken down by endpoint."""
        endpoint_data = {}
        
        for endpoint in self.config.endpoints:
            endpoint_results = [r for r in self.results if r.endpoint == endpoint]
            
            if not endpoint_results:
                continue
            
            successful = [r for r in endpoint_results if r.success]
            response_times = [r.response_time for r in successful]
            
            endpoint_data[endpoint] = {
                'total_requests': len(endpoint_results),
                'successful_requests': len(successful),
                'error_rate': ((len(endpoint_results) - len(successful)) / len(endpoint_results)) * 100,
                'avg_response_time': statistics.mean(response_times) if response_times else 0,
                'p95_response_time': np.percentile(response_times, 95) if response_times else 0,
                'requests_per_second': len(endpoint_results) / self.config.duration_seconds
            }
        
        return endpoint_data
    
    def generate_performance_report(self, results: LoadTestResults, output_dir: str = "reports") -> str:
        """
        Generate comprehensive performance test report.
        
        Args:
            results: Load test results to analyze
            output_dir: Directory to save report files
        
        Returns:
            Path to generated report file
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"performance_report_{timestamp}.json")
        
        # Create detailed report
        report_data = {
            "test_summary": asdict(results),
            "test_configuration": asdict(self.config),
            "performance_thresholds": {
                "response_time_p95_ms": 200,
                "error_rate_max_pct": 1.0,
                "requests_per_second_min": 50,
                "passed": self._evaluate_performance_thresholds(results)
            },
            "recommendations": self._generate_recommendations(results),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Save JSON report
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Generate visualizations
        self._create_performance_charts(results, output_dir, timestamp)
        
        logger.info(f"Performance report generated: {report_path}")
        return report_path
    
    def _evaluate_performance_thresholds(self, results: LoadTestResults) -> bool:
        """Evaluate if performance meets FAANG-level standards."""
        thresholds = {
            'p95_response_time': 200,  # ms
            'error_rate': 1.0,  # %
            'requests_per_second': 50
        }
        
        checks = [
            results.p95_response_time * 1000 <= thresholds['p95_response_time'],
            results.error_rate <= thresholds['error_rate'],
            results.requests_per_second >= thresholds['requests_per_second']
        ]
        
        return all(checks)
    
    def _generate_recommendations(self, results: LoadTestResults) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        if results.p95_response_time > 0.2:  # > 200ms
            recommendations.append("Consider optimizing slow endpoints or adding caching")
        
        if results.error_rate > 1.0:
            recommendations.append("Investigate and fix error sources for better reliability")
        
        if results.requests_per_second < 50:
            recommendations.append("Scale infrastructure or optimize application performance")
        
        # Endpoint-specific recommendations
        for endpoint, metrics in results.endpoint_metrics.items():
            if metrics['error_rate'] > 5.0:
                recommendations.append(f"High error rate on {endpoint}: {metrics['error_rate']:.1f}%")
            
            if metrics['avg_response_time'] > 0.5:
                recommendations.append(f"Slow response time on {endpoint}: {metrics['avg_response_time']:.2f}s")
        
        if not recommendations:
            recommendations.append("Performance meets enterprise standards - consider load balancing for scaling")
        
        return recommendations
    
    def _create_performance_charts(self, results: LoadTestResults, output_dir: str, timestamp: str):
        """Create performance visualization charts."""
        plt.style.use('seaborn-v0_8')
        
        # Response time distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Response time histogram
        response_times = [r.response_time * 1000 for r in self.results if r.success]
        axes[0, 0].hist(response_times, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Response Time Distribution')
        axes[0, 0].set_xlabel('Response Time (ms)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(results.p95_response_time * 1000, color='red', linestyle='--', label='P95')
        axes[0, 0].legend()
        
        # 2. Requests over time
        times = [(r.timestamp - min(r.timestamp for r in self.results)).total_seconds() for r in self.results]
        axes[0, 1].scatter(times, [r.response_time * 1000 for r in self.results], alpha=0.6, s=10)
        axes[0, 1].set_title('Response Time Over Time')
        axes[0, 1].set_xlabel('Time (seconds)')
        axes[0, 1].set_ylabel('Response Time (ms)')
        
        # 3. Error rate by endpoint
        endpoints = list(results.endpoint_metrics.keys())
        error_rates = [results.endpoint_metrics[ep]['error_rate'] for ep in endpoints]
        axes[1, 0].bar(range(len(endpoints)), error_rates, color='lightcoral')
        axes[1, 0].set_title('Error Rate by Endpoint')
        axes[1, 0].set_xlabel('Endpoint')
        axes[1, 0].set_ylabel('Error Rate (%)')
        axes[1, 0].set_xticks(range(len(endpoints)))
        axes[1, 0].set_xticklabels([ep.split('/')[-1] for ep in endpoints], rotation=45)
        
        # 4. Performance summary metrics
        metrics_names = ['Avg Response\nTime (ms)', 'P95 Response\nTime (ms)', 'Requests/sec', 'Error Rate (%)']
        metrics_values = [
            results.average_response_time * 1000,
            results.p95_response_time * 1000,
            results.requests_per_second,
            results.error_rate
        ]
        
        bars = axes[1, 1].bar(metrics_names, metrics_values, color=['lightgreen', 'gold', 'lightblue', 'lightcoral'])
        axes[1, 1].set_title('Key Performance Metrics')
        axes[1, 1].set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics_values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        chart_path = os.path.join(output_dir, f"performance_charts_{timestamp}.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance charts saved: {chart_path}")

# Load testing scenarios for different use cases
class PerformanceTestSuite:
    """Collection of standardized performance test scenarios."""
    
    @staticmethod
    def faang_interview_scenario() -> LoadTestConfig:
        """Performance test scenario designed for FAANG interview demonstration."""
        return LoadTestConfig(
            concurrent_users=200,
            duration_seconds=600,  # 10 minutes
            ramp_up_seconds=120,
            endpoints=[
                "/api/portfolio/optimize",
                "/api/data/market-data",
                "/api/risk/calculate",
                "/api/models/predict",
                "/health"
            ]
        )
    
    @staticmethod
    def production_readiness_test() -> LoadTestConfig:
        """Comprehensive production readiness validation."""
        return LoadTestConfig(
            concurrent_users=500,
            duration_seconds=1800,  # 30 minutes
            ramp_up_seconds=300,
            endpoints=[
                "/api/portfolio/optimize",
                "/api/data/market-data",
                "/api/risk/calculate",
                "/api/models/predict",
                "/api/reports/generate",
                "/health"
            ]
        )
    
    @staticmethod
    def stress_test_scenario() -> LoadTestConfig:
        """Stress test to find breaking points."""
        return LoadTestConfig(
            concurrent_users=1000,
            duration_seconds=300,
            ramp_up_seconds=60,
            endpoints=[
                "/api/portfolio/optimize",
                "/api/data/market-data"
            ]
        )

async def main():
    """Main function to run performance tests."""
    # FAANG-level performance test
    config = PerformanceTestSuite.faang_interview_scenario()
    tester = PortfolioLoadTester(config)
    
    logger.info("Starting FAANG-level performance test...")
    results = await tester.run_load_test()
    
    # Generate comprehensive report
    report_path = tester.generate_performance_report(results)
    
    # Print summary
    print("\n" + "="*80)
    print("PERFORMANCE TEST RESULTS SUMMARY")
    print("="*80)
    print(f"Total Requests: {results.total_requests:,}")
    print(f"Successful Requests: {results.successful_requests:,}")
    print(f"Error Rate: {results.error_rate:.2f}%")
    print(f"Average Response Time: {results.average_response_time*1000:.1f}ms")
    print(f"P95 Response Time: {results.p95_response_time*1000:.1f}ms")
    print(f"P99 Response Time: {results.p99_response_time*1000:.1f}ms")
    print(f"Requests per Second: {results.requests_per_second:.1f}")
    print(f"Concurrent Users: {results.concurrent_users}")
    print(f"Test Duration: {results.test_duration:.1f}s")
    print("="*80)
    
    # Performance threshold check
    threshold_check = tester._evaluate_performance_thresholds(results)
    print(f"FAANG Performance Standards: {'✅ PASSED' if threshold_check else '❌ FAILED'}")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
