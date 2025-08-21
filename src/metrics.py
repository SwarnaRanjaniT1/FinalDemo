"""
Metrics tracking and performance monitoring module.
Implements performance monitoring as described in Objective 4.
"""

import streamlit as st
import time
import statistics
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd


class MetricsTracker:
    """
    Tracks and analyzes system performance metrics.
    Monitors response times, accuracy, and usage patterns.
    """
    
    def __init__(self):
        """Initialize the metrics tracker."""
        self.query_logs = []
        self.system_metrics = {
            'startup_time': time.time(),
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'avg_response_time': 0.0,
            'last_query_time': None
        }
        
        # Performance targets from technical specifications
        self.targets = {
            'retrieval_p95': 500.0,  # milliseconds
            'end_to_end_p95': 700.0,  # milliseconds
            'success_rate': 99.0,     # percentage
            'accuracy_threshold': 0.8  # minimum accuracy score
        }
    
    def record_query(self, query: str, response: str, response_time: float, num_sources: int = 0, success: bool = True) -> None:
        """
        Record a query and its metrics.
        
        Args:
            query: User query
            response: System response
            response_time: Total response time in milliseconds
            num_sources: Number of source documents retrieved
            success: Whether the query was successful
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'response_time': response_time,
            'num_sources': num_sources,
            'success': success,
            'query_length': len(query),
            'response_length': len(response)
        }
        
        self.query_logs.append(log_entry)
        
        # Update system metrics
        self.system_metrics['total_queries'] += 1
        if success:
            self.system_metrics['successful_queries'] += 1
        else:
            self.system_metrics['failed_queries'] += 1
        
        self.system_metrics['last_query_time'] = datetime.now().isoformat()
        
        # Update average response time
        response_times = [log['response_time'] for log in self.query_logs if log['success']]
        if response_times:
            self.system_metrics['avg_response_time'] = sum(response_times) / len(response_times)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of system performance.
        
        Returns:
            Performance summary dictionary
        """
        if not self.query_logs:
            return {
                'total_queries': 0,
                'success_rate': 0.0,
                'avg_response_time': 0.0,
                'p95_response_time': 0.0,
                'meets_targets': False
            }
        
        successful_queries = [log for log in self.query_logs if log['success']]
        response_times = [log['response_time'] for log in successful_queries]
        
        if not response_times:
            return {
                'total_queries': len(self.query_logs),
                'success_rate': 0.0,
                'avg_response_time': 0.0,
                'p95_response_time': 0.0,
                'meets_targets': False
            }
        
        # Calculate percentiles
        response_times_sorted = sorted(response_times)
        n = len(response_times_sorted)
        
        p95_index = min(int(0.95 * n), n - 1)
        p95_response_time = response_times_sorted[p95_index]
        
        success_rate = (len(successful_queries) / len(self.query_logs)) * 100
        avg_response_time = sum(response_times) / len(response_times)
        
        # Check if targets are met
        meets_targets = (
            p95_response_time <= self.targets['end_to_end_p95'] and
            success_rate >= self.targets['success_rate']
        )
        
        return {
            'total_queries': len(self.query_logs),
            'successful_queries': len(successful_queries),
            'failed_queries': len(self.query_logs) - len(successful_queries),
            'success_rate': success_rate,
            'avg_response_time': avg_response_time,
            'p95_response_time': p95_response_time,
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'meets_targets': meets_targets
        }
    
    def get_detailed_percentiles(self) -> Dict[str, float]:
        """
        Get detailed response time percentiles.
        
        Returns:
            Dictionary of percentile values
        """
        successful_queries = [log for log in self.query_logs if log['success']]
        response_times = [log['response_time'] for log in successful_queries]
        
        if not response_times:
            return {}
        
        response_times_sorted = sorted(response_times)
        n = len(response_times_sorted)
        
        percentiles = {}
        for p in [50, 75, 90, 95, 99]:
            index = min(int((p / 100) * n), n - 1)
            percentiles[f'p{p}'] = response_times_sorted[index]
        
        return percentiles
    
    def get_usage_patterns(self) -> Dict[str, Any]:
        """
        Analyze usage patterns from query logs.
        
        Returns:
            Usage pattern analysis
        """
        if not self.query_logs:
            return {}
        
        # Analyze query patterns
        query_lengths = [log['query_length'] for log in self.query_logs]
        response_lengths = [log['response_length'] for log in self.query_logs]
        sources_retrieved = [log['num_sources'] for log in self.query_logs]
        
        # Time-based analysis
        timestamps = [datetime.fromisoformat(log['timestamp']) for log in self.query_logs]
        
        # Calculate query frequency
        if len(timestamps) > 1:
            time_span = timestamps[-1] - timestamps[0]
            queries_per_hour = len(timestamps) / max(time_span.total_seconds() / 3600, 1)
        else:
            queries_per_hour = 0
        
        return {
            'avg_query_length': statistics.mean(query_lengths),
            'avg_response_length': statistics.mean(response_lengths),
            'avg_sources_retrieved': statistics.mean(sources_retrieved),
            'queries_per_hour': queries_per_hour,
            'peak_query_length': max(query_lengths),
            'peak_response_length': max(response_lengths),
            'total_time_span_hours': (timestamps[-1] - timestamps[0]).total_seconds() / 3600 if len(timestamps) > 1 else 0
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health status.
        
        Returns:
            System health dictionary
        """
        performance = self.get_performance_summary()
        
        # Determine health status
        health_score = 100.0
        issues = []
        
        # Check response time targets
        if performance.get('p95_response_time', 0) > self.targets['end_to_end_p95']:
            health_score -= 30
            issues.append(f"P95 response time ({performance['p95_response_time']:.1f}ms) exceeds target ({self.targets['end_to_end_p95']}ms)")
        
        # Check success rate
        if performance.get('success_rate', 100) < self.targets['success_rate']:
            health_score -= 40
            issues.append(f"Success rate ({performance['success_rate']:.1f}%) below target ({self.targets['success_rate']}%)")
        
        # Determine status
        if health_score >= 90:
            status = "Excellent"
            color = "green"
        elif health_score >= 70:
            status = "Good"
            color = "blue"
        elif health_score >= 50:
            status = "Warning"
            color = "orange"
        else:
            status = "Critical"
            color = "red"
        
        return {
            'health_score': health_score,
            'status': status,
            'color': color,
            'issues': issues,
            'uptime_hours': (time.time() - self.system_metrics['startup_time']) / 3600,
            'last_query_time': self.system_metrics['last_query_time']
        }
    
    def export_metrics(self) -> pd.DataFrame:
        """
        Export metrics data as a pandas DataFrame.
        
        Returns:
            DataFrame with query logs
        """
        if not self.query_logs:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.query_logs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def get_target_compliance(self) -> Dict[str, Any]:
        """
        Check compliance with performance targets.
        
        Returns:
            Target compliance status
        """
        performance = self.get_performance_summary()
        
        compliance = {}
        
        # Response time compliance
        p95_time = performance.get('p95_response_time', float('inf'))
        compliance['response_time'] = {
            'target': self.targets['end_to_end_p95'],
            'actual': p95_time,
            'compliant': p95_time <= self.targets['end_to_end_p95'],
            'variance': p95_time - self.targets['end_to_end_p95']
        }
        
        # Success rate compliance
        success_rate = performance.get('success_rate', 0)
        compliance['success_rate'] = {
            'target': self.targets['success_rate'],
            'actual': success_rate,
            'compliant': success_rate >= self.targets['success_rate'],
            'variance': success_rate - self.targets['success_rate']
        }
        
        # Overall compliance
        overall_compliant = all(metric['compliant'] for metric in compliance.values())
        
        return {
            'overall_compliant': overall_compliant,
            'metrics': compliance,
            'targets': self.targets
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics data."""
        self.query_logs = []
        self.system_metrics = {
            'startup_time': time.time(),
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'avg_response_time': 0.0,
            'last_query_time': None
        }
