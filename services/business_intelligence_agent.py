"""
Business Intelligence & Reporting Agent
Automated insights generation and executive dashboards
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

class BusinessIntelligenceAgent:
    """
    Comprehensive Business Intelligence System
    - Automated insights generation
    - Executive dashboard creation
    - KPI monitoring and alerting
    - Predictive analytics
    """
    
    def __init__(self):
        self.name = "Business Intelligence & Reporting Agent"
        self.version = "1.0.0"
        self.capabilities = [
            "KPI Analysis",
            "Automated Insights",
            "Executive Reporting",
            "Trend Analysis",
            "Performance Monitoring",
            "Predictive Analytics"
        ]
        
    def generate_business_intelligence(self, business_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive business intelligence"""
        try:
            company_name = business_data.get('company_name', 'Unknown Company')
            
            # KPI analysis
            kpi_analysis = self._analyze_kpis(business_data)
            
            # Trend analysis
            trend_analysis = self._analyze_trends(business_data)
            
            # Automated insights
            automated_insights = self._generate_automated_insights(kpi_analysis, trend_analysis)
            
            # Executive summary
            executive_summary = self._create_executive_summary(kpi_analysis, trend_analysis, automated_insights)
            
            # Performance alerts
            alerts = self._generate_performance_alerts(kpi_analysis)
            
            return {
                'company': company_name,
                'report_date': datetime.now().isoformat(),
                'kpi_analysis': kpi_analysis,
                'trend_analysis': trend_analysis,
                'automated_insights': automated_insights,
                'executive_summary': executive_summary,
                'performance_alerts': alerts,
                'next_report_date': (datetime.now() + timedelta(days=7)).isoformat()
            }
            
        except Exception as e:
            logging.error(f"Business intelligence generation failed: {str(e)}")
            return {'error': f'Business intelligence generation failed: {str(e)}'}
            
    def _analyze_kpis(self, business_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze key performance indicators"""
        
        # Financial KPIs
        financial_kpis = {
            'revenue': business_data.get('revenue', 0),
            'profit_margin': business_data.get('profit_margin', 0),
            'cash_flow': business_data.get('cash_flow', 0),
            'growth_rate': business_data.get('growth_rate', 0)
        }
        
        # Operational KPIs
        operational_kpis = {
            'customer_satisfaction': business_data.get('customer_satisfaction', 0),
            'employee_satisfaction': business_data.get('employee_satisfaction', 0),
            'operational_efficiency': business_data.get('operational_efficiency', 0),
            'quality_score': business_data.get('quality_score', 0)
        }
        
        # Market KPIs
        market_kpis = {
            'market_share': business_data.get('market_share', 0),
            'customer_acquisition_cost': business_data.get('customer_acquisition_cost', 0),
            'customer_lifetime_value': business_data.get('customer_lifetime_value', 0),
            'churn_rate': business_data.get('churn_rate', 0)
        }
        
        # Calculate KPI health scores
        financial_health = self._calculate_kpi_health(financial_kpis, 'financial')
        operational_health = self._calculate_kpi_health(operational_kpis, 'operational')
        market_health = self._calculate_kpi_health(market_kpis, 'market')
        
        overall_health = (financial_health + operational_health + market_health) / 3
        
        return {
            'financial_kpis': financial_kpis,
            'operational_kpis': operational_kpis,
            'market_kpis': market_kpis,
            'financial_health': financial_health,
            'operational_health': operational_health,
            'market_health': market_health,
            'overall_health': overall_health,
            'health_level': self._categorize_health_level(overall_health)
        }
        
    def _calculate_kpi_health(self, kpis: Dict[str, float], category: str) -> float:
        """Calculate health score for KPI category"""
        
        # Define benchmarks for different categories
        benchmarks = {
            'financial': {
                'revenue': 1000000,      # $1M benchmark
                'profit_margin': 15,     # 15% benchmark
                'cash_flow': 100000,     # $100K benchmark
                'growth_rate': 10        # 10% benchmark
            },
            'operational': {
                'customer_satisfaction': 80,    # 80% benchmark
                'employee_satisfaction': 75,    # 75% benchmark
                'operational_efficiency': 85,   # 85% benchmark
                'quality_score': 90             # 90% benchmark
            },
            'market': {
                'market_share': 10,              # 10% benchmark
                'customer_acquisition_cost': 100, # $100 benchmark (lower is better)
                'customer_lifetime_value': 1000,  # $1000 benchmark
                'churn_rate': 5                   # 5% benchmark (lower is better)
            }
        }
        
        if category not in benchmarks:
            return 0
            
        health_scores = []
        category_benchmarks = benchmarks[category]
        
        for kpi, value in kpis.items():
            if kpi in category_benchmarks:
                benchmark = category_benchmarks[kpi]
                
                # For metrics where lower is better
                if kpi in ['customer_acquisition_cost', 'churn_rate']:
                    health_score = max(0, min(100, (benchmark / max(1, value)) * 100))
                else:
                    health_score = min(100, (value / benchmark) * 100)
                    
                health_scores.append(health_score)
                
        return sum(health_scores) / len(health_scores) if health_scores else 0
        
    def _categorize_health_level(self, health_score: float) -> str:
        """Categorize health level"""
        if health_score >= 90:
            return 'Excellent'
        elif health_score >= 75:
            return 'Good'
        elif health_score >= 60:
            return 'Fair'
        elif health_score >= 45:
            return 'Poor'
        else:
            return 'Critical'
            
    def _analyze_trends(self, business_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze business trends"""
        
        # Historical data analysis
        historical_data = business_data.get('historical_data', {})
        
        trends = {}
        
        # Revenue trend
        revenue_history = historical_data.get('revenue', [])
        revenue_trend = self._calculate_trend(revenue_history)
        trends['revenue_trend'] = revenue_trend
        
        # Customer trend
        customer_history = historical_data.get('customers', [])
        customer_trend = self._calculate_trend(customer_history)
        trends['customer_trend'] = customer_trend
        
        # Profit margin trend
        margin_history = historical_data.get('profit_margin', [])
        margin_trend = self._calculate_trend(margin_history)
        trends['margin_trend'] = margin_trend
        
        # Identify trend patterns
        trend_patterns = self._identify_trend_patterns(trends)
        
        return {
            'trends': trends,
            'trend_patterns': trend_patterns,
            'trend_summary': self._summarize_trends(trends)
        }
        
    def _calculate_trend(self, data_points: List[float]) -> Dict[str, Any]:
        """Calculate trend for data points"""
        if len(data_points) < 2:
            return {'direction': 'Insufficient Data', 'rate': 0, 'confidence': 0}
            
        # Simple linear trend calculation
        n = len(data_points)
        x_sum = sum(range(n))
        y_sum = sum(data_points)
        xy_sum = sum(i * data_points[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        # Calculate slope (trend rate)
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        
        # Determine trend direction
        if slope > 0.05:
            direction = 'Increasing'
        elif slope < -0.05:
            direction = 'Decreasing'
        else:
            direction = 'Stable'
            
        # Calculate trend strength (R-squared approximation)
        y_mean = y_sum / n
        total_variance = sum((y - y_mean) ** 2 for y in data_points)
        explained_variance = sum((slope * i - y_mean) ** 2 for i in range(n))
        confidence = explained_variance / total_variance if total_variance > 0 else 0
        
        return {
            'direction': direction,
            'rate': slope,
            'confidence': min(100, confidence * 100)
        }
        
    def _identify_trend_patterns(self, trends: Dict[str, Any]) -> List[str]:
        """Identify patterns across trends"""
        patterns = []
        
        # All trends increasing
        increasing_trends = [name for name, trend in trends.items() if trend.get('direction') == 'Increasing']
        if len(increasing_trends) >= 2:
            patterns.append(f'Strong growth pattern: {", ".join(increasing_trends)} all increasing')
            
        # Mixed trends
        decreasing_trends = [name for name, trend in trends.items() if trend.get('direction') == 'Decreasing']
        if increasing_trends and decreasing_trends:
            patterns.append('Mixed performance: Some metrics improving while others declining')
            
        # High confidence trends
        high_confidence = [name for name, trend in trends.items() if trend.get('confidence', 0) > 70]
        if high_confidence:
            patterns.append(f'Reliable trends in: {", ".join(high_confidence)}')
            
        return patterns
        
    def _summarize_trends(self, trends: Dict[str, Any]) -> str:
        """Summarize overall trend direction"""
        directions = [trend.get('direction', 'Stable') for trend in trends.values()]
        
        increasing_count = directions.count('Increasing')
        decreasing_count = directions.count('Decreasing')
        stable_count = directions.count('Stable')
        
        if increasing_count > decreasing_count:
            return 'Overall Positive Trend'
        elif decreasing_count > increasing_count:
            return 'Overall Negative Trend'
        else:
            return 'Mixed or Stable Trends'
            
    def _generate_automated_insights(self, kpi_analysis: Dict[str, Any], trend_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate automated business insights"""
        
        insights = []
        
        # Health-based insights
        overall_health = kpi_analysis['overall_health']
        if overall_health >= 90:
            insights.append({
                'type': 'Performance',
                'insight': 'Business performance is excellent across all key areas',
                'impact': 'Positive',
                'confidence': 'High',
                'recommendation': 'Maintain current strategies and look for growth opportunities'
            })
        elif overall_health < 60:
            insights.append({
                'type': 'Performance',
                'insight': 'Business performance requires immediate attention',
                'impact': 'Negative',
                'confidence': 'High',
                'recommendation': 'Focus on underperforming areas and implement corrective actions'
            })
            
        # Trend-based insights
        trend_summary = trend_analysis['trend_summary']
        if trend_summary == 'Overall Positive Trend':
            insights.append({
                'type': 'Growth',
                'insight': 'Multiple business metrics show positive growth trajectory',
                'impact': 'Positive',
                'confidence': 'Medium',
                'recommendation': 'Invest in growth initiatives to accelerate positive trends'
            })
        elif trend_summary == 'Overall Negative Trend':
            insights.append({
                'type': 'Risk',
                'insight': 'Concerning decline across multiple business metrics',
                'impact': 'Negative',
                'confidence': 'Medium',
                'recommendation': 'Conduct thorough analysis and implement turnaround strategies'
            })
            
        # Specific KPI insights
        financial_health = kpi_analysis['financial_health']
        operational_health = kpi_analysis['operational_health']
        
        if financial_health > operational_health + 20:
            insights.append({
                'type': 'Operations',
                'insight': 'Financial performance outpacing operational efficiency',
                'impact': 'Opportunity',
                'confidence': 'Medium',
                'recommendation': 'Invest in operational improvements for sustainable growth'
            })
            
        return insights
        
    def _create_executive_summary(self, kpi_analysis: Dict[str, Any], 
                                trend_analysis: Dict[str, Any], 
                                insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create executive summary"""
        
        # Key highlights
        highlights = []
        
        # Performance summary
        health_level = kpi_analysis['health_level']
        highlights.append(f"Overall business health: {health_level} ({kpi_analysis['overall_health']:.1f}/100)")
        
        # Trend summary
        trend_summary = trend_analysis['trend_summary']
        highlights.append(f"Business trends: {trend_summary}")
        
        # Key insights count
        positive_insights = len([i for i in insights if i['impact'] == 'Positive'])
        negative_insights = len([i for i in insights if i['impact'] == 'Negative'])
        highlights.append(f"Insights: {positive_insights} positive, {negative_insights} areas of concern")
        
        # Priority actions
        priority_actions = []
        
        if kpi_analysis['overall_health'] < 70:
            priority_actions.append('Address underperforming KPIs')
            
        if any(insight['impact'] == 'Negative' and insight['confidence'] == 'High' for insight in insights):
            priority_actions.append('Immediate attention to high-confidence negative insights')
            
        if trend_analysis['trend_summary'] == 'Overall Negative Trend':
            priority_actions.append('Implement trend reversal strategies')
            
        return {
            'report_period': 'Current Period',
            'key_highlights': highlights,
            'priority_actions': priority_actions,
            'overall_status': self._determine_overall_status(kpi_analysis, trend_analysis, insights),
            'executive_recommendations': self._generate_executive_recommendations(kpi_analysis, insights)
        }
        
    def _determine_overall_status(self, kpi_analysis: Dict[str, Any], 
                                trend_analysis: Dict[str, Any], 
                                insights: List[Dict[str, Any]]) -> str:
        """Determine overall business status"""
        
        health_score = kpi_analysis['overall_health']
        trend_summary = trend_analysis['trend_summary']
        negative_insights = len([i for i in insights if i['impact'] == 'Negative'])
        
        if health_score >= 80 and trend_summary == 'Overall Positive Trend':
            return 'Strong Performance'
        elif health_score >= 70 and negative_insights <= 1:
            return 'Good Performance'
        elif health_score >= 60:
            return 'Fair Performance'
        else:
            return 'Performance Issues'
            
    def _generate_executive_recommendations(self, kpi_analysis: Dict[str, Any], insights: List[Dict[str, Any]]) -> List[str]:
        """Generate executive-level recommendations"""
        
        recommendations = []
        
        # Health-based recommendations
        if kpi_analysis['financial_health'] < 70:
            recommendations.append('Focus on financial performance improvement initiatives')
            
        if kpi_analysis['operational_health'] < 70:
            recommendations.append('Invest in operational efficiency and quality improvements')
            
        if kpi_analysis['market_health'] < 70:
            recommendations.append('Strengthen market position and customer relationships')
            
        # Insight-based recommendations
        high_impact_insights = [i for i in insights if i['confidence'] == 'High']
        for insight in high_impact_insights:
            recommendations.append(insight['recommendation'])
            
        return recommendations[:5]  # Top 5 recommendations
        
    def _generate_performance_alerts(self, kpi_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate performance alerts"""
        
        alerts = []
        
        # Critical health alerts
        if kpi_analysis['overall_health'] < 50:
            alerts.append({
                'type': 'Critical',
                'message': 'Overall business health is critically low',
                'metric': 'Overall Health',
                'value': kpi_analysis['overall_health'],
                'threshold': 50,
                'action_required': 'Immediate executive attention required'
            })
            
        # Individual KPI alerts
        if kpi_analysis['financial_health'] < 60:
            alerts.append({
                'type': 'Warning',
                'message': 'Financial performance below acceptable levels',
                'metric': 'Financial Health',
                'value': kpi_analysis['financial_health'],
                'threshold': 60,
                'action_required': 'Review financial strategies and performance'
            })
            
        if kpi_analysis['operational_health'] < 60:
            alerts.append({
                'type': 'Warning',
                'message': 'Operational efficiency needs improvement',
                'metric': 'Operational Health',
                'value': kpi_analysis['operational_health'],
                'threshold': 60,
                'action_required': 'Focus on operational optimization'
            })
            
        return alerts

def test_business_intelligence_agent():
    """Test the Business Intelligence Agent"""
    print("🧪 Testing Business Intelligence & Reporting Agent")
    print("=" * 55)
    
    try:
        agent = BusinessIntelligenceAgent()
        
        test_data = {
            'company_name': 'DataDriven Enterprises',
            'revenue': 2500000,
            'profit_margin': 18,
            'cash_flow': 300000,
            'growth_rate': 15,
            'customer_satisfaction': 85,
            'employee_satisfaction': 78,
            'operational_efficiency': 82,
            'market_share': 12,
            'historical_data': {
                'revenue': [2000000, 2200000, 2400000, 2500000],
                'customers': [1000, 1100, 1250, 1300],
                'profit_margin': [15, 16, 17, 18]
            }
        }
        
        intelligence = agent.generate_business_intelligence(test_data)
        print(f"✅ Business intelligence generated for {test_data['company_name']}")
        print(f"   Overall health: {intelligence['kpi_analysis']['health_level']}")
        print(f"   Trend summary: {intelligence['trend_analysis']['trend_summary']}")
        print(f"   Insights generated: {len(intelligence['automated_insights'])}")
        print(f"   Performance alerts: {len(intelligence['performance_alerts'])}")
        
        return {
            'agent_initialized': True,
            'health_score': intelligence['kpi_analysis']['overall_health'],
            'insights_count': len(intelligence['automated_insights'])
        }
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return {'error': str(e)}

if __name__ == "__main__":
    test_business_intelligence_agent()