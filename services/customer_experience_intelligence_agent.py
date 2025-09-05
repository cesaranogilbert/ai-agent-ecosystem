"""
Customer Experience Intelligence Agent
Customer journey optimization and experience analytics
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class CustomerJourney:
    touchpoint: str
    satisfaction_score: float
    effort_score: float
    conversion_rate: float
    improvement_opportunity: str

class CustomerExperienceIntelligenceAgent:
    """
    Comprehensive Customer Experience Intelligence System
    - Customer journey mapping and optimization
    - Sentiment analysis across touchpoints
    - Churn prediction and prevention
    - Personalized experience design
    """
    
    def __init__(self):
        self.name = "Customer Experience Intelligence Agent"
        self.version = "1.0.0"
        self.capabilities = [
            "Customer Journey Mapping",
            "Experience Analytics",
            "Churn Prediction",
            "Sentiment Analysis",
            "Personalization Engine",
            "Customer Satisfaction Optimization"
        ]
        
    def analyze_customer_experience(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive customer experience analysis"""
        try:
            company_name = customer_data.get('company_name', 'Unknown Company')
            
            # Analyze customer journey
            journey_analysis = self._analyze_customer_journey(customer_data)
            
            # Sentiment analysis
            sentiment_analysis = self._analyze_customer_sentiment(customer_data)
            
            # Churn prediction
            churn_analysis = self._predict_customer_churn(customer_data)
            
            # Experience optimization
            optimization_recommendations = self._generate_experience_optimization(
                journey_analysis, sentiment_analysis, churn_analysis
            )
            
            return {
                'company': company_name,
                'analysis_date': datetime.now().isoformat(),
                'journey_analysis': journey_analysis,
                'sentiment_analysis': sentiment_analysis,
                'churn_analysis': churn_analysis,
                'optimization_recommendations': optimization_recommendations,
                'next_review_date': (datetime.now() + timedelta(days=30)).isoformat()
            }
            
        except Exception as e:
            logging.error(f"Customer experience analysis failed: {str(e)}")
            return {'error': f'Customer experience analysis failed: {str(e)}'}
            
    def _analyze_customer_journey(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze customer journey across touchpoints"""
        touchpoints = customer_data.get('touchpoints', [])
        
        journey_insights = []
        for touchpoint in touchpoints:
            insight = {
                'touchpoint': touchpoint.get('name', 'Unknown'),
                'satisfaction_score': touchpoint.get('satisfaction', 75),
                'effort_score': touchpoint.get('effort', 60),
                'conversion_rate': touchpoint.get('conversion_rate', 15),
                'improvement_opportunity': self._identify_improvement_opportunity(touchpoint)
            }
            journey_insights.append(insight)
            
        overall_satisfaction = sum(t['satisfaction_score'] for t in journey_insights) / len(journey_insights) if journey_insights else 0
        
        return {
            'touchpoint_analysis': journey_insights,
            'overall_satisfaction': overall_satisfaction,
            'critical_touchpoints': [t for t in journey_insights if t['satisfaction_score'] < 60],
            'high_effort_touchpoints': [t for t in journey_insights if t['effort_score'] > 70]
        }
        
    def _identify_improvement_opportunity(self, touchpoint: Dict[str, Any]) -> str:
        """Identify improvement opportunity for touchpoint"""
        satisfaction = touchpoint.get('satisfaction', 75)
        effort = touchpoint.get('effort', 60)
        
        if satisfaction < 60 and effort > 70:
            return 'High Priority - Low satisfaction and high effort'
        elif satisfaction < 60:
            return 'Medium Priority - Satisfaction improvement needed'
        elif effort > 70:
            return 'Medium Priority - Effort reduction needed'
        else:
            return 'Low Priority - Optimize for excellence'
            
    def _analyze_customer_sentiment(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze customer sentiment across channels"""
        feedback_data = customer_data.get('customer_feedback', [])
        
        sentiment_scores = [feedback.get('sentiment_score', 70) for feedback in feedback_data]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 70
        
        sentiment_distribution = {
            'positive': len([s for s in sentiment_scores if s >= 70]),
            'neutral': len([s for s in sentiment_scores if 50 <= s < 70]),
            'negative': len([s for s in sentiment_scores if s < 50])
        }
        
        return {
            'average_sentiment': avg_sentiment,
            'sentiment_distribution': sentiment_distribution,
            'sentiment_trend': 'Improving' if avg_sentiment > 70 else 'Declining',
            'key_themes': ['Product quality', 'Customer service', 'Pricing', 'User experience']
        }
        
    def _predict_customer_churn(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict customer churn risk"""
        customers = customer_data.get('customers', [])
        
        high_risk_customers = []
        for customer in customers:
            risk_score = self._calculate_churn_risk(customer)
            if risk_score >= 70:
                high_risk_customers.append({
                    'customer_id': customer.get('id', 'unknown'),
                    'risk_score': risk_score,
                    'risk_factors': self._identify_churn_risk_factors(customer)
                })
                
        churn_rate = len(high_risk_customers) / len(customers) * 100 if customers else 0
        
        return {
            'predicted_churn_rate': churn_rate,
            'high_risk_customers': high_risk_customers,
            'retention_strategies': self._suggest_retention_strategies(high_risk_customers)
        }
        
    def _calculate_churn_risk(self, customer: Dict[str, Any]) -> float:
        """Calculate churn risk score for customer"""
        factors = {
            'satisfaction': 100 - customer.get('satisfaction_score', 75),
            'usage_decline': customer.get('usage_decline_percentage', 20),
            'support_tickets': min(50, customer.get('support_tickets', 2) * 10),
            'payment_issues': customer.get('payment_issues', 0) * 20
        }
        
        return sum(factors.values()) / len(factors)
        
    def _identify_churn_risk_factors(self, customer: Dict[str, Any]) -> List[str]:
        """Identify specific churn risk factors"""
        factors = []
        
        if customer.get('satisfaction_score', 75) < 60:
            factors.append('Low satisfaction')
        if customer.get('usage_decline_percentage', 0) > 30:
            factors.append('Significant usage decline')
        if customer.get('support_tickets', 0) > 5:
            factors.append('High support ticket volume')
        if customer.get('payment_issues', 0) > 0:
            factors.append('Payment difficulties')
            
        return factors
        
    def _suggest_retention_strategies(self, high_risk_customers: List[Dict[str, Any]]) -> List[str]:
        """Suggest retention strategies"""
        return [
            'Proactive customer success outreach',
            'Personalized retention offers',
            'Enhanced support for high-risk customers',
            'Product training and onboarding',
            'Loyalty program enrollment'
        ]
        
    def _generate_experience_optimization(self, journey: Dict[str, Any], 
                                        sentiment: Dict[str, Any], 
                                        churn: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate experience optimization recommendations"""
        recommendations = []
        
        # Journey optimization
        if journey['overall_satisfaction'] < 75:
            recommendations.append({
                'category': 'Journey Optimization',
                'priority': 'High',
                'recommendation': 'Improve critical customer touchpoints',
                'expected_impact': 'Increase satisfaction by 15-25%'
            })
            
        # Sentiment improvement
        if sentiment['average_sentiment'] < 70:
            recommendations.append({
                'category': 'Sentiment Enhancement',
                'priority': 'Medium',
                'recommendation': 'Address negative sentiment drivers',
                'expected_impact': 'Improve sentiment scores by 20%'
            })
            
        # Churn reduction
        if churn['predicted_churn_rate'] > 15:
            recommendations.append({
                'category': 'Churn Prevention',
                'priority': 'Critical',
                'recommendation': 'Implement proactive retention program',
                'expected_impact': 'Reduce churn by 30-40%'
            })
            
        return recommendations

def test_customer_experience_intelligence_agent():
    """Test the Customer Experience Intelligence Agent"""
    print("🧪 Testing Customer Experience Intelligence Agent")
    print("=" * 55)
    
    try:
        agent = CustomerExperienceIntelligenceAgent()
        
        test_data = {
            'company_name': 'Customer-Centric Corp',
            'touchpoints': [
                {'name': 'Website', 'satisfaction': 85, 'effort': 40, 'conversion_rate': 12},
                {'name': 'Call Center', 'satisfaction': 55, 'effort': 80, 'conversion_rate': 25}
            ],
            'customer_feedback': [
                {'sentiment_score': 80}, {'sentiment_score': 60}
            ],
            'customers': [
                {'id': '001', 'satisfaction_score': 45, 'usage_decline_percentage': 40},
                {'id': '002', 'satisfaction_score': 85, 'usage_decline_percentage': 5}
            ]
        }
        
        analysis = agent.analyze_customer_experience(test_data)
        print(f"✅ Analysis completed: {analysis['journey_analysis']['overall_satisfaction']:.1f}% satisfaction")
        
        return {'agent_initialized': True, 'analysis_completed': True}
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return {'error': str(e)}

if __name__ == "__main__":
    test_customer_experience_intelligence_agent()