"""
Financial Risk Management Agent
Financial risk assessment and mitigation strategies
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

class FinancialRiskManagementAgent:
    """
    Comprehensive Financial Risk Management System
    - Credit risk assessment
    - Market risk analysis
    - Liquidity risk monitoring
    - Operational risk evaluation
    """
    
    def __init__(self):
        self.name = "Financial Risk Management Agent"
        self.version = "1.0.0"
        self.capabilities = [
            "Credit Risk Assessment",
            "Market Risk Analysis",
            "Liquidity Risk Monitoring",
            "Operational Risk Evaluation",
            "Stress Testing",
            "Risk Mitigation Planning"
        ]
        
    def assess_financial_risks(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive financial risk assessment"""
        try:
            company_name = financial_data.get('company_name', 'Unknown Company')
            
            # Credit risk assessment
            credit_risk = self._assess_credit_risk(financial_data)
            
            # Market risk analysis
            market_risk = self._analyze_market_risk(financial_data)
            
            # Liquidity risk evaluation
            liquidity_risk = self._evaluate_liquidity_risk(financial_data)
            
            # Operational risk assessment
            operational_risk = self._assess_operational_risk(financial_data)
            
            # Overall risk profile
            overall_risk = self._calculate_overall_risk_profile(credit_risk, market_risk, liquidity_risk, operational_risk)
            
            # Generate mitigation strategies
            mitigation_strategies = self._generate_risk_mitigation_strategies(
                credit_risk, market_risk, liquidity_risk, operational_risk, overall_risk
            )
            
            return {
                'company': company_name,
                'assessment_date': datetime.now().isoformat(),
                'credit_risk': credit_risk,
                'market_risk': market_risk,
                'liquidity_risk': liquidity_risk,
                'operational_risk': operational_risk,
                'overall_risk_profile': overall_risk,
                'mitigation_strategies': mitigation_strategies,
                'next_assessment_date': (datetime.now() + timedelta(days=30)).isoformat()
            }
            
        except Exception as e:
            logging.error(f"Financial risk assessment failed: {str(e)}")
            return {'error': f'Financial risk assessment failed: {str(e)}'}
            
    def _assess_credit_risk(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess credit risk exposure"""
        
        # Customer concentration analysis
        customer_data = financial_data.get('customer_portfolio', {})
        top_customers_revenue = customer_data.get('top_10_customers_percentage', 30)
        
        # Receivables analysis
        receivables_data = financial_data.get('receivables', {})
        days_sales_outstanding = receivables_data.get('dso', 45)
        bad_debt_rate = receivables_data.get('bad_debt_percentage', 2.0)
        
        # Credit quality metrics
        credit_metrics = {
            'customer_concentration_risk': min(100, top_customers_revenue * 2),  # Higher concentration = higher risk
            'collection_efficiency': max(0, 100 - (days_sales_outstanding - 30) * 2),  # 30 days is benchmark
            'credit_quality': max(0, 100 - bad_debt_rate * 20)  # Lower bad debt = better quality
        }
        
        # Calculate credit risk score
        credit_risk_score = sum(credit_metrics.values()) / len(credit_metrics)
        
        # Identify high-risk customers
        high_risk_customers = customer_data.get('high_risk_customers', [])
        
        return {
            'credit_risk_score': 100 - credit_risk_score,  # Invert to make higher score = higher risk
            'risk_level': self._categorize_risk_level(100 - credit_risk_score),
            'customer_concentration_risk': top_customers_revenue,
            'days_sales_outstanding': days_sales_outstanding,
            'bad_debt_rate': bad_debt_rate,
            'high_risk_customers_count': len(high_risk_customers),
            'credit_risk_factors': self._identify_credit_risk_factors(financial_data)
        }
        
    def _identify_credit_risk_factors(self, financial_data: Dict[str, Any]) -> List[str]:
        """Identify specific credit risk factors"""
        
        factors = []
        customer_data = financial_data.get('customer_portfolio', {})
        receivables_data = financial_data.get('receivables', {})
        
        if customer_data.get('top_10_customers_percentage', 30) > 50:
            factors.append('High customer concentration risk')
            
        if receivables_data.get('dso', 45) > 60:
            factors.append('Slow collections and extended payment terms')
            
        if receivables_data.get('bad_debt_percentage', 2.0) > 3.0:
            factors.append('High bad debt losses')
            
        if customer_data.get('new_customers_percentage', 20) > 40:
            factors.append('High percentage of unproven customers')
            
        return factors
        
    def _analyze_market_risk(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market risk exposure"""
        
        market_data = financial_data.get('market_exposure', {})
        
        # Interest rate risk
        interest_rate_exposure = market_data.get('variable_rate_debt_percentage', 40)
        
        # Foreign exchange risk
        fx_exposure = market_data.get('foreign_revenue_percentage', 20)
        
        # Commodity price risk
        commodity_exposure = market_data.get('commodity_cost_percentage', 15)
        
        # Market concentration risk
        geographic_concentration = market_data.get('primary_market_percentage', 60)
        
        # Calculate market risk components
        risk_components = {
            'interest_rate_risk': min(100, interest_rate_exposure * 1.5),
            'foreign_exchange_risk': min(100, fx_exposure * 2),
            'commodity_price_risk': min(100, commodity_exposure * 3),
            'geographic_concentration_risk': min(100, geographic_concentration * 1.2)
        }
        
        # Overall market risk score
        market_risk_score = sum(risk_components.values()) / len(risk_components)
        
        return {
            'market_risk_score': market_risk_score,
            'risk_level': self._categorize_risk_level(market_risk_score),
            'interest_rate_exposure': interest_rate_exposure,
            'foreign_exchange_exposure': fx_exposure,
            'commodity_exposure': commodity_exposure,
            'geographic_concentration': geographic_concentration,
            'risk_components': risk_components,
            'hedging_recommendations': self._generate_hedging_recommendations(risk_components)
        }
        
    def _generate_hedging_recommendations(self, risk_components: Dict[str, float]) -> List[str]:
        """Generate hedging recommendations based on risk exposure"""
        
        recommendations = []
        
        if risk_components['interest_rate_risk'] > 30:
            recommendations.append('Consider interest rate swaps or caps to hedge variable rate debt')
            
        if risk_components['foreign_exchange_risk'] > 40:
            recommendations.append('Implement FX forward contracts or options for major currency exposures')
            
        if risk_components['commodity_price_risk'] > 45:
            recommendations.append('Use commodity futures or swaps to hedge price volatility')
            
        if risk_components['geographic_concentration_risk'] > 70:
            recommendations.append('Diversify into new geographic markets to reduce concentration')
            
        return recommendations
        
    def _evaluate_liquidity_risk(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate liquidity risk position"""
        
        liquidity_data = financial_data.get('liquidity_metrics', {})
        
        # Liquidity ratios
        current_ratio = liquidity_data.get('current_ratio', 1.5)
        quick_ratio = liquidity_data.get('quick_ratio', 1.0)
        cash_ratio = liquidity_data.get('cash_ratio', 0.3)
        
        # Cash flow metrics
        operating_cash_flow = liquidity_data.get('operating_cash_flow', 1000000)
        free_cash_flow = liquidity_data.get('free_cash_flow', 500000)
        
        # Credit facilities
        available_credit = liquidity_data.get('available_credit_lines', 2000000)
        credit_utilization = liquidity_data.get('credit_utilization_percentage', 30)
        
        # Calculate liquidity scores
        liquidity_scores = {
            'current_liquidity': min(100, current_ratio * 50),  # 2.0 ratio = 100 score
            'immediate_liquidity': min(100, quick_ratio * 75),  # 1.33 ratio = 100 score
            'cash_liquidity': min(100, cash_ratio * 200),      # 0.5 ratio = 100 score
            'operating_liquidity': 80 if operating_cash_flow > 0 else 20,
            'credit_availability': max(0, 100 - credit_utilization * 1.5)
        }
        
        overall_liquidity_score = sum(liquidity_scores.values()) / len(liquidity_scores)
        
        # Days cash on hand
        daily_expenses = financial_data.get('daily_operating_expenses', 10000)
        cash_on_hand = liquidity_data.get('cash_and_equivalents', 500000)
        days_cash = cash_on_hand / daily_expenses if daily_expenses > 0 else 0
        
        return {
            'liquidity_score': overall_liquidity_score,
            'risk_level': self._categorize_risk_level(100 - overall_liquidity_score),  # Invert for risk
            'current_ratio': current_ratio,
            'quick_ratio': quick_ratio,
            'cash_ratio': cash_ratio,
            'days_cash_on_hand': days_cash,
            'credit_utilization': credit_utilization,
            'liquidity_adequacy': self._assess_liquidity_adequacy(overall_liquidity_score, days_cash),
            'liquidity_recommendations': self._generate_liquidity_recommendations(liquidity_scores, days_cash)
        }
        
    def _assess_liquidity_adequacy(self, liquidity_score: float, days_cash: float) -> str:
        """Assess liquidity adequacy"""
        
        if liquidity_score >= 80 and days_cash >= 60:
            return 'Excellent liquidity position'
        elif liquidity_score >= 65 and days_cash >= 30:
            return 'Adequate liquidity position'
        elif liquidity_score >= 50 and days_cash >= 15:
            return 'Marginal liquidity position'
        else:
            return 'Insufficient liquidity - immediate attention required'
            
    def _generate_liquidity_recommendations(self, scores: Dict[str, float], days_cash: float) -> List[str]:
        """Generate liquidity improvement recommendations"""
        
        recommendations = []
        
        if scores['current_liquidity'] < 60:
            recommendations.append('Improve current ratio through working capital management')
            
        if scores['cash_liquidity'] < 40:
            recommendations.append('Increase cash reserves through cash management optimization')
            
        if days_cash < 30:
            recommendations.append('Build cash reserves to maintain at least 30 days of operating expenses')
            
        if scores['credit_availability'] < 50:
            recommendations.append('Negotiate additional credit facilities or reduce credit utilization')
            
        return recommendations
        
    def _assess_operational_risk(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess operational risk factors"""
        
        operational_data = financial_data.get('operational_metrics', {})
        
        # Key person dependency
        key_person_risk = operational_data.get('key_person_dependency_score', 50)
        
        # System and process risk
        system_reliability = operational_data.get('system_reliability_score', 80)
        process_automation = operational_data.get('process_automation_percentage', 60)
        
        # Compliance and regulatory risk
        compliance_score = operational_data.get('compliance_score', 85)
        regulatory_changes = operational_data.get('pending_regulatory_changes', 2)
        
        # Cybersecurity risk
        cybersecurity_score = operational_data.get('cybersecurity_score', 75)
        
        # Calculate operational risk components
        risk_components = {
            'key_person_risk': key_person_risk,
            'system_process_risk': 100 - ((system_reliability + process_automation) / 2),
            'compliance_risk': 100 - compliance_score + (regulatory_changes * 10),
            'cybersecurity_risk': 100 - cybersecurity_score
        }
        
        operational_risk_score = sum(risk_components.values()) / len(risk_components)
        
        return {
            'operational_risk_score': operational_risk_score,
            'risk_level': self._categorize_risk_level(operational_risk_score),
            'key_person_dependency': key_person_risk,
            'system_reliability': system_reliability,
            'compliance_score': compliance_score,
            'cybersecurity_score': cybersecurity_score,
            'risk_components': risk_components,
            'operational_risk_factors': self._identify_operational_risk_factors(operational_data)
        }
        
    def _identify_operational_risk_factors(self, operational_data: Dict[str, Any]) -> List[str]:
        """Identify specific operational risk factors"""
        
        factors = []
        
        if operational_data.get('key_person_dependency_score', 50) > 70:
            factors.append('High dependency on key personnel')
            
        if operational_data.get('system_reliability_score', 80) < 70:
            factors.append('System reliability concerns')
            
        if operational_data.get('process_automation_percentage', 60) < 40:
            factors.append('Low process automation increases manual error risk')
            
        if operational_data.get('compliance_score', 85) < 75:
            factors.append('Compliance gaps and regulatory risk')
            
        if operational_data.get('cybersecurity_score', 75) < 70:
            factors.append('Cybersecurity vulnerabilities')
            
        return factors
        
    def _categorize_risk_level(self, risk_score: float) -> str:
        """Categorize risk level"""
        
        if risk_score >= 70:
            return 'High Risk'
        elif risk_score >= 50:
            return 'Medium Risk'
        elif risk_score >= 30:
            return 'Low-Medium Risk'
        else:
            return 'Low Risk'
            
    def _calculate_overall_risk_profile(self, credit: Dict[str, Any], market: Dict[str, Any], 
                                      liquidity: Dict[str, Any], operational: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall risk profile"""
        
        # Weight different risk types
        risk_weights = {
            'credit': 0.25,
            'market': 0.20,
            'liquidity': 0.30,  # Higher weight for liquidity risk
            'operational': 0.25
        }
        
        # Calculate weighted risk score
        overall_risk_score = (
            credit['credit_risk_score'] * risk_weights['credit'] +
            market['market_risk_score'] * risk_weights['market'] +
            (100 - liquidity['liquidity_score']) * risk_weights['liquidity'] +  # Invert liquidity score
            operational['operational_risk_score'] * risk_weights['operational']
        )
        
        # Identify primary risk areas
        risk_scores = {
            'Credit Risk': credit['credit_risk_score'],
            'Market Risk': market['market_risk_score'],
            'Liquidity Risk': 100 - liquidity['liquidity_score'],
            'Operational Risk': operational['operational_risk_score']
        }
        
        primary_risks = [risk_type for risk_type, score in risk_scores.items() if score >= 50]
        
        return {
            'overall_risk_score': overall_risk_score,
            'overall_risk_level': self._categorize_risk_level(overall_risk_score),
            'risk_distribution': risk_scores,
            'primary_risk_areas': primary_risks,
            'risk_trend': self._assess_risk_trend(overall_risk_score),
            'risk_capacity': self._assess_risk_capacity(overall_risk_score)
        }
        
    def _assess_risk_trend(self, current_score: float) -> str:
        """Assess risk trend direction"""
        # Simplified trend assessment
        if current_score > 60:
            return 'Increasing risk - immediate attention required'
        elif current_score > 40:
            return 'Stable risk - monitor closely'
        else:
            return 'Decreasing risk - maintain current practices'
            
    def _assess_risk_capacity(self, risk_score: float) -> str:
        """Assess risk capacity and tolerance"""
        if risk_score < 30:
            return 'High risk capacity - can take on additional risk'
        elif risk_score < 50:
            return 'Moderate risk capacity - selective risk taking'
        else:
            return 'Low risk capacity - focus on risk reduction'
            
    def _generate_risk_mitigation_strategies(self, credit: Dict[str, Any], market: Dict[str, Any], 
                                           liquidity: Dict[str, Any], operational: Dict[str, Any], 
                                           overall: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate comprehensive risk mitigation strategies"""
        
        strategies = []
        
        # Credit risk mitigation
        if credit['risk_level'] in ['High Risk', 'Medium Risk']:
            strategies.append({
                'risk_type': 'Credit Risk',
                'priority': 'High' if credit['risk_level'] == 'High Risk' else 'Medium',
                'strategies': [
                    'Implement stricter credit approval processes',
                    'Diversify customer base to reduce concentration',
                    'Improve collections and reduce DSO',
                    'Consider credit insurance for large exposures'
                ],
                'expected_impact': 'Reduce credit losses by 30-50%',
                'timeline': '3-6 months'
            })
            
        # Market risk mitigation
        if market['risk_level'] in ['High Risk', 'Medium Risk']:
            strategies.append({
                'risk_type': 'Market Risk',
                'priority': 'Medium',
                'strategies': market['hedging_recommendations'],
                'expected_impact': 'Reduce market volatility impact by 40-60%',
                'timeline': '1-3 months'
            })
            
        # Liquidity risk mitigation
        if liquidity['risk_level'] in ['High Risk', 'Medium Risk']:
            strategies.append({
                'risk_type': 'Liquidity Risk',
                'priority': 'Critical' if liquidity['risk_level'] == 'High Risk' else 'High',
                'strategies': liquidity['liquidity_recommendations'],
                'expected_impact': 'Improve liquidity position and financial flexibility',
                'timeline': '1-2 months'
            })
            
        # Operational risk mitigation
        if operational['risk_level'] in ['High Risk', 'Medium Risk']:
            strategies.append({
                'risk_type': 'Operational Risk',
                'priority': 'Medium',
                'strategies': [
                    'Reduce key person dependencies through cross-training',
                    'Improve system reliability and automation',
                    'Strengthen compliance and governance processes',
                    'Enhance cybersecurity measures'
                ],
                'expected_impact': 'Reduce operational disruption risk by 40-60%',
                'timeline': '6-12 months'
            })
            
        # Overall risk management
        if overall['overall_risk_level'] == 'High Risk':
            strategies.append({
                'risk_type': 'Enterprise Risk Management',
                'priority': 'Strategic',
                'strategies': [
                    'Establish comprehensive risk management framework',
                    'Implement regular stress testing',
                    'Create risk committee and governance structure',
                    'Develop risk appetite statement and limits'
                ],
                'expected_impact': 'Improve overall risk profile and management capability',
                'timeline': '9-18 months'
            })
            
        return strategies

def test_financial_risk_management_agent():
    """Test the Financial Risk Management Agent"""
    print("🧪 Testing Financial Risk Management Agent")
    print("=" * 45)
    
    try:
        agent = FinancialRiskManagementAgent()
        
        test_data = {
            'company_name': 'Risk Aware Enterprises',
            'customer_portfolio': {
                'top_10_customers_percentage': 45,
                'high_risk_customers': ['Customer A', 'Customer B']
            },
            'receivables': {
                'dso': 55,
                'bad_debt_percentage': 2.5
            },
            'market_exposure': {
                'variable_rate_debt_percentage': 60,
                'foreign_revenue_percentage': 30,
                'commodity_cost_percentage': 20
            },
            'liquidity_metrics': {
                'current_ratio': 1.3,
                'quick_ratio': 0.9,
                'cash_ratio': 0.25,
                'operating_cash_flow': 800000,
                'credit_utilization_percentage': 45,
                'cash_and_equivalents': 400000
            },
            'operational_metrics': {
                'key_person_dependency_score': 65,
                'system_reliability_score': 75,
                'compliance_score': 82,
                'cybersecurity_score': 70
            },
            'daily_operating_expenses': 15000
        }
        
        assessment = agent.assess_financial_risks(test_data)
        print(f"✅ Financial risk assessment completed for {test_data['company_name']}")
        print(f"   Overall risk level: {assessment['overall_risk_profile']['overall_risk_level']}")
        print(f"   Credit risk: {assessment['credit_risk']['risk_level']}")
        print(f"   Liquidity risk: {assessment['liquidity_risk']['risk_level']}")
        print(f"   Mitigation strategies: {len(assessment['mitigation_strategies'])}")
        
        return {
            'agent_initialized': True,
            'overall_risk_score': assessment['overall_risk_profile']['overall_risk_score'],
            'risk_level': assessment['overall_risk_profile']['overall_risk_level']
        }
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return {'error': str(e)}

if __name__ == "__main__":
    test_financial_risk_management_agent()