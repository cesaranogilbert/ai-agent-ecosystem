"""
Financial Performance Prediction Agent
Real-time financial forecasting and scenario modeling
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

@dataclass
class FinancialForecast:
    revenue_forecast: float
    expense_forecast: float
    profit_margin: float
    cash_flow: float
    confidence_level: float

@dataclass
class ScenarioResults:
    best_case: FinancialForecast
    most_likely: FinancialForecast
    worst_case: FinancialForecast

class FinancialPerformancePredictionAgent:
    """
    Comprehensive Financial Performance Prediction System
    - Real-time financial forecasting
    - Scenario modeling and stress testing
    - Cash flow optimization
    - Investment ROI prediction
    """
    
    def __init__(self):
        self.name = "Financial Performance Prediction Agent"
        self.version = "1.0.0"
        self.capabilities = [
            "Financial Forecasting",
            "Scenario Modeling",
            "Cash Flow Prediction",
            "Investment ROI Analysis",
            "Risk Assessment",
            "Performance Analytics"
        ]
        
        # Financial ratios and benchmarks
        self.industry_benchmarks = {
            'technology': {
                'profit_margin': 15.5,
                'revenue_growth': 12.3,
                'roa': 8.7,
                'current_ratio': 2.1
            },
            'manufacturing': {
                'profit_margin': 8.2,
                'revenue_growth': 5.4,
                'roa': 6.1,
                'current_ratio': 1.8
            },
            'retail': {
                'profit_margin': 4.1,
                'revenue_growth': 3.2,
                'roa': 5.3,
                'current_ratio': 1.4
            },
            'financial': {
                'profit_margin': 22.1,
                'revenue_growth': 6.8,
                'roa': 1.1,
                'current_ratio': 1.1
            }
        }
        
    def predict_financial_performance(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive financial performance prediction
        """
        try:
            company_name = financial_data.get('company_name', 'Unknown Company')
            
            # Generate base forecast
            base_forecast = self._generate_base_forecast(financial_data)
            
            # Create scenario analysis
            scenario_analysis = self._perform_scenario_analysis(financial_data, base_forecast)
            
            # Analyze financial health
            health_assessment = self._assess_financial_health(financial_data)
            
            # Generate investment analysis
            investment_analysis = self._analyze_investment_opportunities(financial_data, base_forecast)
            
            # Risk assessment
            risk_assessment = self._assess_financial_risks(financial_data, scenario_analysis)
            
            # Strategic recommendations
            recommendations = self._generate_financial_recommendations(
                base_forecast, scenario_analysis, health_assessment
            )
            
            return {
                'company': company_name,
                'forecast_date': datetime.now().isoformat(),
                'base_forecast': base_forecast.__dict__,
                'scenario_analysis': {
                    'best_case': scenario_analysis.best_case.__dict__,
                    'most_likely': scenario_analysis.most_likely.__dict__,
                    'worst_case': scenario_analysis.worst_case.__dict__
                },
                'health_assessment': health_assessment,
                'investment_analysis': investment_analysis,
                'risk_assessment': risk_assessment,
                'recommendations': recommendations,
                'next_review_date': (datetime.now() + timedelta(days=30)).isoformat()
            }
            
        except Exception as e:
            logging.error(f"Financial prediction failed: {str(e)}")
            return {'error': f'Financial prediction failed: {str(e)}'}
            
    def _generate_base_forecast(self, financial_data: Dict[str, Any]) -> FinancialForecast:
        """Generate base financial forecast"""
        
        # Historical data analysis
        historical_revenue = financial_data.get('historical_revenue', [100, 110, 121])
        historical_expenses = financial_data.get('historical_expenses', [80, 85, 88])
        
        # Calculate growth trends
        revenue_growth_rate = self._calculate_growth_rate(historical_revenue)
        expense_growth_rate = self._calculate_growth_rate(historical_expenses)
        
        # Current financial position
        current_revenue = historical_revenue[-1] if historical_revenue else 100
        current_expenses = historical_expenses[-1] if historical_expenses else 80
        
        # Forecast next period
        forecasted_revenue = current_revenue * (1 + revenue_growth_rate)
        forecasted_expenses = current_expenses * (1 + expense_growth_rate)
        
        # Calculate metrics
        profit_margin = ((forecasted_revenue - forecasted_expenses) / forecasted_revenue) * 100
        cash_flow = self._forecast_cash_flow(financial_data, forecasted_revenue, forecasted_expenses)
        
        # Confidence calculation based on data quality and volatility
        confidence_level = self._calculate_forecast_confidence(financial_data)
        
        return FinancialForecast(
            revenue_forecast=forecasted_revenue,
            expense_forecast=forecasted_expenses,
            profit_margin=profit_margin,
            cash_flow=cash_flow,
            confidence_level=confidence_level
        )
        
    def _calculate_growth_rate(self, historical_data: List[float]) -> float:
        """Calculate compound annual growth rate"""
        if len(historical_data) < 2:
            return 0.05  # Default 5% growth
            
        periods = len(historical_data) - 1
        start_value = historical_data[0]
        end_value = historical_data[-1]
        
        if start_value <= 0:
            return 0.05
            
        growth_rate = (end_value / start_value) ** (1/periods) - 1
        
        # Cap extreme growth rates
        return max(-0.5, min(0.5, growth_rate))
        
    def _forecast_cash_flow(self, financial_data: Dict[str, Any], revenue: float, expenses: float) -> float:
        """Forecast cash flow based on revenue and expense predictions"""
        
        # Operating cash flow calculation
        operating_income = revenue - expenses
        
        # Adjustments for non-cash items
        depreciation = financial_data.get('depreciation', revenue * 0.05)
        working_capital_change = financial_data.get('working_capital_change', revenue * 0.02)
        
        # Capital expenditures
        capex = financial_data.get('planned_capex', revenue * 0.03)
        
        # Free cash flow calculation
        free_cash_flow = operating_income + depreciation - working_capital_change - capex
        
        return free_cash_flow
        
    def _calculate_forecast_confidence(self, financial_data: Dict[str, Any]) -> float:
        """Calculate confidence level for forecast"""
        
        confidence_factors = []
        
        # Data quality assessment
        historical_revenue = financial_data.get('historical_revenue', [])
        if len(historical_revenue) >= 5:
            confidence_factors.append(0.9)
        elif len(historical_revenue) >= 3:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
            
        # Market stability
        market_volatility = financial_data.get('market_volatility_score', 50)
        market_confidence = (100 - market_volatility) / 100
        confidence_factors.append(market_confidence)
        
        # Business model maturity
        company_age = financial_data.get('company_age_years', 5)
        maturity_confidence = min(1.0, company_age / 10)
        confidence_factors.append(maturity_confidence)
        
        # External factors
        economic_stability = financial_data.get('economic_stability_score', 75) / 100
        confidence_factors.append(economic_stability)
        
        return sum(confidence_factors) / len(confidence_factors)
        
    def _perform_scenario_analysis(self, financial_data: Dict[str, Any], base_forecast: FinancialForecast) -> ScenarioResults:
        """Perform comprehensive scenario analysis"""
        
        # Scenario parameters
        optimistic_revenue_uplift = 1.15  # 15% upside
        pessimistic_revenue_impact = 0.85  # 15% downside
        
        optimistic_expense_efficiency = 0.95  # 5% cost reduction
        pessimistic_expense_increase = 1.10  # 10% cost increase
        
        # Best case scenario
        best_case = FinancialForecast(
            revenue_forecast=base_forecast.revenue_forecast * optimistic_revenue_uplift,
            expense_forecast=base_forecast.expense_forecast * optimistic_expense_efficiency,
            profit_margin=0,  # Will be calculated
            cash_flow=0,  # Will be calculated
            confidence_level=base_forecast.confidence_level * 0.8  # Lower confidence for extreme scenarios
        )
        
        best_case.profit_margin = ((best_case.revenue_forecast - best_case.expense_forecast) / 
                                  best_case.revenue_forecast) * 100
        best_case.cash_flow = self._forecast_cash_flow(
            financial_data, best_case.revenue_forecast, best_case.expense_forecast
        )
        
        # Most likely scenario (base forecast)
        most_likely = base_forecast
        
        # Worst case scenario
        worst_case = FinancialForecast(
            revenue_forecast=base_forecast.revenue_forecast * pessimistic_revenue_impact,
            expense_forecast=base_forecast.expense_forecast * pessimistic_expense_increase,
            profit_margin=0,  # Will be calculated
            cash_flow=0,  # Will be calculated
            confidence_level=base_forecast.confidence_level * 0.7
        )
        
        worst_case.profit_margin = ((worst_case.revenue_forecast - worst_case.expense_forecast) / 
                                   worst_case.revenue_forecast) * 100
        worst_case.cash_flow = self._forecast_cash_flow(
            financial_data, worst_case.revenue_forecast, worst_case.expense_forecast
        )
        
        return ScenarioResults(
            best_case=best_case,
            most_likely=most_likely,
            worst_case=worst_case
        )
        
    def _assess_financial_health(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall financial health"""
        
        # Extract key financial metrics
        current_assets = financial_data.get('current_assets', 1000000)
        current_liabilities = financial_data.get('current_liabilities', 500000)
        total_debt = financial_data.get('total_debt', 800000)
        total_equity = financial_data.get('total_equity', 1200000)
        revenue = financial_data.get('current_revenue', 2000000)
        net_income = financial_data.get('net_income', 200000)
        
        # Calculate key ratios
        current_ratio = current_assets / max(1, current_liabilities)
        debt_to_equity = total_debt / max(1, total_equity)
        profit_margin = (net_income / max(1, revenue)) * 100
        
        # Assess liquidity
        liquidity_score = min(100, current_ratio * 50)
        liquidity_status = self._categorize_score(liquidity_score, 'liquidity')
        
        # Assess leverage
        leverage_score = max(0, 100 - (debt_to_equity * 100))
        leverage_status = self._categorize_score(leverage_score, 'leverage')
        
        # Assess profitability
        profitability_score = max(0, min(100, profit_margin * 5))
        profitability_status = self._categorize_score(profitability_score, 'profitability')
        
        # Overall health score
        overall_score = (liquidity_score + leverage_score + profitability_score) / 3
        overall_status = self._categorize_score(overall_score, 'overall')
        
        return {
            'overall_health_score': overall_score,
            'overall_status': overall_status,
            'liquidity': {
                'current_ratio': current_ratio,
                'score': liquidity_score,
                'status': liquidity_status
            },
            'leverage': {
                'debt_to_equity': debt_to_equity,
                'score': leverage_score,
                'status': leverage_status
            },
            'profitability': {
                'profit_margin': profit_margin,
                'score': profitability_score,
                'status': profitability_status
            },
            'improvement_areas': self._identify_improvement_areas(liquidity_score, leverage_score, profitability_score)
        }
        
    def _categorize_score(self, score: float, category: str) -> str:
        """Categorize financial scores"""
        if score >= 80:
            return 'Excellent'
        elif score >= 70:
            return 'Good'
        elif score >= 60:
            return 'Fair'
        elif score >= 50:
            return 'Poor'
        else:
            return 'Critical'
            
    def _identify_improvement_areas(self, liquidity: float, leverage: float, profitability: float) -> List[str]:
        """Identify areas for financial improvement"""
        
        areas = []
        
        if liquidity < 60:
            areas.append('Improve liquidity position through cash management')
        if leverage < 60:
            areas.append('Optimize capital structure and reduce debt burden')
        if profitability < 60:
            areas.append('Enhance profitability through cost management and revenue growth')
            
        return areas
        
    def _analyze_investment_opportunities(self, financial_data: Dict[str, Any], forecast: FinancialForecast) -> Dict[str, Any]:
        """Analyze investment opportunities and ROI predictions"""
        
        # Current financial position
        available_cash = financial_data.get('available_cash', 500000)
        debt_capacity = financial_data.get('debt_capacity', 1000000)
        
        # Investment scenarios
        investment_opportunities = []
        
        # Growth investments
        growth_investment = min(available_cash * 0.6, 2000000)
        growth_roi = self._calculate_investment_roi(growth_investment, 'growth', forecast)
        
        investment_opportunities.append({
            'type': 'Growth Investment',
            'investment_amount': growth_investment,
            'expected_roi': growth_roi,
            'payback_period': self._calculate_payback_period(growth_investment, growth_roi),
            'risk_level': 'Medium',
            'description': 'Investment in market expansion and product development'
        })
        
        # Efficiency investments
        efficiency_investment = min(available_cash * 0.3, 1000000)
        efficiency_roi = self._calculate_investment_roi(efficiency_investment, 'efficiency', forecast)
        
        investment_opportunities.append({
            'type': 'Efficiency Investment',
            'investment_amount': efficiency_investment,
            'expected_roi': efficiency_roi,
            'payback_period': self._calculate_payback_period(efficiency_investment, efficiency_roi),
            'risk_level': 'Low',
            'description': 'Investment in process automation and cost reduction'
        })
        
        # Technology investments
        tech_investment = min(available_cash * 0.4, 1500000)
        tech_roi = self._calculate_investment_roi(tech_investment, 'technology', forecast)
        
        investment_opportunities.append({
            'type': 'Technology Investment',
            'investment_amount': tech_investment,
            'expected_roi': tech_roi,
            'payback_period': self._calculate_payback_period(tech_investment, tech_roi),
            'risk_level': 'Medium-High',
            'description': 'Investment in digital transformation and innovation'
        })
        
        return {
            'available_capital': available_cash + debt_capacity,
            'recommended_investments': sorted(investment_opportunities, key=lambda x: x['expected_roi'], reverse=True),
            'optimal_investment_mix': self._optimize_investment_portfolio(investment_opportunities),
            'expected_impact': self._calculate_investment_impact(investment_opportunities, forecast)
        }
        
    def _calculate_investment_roi(self, investment_amount: float, investment_type: str, forecast: FinancialForecast) -> float:
        """Calculate expected ROI for investment type"""
        
        roi_multipliers = {
            'growth': 0.18,      # 18% ROI for growth investments
            'efficiency': 0.25,   # 25% ROI for efficiency investments
            'technology': 0.15    # 15% ROI for technology investments
        }
        
        base_roi = roi_multipliers.get(investment_type, 0.12)
        
        # Adjust for company performance
        performance_adjustment = (forecast.profit_margin - 10) / 100  # Adjust based on profit margin
        adjusted_roi = base_roi + performance_adjustment
        
        return max(0.05, min(0.35, adjusted_roi))  # Cap between 5% and 35%
        
    def _calculate_payback_period(self, investment: float, roi: float) -> str:
        """Calculate investment payback period"""
        
        annual_return = investment * roi
        payback_years = investment / max(1, annual_return)
        
        if payback_years <= 2:
            return '1-2 years'
        elif payback_years <= 3:
            return '2-3 years'
        elif payback_years <= 5:
            return '3-5 years'
        else:
            return '5+ years'
            
    def _optimize_investment_portfolio(self, opportunities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize investment portfolio allocation"""
        
        total_investment = sum(opp['investment_amount'] for opp in opportunities)
        weighted_roi = sum(opp['investment_amount'] * opp['expected_roi'] for opp in opportunities) / max(1, total_investment)
        
        # Risk-adjusted allocation
        allocations = []
        for opp in opportunities:
            risk_factor = {'Low': 1.2, 'Medium': 1.0, 'Medium-High': 0.8, 'High': 0.6}[opp['risk_level']]
            adjusted_score = opp['expected_roi'] * risk_factor
            allocations.append((opp['type'], adjusted_score))
            
        return {
            'recommended_allocation': sorted(allocations, key=lambda x: x[1], reverse=True),
            'portfolio_roi': weighted_roi,
            'diversification_score': len(opportunities) / 5.0  # Normalize to max 5 types
        }
        
    def _calculate_investment_impact(self, opportunities: List[Dict[str, Any]], forecast: FinancialForecast) -> Dict[str, Any]:
        """Calculate expected impact of investments on financial performance"""
        
        total_investment = sum(opp['investment_amount'] for opp in opportunities)
        weighted_roi = sum(opp['investment_amount'] * opp['expected_roi'] for opp in opportunities) / max(1, total_investment)
        
        annual_return = total_investment * weighted_roi
        
        # Impact on key metrics
        revenue_impact = annual_return * 0.7  # 70% of returns flow to revenue
        cost_savings = annual_return * 0.3    # 30% from cost savings
        
        new_revenue = forecast.revenue_forecast + revenue_impact
        new_expenses = forecast.expense_forecast - cost_savings
        new_profit_margin = ((new_revenue - new_expenses) / new_revenue) * 100
        
        return {
            'total_investment': total_investment,
            'expected_annual_return': annual_return,
            'revenue_impact': revenue_impact,
            'cost_savings': cost_savings,
            'new_profit_margin': new_profit_margin,
            'margin_improvement': new_profit_margin - forecast.profit_margin
        }
        
    def _assess_financial_risks(self, financial_data: Dict[str, Any], scenarios: ScenarioResults) -> Dict[str, Any]:
        """Assess financial risks and vulnerabilities"""
        
        risks = {}
        
        # Market risk
        revenue_volatility = abs(scenarios.best_case.revenue_forecast - scenarios.worst_case.revenue_forecast) / scenarios.most_likely.revenue_forecast
        market_risk_score = min(100, revenue_volatility * 100)
        risks['market_risk'] = {
            'score': market_risk_score,
            'level': self._categorize_risk_level(market_risk_score),
            'description': 'Risk from market volatility and demand fluctuations'
        }
        
        # Operational risk
        margin_volatility = abs(scenarios.best_case.profit_margin - scenarios.worst_case.profit_margin)
        operational_risk_score = min(100, margin_volatility * 2)
        risks['operational_risk'] = {
            'score': operational_risk_score,
            'level': self._categorize_risk_level(operational_risk_score),
            'description': 'Risk from operational inefficiencies and cost variations'
        }
        
        # Liquidity risk
        current_ratio = financial_data.get('current_assets', 1000000) / max(1, financial_data.get('current_liabilities', 500000))
        liquidity_risk_score = max(0, 100 - (current_ratio * 50))
        risks['liquidity_risk'] = {
            'score': liquidity_risk_score,
            'level': self._categorize_risk_level(liquidity_risk_score),
            'description': 'Risk of insufficient liquidity to meet obligations'
        }
        
        # Credit risk
        debt_to_equity = financial_data.get('total_debt', 800000) / max(1, financial_data.get('total_equity', 1200000))
        credit_risk_score = min(100, debt_to_equity * 50)
        risks['credit_risk'] = {
            'score': credit_risk_score,
            'level': self._categorize_risk_level(credit_risk_score),
            'description': 'Risk from excessive leverage and debt burden'
        }
        
        # Overall risk assessment
        risk_scores = [risk['score'] for risk in risks.values()]
        overall_risk = sum(risk_scores) / len(risk_scores)
        
        return {
            'individual_risks': risks,
            'overall_risk_score': overall_risk,
            'risk_level': self._categorize_risk_level(overall_risk),
            'risk_mitigation_priorities': self._prioritize_risk_mitigation(risks),
            'stress_test_results': self._perform_stress_tests(scenarios)
        }
        
    def _categorize_risk_level(self, score: float) -> str:
        """Categorize risk level"""
        if score >= 75:
            return 'High'
        elif score >= 50:
            return 'Medium'
        elif score >= 25:
            return 'Low'
        else:
            return 'Very Low'
            
    def _prioritize_risk_mitigation(self, risks: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prioritize risk mitigation actions"""
        
        priorities = []
        
        for risk_type, risk_data in risks.items():
            if risk_data['score'] >= 60:  # High risk threshold
                priority = {
                    'risk_type': risk_type.replace('_', ' ').title(),
                    'priority': 'High' if risk_data['score'] >= 75 else 'Medium',
                    'recommended_actions': self._get_risk_mitigation_actions(risk_type),
                    'timeline': 'Immediate' if risk_data['score'] >= 75 else '3-6 months'
                }
                priorities.append(priority)
                
        return sorted(priorities, key=lambda x: x['priority'], reverse=True)
        
    def _get_risk_mitigation_actions(self, risk_type: str) -> List[str]:
        """Get specific mitigation actions for risk type"""
        
        actions_map = {
            'market_risk': [
                'Diversify revenue streams',
                'Implement flexible pricing strategies',
                'Develop countercyclical products'
            ],
            'operational_risk': [
                'Improve cost management processes',
                'Implement operational efficiency programs',
                'Enhance supply chain resilience'
            ],
            'liquidity_risk': [
                'Establish credit facilities',
                'Improve cash management',
                'Optimize working capital'
            ],
            'credit_risk': [
                'Reduce debt levels',
                'Improve debt structure',
                'Strengthen equity position'
            ]
        }
        
        return actions_map.get(risk_type, ['Implement comprehensive risk management'])
        
    def _perform_stress_tests(self, scenarios: ScenarioResults) -> Dict[str, Any]:
        """Perform financial stress tests"""
        
        return {
            'revenue_stress_test': {
                'scenario': '30% revenue decline',
                'impact': scenarios.worst_case.revenue_forecast * 0.7,
                'survival_months': self._calculate_survival_months(scenarios.worst_case)
            },
            'margin_stress_test': {
                'scenario': 'Margin compression to 5%',
                'impact': scenarios.most_likely.revenue_forecast * 0.05,
                'sustainability': 'Low' if scenarios.worst_case.profit_margin < 5 else 'Medium'
            },
            'cash_flow_stress_test': {
                'scenario': 'Negative cash flow for 6 months',
                'impact': scenarios.worst_case.cash_flow * 6,
                'liquidity_requirement': abs(scenarios.worst_case.cash_flow * 6) if scenarios.worst_case.cash_flow < 0 else 0
            }
        }
        
    def _calculate_survival_months(self, worst_case: FinancialForecast) -> int:
        """Calculate survival months in worst case scenario"""
        
        if worst_case.cash_flow >= 0:
            return 12  # Indefinite if cash flow positive
            
        # Assume current cash reserves (simplified)
        cash_reserves = abs(worst_case.cash_flow) * 6  # 6 months of negative cash flow
        monthly_burn = abs(worst_case.cash_flow) / 12
        
        return int(cash_reserves / max(1, monthly_burn))
        
    def _generate_financial_recommendations(self, forecast: FinancialForecast, 
                                          scenarios: ScenarioResults, 
                                          health: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate financial performance recommendations"""
        
        recommendations = []
        
        # Performance improvement recommendations
        if forecast.profit_margin < 15:
            recommendations.append({
                'category': 'Profitability',
                'priority': 'High',
                'recommendation': 'Implement comprehensive margin improvement program',
                'actions': [
                    'Optimize pricing strategies',
                    'Reduce operational costs',
                    'Improve product mix'
                ],
                'expected_impact': f'Increase margin by {20 - forecast.profit_margin:.1f} percentage points',
                'timeline': '6-12 months'
            })
            
        # Cash flow optimization
        if forecast.cash_flow < 0:
            recommendations.append({
                'category': 'Cash Flow',
                'priority': 'Critical',
                'recommendation': 'Urgent cash flow improvement required',
                'actions': [
                    'Accelerate accounts receivable collection',
                    'Optimize inventory levels',
                    'Defer non-critical expenditures'
                ],
                'expected_impact': 'Positive cash flow within 3 months',
                'timeline': '1-3 months'
            })
            
        # Financial health improvements
        if health['overall_health_score'] < 70:
            recommendations.append({
                'category': 'Financial Health',
                'priority': 'Medium',
                'recommendation': 'Strengthen overall financial position',
                'actions': health['improvement_areas'],
                'expected_impact': 'Improve financial health score to 80+',
                'timeline': '6-18 months'
            })
            
        # Growth investment recommendations
        if forecast.confidence_level > 0.7 and forecast.profit_margin > 10:
            recommendations.append({
                'category': 'Growth',
                'priority': 'Medium',
                'recommendation': 'Consider strategic growth investments',
                'actions': [
                    'Evaluate market expansion opportunities',
                    'Invest in product development',
                    'Consider strategic acquisitions'
                ],
                'expected_impact': 'Accelerate revenue growth by 5-10%',
                'timeline': '12-24 months'
            })
            
        return recommendations

def test_financial_performance_prediction_agent():
    """Test the Financial Performance Prediction Agent"""
    print("🧪 Testing Financial Performance Prediction Agent")
    print("=" * 55)
    
    try:
        agent = FinancialPerformancePredictionAgent()
        
        # Test data
        test_financial_data = {
            'company_name': 'Global Enterprise Corp',
            'historical_revenue': [2000000, 2200000, 2420000, 2662000],
            'historical_expenses': [1600000, 1700000, 1800000, 1900000],
            'current_assets': 1500000,
            'current_liabilities': 800000,
            'total_debt': 1200000,
            'total_equity': 2000000,
            'current_revenue': 2662000,
            'net_income': 762000,
            'available_cash': 800000,
            'market_volatility_score': 35,
            'company_age_years': 8
        }
        
        # Test financial prediction
        prediction = agent.predict_financial_performance(test_financial_data)
        print(f"✅ Financial prediction completed for {test_financial_data['company_name']}")
        print(f"   Revenue forecast: ${prediction['base_forecast']['revenue_forecast']:,.0f}")
        print(f"   Profit margin: {prediction['base_forecast']['profit_margin']:.1f}%")
        print(f"   Confidence level: {prediction['base_forecast']['confidence_level']:.1f}")
        print(f"   Health score: {prediction['health_assessment']['overall_health_score']:.1f}")
        print(f"   Investment opportunities: {len(prediction['investment_analysis']['recommended_investments'])}")
        
        return {
            'agent_initialized': True,
            'revenue_forecast': prediction['base_forecast']['revenue_forecast'],
            'profit_margin': prediction['base_forecast']['profit_margin'],
            'health_score': prediction['health_assessment']['overall_health_score'],
            'investment_opportunities': len(prediction['investment_analysis']['recommended_investments'])
        }
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return {'error': str(e)}

if __name__ == "__main__":
    test_financial_performance_prediction_agent()