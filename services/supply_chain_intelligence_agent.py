"""
Supply Chain Intelligence Agent
End-to-end supply chain optimization and risk management
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

@dataclass
class SupplierRisk:
    supplier_id: str
    risk_score: float
    risk_category: str
    mitigation_actions: List[str]

@dataclass
class SupplyChainMetrics:
    efficiency_score: float
    resilience_score: float
    cost_optimization: float
    sustainability_score: float

class SupplyChainIntelligenceAgent:
    """
    Comprehensive Supply Chain Intelligence System
    - End-to-end supply chain optimization
    - Disruption prediction and mitigation
    - Vendor risk assessment
    - Just-in-time inventory optimization
    """
    
    def __init__(self):
        self.name = "Supply Chain Intelligence Agent"
        self.version = "1.0.0"
        self.capabilities = [
            "Supply Chain Optimization",
            "Disruption Prediction and Mitigation",
            "Vendor Risk Assessment",
            "Inventory Optimization",
            "Logistics Planning",
            "Sustainability Analysis"
        ]
        
        # Risk factors for supply chain analysis
        self.risk_factors = {
            'geographic': ['political_stability', 'natural_disasters', 'infrastructure'],
            'economic': ['currency_volatility', 'inflation', 'trade_policies'],
            'operational': ['capacity_constraints', 'quality_issues', 'delivery_performance'],
            'financial': ['financial_stability', 'credit_risk', 'payment_terms'],
            'environmental': ['sustainability_practices', 'carbon_footprint', 'waste_management']
        }
        
        # Supply chain optimization parameters
        self.optimization_parameters = {
            'cost': 0.3,
            'quality': 0.25,
            'delivery': 0.2,
            'sustainability': 0.15,
            'risk': 0.1
        }
        
    def analyze_supply_chain_performance(self, supply_chain_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive supply chain performance analysis
        """
        try:
            company_name = supply_chain_data.get('company_name', 'Unknown Company')
            
            # Analyze current performance
            performance_metrics = self._calculate_performance_metrics(supply_chain_data)
            
            # Assess supplier risks
            supplier_risks = self._assess_supplier_risks(supply_chain_data.get('suppliers', []))
            
            # Identify optimization opportunities
            optimization_opportunities = self._identify_optimization_opportunities(performance_metrics, supply_chain_data)
            
            # Predict potential disruptions
            disruption_analysis = self._predict_supply_chain_disruptions(supply_chain_data)
            
            # Generate recommendations
            recommendations = self._generate_supply_chain_recommendations(
                performance_metrics, supplier_risks, optimization_opportunities
            )
            
            return {
                'company': company_name,
                'analysis_date': datetime.now().isoformat(),
                'performance_metrics': performance_metrics.__dict__,
                'supplier_risks': [risk.__dict__ for risk in supplier_risks],
                'optimization_opportunities': optimization_opportunities,
                'disruption_analysis': disruption_analysis,
                'recommendations': recommendations,
                'next_review_date': (datetime.now() + timedelta(days=30)).isoformat()
            }
            
        except Exception as e:
            logging.error(f"Supply chain analysis failed: {str(e)}")
            return {'error': f'Supply chain analysis failed: {str(e)}'}
            
    def _calculate_performance_metrics(self, supply_chain_data: Dict[str, Any]) -> SupplyChainMetrics:
        """Calculate comprehensive supply chain performance metrics"""
        
        # Efficiency score calculation
        delivery_performance = supply_chain_data.get('on_time_delivery_rate', 85)
        cost_variance = 100 - supply_chain_data.get('cost_variance_percentage', 15)
        inventory_turnover = min(100, supply_chain_data.get('inventory_turnover', 6) * 10)
        efficiency_score = (delivery_performance + cost_variance + inventory_turnover) / 3
        
        # Resilience score calculation
        supplier_diversification = min(100, supply_chain_data.get('supplier_count', 5) * 10)
        geographic_distribution = supply_chain_data.get('geographic_diversification_score', 70)
        backup_suppliers = supply_chain_data.get('backup_supplier_percentage', 50)
        resilience_score = (supplier_diversification + geographic_distribution + backup_suppliers) / 3
        
        # Cost optimization score
        total_cost_improvement = supply_chain_data.get('cost_reduction_percentage', 10)
        procurement_efficiency = supply_chain_data.get('procurement_efficiency_score', 75)
        logistics_optimization = supply_chain_data.get('logistics_cost_optimization', 80)
        cost_optimization = (total_cost_improvement * 5 + procurement_efficiency + logistics_optimization) / 3
        
        # Sustainability score
        sustainable_suppliers = supply_chain_data.get('sustainable_supplier_percentage', 60)
        carbon_footprint_reduction = supply_chain_data.get('carbon_reduction_percentage', 20)
        circular_economy_practices = supply_chain_data.get('circular_economy_score', 50)
        sustainability_score = (sustainable_suppliers + carbon_footprint_reduction * 2 + circular_economy_practices) / 3
        
        return SupplyChainMetrics(
            efficiency_score=efficiency_score,
            resilience_score=resilience_score,
            cost_optimization=cost_optimization,
            sustainability_score=sustainability_score
        )
        
    def _assess_supplier_risks(self, suppliers: List[Dict[str, Any]]) -> List[SupplierRisk]:
        """Assess risks for each supplier"""
        
        supplier_risks = []
        
        for supplier in suppliers:
            risk_score = self._calculate_supplier_risk_score(supplier)
            risk_category = self._categorize_supplier_risk(risk_score)
            mitigation_actions = self._generate_mitigation_actions(supplier, risk_score)
            
            supplier_risk = SupplierRisk(
                supplier_id=supplier.get('id', 'unknown'),
                risk_score=risk_score,
                risk_category=risk_category,
                mitigation_actions=mitigation_actions
            )
            
            supplier_risks.append(supplier_risk)
            
        return supplier_risks
        
    def _calculate_supplier_risk_score(self, supplier: Dict[str, Any]) -> float:
        """Calculate comprehensive risk score for a supplier"""
        
        risk_scores = {}
        
        # Geographic risk
        political_stability = supplier.get('political_stability_score', 70)
        natural_disaster_risk = 100 - supplier.get('natural_disaster_exposure', 30)
        infrastructure_quality = supplier.get('infrastructure_score', 80)
        risk_scores['geographic'] = (political_stability + natural_disaster_risk + infrastructure_quality) / 3
        
        # Economic risk
        currency_stability = supplier.get('currency_stability_score', 75)
        economic_indicators = supplier.get('economic_health_score', 80)
        trade_policy_risk = 100 - supplier.get('trade_policy_uncertainty', 25)
        risk_scores['economic'] = (currency_stability + economic_indicators + trade_policy_risk) / 3
        
        # Operational risk
        quality_performance = supplier.get('quality_score', 85)
        delivery_reliability = supplier.get('delivery_reliability', 90)
        capacity_utilization = min(100, 100 - abs(supplier.get('capacity_utilization', 80) - 80))
        risk_scores['operational'] = (quality_performance + delivery_reliability + capacity_utilization) / 3
        
        # Financial risk
        financial_stability = supplier.get('financial_health_score', 75)
        credit_rating = supplier.get('credit_rating_score', 80)
        payment_terms_risk = 100 - supplier.get('payment_risk_score', 20)
        risk_scores['financial'] = (financial_stability + credit_rating + payment_terms_risk) / 3
        
        # Environmental risk
        sustainability_practices = supplier.get('sustainability_score', 60)
        environmental_compliance = supplier.get('environmental_compliance_score', 80)
        carbon_footprint = 100 - supplier.get('carbon_intensity_score', 40)
        risk_scores['environmental'] = (sustainability_practices + environmental_compliance + carbon_footprint) / 3
        
        # Calculate weighted risk score (lower is riskier)
        total_risk = sum(risk_scores.values()) / len(risk_scores)
        
        # Convert to risk score (higher is riskier)
        risk_score = 100 - total_risk
        
        return risk_score
        
    def _categorize_supplier_risk(self, risk_score: float) -> str:
        """Categorize supplier risk level"""
        if risk_score >= 70:
            return "Critical"
        elif risk_score >= 50:
            return "High"
        elif risk_score >= 30:
            return "Medium"
        else:
            return "Low"
            
    def _generate_mitigation_actions(self, supplier: Dict[str, Any], risk_score: float) -> List[str]:
        """Generate risk mitigation actions for supplier"""
        
        actions = []
        
        if risk_score >= 70:
            actions.extend([
                "Identify alternative suppliers immediately",
                "Increase safety stock for critical items",
                "Implement enhanced monitoring and reporting"
            ])
        elif risk_score >= 50:
            actions.extend([
                "Develop backup supplier relationships",
                "Negotiate improved contract terms",
                "Implement supplier improvement program"
            ])
        elif risk_score >= 30:
            actions.extend([
                "Monitor performance closely",
                "Conduct regular supplier audits",
                "Maintain contingency plans"
            ])
        else:
            actions.extend([
                "Continue standard monitoring",
                "Explore partnership opportunities",
                "Consider preferred supplier status"
            ])
            
        return actions
        
    def _identify_optimization_opportunities(self, metrics: SupplyChainMetrics, supply_chain_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify supply chain optimization opportunities"""
        
        opportunities = []
        
        # Efficiency opportunities
        if metrics.efficiency_score < 80:
            opportunities.append({
                'category': 'Efficiency',
                'opportunity': 'Improve delivery performance and cost management',
                'potential_impact': f'{85 - metrics.efficiency_score:.1f}% efficiency improvement',
                'investment_required': 'Medium',
                'timeline': '6-12 months',
                'priority': 'High'
            })
            
        # Resilience opportunities
        if metrics.resilience_score < 75:
            opportunities.append({
                'category': 'Resilience',
                'opportunity': 'Enhance supplier diversification and geographic distribution',
                'potential_impact': f'{80 - metrics.resilience_score:.1f}% resilience improvement',
                'investment_required': 'Medium-High',
                'timeline': '9-18 months',
                'priority': 'Medium'
            })
            
        # Cost optimization opportunities
        if metrics.cost_optimization < 85:
            opportunities.append({
                'category': 'Cost Optimization',
                'opportunity': 'Implement advanced procurement and logistics optimization',
                'potential_impact': f'${(90 - metrics.cost_optimization) * 50000:,.0f} annual savings',
                'investment_required': 'Medium',
                'timeline': '3-9 months',
                'priority': 'High'
            })
            
        # Sustainability opportunities
        if metrics.sustainability_score < 70:
            opportunities.append({
                'category': 'Sustainability',
                'opportunity': 'Increase sustainable supplier adoption and circular economy practices',
                'potential_impact': f'{75 - metrics.sustainability_score:.1f}% sustainability improvement',
                'investment_required': 'Medium',
                'timeline': '12-24 months',
                'priority': 'Medium'
            })
            
        return opportunities
        
    def _predict_supply_chain_disruptions(self, supply_chain_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict potential supply chain disruptions"""
        
        # Analyze disruption probability
        disruption_factors = {
            'geopolitical_tensions': supply_chain_data.get('geopolitical_risk_score', 30),
            'climate_events': supply_chain_data.get('climate_risk_score', 25),
            'economic_volatility': supply_chain_data.get('economic_volatility_score', 35),
            'cyber_threats': supply_chain_data.get('cyber_risk_score', 40),
            'pandemic_impact': supply_chain_data.get('pandemic_preparedness_score', 60)
        }
        
        # Calculate overall disruption probability
        avg_risk = sum(disruption_factors.values()) / len(disruption_factors)
        disruption_probability = min(100, avg_risk)
        
        # Identify most likely disruptions
        likely_disruptions = []
        for factor, score in disruption_factors.items():
            if score >= 40:
                likely_disruptions.append({
                    'type': factor.replace('_', ' ').title(),
                    'probability': score,
                    'potential_impact': self._assess_disruption_impact(factor, score),
                    'mitigation_strategies': self._get_disruption_mitigation_strategies(factor)
                })
                
        return {
            'overall_disruption_probability': disruption_probability,
            'risk_level': 'High' if disruption_probability >= 60 else 'Medium' if disruption_probability >= 40 else 'Low',
            'likely_disruptions': likely_disruptions,
            'preparedness_score': 100 - disruption_probability,
            'recommendation': self._generate_disruption_recommendation(disruption_probability)
        }
        
    def _assess_disruption_impact(self, factor: str, score: float) -> str:
        """Assess potential impact of disruption"""
        impact_map = {
            'geopolitical_tensions': 'Supply delays, increased costs, sourcing restrictions',
            'climate_events': 'Facility damage, transportation disruptions, material shortages',
            'economic_volatility': 'Currency fluctuations, demand changes, supplier financial stress',
            'cyber_threats': 'System downtime, data breaches, operational disruptions',
            'pandemic_impact': 'Workforce shortages, facility closures, demand volatility'
        }
        
        return impact_map.get(factor, 'Operational and financial impact')
        
    def _get_disruption_mitigation_strategies(self, factor: str) -> List[str]:
        """Get mitigation strategies for specific disruption type"""
        strategies_map = {
            'geopolitical_tensions': [
                'Diversify supplier base across regions',
                'Develop local sourcing capabilities',
                'Monitor political risk indicators'
            ],
            'climate_events': [
                'Assess climate risk exposure',
                'Implement backup facilities',
                'Develop emergency response plans'
            ],
            'economic_volatility': [
                'Use currency hedging strategies',
                'Maintain flexible supplier contracts',
                'Monitor economic indicators'
            ],
            'cyber_threats': [
                'Implement cybersecurity measures',
                'Develop backup systems',
                'Train staff on security protocols'
            ],
            'pandemic_impact': [
                'Develop workforce contingency plans',
                'Implement remote work capabilities',
                'Maintain higher inventory levels'
            ]
        }
        
        return strategies_map.get(factor, ['Develop specific mitigation strategies'])
        
    def _generate_disruption_recommendation(self, probability: float) -> str:
        """Generate recommendation based on disruption probability"""
        if probability >= 70:
            return "Immediate action required: Implement comprehensive risk mitigation strategies"
        elif probability >= 50:
            return "High priority: Develop and test contingency plans"
        elif probability >= 30:
            return "Moderate priority: Enhance monitoring and preparedness"
        else:
            return "Continue standard risk management practices"
            
    def _generate_supply_chain_recommendations(self, metrics: SupplyChainMetrics, 
                                             supplier_risks: List[SupplierRisk], 
                                             opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate comprehensive supply chain recommendations"""
        
        recommendations = []
        
        # High-risk supplier recommendations
        critical_suppliers = [risk for risk in supplier_risks if risk.risk_category == "Critical"]
        if critical_suppliers:
            recommendations.append({
                'category': 'Risk Management',
                'priority': 'Critical',
                'recommendation': f'Address {len(critical_suppliers)} critical supplier risks immediately',
                'actions': [
                    'Conduct emergency supplier assessments',
                    'Activate backup supplier agreements',
                    'Implement enhanced monitoring protocols'
                ],
                'timeline': 'Immediate (1-4 weeks)',
                'investment': 'High'
            })
            
        # Performance improvement recommendations
        if metrics.efficiency_score < 80:
            recommendations.append({
                'category': 'Performance Optimization',
                'priority': 'High',
                'recommendation': 'Implement comprehensive efficiency improvement program',
                'actions': [
                    'Optimize delivery schedules and routes',
                    'Implement demand forecasting improvements',
                    'Enhance supplier performance management'
                ],
                'timeline': '3-6 months',
                'investment': 'Medium'
            })
            
        # Technology recommendations
        recommendations.append({
            'category': 'Digital Transformation',
            'priority': 'Medium',
            'recommendation': 'Implement advanced supply chain technologies',
            'actions': [
                'Deploy IoT sensors for real-time visibility',
                'Implement AI-powered demand forecasting',
                'Adopt blockchain for supply chain transparency'
            ],
            'timeline': '6-18 months',
            'investment': 'High'
        })
        
        # Sustainability recommendations
        if metrics.sustainability_score < 70:
            recommendations.append({
                'category': 'Sustainability',
                'priority': 'Medium',
                'recommendation': 'Enhance supply chain sustainability practices',
                'actions': [
                    'Implement supplier sustainability assessments',
                    'Develop carbon reduction initiatives',
                    'Adopt circular economy principles'
                ],
                'timeline': '12-24 months',
                'investment': 'Medium'
            })
            
        return recommendations
        
    def optimize_inventory_levels(self, inventory_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize inventory levels using advanced analytics
        """
        try:
            # Extract inventory parameters
            current_items = inventory_data.get('inventory_items', [])
            demand_forecast = inventory_data.get('demand_forecast', {})
            service_level_target = inventory_data.get('service_level_target', 95)
            
            optimized_inventory = []
            total_cost_savings = 0
            
            for item in current_items:
                optimization = self._optimize_item_inventory(item, demand_forecast, service_level_target)
                optimized_inventory.append(optimization)
                total_cost_savings += optimization.get('cost_savings', 0)
                
            return {
                'optimized_inventory': optimized_inventory,
                'total_cost_savings': total_cost_savings,
                'service_level_achievement': service_level_target,
                'inventory_reduction': sum(opt.get('quantity_reduction', 0) for opt in optimized_inventory),
                'implementation_timeline': '2-6 months',
                'monitoring_recommendations': self._generate_inventory_monitoring_recommendations()
            }
            
        except Exception as e:
            logging.error(f"Inventory optimization failed: {str(e)}")
            return {'error': f'Inventory optimization failed: {str(e)}'}
            
    def _optimize_item_inventory(self, item: Dict[str, Any], demand_forecast: Dict[str, Any], service_level: float) -> Dict[str, Any]:
        """Optimize inventory for a specific item"""
        
        item_id = item.get('id', 'unknown')
        current_quantity = item.get('current_quantity', 100)
        unit_cost = item.get('unit_cost', 10)
        lead_time = item.get('lead_time_days', 7)
        
        # Get demand forecast for this item
        forecast = demand_forecast.get(item_id, {})
        avg_demand = forecast.get('average_daily_demand', 10)
        demand_variability = forecast.get('demand_std_dev', 2)
        
        # Calculate optimal order quantity (simplified EOQ)
        ordering_cost = item.get('ordering_cost', 50)
        holding_cost_rate = item.get('holding_cost_rate', 0.2)
        
        eoq = np.sqrt((2 * avg_demand * 365 * ordering_cost) / (unit_cost * holding_cost_rate))
        
        # Calculate safety stock
        z_score = 1.96 if service_level >= 95 else 1.65 if service_level >= 90 else 1.28
        safety_stock = z_score * demand_variability * np.sqrt(lead_time)
        
        # Calculate reorder point
        reorder_point = (avg_demand * lead_time) + safety_stock
        
        # Calculate optimal stock level
        optimal_quantity = reorder_point + (eoq / 2)
        
        # Calculate savings
        quantity_difference = current_quantity - optimal_quantity
        cost_savings = quantity_difference * unit_cost * holding_cost_rate if quantity_difference > 0 else 0
        
        return {
            'item_id': item_id,
            'current_quantity': current_quantity,
            'optimal_quantity': optimal_quantity,
            'quantity_reduction': max(0, quantity_difference),
            'reorder_point': reorder_point,
            'order_quantity': eoq,
            'safety_stock': safety_stock,
            'cost_savings': cost_savings,
            'service_level_expected': service_level
        }
        
    def _generate_inventory_monitoring_recommendations(self) -> List[Dict[str, Any]]:
        """Generate inventory monitoring recommendations"""
        
        return [
            {
                'metric': 'Inventory Turnover',
                'target': '>6 turns per year',
                'frequency': 'Monthly',
                'action': 'Adjust order quantities if turnover is too low'
            },
            {
                'metric': 'Stock-out Rate',
                'target': '<5% of orders',
                'frequency': 'Weekly',
                'action': 'Increase safety stock if stock-outs exceed target'
            },
            {
                'metric': 'Carrying Cost',
                'target': '<25% of inventory value',
                'frequency': 'Monthly',
                'action': 'Optimize inventory levels if costs are too high'
            },
            {
                'metric': 'Demand Forecast Accuracy',
                'target': '>80% accuracy',
                'frequency': 'Monthly',
                'action': 'Improve forecasting methods if accuracy is low'
            }
        ]

def test_supply_chain_intelligence_agent():
    """Test the Supply Chain Intelligence Agent"""
    print("🧪 Testing Supply Chain Intelligence Agent")
    print("=" * 50)
    
    try:
        agent = SupplyChainIntelligenceAgent()
        
        # Test data
        test_supply_chain = {
            'company_name': 'Global Manufacturing Corp',
            'on_time_delivery_rate': 88,
            'cost_variance_percentage': 12,
            'inventory_turnover': 8,
            'supplier_count': 15,
            'suppliers': [
                {
                    'id': 'SUP001',
                    'political_stability_score': 85,
                    'quality_score': 90,
                    'financial_health_score': 80,
                    'sustainability_score': 70
                },
                {
                    'id': 'SUP002',
                    'political_stability_score': 60,
                    'quality_score': 75,
                    'financial_health_score': 65,
                    'sustainability_score': 50
                }
            ]
        }
        
        # Test supply chain analysis
        analysis = agent.analyze_supply_chain_performance(test_supply_chain)
        print(f"✅ Supply chain analysis completed for {test_supply_chain['company_name']}")
        print(f"   Efficiency score: {analysis['performance_metrics']['efficiency_score']:.1f}")
        print(f"   Supplier risks identified: {len(analysis['supplier_risks'])}")
        print(f"   Optimization opportunities: {len(analysis['optimization_opportunities'])}")
        
        # Test inventory optimization
        inventory_data = {
            'inventory_items': [
                {
                    'id': 'ITEM001',
                    'current_quantity': 500,
                    'unit_cost': 25,
                    'lead_time_days': 10,
                    'ordering_cost': 100,
                    'holding_cost_rate': 0.15
                }
            ],
            'demand_forecast': {
                'ITEM001': {
                    'average_daily_demand': 15,
                    'demand_std_dev': 3
                }
            },
            'service_level_target': 95
        }
        
        inventory_optimization = agent.optimize_inventory_levels(inventory_data)
        print(f"✅ Inventory optimization completed")
        print(f"   Total cost savings: ${inventory_optimization['total_cost_savings']:,.0f}")
        print(f"   Inventory reduction: {inventory_optimization['inventory_reduction']:.0f} units")
        
        return {
            'agent_initialized': True,
            'efficiency_score': analysis['performance_metrics']['efficiency_score'],
            'supplier_risks': len(analysis['supplier_risks']),
            'cost_savings': inventory_optimization['total_cost_savings']
        }
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return {'error': str(e)}

if __name__ == "__main__":
    test_supply_chain_intelligence_agent()