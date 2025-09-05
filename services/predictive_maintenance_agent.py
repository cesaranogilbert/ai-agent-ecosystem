"""
Predictive Maintenance Intelligence Agent
Equipment failure prediction and maintenance optimization
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class MaintenanceAlert:
    equipment_id: str
    failure_probability: float
    recommended_action: str
    urgency_level: str

class PredictiveMaintenanceAgent:
    """
    Comprehensive Predictive Maintenance System
    - Equipment failure prediction
    - Maintenance optimization
    - Cost reduction analysis
    - Performance monitoring
    """
    
    def __init__(self):
        self.name = "Predictive Maintenance Intelligence Agent"
        self.version = "1.0.0"
        self.capabilities = [
            "Failure Prediction",
            "Maintenance Scheduling",
            "Cost Optimization",
            "Performance Analytics",
            "Resource Planning",
            "Risk Assessment"
        ]
        
    def analyze_equipment_health(self, equipment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive equipment health analysis"""
        try:
            company_name = equipment_data.get('company_name', 'Unknown Company')
            
            # Equipment health assessment
            health_assessment = self._assess_equipment_health(equipment_data)
            
            # Failure prediction
            failure_predictions = self._predict_equipment_failures(equipment_data)
            
            # Maintenance optimization
            maintenance_optimization = self._optimize_maintenance_schedule(equipment_data, failure_predictions)
            
            # Generate recommendations
            recommendations = self._generate_maintenance_recommendations(
                health_assessment, failure_predictions, maintenance_optimization
            )
            
            return {
                'company': company_name,
                'analysis_date': datetime.now().isoformat(),
                'health_assessment': health_assessment,
                'failure_predictions': failure_predictions,
                'maintenance_optimization': maintenance_optimization,
                'recommendations': recommendations,
                'next_review_date': (datetime.now() + timedelta(days=7)).isoformat()
            }
            
        except Exception as e:
            logging.error(f"Predictive maintenance analysis failed: {str(e)}")
            return {'error': f'Predictive maintenance analysis failed: {str(e)}'}
            
    def _assess_equipment_health(self, equipment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall equipment health"""
        equipment_list = equipment_data.get('equipment', [])
        
        health_scores = []
        critical_equipment = []
        
        for equipment in equipment_list:
            health_score = self._calculate_equipment_health_score(equipment)
            health_scores.append(health_score)
            
            if health_score < 60:
                critical_equipment.append({
                    'equipment_id': equipment.get('id', 'unknown'),
                    'name': equipment.get('name', 'Unknown'),
                    'health_score': health_score,
                    'criticality': equipment.get('criticality', 'Medium')
                })
                
        avg_health = sum(health_scores) / len(health_scores) if health_scores else 0
        
        return {
            'average_health_score': avg_health,
            'total_equipment': len(equipment_list),
            'critical_equipment_count': len(critical_equipment),
            'critical_equipment': critical_equipment,
            'overall_status': 'Good' if avg_health >= 80 else 'Fair' if avg_health >= 60 else 'Poor'
        }
        
    def _calculate_equipment_health_score(self, equipment: Dict[str, Any]) -> float:
        """Calculate health score for individual equipment"""
        factors = {
            'age': 100 - min(100, equipment.get('age_years', 5) * 10),  # Newer is better
            'usage': 100 - equipment.get('usage_hours_percentage', 50),  # Lower usage is better
            'maintenance_compliance': equipment.get('maintenance_compliance', 80),
            'performance_efficiency': equipment.get('performance_efficiency', 85),
            'vibration_levels': 100 - equipment.get('abnormal_vibration_score', 20),
            'temperature_variance': 100 - equipment.get('temperature_variance_score', 15)
        }
        
        # Weighted average
        weights = {
            'age': 0.15,
            'usage': 0.20,
            'maintenance_compliance': 0.25,
            'performance_efficiency': 0.20,
            'vibration_levels': 0.10,
            'temperature_variance': 0.10
        }
        
        health_score = sum(factors[factor] * weights[factor] for factor in factors)
        
        return max(0, min(100, health_score))
        
    def _predict_equipment_failures(self, equipment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict equipment failures"""
        equipment_list = equipment_data.get('equipment', [])
        
        failure_predictions = []
        high_risk_equipment = []
        
        for equipment in equipment_list:
            failure_prob = self._calculate_failure_probability(equipment)
            days_to_failure = self._estimate_days_to_failure(equipment, failure_prob)
            
            prediction = {
                'equipment_id': equipment.get('id', 'unknown'),
                'equipment_name': equipment.get('name', 'Unknown'),
                'failure_probability': failure_prob,
                'days_to_potential_failure': days_to_failure,
                'risk_level': self._categorize_risk_level(failure_prob),
                'recommended_action': self._recommend_action(failure_prob, days_to_failure)
            }
            
            failure_predictions.append(prediction)
            
            if failure_prob >= 70:
                high_risk_equipment.append(prediction)
                
        return {
            'failure_predictions': failure_predictions,
            'high_risk_equipment': high_risk_equipment,
            'immediate_attention_needed': len([p for p in failure_predictions if p['failure_probability'] >= 85]),
            'average_failure_probability': sum(p['failure_probability'] for p in failure_predictions) / len(failure_predictions) if failure_predictions else 0
        }
        
    def _calculate_failure_probability(self, equipment: Dict[str, Any]) -> float:
        """Calculate failure probability for equipment"""
        risk_factors = {
            'age_factor': min(100, equipment.get('age_years', 5) * 8),  # 8% per year
            'usage_factor': equipment.get('usage_hours_percentage', 50) * 0.6,
            'maintenance_factor': max(0, 80 - equipment.get('maintenance_compliance', 80)),
            'performance_degradation': max(0, 90 - equipment.get('performance_efficiency', 85)) * 2,
            'sensor_anomalies': equipment.get('sensor_anomaly_score', 20),
            'historical_failures': equipment.get('historical_failure_rate', 10) * 3
        }
        
        total_risk = sum(risk_factors.values())
        failure_probability = min(95, total_risk / 6)  # Average and cap at 95%
        
        return failure_probability
        
    def _estimate_days_to_failure(self, equipment: Dict[str, Any], failure_prob: float) -> int:
        """Estimate days until potential failure"""
        if failure_prob >= 90:
            return 7  # 1 week
        elif failure_prob >= 80:
            return 14  # 2 weeks
        elif failure_prob >= 70:
            return 30  # 1 month
        elif failure_prob >= 60:
            return 60  # 2 months
        else:
            return 90  # 3+ months
            
    def _categorize_risk_level(self, failure_prob: float) -> str:
        """Categorize risk level"""
        if failure_prob >= 85:
            return 'Critical'
        elif failure_prob >= 70:
            return 'High'
        elif failure_prob >= 50:
            return 'Medium'
        else:
            return 'Low'
            
    def _recommend_action(self, failure_prob: float, days_to_failure: int) -> str:
        """Recommend maintenance action"""
        if failure_prob >= 85:
            return 'Immediate maintenance required'
        elif failure_prob >= 70:
            return 'Schedule maintenance within 1-2 weeks'
        elif failure_prob >= 50:
            return 'Plan maintenance within next month'
        else:
            return 'Continue routine monitoring'
            
    def _optimize_maintenance_schedule(self, equipment_data: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize maintenance scheduling"""
        
        # Extract high-priority maintenance needs
        urgent_maintenance = []
        scheduled_maintenance = []
        
        for prediction in predictions['failure_predictions']:
            if prediction['risk_level'] in ['Critical', 'High']:
                urgent_maintenance.append({
                    'equipment_id': prediction['equipment_id'],
                    'priority': prediction['risk_level'],
                    'recommended_date': datetime.now() + timedelta(days=min(7, prediction['days_to_potential_failure'])),
                    'estimated_duration': self._estimate_maintenance_duration(prediction['equipment_id'], equipment_data),
                    'estimated_cost': self._estimate_maintenance_cost(prediction['equipment_id'], equipment_data)
                })
            else:
                scheduled_maintenance.append({
                    'equipment_id': prediction['equipment_id'],
                    'priority': prediction['risk_level'],
                    'recommended_date': datetime.now() + timedelta(days=prediction['days_to_potential_failure']),
                    'estimated_duration': self._estimate_maintenance_duration(prediction['equipment_id'], equipment_data),
                    'estimated_cost': self._estimate_maintenance_cost(prediction['equipment_id'], equipment_data)
                })
                
        # Calculate resource requirements
        total_cost = sum(m['estimated_cost'] for m in urgent_maintenance + scheduled_maintenance)
        total_hours = sum(m['estimated_duration'] for m in urgent_maintenance + scheduled_maintenance)
        
        return {
            'urgent_maintenance': urgent_maintenance,
            'scheduled_maintenance': scheduled_maintenance,
            'total_estimated_cost': total_cost,
            'total_estimated_hours': total_hours,
            'resource_optimization': self._optimize_maintenance_resources(urgent_maintenance, scheduled_maintenance)
        }
        
    def _estimate_maintenance_duration(self, equipment_id: str, equipment_data: Dict[str, Any]) -> float:
        """Estimate maintenance duration in hours"""
        equipment_list = equipment_data.get('equipment', [])
        
        for equipment in equipment_list:
            if equipment.get('id') == equipment_id:
                complexity = equipment.get('maintenance_complexity', 'Medium')
                if complexity == 'High':
                    return 16.0  # 2 days
                elif complexity == 'Medium':
                    return 8.0   # 1 day
                else:
                    return 4.0   # 0.5 day
                    
        return 8.0  # Default
        
    def _estimate_maintenance_cost(self, equipment_id: str, equipment_data: Dict[str, Any]) -> float:
        """Estimate maintenance cost"""
        equipment_list = equipment_data.get('equipment', [])
        
        for equipment in equipment_list:
            if equipment.get('id') == equipment_id:
                base_cost = equipment.get('maintenance_base_cost', 5000)
                criticality_multiplier = {'High': 1.5, 'Medium': 1.0, 'Low': 0.7}.get(equipment.get('criticality', 'Medium'), 1.0)
                return base_cost * criticality_multiplier
                
        return 5000  # Default
        
    def _optimize_maintenance_resources(self, urgent: List[Dict[str, Any]], scheduled: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize maintenance resource allocation"""
        
        # Calculate weekly resource needs
        urgent_hours_week1 = sum(m['estimated_duration'] for m in urgent)
        
        # Suggest resource optimization
        suggestions = []
        
        if urgent_hours_week1 > 40:  # More than 1 person-week
            suggestions.append('Consider hiring temporary maintenance staff or contractors')
            
        if len(urgent) > 5:
            suggestions.append('Prioritize critical equipment and defer lower-priority items')
            
        if not suggestions:
            suggestions.append('Current resource allocation appears adequate')
            
        return {
            'urgent_hours_next_week': urgent_hours_week1,
            'scheduled_hours_next_month': sum(m['estimated_duration'] for m in scheduled),
            'resource_optimization_suggestions': suggestions,
            'recommended_team_size': max(1, int(urgent_hours_week1 / 40) + 1)
        }
        
    def _generate_maintenance_recommendations(self, health: Dict[str, Any], 
                                           predictions: Dict[str, Any], 
                                           optimization: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate maintenance recommendations"""
        
        recommendations = []
        
        # Critical equipment recommendations
        if predictions['immediate_attention_needed'] > 0:
            recommendations.append({
                'category': 'Immediate Action',
                'priority': 'Critical',
                'recommendation': f'Address {predictions["immediate_attention_needed"]} critical equipment immediately',
                'actions': [
                    'Perform emergency maintenance on critical equipment',
                    'Prepare backup equipment if available',
                    'Monitor equipment continuously until maintenance'
                ],
                'timeline': '1-3 days',
                'estimated_cost': sum(m['estimated_cost'] for m in optimization['urgent_maintenance'])
            })
            
        # Preventive maintenance recommendations
        if health['critical_equipment_count'] > 0:
            recommendations.append({
                'category': 'Preventive Maintenance',
                'priority': 'High',
                'recommendation': 'Implement enhanced preventive maintenance program',
                'actions': [
                    'Increase monitoring frequency for critical equipment',
                    'Implement condition-based maintenance',
                    'Review and update maintenance procedures'
                ],
                'timeline': '1-4 weeks',
                'estimated_cost': optimization['total_estimated_cost'] * 0.3
            })
            
        # Optimization recommendations
        recommendations.append({
            'category': 'Maintenance Optimization',
            'priority': 'Medium',
            'recommendation': 'Optimize maintenance operations',
            'actions': [
                'Implement predictive maintenance technologies',
                'Train maintenance staff on new procedures',
                'Establish maintenance performance metrics'
            ],
            'timeline': '3-6 months',
            'estimated_cost': 50000
        })
        
        return recommendations

def test_predictive_maintenance_agent():
    """Test the Predictive Maintenance Agent"""
    print("🧪 Testing Predictive Maintenance Intelligence Agent")
    print("=" * 55)
    
    try:
        agent = PredictiveMaintenanceAgent()
        
        test_data = {
            'company_name': 'Manufacturing Pro Inc',
            'equipment': [
                {
                    'id': 'EQ001',
                    'name': 'Production Line A',
                    'age_years': 8,
                    'usage_hours_percentage': 85,
                    'maintenance_compliance': 70,
                    'performance_efficiency': 75,
                    'criticality': 'High',
                    'maintenance_base_cost': 15000
                },
                {
                    'id': 'EQ002',
                    'name': 'Conveyor System B',
                    'age_years': 3,
                    'usage_hours_percentage': 60,
                    'maintenance_compliance': 90,
                    'performance_efficiency': 92,
                    'criticality': 'Medium',
                    'maintenance_base_cost': 8000
                }
            ]
        }
        
        analysis = agent.analyze_equipment_health(test_data)
        print(f"✅ Equipment analysis completed for {test_data['company_name']}")
        print(f"   Equipment analyzed: {analysis['health_assessment']['total_equipment']}")
        print(f"   Critical equipment: {analysis['health_assessment']['critical_equipment_count']}")
        print(f"   High-risk failures: {len(analysis['failure_predictions']['high_risk_equipment'])}")
        
        return {
            'agent_initialized': True,
            'equipment_analyzed': analysis['health_assessment']['total_equipment'],
            'predictions_generated': len(analysis['failure_predictions']['failure_predictions'])
        }
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return {'error': str(e)}

if __name__ == "__main__":
    test_predictive_maintenance_agent()