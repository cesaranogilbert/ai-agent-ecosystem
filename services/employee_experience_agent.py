"""
Employee Experience & Engagement Agent
Workplace satisfaction optimization and engagement analytics
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

class EmployeeExperienceAgent:
    """
    Comprehensive Employee Experience & Engagement System
    - Satisfaction surveys and analytics
    - Engagement optimization
    - Retention prediction
    - Culture assessment
    """
    
    def __init__(self):
        self.name = "Employee Experience & Engagement Agent"
        self.version = "1.0.0"
        self.capabilities = [
            "Satisfaction Analysis",
            "Engagement Optimization",
            "Retention Prediction",
            "Culture Assessment",
            "Performance Insights",
            "Wellbeing Monitoring"
        ]
        
    def analyze_employee_experience(self, employee_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive employee experience analysis"""
        try:
            company_name = employee_data.get('company_name', 'Unknown Company')
            
            # Satisfaction analysis
            satisfaction_analysis = self._analyze_satisfaction(employee_data)
            
            # Engagement metrics
            engagement_analysis = self._analyze_engagement(employee_data)
            
            # Retention prediction
            retention_analysis = self._predict_retention(employee_data)
            
            # Culture assessment
            culture_assessment = self._assess_culture(employee_data)
            
            # Generate recommendations
            recommendations = self._generate_experience_recommendations(
                satisfaction_analysis, engagement_analysis, retention_analysis, culture_assessment
            )
            
            return {
                'company': company_name,
                'analysis_date': datetime.now().isoformat(),
                'satisfaction_analysis': satisfaction_analysis,
                'engagement_analysis': engagement_analysis,
                'retention_analysis': retention_analysis,
                'culture_assessment': culture_assessment,
                'recommendations': recommendations,
                'next_review_date': (datetime.now() + timedelta(days=30)).isoformat()
            }
            
        except Exception as e:
            logging.error(f"Employee experience analysis failed: {str(e)}")
            return {'error': f'Employee experience analysis failed: {str(e)}'}
            
    def _analyze_satisfaction(self, employee_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze employee satisfaction"""
        employees = employee_data.get('employees', [])
        
        if not employees:
            return {'overall_satisfaction': 0, 'satisfaction_categories': {}}
            
        # Calculate satisfaction scores
        satisfaction_scores = [emp.get('satisfaction_score', 70) for emp in employees]
        overall_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores)
        
        # Category-wise satisfaction
        categories = {
            'compensation': [emp.get('compensation_satisfaction', 70) for emp in employees],
            'work_life_balance': [emp.get('work_life_balance', 70) for emp in employees],
            'career_growth': [emp.get('career_growth_satisfaction', 70) for emp in employees],
            'management': [emp.get('management_satisfaction', 70) for emp in employees],
            'workplace_culture': [emp.get('culture_satisfaction', 70) for emp in employees]
        }
        
        category_averages = {}
        for category, scores in categories.items():
            category_averages[category] = sum(scores) / len(scores)
            
        # Identify satisfaction trends
        satisfaction_distribution = {
            'highly_satisfied': len([s for s in satisfaction_scores if s >= 85]),
            'satisfied': len([s for s in satisfaction_scores if 70 <= s < 85]),
            'neutral': len([s for s in satisfaction_scores if 50 <= s < 70]),
            'dissatisfied': len([s for s in satisfaction_scores if s < 50])
        }
        
        return {
            'overall_satisfaction': overall_satisfaction,
            'satisfaction_categories': category_averages,
            'satisfaction_distribution': satisfaction_distribution,
            'satisfaction_level': 'High' if overall_satisfaction >= 80 else 'Medium' if overall_satisfaction >= 65 else 'Low',
            'improvement_priorities': self._identify_satisfaction_priorities(category_averages)
        }
        
    def _identify_satisfaction_priorities(self, categories: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify satisfaction improvement priorities"""
        priorities = []
        
        for category, score in categories.items():
            if score < 65:
                priority = {
                    'category': category,
                    'current_score': score,
                    'priority_level': 'High' if score < 55 else 'Medium',
                    'improvement_potential': 80 - score
                }
                priorities.append(priority)
                
        return sorted(priorities, key=lambda x: x['improvement_potential'], reverse=True)
        
    def _analyze_engagement(self, employee_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze employee engagement"""
        employees = employee_data.get('employees', [])
        
        if not employees:
            return {'overall_engagement': 0}
            
        # Engagement factors
        engagement_factors = {
            'job_enthusiasm': [emp.get('job_enthusiasm', 70) for emp in employees],
            'commitment_level': [emp.get('commitment_level', 70) for emp in employees],
            'advocacy_score': [emp.get('advocacy_score', 70) for emp in employees],
            'productivity_score': [emp.get('productivity_score', 70) for emp in employees]
        }
        
        # Calculate engagement scores
        factor_averages = {}
        for factor, scores in engagement_factors.items():
            factor_averages[factor] = sum(scores) / len(scores)
            
        overall_engagement = sum(factor_averages.values()) / len(factor_averages)
        
        # Engagement levels
        engagement_levels = {
            'highly_engaged': len([emp for emp in employees if emp.get('engagement_score', 70) >= 85]),
            'engaged': len([emp for emp in employees if 70 <= emp.get('engagement_score', 70) < 85]),
            'somewhat_engaged': len([emp for emp in employees if 50 <= emp.get('engagement_score', 70) < 70]),
            'disengaged': len([emp for emp in employees if emp.get('engagement_score', 70) < 50])
        }
        
        return {
            'overall_engagement': overall_engagement,
            'engagement_factors': factor_averages,
            'engagement_distribution': engagement_levels,
            'engagement_level': 'High' if overall_engagement >= 80 else 'Medium' if overall_engagement >= 65 else 'Low',
            'engagement_drivers': self._identify_engagement_drivers(factor_averages)
        }
        
    def _identify_engagement_drivers(self, factors: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify key engagement drivers"""
        drivers = []
        
        # Top performing factors
        top_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)
        
        for factor, score in top_factors:
            if score >= 75:
                drivers.append({
                    'driver': factor,
                    'score': score,
                    'impact': 'Positive',
                    'recommendation': f'Leverage {factor} as engagement strength'
                })
            elif score < 60:
                drivers.append({
                    'driver': factor,
                    'score': score,
                    'impact': 'Negative',
                    'recommendation': f'Focus improvement efforts on {factor}'
                })
                
        return drivers
        
    def _predict_retention(self, employee_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict employee retention"""
        employees = employee_data.get('employees', [])
        
        if not employees:
            return {'overall_retention_risk': 0}
            
        # Calculate retention risk for each employee
        high_risk_employees = []
        retention_scores = []
        
        for emp in employees:
            risk_factors = {
                'satisfaction': 100 - emp.get('satisfaction_score', 70),
                'engagement': 100 - emp.get('engagement_score', 70),
                'tenure': min(50, max(0, 60 - emp.get('tenure_months', 24))),
                'career_growth': 100 - emp.get('career_growth_satisfaction', 70),
                'compensation': 100 - emp.get('compensation_satisfaction', 70)
            }
            
            risk_score = sum(risk_factors.values()) / len(risk_factors)
            retention_scores.append(100 - risk_score)
            
            if risk_score >= 60:  # High risk threshold
                high_risk_employees.append({
                    'employee_id': emp.get('id', 'unknown'),
                    'name': emp.get('name', 'Unknown'),
                    'department': emp.get('department', 'Unknown'),
                    'risk_score': risk_score,
                    'key_risk_factors': [factor for factor, score in risk_factors.items() if score >= 30],
                    'recommended_actions': self._suggest_retention_actions(risk_factors)
                })
                
        overall_retention = sum(retention_scores) / len(retention_scores)
        
        return {
            'overall_retention_score': overall_retention,
            'retention_level': 'High' if overall_retention >= 80 else 'Medium' if overall_retention >= 65 else 'Low',
            'high_risk_employees': high_risk_employees,
            'at_risk_count': len(high_risk_employees),
            'retention_strategies': self._generate_retention_strategies(high_risk_employees)
        }
        
    def _suggest_retention_actions(self, risk_factors: Dict[str, float]) -> List[str]:
        """Suggest retention actions based on risk factors"""
        actions = []
        
        for factor, score in risk_factors.items():
            if score >= 30:  # Significant risk factor
                if factor == 'satisfaction':
                    actions.append('Conduct stay interview to understand concerns')
                elif factor == 'engagement':
                    actions.append('Provide challenging projects and recognition')
                elif factor == 'career_growth':
                    actions.append('Discuss career development opportunities')
                elif factor == 'compensation':
                    actions.append('Review compensation and benefits package')
                elif factor == 'tenure':
                    actions.append('Provide onboarding support and mentoring')
                    
        return actions
        
    def _generate_retention_strategies(self, high_risk_employees: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate overall retention strategies"""
        strategies = []
        
        if len(high_risk_employees) > 0:
            # Common risk factors
            all_risk_factors = []
            for emp in high_risk_employees:
                all_risk_factors.extend(emp['key_risk_factors'])
                
            factor_counts = {}
            for factor in all_risk_factors:
                factor_counts[factor] = factor_counts.get(factor, 0) + 1
                
            # Top risk factors
            top_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            
            for factor, count in top_factors:
                strategies.append({
                    'strategy': f'Address {factor} issues',
                    'affected_employees': count,
                    'priority': 'High',
                    'timeline': '1-3 months'
                })
                
        return strategies
        
    def _assess_culture(self, employee_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess organizational culture"""
        employees = employee_data.get('employees', [])
        
        if not employees:
            return {'culture_score': 0}
            
        # Culture dimensions
        culture_dimensions = {
            'collaboration': [emp.get('collaboration_score', 70) for emp in employees],
            'innovation': [emp.get('innovation_culture', 70) for emp in employees],
            'diversity_inclusion': [emp.get('diversity_inclusion', 70) for emp in employees],
            'leadership_trust': [emp.get('leadership_trust', 70) for emp in employees],
            'learning_development': [emp.get('learning_opportunities', 70) for emp in employees],
            'work_environment': [emp.get('work_environment', 70) for emp in employees]
        }
        
        # Calculate dimension averages
        dimension_scores = {}
        for dimension, scores in culture_dimensions.items():
            dimension_scores[dimension] = sum(scores) / len(scores)
            
        overall_culture = sum(dimension_scores.values()) / len(dimension_scores)
        
        # Culture strengths and gaps
        strengths = [dim for dim, score in dimension_scores.items() if score >= 80]
        gaps = [dim for dim, score in dimension_scores.items() if score < 65]
        
        return {
            'overall_culture_score': overall_culture,
            'culture_dimensions': dimension_scores,
            'culture_level': 'Strong' if overall_culture >= 80 else 'Developing' if overall_culture >= 65 else 'Needs Improvement',
            'culture_strengths': strengths,
            'culture_gaps': gaps,
            'culture_improvement_plan': self._create_culture_improvement_plan(gaps, dimension_scores)
        }
        
    def _create_culture_improvement_plan(self, gaps: List[str], dimension_scores: Dict[str, float]) -> List[Dict[str, Any]]:
        """Create culture improvement plan"""
        improvement_plan = []
        
        for gap in gaps:
            score = dimension_scores[gap]
            improvement_plan.append({
                'dimension': gap,
                'current_score': score,
                'target_score': 75,
                'improvement_needed': 75 - score,
                'recommended_initiatives': self._get_culture_initiatives(gap),
                'timeline': '6-12 months'
            })
            
        return improvement_plan
        
    def _get_culture_initiatives(self, dimension: str) -> List[str]:
        """Get culture improvement initiatives for dimension"""
        initiatives_map = {
            'collaboration': [
                'Implement cross-functional team projects',
                'Create collaboration tools and spaces',
                'Establish team-building activities'
            ],
            'innovation': [
                'Launch innovation challenges',
                'Create idea management platform',
                'Allocate time for innovation projects'
            ],
            'diversity_inclusion': [
                'Implement diversity training programs',
                'Create employee resource groups',
                'Review hiring and promotion practices'
            ],
            'leadership_trust': [
                'Increase leadership transparency',
                'Implement regular town halls',
                'Provide leadership development training'
            ],
            'learning_development': [
                'Expand learning and development programs',
                'Create mentoring programs',
                'Offer tuition reimbursement'
            ],
            'work_environment': [
                'Improve physical workspace',
                'Enhance remote work policies',
                'Focus on work-life balance initiatives'
            ]
        }
        
        return initiatives_map.get(dimension, ['Develop targeted improvement initiatives'])
        
    def _generate_experience_recommendations(self, satisfaction: Dict[str, Any], 
                                           engagement: Dict[str, Any], 
                                           retention: Dict[str, Any], 
                                           culture: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate employee experience recommendations"""
        
        recommendations = []
        
        # Satisfaction recommendations
        if satisfaction['satisfaction_level'] == 'Low':
            recommendations.append({
                'category': 'Satisfaction Improvement',
                'priority': 'High',
                'recommendation': 'Address low employee satisfaction through targeted interventions',
                'actions': [
                    f'Focus on {satisfaction["improvement_priorities"][0]["category"]} improvements',
                    'Conduct comprehensive employee feedback sessions',
                    'Implement satisfaction monitoring dashboard',
                    'Create employee satisfaction task force'
                ],
                'expected_impact': 'Improve satisfaction by 20-30%'
            })
            
        # Engagement recommendations
        if engagement['engagement_level'] == 'Low':
            recommendations.append({
                'category': 'Engagement Enhancement',
                'priority': 'High',
                'recommendation': 'Boost employee engagement through strategic initiatives',
                'actions': [
                    'Implement recognition and rewards programs',
                    'Enhance communication and feedback mechanisms',
                    'Provide more autonomy and decision-making authority',
                    'Create career development pathways'
                ],
                'expected_impact': 'Increase engagement scores by 15-25%'
            })
            
        # Retention recommendations
        if retention['at_risk_count'] > 0:
            recommendations.append({
                'category': 'Retention Management',
                'priority': 'Critical',
                'recommendation': f'Implement immediate retention strategies for {retention["at_risk_count"]} at-risk employees',
                'actions': [
                    'Conduct stay interviews with high-risk employees',
                    'Develop personalized retention plans',
                    'Address systemic retention issues',
                    'Implement proactive retention monitoring'
                ],
                'expected_impact': 'Reduce turnover risk by 40-60%'
            })
            
        # Culture recommendations
        if culture['culture_level'] == 'Needs Improvement':
            recommendations.append({
                'category': 'Culture Development',
                'priority': 'Medium',
                'recommendation': 'Strengthen organizational culture through comprehensive initiatives',
                'actions': [
                    'Implement culture transformation program',
                    'Align leadership behaviors with desired culture',
                    'Create culture measurement and feedback systems',
                    'Celebrate and reinforce positive cultural behaviors'
                ],
                'expected_impact': 'Improve culture score by 10-20%'
            })
            
        return recommendations

def test_employee_experience_agent():
    """Test the Employee Experience Agent"""
    print("🧪 Testing Employee Experience & Engagement Agent")
    print("=" * 55)
    
    try:
        agent = EmployeeExperienceAgent()
        
        test_data = {
            'company_name': 'PeopleFirst Corp',
            'employees': [
                {
                    'id': 'EMP001',
                    'name': 'John Smith',
                    'department': 'Engineering',
                    'satisfaction_score': 65,
                    'engagement_score': 70,
                    'tenure_months': 18,
                    'compensation_satisfaction': 60,
                    'career_growth_satisfaction': 55
                },
                {
                    'id': 'EMP002',
                    'name': 'Jane Doe',
                    'department': 'Marketing',
                    'satisfaction_score': 85,
                    'engagement_score': 88,
                    'tenure_months': 36,
                    'compensation_satisfaction': 80,
                    'career_growth_satisfaction': 85
                }
            ]
        }
        
        analysis = agent.analyze_employee_experience(test_data)
        print(f"✅ Employee experience analysis completed for {test_data['company_name']}")
        print(f"   Overall satisfaction: {analysis['satisfaction_analysis']['overall_satisfaction']:.1f}")
        print(f"   Engagement level: {analysis['engagement_analysis']['engagement_level']}")
        print(f"   At-risk employees: {analysis['retention_analysis']['at_risk_count']}")
        
        return {
            'agent_initialized': True,
            'employees_analyzed': len(test_data['employees']),
            'satisfaction_score': analysis['satisfaction_analysis']['overall_satisfaction']
        }
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return {'error': str(e)}

if __name__ == "__main__":
    test_employee_experience_agent()