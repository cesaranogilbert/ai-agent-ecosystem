"""
Crisis Management & Response Agent
Crisis scenario simulation and response optimization
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class CrisisScenario:
    scenario_id: str
    crisis_type: str
    severity_level: str
    probability: float
    impact_score: float

class CrisisManagementResponseAgent:
    """
    Comprehensive Crisis Management & Response System
    - Crisis scenario simulation
    - Stakeholder communication optimization
    - Business continuity planning
    - Reputation risk management
    """
    
    def __init__(self):
        self.name = "Crisis Management & Response Agent"
        self.version = "1.0.0"
        self.capabilities = [
            "Crisis Scenario Planning",
            "Response Strategy Development",
            "Stakeholder Communication",
            "Business Continuity Planning",
            "Reputation Management",
            "Recovery Planning"
        ]
        
        # Crisis types and response frameworks
        self.crisis_types = {
            'cybersecurity': {
                'typical_duration': '1-7 days',
                'stakeholders': ['customers', 'regulators', 'employees', 'media'],
                'response_priorities': ['containment', 'communication', 'investigation', 'recovery']
            },
            'financial': {
                'typical_duration': '2-8 weeks',
                'stakeholders': ['investors', 'customers', 'employees', 'regulators'],
                'response_priorities': ['stakeholder_communication', 'financial_stabilization', 'operational_continuity']
            },
            'operational': {
                'typical_duration': '1-4 weeks',
                'stakeholders': ['customers', 'employees', 'suppliers', 'community'],
                'response_priorities': ['safety', 'operations_restoration', 'communication', 'prevention']
            },
            'reputational': {
                'typical_duration': '2-12 weeks',
                'stakeholders': ['customers', 'media', 'community', 'employees'],
                'response_priorities': ['communication', 'corrective_action', 'relationship_rebuilding']
            }
        }
        
    def develop_crisis_response_plan(self, crisis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Develop comprehensive crisis response plan"""
        try:
            company_name = crisis_data.get('company_name', 'Unknown Company')
            
            # Analyze potential crisis scenarios
            scenario_analysis = self._analyze_crisis_scenarios(crisis_data)
            
            # Develop response strategies
            response_strategies = self._develop_response_strategies(scenario_analysis)
            
            # Create communication plans
            communication_plans = self._create_communication_plans(crisis_data, scenario_analysis)
            
            # Business continuity planning
            continuity_plan = self._develop_continuity_plan(crisis_data)
            
            # Generate recommendations
            recommendations = self._generate_crisis_recommendations(
                scenario_analysis, response_strategies, continuity_plan
            )
            
            return {
                'company': company_name,
                'plan_date': datetime.now().isoformat(),
                'scenario_analysis': scenario_analysis,
                'response_strategies': response_strategies,
                'communication_plans': communication_plans,
                'continuity_plan': continuity_plan,
                'recommendations': recommendations,
                'next_review_date': (datetime.now() + timedelta(days=90)).isoformat()
            }
            
        except Exception as e:
            logging.error(f"Crisis response planning failed: {str(e)}")
            return {'error': f'Crisis response planning failed: {str(e)}'}
            
    def _analyze_crisis_scenarios(self, crisis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze potential crisis scenarios"""
        industry = crisis_data.get('industry', 'general').lower()
        company_size = crisis_data.get('company_size', 'medium')
        
        # Identify relevant crisis scenarios
        relevant_scenarios = self._identify_relevant_scenarios(industry, company_size)
        
        # Calculate risk scores
        scenario_risks = []
        for scenario in relevant_scenarios:
            risk_assessment = self._assess_scenario_risk(scenario, crisis_data)
            scenario_risks.append(risk_assessment)
            
        # Sort by risk level
        scenario_risks.sort(key=lambda x: x['risk_score'], reverse=True)
        
        return {
            'high_risk_scenarios': [s for s in scenario_risks if s['risk_score'] >= 70],
            'medium_risk_scenarios': [s for s in scenario_risks if 40 <= s['risk_score'] < 70],
            'low_risk_scenarios': [s for s in scenario_risks if s['risk_score'] < 40],
            'overall_risk_profile': self._calculate_overall_risk_profile(scenario_risks)
        }
        
    def _identify_relevant_scenarios(self, industry: str, company_size: str) -> List[Dict[str, Any]]:
        """Identify crisis scenarios relevant to company"""
        
        base_scenarios = [
            {'type': 'cybersecurity', 'name': 'Data Breach', 'base_probability': 25},
            {'type': 'operational', 'name': 'Supply Chain Disruption', 'base_probability': 30},
            {'type': 'financial', 'name': 'Cash Flow Crisis', 'base_probability': 15},
            {'type': 'reputational', 'name': 'Public Relations Crisis', 'base_probability': 20},
            {'type': 'operational', 'name': 'Key Personnel Loss', 'base_probability': 35},
            {'type': 'cybersecurity', 'name': 'Ransomware Attack', 'base_probability': 20}
        ]
        
        # Adjust probabilities based on industry and size
        industry_adjustments = {
            'technology': {'cybersecurity': 1.5, 'operational': 0.8},
            'manufacturing': {'operational': 1.4, 'cybersecurity': 0.9},
            'financial': {'financial': 1.3, 'reputational': 1.2},
            'healthcare': {'reputational': 1.4, 'cybersecurity': 1.3}
        }
        
        size_adjustments = {
            'small': {'financial': 1.3, 'operational': 1.2},
            'large': {'reputational': 1.2, 'cybersecurity': 1.1}
        }
        
        adjusted_scenarios = []
        for scenario in base_scenarios:
            adjusted_prob = scenario['base_probability']
            
            # Apply industry adjustments
            if industry in industry_adjustments:
                adjustment = industry_adjustments[industry].get(scenario['type'], 1.0)
                adjusted_prob *= adjustment
                
            # Apply size adjustments
            if company_size in size_adjustments:
                adjustment = size_adjustments[company_size].get(scenario['type'], 1.0)
                adjusted_prob *= adjustment
                
            scenario['adjusted_probability'] = min(80, adjusted_prob)  # Cap at 80%
            adjusted_scenarios.append(scenario)
            
        return adjusted_scenarios
        
    def _assess_scenario_risk(self, scenario: Dict[str, Any], crisis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk level for specific scenario"""
        
        probability = scenario['adjusted_probability']
        
        # Calculate impact based on scenario type and company factors
        impact_factors = {
            'revenue_at_risk': crisis_data.get('annual_revenue', 10000000) * 0.1,  # 10% revenue impact
            'customer_base': crisis_data.get('customer_count', 1000),
            'employee_count': crisis_data.get('employee_count', 100),
            'regulatory_exposure': crisis_data.get('regulatory_complexity', 50)
        }
        
        # Scenario-specific impact calculation
        if scenario['type'] == 'cybersecurity':
            impact_score = min(100, (impact_factors['customer_base'] / 1000 + impact_factors['regulatory_exposure']) / 2)
        elif scenario['type'] == 'financial':
            impact_score = min(100, impact_factors['revenue_at_risk'] / 1000000 * 10)
        elif scenario['type'] == 'operational':
            impact_score = min(100, (impact_factors['employee_count'] / 10 + impact_factors['customer_base'] / 1000) / 2)
        else:  # reputational
            impact_score = min(100, (impact_factors['customer_base'] / 1000 + 50) / 2)
            
        # Calculate overall risk score
        risk_score = (probability + impact_score) / 2
        
        return {
            'scenario_name': scenario['name'],
            'scenario_type': scenario['type'],
            'probability': probability,
            'impact_score': impact_score,
            'risk_score': risk_score,
            'risk_level': self._categorize_risk_level(risk_score),
            'estimated_cost': self._estimate_crisis_cost(scenario, impact_score)
        }
        
    def _categorize_risk_level(self, risk_score: float) -> str:
        """Categorize risk level"""
        if risk_score >= 70:
            return 'High'
        elif risk_score >= 40:
            return 'Medium'
        else:
            return 'Low'
            
    def _estimate_crisis_cost(self, scenario: Dict[str, Any], impact_score: float) -> Dict[str, Any]:
        """Estimate potential cost of crisis"""
        
        base_costs = {
            'cybersecurity': 2000000,  # $2M base cost
            'financial': 5000000,     # $5M base cost
            'operational': 1500000,   # $1.5M base cost
            'reputational': 3000000   # $3M base cost
        }
        
        base_cost = base_costs.get(scenario['type'], 1000000)
        impact_multiplier = impact_score / 50  # Scale impact
        
        estimated_cost = base_cost * impact_multiplier
        
        return {
            'direct_costs': estimated_cost * 0.6,
            'indirect_costs': estimated_cost * 0.4,
            'total_estimated_cost': estimated_cost,
            'recovery_timeline': self.crisis_types[scenario['type']]['typical_duration']
        }
        
    def _calculate_overall_risk_profile(self, scenario_risks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall risk profile"""
        
        total_scenarios = len(scenario_risks)
        high_risk_count = len([s for s in scenario_risks if s['risk_level'] == 'High'])
        
        avg_risk_score = sum(s['risk_score'] for s in scenario_risks) / total_scenarios if total_scenarios > 0 else 0
        
        overall_risk_level = 'High' if avg_risk_score >= 60 else 'Medium' if avg_risk_score >= 35 else 'Low'
        
        return {
            'overall_risk_level': overall_risk_level,
            'average_risk_score': avg_risk_score,
            'high_risk_scenario_count': high_risk_count,
            'crisis_preparedness_needed': high_risk_count > 0 or avg_risk_score >= 50
        }
        
    def _develop_response_strategies(self, scenario_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Develop crisis response strategies"""
        
        strategies = {}
        
        # High-risk scenarios require detailed strategies
        for scenario in scenario_analysis['high_risk_scenarios']:
            scenario_type = scenario['scenario_type']
            
            if scenario_type in self.crisis_types:
                framework = self.crisis_types[scenario_type]
                
                strategy = {
                    'scenario': scenario['scenario_name'],
                    'response_priorities': framework['response_priorities'],
                    'stakeholders': framework['stakeholders'],
                    'response_timeline': self._create_response_timeline(framework),
                    'key_actions': self._define_key_actions(scenario_type),
                    'success_metrics': self._define_success_metrics(scenario_type)
                }
                
                strategies[scenario['scenario_name']] = strategy
                
        return {
            'detailed_strategies': strategies,
            'general_response_principles': self._define_general_response_principles(),
            'escalation_procedures': self._define_escalation_procedures()
        }
        
    def _create_response_timeline(self, framework: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create response timeline"""
        return [
            {'phase': 'Immediate (0-4 hours)', 'actions': ['Assess situation', 'Activate crisis team', 'Initial stakeholder notification']},
            {'phase': 'Short-term (4-24 hours)', 'actions': ['Implement containment', 'Detailed communication', 'Resource mobilization']},
            {'phase': 'Medium-term (1-7 days)', 'actions': ['Execute response plan', 'Monitor progress', 'Adjust strategy']},
            {'phase': 'Long-term (1+ weeks)', 'actions': ['Recovery implementation', 'Lessons learned', 'Prevention measures']}
        ]
        
    def _define_key_actions(self, scenario_type: str) -> List[str]:
        """Define key actions for scenario type"""
        action_map = {
            'cybersecurity': [
                'Isolate affected systems',
                'Notify law enforcement if required',
                'Implement incident response procedures',
                'Communicate with affected parties'
            ],
            'financial': [
                'Assess financial position',
                'Contact financial institutions',
                'Implement cost reduction measures',
                'Communicate with investors'
            ],
            'operational': [
                'Ensure employee safety',
                'Activate backup procedures',
                'Communicate with customers',
                'Implement recovery plan'
            ],
            'reputational': [
                'Assess situation accurately',
                'Develop key messages',
                'Engage with media proactively',
                'Implement corrective actions'
            ]
        }
        
        return action_map.get(scenario_type, ['Assess situation', 'Communicate with stakeholders', 'Implement response'])
        
    def _define_success_metrics(self, scenario_type: str) -> List[str]:
        """Define success metrics for crisis response"""
        return [
            'Time to containment',
            'Stakeholder satisfaction',
            'Financial impact minimization',
            'Reputation recovery speed',
            'Operational restoration time'
        ]
        
    def _define_general_response_principles(self) -> List[str]:
        """Define general crisis response principles"""
        return [
            'Safety first - prioritize people over profits',
            'Communicate early and transparently',
            'Take responsibility and show accountability',
            'Focus on solutions, not blame',
            'Learn and improve from every crisis'
        ]
        
    def _define_escalation_procedures(self) -> Dict[str, Any]:
        """Define crisis escalation procedures"""
        return {
            'level_1': 'Department manager handles local issues',
            'level_2': 'Crisis team activated for significant incidents',
            'level_3': 'Executive leadership involved for major crises',
            'level_4': 'Board notification for company-threatening events',
            'escalation_criteria': [
                'Potential financial impact > $1M',
                'Regulatory investigation likely',
                'Media attention probable',
                'Customer safety at risk'
            ]
        }
        
    def _create_communication_plans(self, crisis_data: Dict[str, Any], scenario_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create stakeholder communication plans"""
        
        stakeholder_groups = crisis_data.get('stakeholder_groups', [
            'customers', 'employees', 'investors', 'media', 'regulators', 'community'
        ])
        
        communication_templates = {}
        for group in stakeholder_groups:
            template = {
                'stakeholder_group': group,
                'communication_channels': self._get_communication_channels(group),
                'message_framework': self._create_message_framework(group),
                'update_frequency': self._determine_update_frequency(group),
                'spokesperson': self._assign_spokesperson(group)
            }
            communication_templates[group] = template
            
        return {
            'stakeholder_templates': communication_templates,
            'crisis_communication_principles': self._define_communication_principles(),
            'media_strategy': self._develop_media_strategy()
        }
        
    def _get_communication_channels(self, stakeholder_group: str) -> List[str]:
        """Get appropriate communication channels for stakeholder group"""
        channels_map = {
            'customers': ['email', 'website', 'social_media', 'customer_service'],
            'employees': ['internal_email', 'intranet', 'town_halls', 'direct_manager'],
            'investors': ['investor_relations', 'sec_filings', 'conference_calls'],
            'media': ['press_releases', 'media_interviews', 'press_conferences'],
            'regulators': ['official_correspondence', 'regulatory_filings', 'direct_contact'],
            'community': ['local_media', 'community_meetings', 'social_media']
        }
        
        return channels_map.get(stakeholder_group, ['direct_communication'])
        
    def _create_message_framework(self, stakeholder_group: str) -> Dict[str, str]:
        """Create message framework for stakeholder group"""
        return {
            'opening': f'Acknowledge the situation and impact on {stakeholder_group}',
            'facts': 'Provide clear, factual information about what happened',
            'actions': 'Explain what is being done to address the situation',
            'timeline': 'Give realistic expectations for resolution',
            'contact': 'Provide contact information for further questions',
            'closing': 'Reaffirm commitment to stakeholder interests'
        }
        
    def _determine_update_frequency(self, stakeholder_group: str) -> str:
        """Determine communication update frequency"""
        frequency_map = {
            'customers': 'Daily during active crisis',
            'employees': 'Twice daily during crisis',
            'investors': 'As material developments occur',
            'media': 'As requested or significant updates',
            'regulators': 'As required by regulation',
            'community': 'Weekly or as significant updates occur'
        }
        
        return frequency_map.get(stakeholder_group, 'As needed')
        
    def _assign_spokesperson(self, stakeholder_group: str) -> str:
        """Assign appropriate spokesperson for stakeholder group"""
        spokesperson_map = {
            'customers': 'Customer Service Director',
            'employees': 'Chief Human Resources Officer',
            'investors': 'Chief Financial Officer',
            'media': 'Chief Communications Officer',
            'regulators': 'Chief Legal Officer',
            'community': 'Chief Executive Officer'
        }
        
        return spokesperson_map.get(stakeholder_group, 'Crisis Team Leader')
        
    def _define_communication_principles(self) -> List[str]:
        """Define crisis communication principles"""
        return [
            'Be first, be right, be credible',
            'Communicate with empathy and concern',
            'Provide regular updates even if no new information',
            'Be consistent across all channels and stakeholders',
            'Address rumors and misinformation quickly'
        ]
        
    def _develop_media_strategy(self) -> Dict[str, Any]:
        """Develop media engagement strategy"""
        return {
            'proactive_approach': 'Engage media before they engage you',
            'key_messages': 'Develop 3-5 core messages for all communications',
            'media_training': 'Ensure spokespersons are media trained',
            'monitoring': 'Monitor media coverage and social media sentiment',
            'response_protocol': 'Respond to media inquiries within 2 hours'
        }
        
    def _develop_continuity_plan(self, crisis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Develop business continuity plan"""
        
        critical_functions = crisis_data.get('critical_business_functions', [
            'Customer service', 'Production', 'IT systems', 'Finance', 'Supply chain'
        ])
        
        continuity_strategies = {}
        for function in critical_functions:
            strategy = {
                'function': function,
                'recovery_time_objective': self._determine_rto(function),
                'backup_procedures': self._define_backup_procedures(function),
                'resource_requirements': self._identify_resource_requirements(function),
                'alternative_solutions': self._identify_alternative_solutions(function)
            }
            continuity_strategies[function] = strategy
            
        return {
            'continuity_strategies': continuity_strategies,
            'overall_recovery_plan': self._create_overall_recovery_plan(),
            'testing_schedule': 'Quarterly continuity plan testing recommended'
        }
        
    def _determine_rto(self, function: str) -> str:
        """Determine Recovery Time Objective for function"""
        rto_map = {
            'Customer service': '4 hours',
            'IT systems': '2 hours',
            'Production': '24 hours',
            'Finance': '8 hours',
            'Supply chain': '48 hours'
        }
        
        return rto_map.get(function, '24 hours')
        
    def _define_backup_procedures(self, function: str) -> List[str]:
        """Define backup procedures for business function"""
        procedures_map = {
            'Customer service': ['Activate call center backup', 'Implement remote support', 'Use partner support services'],
            'IT systems': ['Activate backup data center', 'Implement cloud failover', 'Use manual processes'],
            'Production': ['Activate alternate facility', 'Use partner manufacturing', 'Implement reduced capacity'],
            'Finance': ['Use backup financial systems', 'Implement manual processes', 'Access off-site records'],
            'Supply chain': ['Activate alternate suppliers', 'Use emergency inventory', 'Implement expedited shipping']
        }
        
        return procedures_map.get(function, ['Implement manual backup procedures'])
        
    def _identify_resource_requirements(self, function: str) -> Dict[str, Any]:
        """Identify resource requirements for continuity"""
        return {
            'personnel': 'Backup staff and emergency contacts',
            'technology': 'Backup systems and equipment',
            'facilities': 'Alternative work locations',
            'communications': 'Emergency communication systems'
        }
        
    def _identify_alternative_solutions(self, function: str) -> List[str]:
        """Identify alternative solutions for business function"""
        return [
            'Outsourcing to third parties',
            'Manual workaround procedures',
            'Reduced service level operations',
            'Partner collaboration agreements'
        ]
        
    def _create_overall_recovery_plan(self) -> Dict[str, Any]:
        """Create overall recovery plan"""
        return {
            'recovery_phases': [
                'Immediate response (0-24 hours)',
                'Short-term recovery (1-7 days)',
                'Long-term recovery (1+ weeks)',
                'Return to normal operations'
            ],
            'recovery_priorities': [
                'Ensure employee safety',
                'Restore critical functions',
                'Communicate with stakeholders',
                'Minimize financial impact'
            ],
            'success_criteria': [
                'All critical functions operational',
                'Stakeholder confidence restored',
                'Financial impact contained',
                'Lessons learned documented'
            ]
        }
        
    def _generate_crisis_recommendations(self, scenario_analysis: Dict[str, Any], 
                                       response_strategies: Dict[str, Any], 
                                       continuity_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate crisis management recommendations"""
        
        recommendations = []
        
        # High-risk scenario recommendations
        if scenario_analysis['overall_risk_profile']['crisis_preparedness_needed']:
            recommendations.append({
                'category': 'Crisis Preparedness',
                'priority': 'High',
                'recommendation': 'Implement comprehensive crisis management program',
                'actions': [
                    'Establish crisis management team',
                    'Develop detailed response procedures',
                    'Conduct regular crisis simulations'
                ],
                'timeline': '3-6 months',
                'investment': 'Medium'
            })
            
        # Communication planning recommendations
        recommendations.append({
            'category': 'Communication Readiness',
            'priority': 'Medium',
            'recommendation': 'Enhance crisis communication capabilities',
            'actions': [
                'Develop stakeholder communication templates',
                'Train spokespersons',
                'Establish media monitoring systems'
            ],
            'timeline': '2-4 months',
            'investment': 'Low-Medium'
        })
        
        # Business continuity recommendations
        recommendations.append({
            'category': 'Business Continuity',
            'priority': 'Medium',
            'recommendation': 'Strengthen business continuity planning',
            'actions': [
                'Implement backup procedures for critical functions',
                'Establish alternative supplier relationships',
                'Test continuity plans regularly'
            ],
            'timeline': '6-12 months',
            'investment': 'Medium-High'
        })
        
        return recommendations

def test_crisis_management_response_agent():
    """Test the Crisis Management & Response Agent"""
    print("🧪 Testing Crisis Management & Response Agent")
    print("=" * 50)
    
    try:
        agent = CrisisManagementResponseAgent()
        
        test_data = {
            'company_name': 'Resilient Enterprises Inc',
            'industry': 'technology',
            'company_size': 'medium',
            'annual_revenue': 50000000,
            'customer_count': 5000,
            'employee_count': 250,
            'stakeholder_groups': ['customers', 'employees', 'investors'],
            'critical_business_functions': ['Customer service', 'IT systems', 'Production']
        }
        
        plan = agent.develop_crisis_response_plan(test_data)
        print(f"✅ Crisis response plan developed for {test_data['company_name']}")
        print(f"   High-risk scenarios: {len(plan['scenario_analysis']['high_risk_scenarios'])}")
        print(f"   Response strategies: {len(plan['response_strategies']['detailed_strategies'])}")
        
        return {'agent_initialized': True, 'plan_developed': True}
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return {'error': str(e)}

if __name__ == "__main__":
    test_crisis_management_response_agent()