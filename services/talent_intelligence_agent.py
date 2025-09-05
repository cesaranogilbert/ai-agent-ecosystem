"""
Talent Intelligence & Workforce Agent
Skills gap analysis, succession planning, and workforce optimization
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

@dataclass
class SkillGap:
    skill_name: str
    current_level: float
    required_level: float
    gap_size: float
    criticality: str
    development_time: str

@dataclass
class TalentMetrics:
    workforce_efficiency: float
    skill_alignment: float
    retention_risk: float
    development_readiness: float

class TalentIntelligenceAgent:
    """
    Comprehensive Talent Intelligence System
    - Skills gap analysis and prediction
    - Executive succession planning
    - Workforce optimization modeling
    - Cultural fit assessment for acquisitions
    """
    
    def __init__(self):
        self.name = "Talent Intelligence & Workforce Agent"
        self.version = "1.0.0"
        self.capabilities = [
            "Skills Gap Analysis",
            "Executive Succession Planning",
            "Workforce Optimization",
            "Cultural Fit Assessment",
            "Talent Pipeline Development",
            "Performance Prediction"
        ]
        
        # Skills taxonomy by role and industry
        self.skills_taxonomy = {
            'technology': {
                'technical_skills': [
                    'software_development', 'cloud_architecture', 'data_science',
                    'cybersecurity', 'ai_machine_learning', 'devops'
                ],
                'leadership_skills': [
                    'strategic_thinking', 'team_management', 'innovation_leadership',
                    'digital_transformation', 'agile_methodology'
                ],
                'soft_skills': [
                    'communication', 'problem_solving', 'adaptability',
                    'collaboration', 'critical_thinking'
                ]
            },
            'finance': {
                'technical_skills': [
                    'financial_analysis', 'risk_management', 'compliance',
                    'financial_modeling', 'treasury_management', 'audit'
                ],
                'leadership_skills': [
                    'strategic_planning', 'stakeholder_management', 'decision_making',
                    'crisis_management', 'regulatory_expertise'
                ],
                'soft_skills': [
                    'analytical_thinking', 'attention_to_detail', 'integrity',
                    'communication', 'negotiation'
                ]
            }
        }
        
        # Performance indicators
        self.performance_indicators = [
            'productivity_score', 'quality_metrics', 'collaboration_score',
            'innovation_index', 'leadership_potential', 'learning_agility'
        ]
        
    def analyze_workforce_capabilities(self, workforce_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive workforce capabilities analysis
        """
        try:
            company_name = workforce_data.get('company_name', 'Unknown Company')
            industry = workforce_data.get('industry', 'general').lower()
            
            # Calculate talent metrics
            talent_metrics = self._calculate_talent_metrics(workforce_data)
            
            # Identify skills gaps
            skills_gaps = self._identify_skills_gaps(workforce_data, industry)
            
            # Assess retention risks
            retention_analysis = self._assess_retention_risks(workforce_data)
            
            # Generate succession plans
            succession_plans = self._generate_succession_plans(workforce_data)
            
            # Workforce optimization recommendations
            optimization_recommendations = self._generate_workforce_optimization(talent_metrics, skills_gaps)
            
            return {
                'company': company_name,
                'analysis_date': datetime.now().isoformat(),
                'talent_metrics': talent_metrics.__dict__,
                'skills_gaps': [gap.__dict__ for gap in skills_gaps],
                'retention_analysis': retention_analysis,
                'succession_plans': succession_plans,
                'optimization_recommendations': optimization_recommendations,
                'next_assessment_date': (datetime.now() + timedelta(days=90)).isoformat()
            }
            
        except Exception as e:
            logging.error(f"Workforce analysis failed: {str(e)}")
            return {'error': f'Workforce analysis failed: {str(e)}'}
            
    def _calculate_talent_metrics(self, workforce_data: Dict[str, Any]) -> TalentMetrics:
        """Calculate comprehensive talent metrics"""
        
        employees = workforce_data.get('employees', [])
        
        if not employees:
            return TalentMetrics(0, 0, 0, 0)
            
        # Workforce efficiency calculation
        productivity_scores = [emp.get('productivity_score', 70) for emp in employees]
        workforce_efficiency = sum(productivity_scores) / len(productivity_scores)
        
        # Skill alignment calculation
        skill_alignments = [emp.get('skill_role_alignment', 75) for emp in employees]
        skill_alignment = sum(skill_alignments) / len(skill_alignments)
        
        # Retention risk calculation (inverse of retention probability)
        retention_scores = [emp.get('retention_probability', 80) for emp in employees]
        retention_risk = 100 - (sum(retention_scores) / len(retention_scores))
        
        # Development readiness calculation
        learning_agility = [emp.get('learning_agility_score', 65) for emp in employees]
        development_readiness = sum(learning_agility) / len(learning_agility)
        
        return TalentMetrics(
            workforce_efficiency=workforce_efficiency,
            skill_alignment=skill_alignment,
            retention_risk=retention_risk,
            development_readiness=development_readiness
        )
        
    def _identify_skills_gaps(self, workforce_data: Dict[str, Any], industry: str) -> List[SkillGap]:
        """Identify critical skills gaps in the workforce"""
        
        skills_gaps = []
        employees = workforce_data.get('employees', [])
        required_skills = workforce_data.get('required_skills', {})
        
        # Get industry-specific skills
        if industry in self.skills_taxonomy:
            taxonomy = self.skills_taxonomy[industry]
            all_required_skills = (
                taxonomy.get('technical_skills', []) +
                taxonomy.get('leadership_skills', []) +
                taxonomy.get('soft_skills', [])
            )
        else:
            all_required_skills = list(required_skills.keys())
            
        for skill in all_required_skills:
            current_level = self._calculate_current_skill_level(skill, employees)
            required_level = required_skills.get(skill, 80)  # Default requirement
            gap_size = max(0, required_level - current_level)
            
            if gap_size > 10:  # Significant gap threshold
                criticality = self._assess_skill_criticality(skill, gap_size)
                development_time = self._estimate_development_time(skill, gap_size)
                
                skill_gap = SkillGap(
                    skill_name=skill,
                    current_level=current_level,
                    required_level=required_level,
                    gap_size=gap_size,
                    criticality=criticality,
                    development_time=development_time
                )
                
                skills_gaps.append(skill_gap)
                
        # Sort by criticality and gap size
        skills_gaps.sort(key=lambda x: (
            {'Critical': 3, 'High': 2, 'Medium': 1, 'Low': 0}[x.criticality],
            x.gap_size
        ), reverse=True)
        
        return skills_gaps
        
    def _calculate_current_skill_level(self, skill: str, employees: List[Dict[str, Any]]) -> float:
        """Calculate current average skill level across workforce"""
        
        skill_levels = []
        for employee in employees:
            skills = employee.get('skills', {})
            skill_level = skills.get(skill, 0)
            skill_levels.append(skill_level)
            
        return sum(skill_levels) / len(skill_levels) if skill_levels else 0
        
    def _assess_skill_criticality(self, skill: str, gap_size: float) -> str:
        """Assess criticality of skill gap"""
        
        # Critical skills that significantly impact business operations
        critical_skills = [
            'strategic_thinking', 'cybersecurity', 'ai_machine_learning',
            'risk_management', 'digital_transformation', 'leadership'
        ]
        
        if skill.lower() in [s.lower() for s in critical_skills]:
            if gap_size >= 30:
                return 'Critical'
            elif gap_size >= 20:
                return 'High'
            else:
                return 'Medium'
        else:
            if gap_size >= 40:
                return 'High'
            elif gap_size >= 25:
                return 'Medium'
            else:
                return 'Low'
                
    def _estimate_development_time(self, skill: str, gap_size: float) -> str:
        """Estimate time required to close skill gap"""
        
        # Complex skills require more development time
        complex_skills = [
            'ai_machine_learning', 'strategic_thinking', 'cybersecurity',
            'financial_modeling', 'leadership', 'innovation_leadership'
        ]
        
        base_months = 6 if skill.lower() in [s.lower() for s in complex_skills] else 3
        additional_months = gap_size / 10  # 1 month per 10 point gap
        
        total_months = base_months + additional_months
        
        if total_months <= 3:
            return '1-3 months'
        elif total_months <= 6:
            return '3-6 months'
        elif total_months <= 12:
            return '6-12 months'
        else:
            return '12+ months'
            
    def _assess_retention_risks(self, workforce_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess employee retention risks"""
        
        employees = workforce_data.get('employees', [])
        
        # Identify high-risk employees
        high_risk_employees = []
        flight_risk_factors = []
        
        for employee in employees:
            risk_score = self._calculate_retention_risk_score(employee)
            
            if risk_score >= 70:  # High risk threshold
                high_risk_employees.append({
                    'employee_id': employee.get('id', 'unknown'),
                    'name': employee.get('name', 'Unknown'),
                    'role': employee.get('role', 'Unknown'),
                    'risk_score': risk_score,
                    'key_risk_factors': self._identify_risk_factors(employee),
                    'retention_actions': self._suggest_retention_actions(employee, risk_score)
                })
                
        # Analyze flight risk factors across workforce
        all_risk_factors = []
        for employee in employees:
            all_risk_factors.extend(self._identify_risk_factors(employee))
            
        # Count frequency of risk factors
        factor_counts = {}
        for factor in all_risk_factors:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
            
        top_risk_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'high_risk_employees': high_risk_employees,
            'total_high_risk_count': len(high_risk_employees),
            'workforce_retention_rate': self._calculate_retention_rate(employees),
            'top_risk_factors': [{'factor': factor, 'frequency': count} for factor, count in top_risk_factors],
            'recommended_interventions': self._recommend_retention_interventions(top_risk_factors)
        }
        
    def _calculate_retention_risk_score(self, employee: Dict[str, Any]) -> float:
        """Calculate retention risk score for individual employee"""
        
        risk_factors = {
            'job_satisfaction': 100 - employee.get('job_satisfaction_score', 75),
            'compensation_satisfaction': 100 - employee.get('compensation_satisfaction', 80),
            'career_growth_satisfaction': 100 - employee.get('career_growth_satisfaction', 70),
            'work_life_balance': 100 - employee.get('work_life_balance_score', 75),
            'manager_relationship': 100 - employee.get('manager_relationship_score', 80),
            'tenure_factor': min(50, max(0, 60 - employee.get('tenure_months', 24)))  # Risk increases with low tenure
        }
        
        # Weighted risk calculation
        weights = {
            'job_satisfaction': 0.25,
            'compensation_satisfaction': 0.2,
            'career_growth_satisfaction': 0.2,
            'work_life_balance': 0.15,
            'manager_relationship': 0.15,
            'tenure_factor': 0.05
        }
        
        total_risk = sum(risk_factors[factor] * weights[factor] for factor in risk_factors)
        
        return min(100, total_risk)
        
    def _identify_risk_factors(self, employee: Dict[str, Any]) -> List[str]:
        """Identify specific risk factors for employee"""
        
        risk_factors = []
        
        if employee.get('job_satisfaction_score', 75) < 60:
            risk_factors.append('Low job satisfaction')
        if employee.get('compensation_satisfaction', 80) < 65:
            risk_factors.append('Compensation concerns')
        if employee.get('career_growth_satisfaction', 70) < 60:
            risk_factors.append('Limited career growth')
        if employee.get('work_life_balance_score', 75) < 60:
            risk_factors.append('Poor work-life balance')
        if employee.get('manager_relationship_score', 80) < 65:
            risk_factors.append('Manager relationship issues')
        if employee.get('tenure_months', 24) < 12:
            risk_factors.append('New employee adjustment')
            
        return risk_factors
        
    def _suggest_retention_actions(self, employee: Dict[str, Any], risk_score: float) -> List[str]:
        """Suggest retention actions for high-risk employee"""
        
        actions = []
        risk_factors = self._identify_risk_factors(employee)
        
        if 'Low job satisfaction' in risk_factors:
            actions.append('Conduct detailed stay interview')
            actions.append('Review role responsibilities and alignment')
            
        if 'Compensation concerns' in risk_factors:
            actions.append('Review compensation against market rates')
            actions.append('Consider promotion or salary adjustment')
            
        if 'Limited career growth' in risk_factors:
            actions.append('Develop personalized career development plan')
            actions.append('Provide mentoring and stretch assignments')
            
        if 'Poor work-life balance' in risk_factors:
            actions.append('Explore flexible work arrangements')
            actions.append('Review workload and priorities')
            
        if 'Manager relationship issues' in risk_factors:
            actions.append('Provide manager coaching and feedback')
            actions.append('Consider team or manager reassignment')
            
        # General high-risk actions
        if risk_score >= 80:
            actions.append('Immediate manager intervention required')
            actions.append('Consider retention bonus or special assignment')
            
        return actions
        
    def _calculate_retention_rate(self, employees: List[Dict[str, Any]]) -> float:
        """Calculate overall workforce retention rate"""
        
        if not employees:
            return 0
            
        retention_scores = [employee.get('retention_probability', 80) for employee in employees]
        return sum(retention_scores) / len(retention_scores)
        
    def _recommend_retention_interventions(self, top_risk_factors: List[tuple]) -> List[Dict[str, Any]]:
        """Recommend workforce-wide retention interventions"""
        
        interventions = []
        
        for factor, frequency in top_risk_factors:
            if 'job satisfaction' in factor.lower():
                interventions.append({
                    'intervention': 'Employee Engagement Initiative',
                    'description': 'Comprehensive program to improve job satisfaction',
                    'timeline': '3-6 months',
                    'investment': 'Medium'
                })
            elif 'compensation' in factor.lower():
                interventions.append({
                    'intervention': 'Compensation Review and Adjustment',
                    'description': 'Market-based compensation analysis and adjustments',
                    'timeline': '1-3 months',
                    'investment': 'High'
                })
            elif 'career growth' in factor.lower():
                interventions.append({
                    'intervention': 'Career Development Program',
                    'description': 'Structured career pathing and development opportunities',
                    'timeline': '6-12 months',
                    'investment': 'Medium'
                })
                
        return interventions
        
    def _generate_succession_plans(self, workforce_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate executive succession plans"""
        
        employees = workforce_data.get('employees', [])
        key_positions = workforce_data.get('key_positions', [])
        
        succession_plans = []
        
        for position in key_positions:
            position_name = position.get('title', 'Unknown Position')
            required_skills = position.get('required_skills', [])
            
            # Find potential successors
            potential_successors = self._identify_potential_successors(
                employees, required_skills, position
            )
            
            succession_plan = {
                'position': position_name,
                'current_incumbent': position.get('current_incumbent', 'Unknown'),
                'succession_risk': self._assess_succession_risk(position, potential_successors),
                'potential_successors': potential_successors,
                'development_needs': self._identify_succession_development_needs(potential_successors, required_skills),
                'timeline': self._estimate_succession_timeline(potential_successors),
                'recommended_actions': self._recommend_succession_actions(position, potential_successors)
            }
            
            succession_plans.append(succession_plan)
            
        return succession_plans
        
    def _identify_potential_successors(self, employees: List[Dict[str, Any]], 
                                     required_skills: List[str], 
                                     position: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential successors for key position"""
        
        potential_successors = []
        
        for employee in employees:
            if employee.get('id') == position.get('current_incumbent_id'):
                continue  # Skip current incumbent
                
            readiness_score = self._calculate_succession_readiness(employee, required_skills, position)
            
            if readiness_score >= 60:  # Minimum readiness threshold
                successor = {
                    'employee_id': employee.get('id', 'unknown'),
                    'name': employee.get('name', 'Unknown'),
                    'current_role': employee.get('role', 'Unknown'),
                    'readiness_score': readiness_score,
                    'readiness_level': self._categorize_readiness_level(readiness_score),
                    'development_timeline': self._estimate_development_timeline(readiness_score),
                    'strengths': self._identify_successor_strengths(employee, required_skills),
                    'development_areas': self._identify_development_areas(employee, required_skills)
                }
                
                potential_successors.append(successor)
                
        # Sort by readiness score
        potential_successors.sort(key=lambda x: x['readiness_score'], reverse=True)
        
        return potential_successors[:5]  # Top 5 candidates
        
    def _calculate_succession_readiness(self, employee: Dict[str, Any], 
                                      required_skills: List[str], 
                                      position: Dict[str, Any]) -> float:
        """Calculate succession readiness score"""
        
        # Performance and potential
        performance_score = employee.get('performance_rating', 75)
        leadership_potential = employee.get('leadership_potential', 70)
        
        # Skills alignment
        employee_skills = employee.get('skills', {})
        skill_alignment = []
        
        for skill in required_skills:
            current_level = employee_skills.get(skill, 0)
            required_level = 80  # Standard requirement for key positions
            alignment = min(100, (current_level / required_level) * 100)
            skill_alignment.append(alignment)
            
        avg_skill_alignment = sum(skill_alignment) / len(skill_alignment) if skill_alignment else 0
        
        # Experience relevance
        experience_years = employee.get('relevant_experience_years', 0)
        experience_score = min(100, experience_years * 10)  # 10 points per year, max 100
        
        # Calculate weighted readiness score
        readiness_score = (
            performance_score * 0.3 +
            leadership_potential * 0.3 +
            avg_skill_alignment * 0.25 +
            experience_score * 0.15
        )
        
        return readiness_score
        
    def _categorize_readiness_level(self, score: float) -> str:
        """Categorize succession readiness level"""
        if score >= 85:
            return 'Ready Now'
        elif score >= 75:
            return 'Ready in 1-2 years'
        elif score >= 65:
            return 'Ready in 3-5 years'
        else:
            return 'Long-term potential'
            
    def _estimate_development_timeline(self, readiness_score: float) -> str:
        """Estimate development timeline for succession"""
        if readiness_score >= 85:
            return '0-6 months'
        elif readiness_score >= 75:
            return '1-2 years'
        elif readiness_score >= 65:
            return '3-5 years'
        else:
            return '5+ years'
            
    def _identify_successor_strengths(self, employee: Dict[str, Any], required_skills: List[str]) -> List[str]:
        """Identify strengths of potential successor"""
        
        strengths = []
        employee_skills = employee.get('skills', {})
        
        # High-performing skills
        for skill, level in employee_skills.items():
            if level >= 80:
                strengths.append(f'Strong {skill.replace("_", " ").title()}')
                
        # Performance indicators
        if employee.get('performance_rating', 0) >= 85:
            strengths.append('High performance track record')
        if employee.get('leadership_potential', 0) >= 80:
            strengths.append('Strong leadership potential')
        if employee.get('learning_agility_score', 0) >= 80:
            strengths.append('High learning agility')
            
        return strengths[:5]  # Top 5 strengths
        
    def _identify_development_areas(self, employee: Dict[str, Any], required_skills: List[str]) -> List[str]:
        """Identify development areas for potential successor"""
        
        development_areas = []
        employee_skills = employee.get('skills', {})
        
        # Skills gaps
        for skill in required_skills:
            current_level = employee_skills.get(skill, 0)
            if current_level < 75:  # Development threshold
                development_areas.append(f'Develop {skill.replace("_", " ").title()}')
                
        # Experience gaps
        if employee.get('relevant_experience_years', 0) < 5:
            development_areas.append('Gain additional relevant experience')
            
        return development_areas[:5]  # Top 5 development areas
        
    def _assess_succession_risk(self, position: Dict[str, Any], potential_successors: List[Dict[str, Any]]) -> str:
        """Assess succession risk for position"""
        
        if not potential_successors:
            return 'Critical'
            
        best_successor = potential_successors[0]
        readiness_score = best_successor['readiness_score']
        
        if readiness_score >= 85:
            return 'Low'
        elif readiness_score >= 75:
            return 'Medium'
        elif readiness_score >= 65:
            return 'High'
        else:
            return 'Critical'
            
    def _identify_succession_development_needs(self, successors: List[Dict[str, Any]], required_skills: List[str]) -> List[Dict[str, Any]]:
        """Identify development needs for succession planning"""
        
        development_needs = []
        
        # Aggregate development areas across successors
        all_development_areas = []
        for successor in successors:
            all_development_areas.extend(successor.get('development_areas', []))
            
        # Count frequency of development needs
        area_counts = {}
        for area in all_development_areas:
            area_counts[area] = area_counts.get(area, 0) + 1
            
        # Top development needs
        top_needs = sorted(area_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for need, frequency in top_needs:
            development_needs.append({
                'development_area': need,
                'priority': 'High' if frequency >= len(successors) * 0.6 else 'Medium',
                'affected_successors': frequency,
                'recommended_approach': self._recommend_development_approach(need)
            })
            
        return development_needs
        
    def _recommend_development_approach(self, development_area: str) -> str:
        """Recommend development approach for specific area"""
        
        if 'leadership' in development_area.lower():
            return 'Executive coaching and leadership programs'
        elif 'experience' in development_area.lower():
            return 'Stretch assignments and cross-functional projects'
        elif 'strategic' in development_area.lower():
            return 'Strategy workshops and mentoring'
        else:
            return 'Targeted training and certification programs'
            
    def _estimate_succession_timeline(self, successors: List[Dict[str, Any]]) -> str:
        """Estimate overall succession timeline"""
        
        if not successors:
            return 'No suitable successors identified'
            
        best_successor = successors[0]
        return best_successor.get('development_timeline', 'Unknown')
        
    def _recommend_succession_actions(self, position: Dict[str, Any], successors: List[Dict[str, Any]]) -> List[str]:
        """Recommend actions for succession planning"""
        
        actions = []
        
        if not successors:
            actions.extend([
                'Conduct external talent search',
                'Develop internal succession pipeline',
                'Implement knowledge transfer program'
            ])
        else:
            best_successor = successors[0]
            readiness_level = best_successor['readiness_level']
            
            if readiness_level == 'Ready Now':
                actions.extend([
                    'Begin immediate succession preparation',
                    'Implement knowledge transfer',
                    'Prepare succession announcement'
                ])
            else:
                actions.extend([
                    'Develop accelerated development program',
                    'Provide stretch assignments',
                    'Implement mentoring program'
                ])
                
        return actions
        
    def _generate_workforce_optimization(self, metrics: TalentMetrics, skills_gaps: List[SkillGap]) -> List[Dict[str, Any]]:
        """Generate workforce optimization recommendations"""
        
        recommendations = []
        
        # Skills development recommendations
        if skills_gaps:
            critical_gaps = [gap for gap in skills_gaps if gap.criticality == 'Critical']
            if critical_gaps:
                recommendations.append({
                    'category': 'Skills Development',
                    'priority': 'Critical',
                    'recommendation': f'Address {len(critical_gaps)} critical skills gaps immediately',
                    'actions': [
                        'Implement emergency skills training programs',
                        'Consider external hiring for critical roles',
                        'Develop accelerated development programs'
                    ],
                    'timeline': '1-6 months',
                    'investment': 'High'
                })
                
        # Retention optimization
        if metrics.retention_risk > 30:
            recommendations.append({
                'category': 'Retention',
                'priority': 'High',
                'recommendation': 'Implement comprehensive retention strategy',
                'actions': [
                    'Address top retention risk factors',
                    'Enhance employee engagement programs',
                    'Review compensation and benefits'
                ],
                'timeline': '3-12 months',
                'investment': 'Medium-High'
            })
            
        # Performance optimization
        if metrics.workforce_efficiency < 80:
            recommendations.append({
                'category': 'Performance',
                'priority': 'Medium',
                'recommendation': 'Optimize workforce performance and productivity',
                'actions': [
                    'Implement performance management improvements',
                    'Enhance training and development programs',
                    'Optimize role assignments and workflows'
                ],
                'timeline': '6-18 months',
                'investment': 'Medium'
            })
            
        return recommendations

def test_talent_intelligence_agent():
    """Test the Talent Intelligence Agent"""
    print("🧪 Testing Talent Intelligence & Workforce Agent")
    print("=" * 55)
    
    try:
        agent = TalentIntelligenceAgent()
        
        # Test data
        test_workforce = {
            'company_name': 'Tech Innovation Corp',
            'industry': 'technology',
            'employees': [
                {
                    'id': 'EMP001',
                    'name': 'John Smith',
                    'role': 'Senior Developer',
                    'productivity_score': 85,
                    'skill_role_alignment': 80,
                    'retention_probability': 75,
                    'learning_agility_score': 80,
                    'skills': {
                        'software_development': 90,
                        'cloud_architecture': 70,
                        'leadership': 60
                    },
                    'performance_rating': 85,
                    'leadership_potential': 75
                },
                {
                    'id': 'EMP002',
                    'name': 'Jane Doe',
                    'role': 'Engineering Manager',
                    'productivity_score': 90,
                    'skill_role_alignment': 85,
                    'retention_probability': 90,
                    'learning_agility_score': 85,
                    'skills': {
                        'software_development': 85,
                        'team_management': 90,
                        'strategic_thinking': 80
                    },
                    'performance_rating': 92,
                    'leadership_potential': 88
                }
            ],
            'required_skills': {
                'software_development': 85,
                'cloud_architecture': 80,
                'ai_machine_learning': 75,
                'leadership': 80
            },
            'key_positions': [
                {
                    'title': 'CTO',
                    'current_incumbent': 'Current CTO',
                    'required_skills': ['strategic_thinking', 'technology_leadership', 'team_management']
                }
            ]
        }
        
        # Test workforce analysis
        analysis = agent.analyze_workforce_capabilities(test_workforce)
        print(f"✅ Workforce analysis completed for {test_workforce['company_name']}")
        print(f"   Workforce efficiency: {analysis['talent_metrics']['workforce_efficiency']:.1f}")
        print(f"   Skills gaps identified: {len(analysis['skills_gaps'])}")
        print(f"   Retention risk: {analysis['talent_metrics']['retention_risk']:.1f}")
        print(f"   Succession plans: {len(analysis['succession_plans'])}")
        
        return {
            'agent_initialized': True,
            'workforce_efficiency': analysis['talent_metrics']['workforce_efficiency'],
            'skills_gaps': len(analysis['skills_gaps']),
            'succession_plans': len(analysis['succession_plans'])
        }
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return {'error': str(e)}

if __name__ == "__main__":
    test_talent_intelligence_agent()