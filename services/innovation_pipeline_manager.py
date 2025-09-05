"""
Innovation Pipeline Manager
R&D project prioritization and innovation portfolio optimization
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class InnovationProject:
    project_id: str
    name: str
    stage: str
    priority_score: float
    resource_requirements: Dict[str, Any]
    expected_roi: float

class InnovationPipelineManager:
    """
    Comprehensive Innovation Pipeline Management System
    - R&D project prioritization
    - Innovation portfolio optimization
    - Patent landscape analysis
    - Competitive innovation tracking
    """
    
    def __init__(self):
        self.name = "Innovation Pipeline Manager"
        self.version = "1.0.0"
        self.capabilities = [
            "Project Portfolio Optimization",
            "Innovation Prioritization",
            "Patent Analysis",
            "Competitive Intelligence",
            "Resource Allocation",
            "ROI Prediction"
        ]
        
    def optimize_innovation_portfolio(self, innovation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize innovation project portfolio"""
        try:
            company_name = innovation_data.get('company_name', 'Unknown Company')
            
            # Analyze current portfolio
            portfolio_analysis = self._analyze_current_portfolio(innovation_data)
            
            # Optimize project prioritization
            prioritization = self._prioritize_innovation_projects(innovation_data)
            
            # Resource allocation optimization
            resource_optimization = self._optimize_resource_allocation(innovation_data)
            
            # Generate recommendations
            recommendations = self._generate_innovation_recommendations(
                portfolio_analysis, prioritization, resource_optimization
            )
            
            return {
                'company': company_name,
                'analysis_date': datetime.now().isoformat(),
                'portfolio_analysis': portfolio_analysis,
                'project_prioritization': prioritization,
                'resource_optimization': resource_optimization,
                'recommendations': recommendations,
                'next_review_date': (datetime.now() + timedelta(days=90)).isoformat()
            }
            
        except Exception as e:
            logging.error(f"Innovation portfolio optimization failed: {str(e)}")
            return {'error': f'Innovation portfolio optimization failed: {str(e)}'}
            
    def _analyze_current_portfolio(self, innovation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current innovation portfolio"""
        projects = innovation_data.get('projects', [])
        
        stage_distribution = {}
        total_investment = 0
        
        for project in projects:
            stage = project.get('stage', 'Unknown')
            investment = project.get('investment', 0)
            
            stage_distribution[stage] = stage_distribution.get(stage, 0) + 1
            total_investment += investment
            
        portfolio_balance = self._assess_portfolio_balance(stage_distribution)
        
        return {
            'total_projects': len(projects),
            'total_investment': total_investment,
            'stage_distribution': stage_distribution,
            'portfolio_balance': portfolio_balance,
            'average_project_investment': total_investment / len(projects) if projects else 0
        }
        
    def _assess_portfolio_balance(self, stage_distribution: Dict[str, int]) -> str:
        """Assess innovation portfolio balance"""
        total_projects = sum(stage_distribution.values())
        
        if total_projects == 0:
            return 'No projects'
            
        research_pct = stage_distribution.get('Research', 0) / total_projects * 100
        development_pct = stage_distribution.get('Development', 0) / total_projects * 100
        
        if research_pct > 60:
            return 'Too research-heavy'
        elif development_pct > 70:
            return 'Too development-heavy'
        else:
            return 'Well-balanced'
            
    def _prioritize_innovation_projects(self, innovation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prioritize innovation projects"""
        projects = innovation_data.get('projects', [])
        
        prioritized_projects = []
        for project in projects:
            priority_score = self._calculate_project_priority(project)
            
            prioritized_project = {
                'project_id': project.get('id', 'unknown'),
                'name': project.get('name', 'Unknown Project'),
                'priority_score': priority_score,
                'expected_roi': project.get('expected_roi', 15),
                'resource_requirements': project.get('resource_requirements', {}),
                'recommendation': self._get_project_recommendation(priority_score)
            }
            
            prioritized_projects.append(prioritized_project)
            
        # Sort by priority score
        prioritized_projects.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return {
            'prioritized_projects': prioritized_projects,
            'high_priority_count': len([p for p in prioritized_projects if p['priority_score'] >= 80]),
            'funding_recommendations': self._generate_funding_recommendations(prioritized_projects)
        }
        
    def _calculate_project_priority(self, project: Dict[str, Any]) -> float:
        """Calculate project priority score"""
        factors = {
            'market_potential': project.get('market_potential', 60),
            'technical_feasibility': project.get('technical_feasibility', 70),
            'competitive_advantage': project.get('competitive_advantage', 65),
            'strategic_alignment': project.get('strategic_alignment', 75),
            'resource_availability': project.get('resource_availability', 80)
        }
        
        # Weighted scoring
        weights = {
            'market_potential': 0.25,
            'technical_feasibility': 0.20,
            'competitive_advantage': 0.20,
            'strategic_alignment': 0.20,
            'resource_availability': 0.15
        }
        
        priority_score = sum(factors[factor] * weights[factor] for factor in factors)
        
        return priority_score
        
    def _get_project_recommendation(self, priority_score: float) -> str:
        """Get recommendation based on priority score"""
        if priority_score >= 80:
            return 'Fast-track funding'
        elif priority_score >= 70:
            return 'Recommended for funding'
        elif priority_score >= 60:
            return 'Consider for future funding'
        else:
            return 'Defer or discontinue'
            
    def _generate_funding_recommendations(self, projects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate funding recommendations"""
        recommendations = []
        
        high_priority = [p for p in projects if p['priority_score'] >= 80]
        medium_priority = [p for p in projects if 70 <= p['priority_score'] < 80]
        
        if high_priority:
            total_funding_needed = sum(p.get('resource_requirements', {}).get('funding', 500000) for p in high_priority)
            recommendations.append({
                'priority': 'High',
                'projects': len(high_priority),
                'funding_needed': total_funding_needed,
                'recommendation': 'Immediate funding approval'
            })
            
        if medium_priority:
            total_funding_needed = sum(p.get('resource_requirements', {}).get('funding', 300000) for p in medium_priority)
            recommendations.append({
                'priority': 'Medium',
                'projects': len(medium_priority),
                'funding_needed': total_funding_needed,
                'recommendation': 'Conditional funding based on milestones'
            })
            
        return recommendations
        
    def _optimize_resource_allocation(self, innovation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize resource allocation across projects"""
        projects = innovation_data.get('projects', [])
        available_resources = innovation_data.get('available_resources', {})
        
        resource_demand = {}
        for project in projects:
            requirements = project.get('resource_requirements', {})
            for resource_type, amount in requirements.items():
                resource_demand[resource_type] = resource_demand.get(resource_type, 0) + amount
                
        resource_gaps = {}
        for resource_type, demand in resource_demand.items():
            available = available_resources.get(resource_type, 0)
            if demand > available:
                resource_gaps[resource_type] = demand - available
                
        return {
            'resource_demand': resource_demand,
            'resource_gaps': resource_gaps,
            'allocation_efficiency': self._calculate_allocation_efficiency(resource_demand, available_resources),
            'optimization_suggestions': self._suggest_resource_optimizations(resource_gaps)
        }
        
    def _calculate_allocation_efficiency(self, demand: Dict[str, Any], available: Dict[str, Any]) -> float:
        """Calculate resource allocation efficiency"""
        if not demand:
            return 100
            
        total_demand = sum(demand.values())
        total_available = sum(available.values())
        
        return min(100, (total_available / total_demand) * 100) if total_demand > 0 else 100
        
    def _suggest_resource_optimizations(self, gaps: Dict[str, Any]) -> List[str]:
        """Suggest resource optimization strategies"""
        suggestions = []
        
        if gaps:
            suggestions.extend([
                'Consider external partnerships for resource gaps',
                'Implement phased project execution',
                'Evaluate project portfolio pruning',
                'Explore cross-functional resource sharing'
            ])
        else:
            suggestions.append('Current resource allocation is adequate')
            
        return suggestions
        
    def _generate_innovation_recommendations(self, portfolio: Dict[str, Any], 
                                           prioritization: Dict[str, Any], 
                                           resources: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate innovation management recommendations"""
        recommendations = []
        
        # Portfolio balance recommendations
        if portfolio['portfolio_balance'] != 'Well-balanced':
            recommendations.append({
                'category': 'Portfolio Balance',
                'priority': 'Medium',
                'recommendation': f'Rebalance portfolio - currently {portfolio["portfolio_balance"]}',
                'actions': ['Adjust new project intake', 'Review stage gate criteria']
            })
            
        # Resource optimization recommendations
        if resources['allocation_efficiency'] < 80:
            recommendations.append({
                'category': 'Resource Optimization',
                'priority': 'High',
                'recommendation': 'Address resource allocation inefficiencies',
                'actions': resources['optimization_suggestions']
            })
            
        # High-priority project recommendations
        if prioritization['high_priority_count'] > 0:
            recommendations.append({
                'category': 'Project Execution',
                'priority': 'High',
                'recommendation': f'Fast-track {prioritization["high_priority_count"]} high-priority projects',
                'actions': ['Accelerate funding approval', 'Assign dedicated resources']
            })
            
        return recommendations

def test_innovation_pipeline_manager():
    """Test the Innovation Pipeline Manager"""
    print("🧪 Testing Innovation Pipeline Manager")
    print("=" * 40)
    
    try:
        manager = InnovationPipelineManager()
        
        test_data = {
            'company_name': 'Innovation Leaders Inc',
            'projects': [
                {
                    'id': 'PROJ001',
                    'name': 'AI-Powered Analytics',
                    'stage': 'Development',
                    'investment': 500000,
                    'market_potential': 90,
                    'technical_feasibility': 85,
                    'expected_roi': 250,
                    'resource_requirements': {'funding': 750000, 'researchers': 5}
                },
                {
                    'id': 'PROJ002',
                    'name': 'Blockchain Integration',
                    'stage': 'Research',
                    'investment': 300000,
                    'market_potential': 70,
                    'technical_feasibility': 60,
                    'expected_roi': 180
                }
            ],
            'available_resources': {'funding': 1000000, 'researchers': 8}
        }
        
        analysis = manager.optimize_innovation_portfolio(test_data)
        print(f"✅ Portfolio optimization completed: {analysis['portfolio_analysis']['total_projects']} projects")
        
        return {'manager_initialized': True, 'optimization_completed': True}
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return {'error': str(e)}

if __name__ == "__main__":
    test_innovation_pipeline_manager()