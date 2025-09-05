"""
Workspace Intelligence Service
Provides intelligent assistance and recommendations across all applications in the workspace
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import json

from .agent_coordination_service import (
    coordination_service, 
    CoordinationRequest, 
    RequestType, 
    AgentCapability
)

@dataclass
class WorkspaceContext:
    """Context about the current workspace state"""
    active_apps: List[Dict[str, Any]]
    recent_activities: List[Dict[str, Any]]
    user_preferences: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    optimization_opportunities: List[Dict[str, Any]]

@dataclass
class IntelligenceRequest:
    """Request for workspace intelligence assistance"""
    request_type: str
    description: str
    context: Dict[str, Any]
    target_apps: Optional[List[str]] = None
    priority: int = 5

@dataclass
class IntelligenceResponse:
    """Response with coordinated intelligence insights"""
    primary_insight: Dict[str, Any]
    recommendations: List[str]
    implementation_plan: List[Dict[str, str]]
    affected_apps: List[str]
    estimated_impact: Dict[str, Any]
    confidence_score: float
    supporting_data: List[Dict[str, Any]]

class WorkspaceIntelligenceService:
    """
    Provides intelligent assistance leveraging coordinated AI agents
    for comprehensive workspace optimization and development support
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.workspace_context: Optional[WorkspaceContext] = None
        self.intelligence_cache: Dict[str, Any] = {}
        
        # Intelligence patterns and templates
        self.intelligence_patterns = {
            'code_optimization': {
                'required_capabilities': [
                    AgentCapability.CODE_ANALYSIS,
                    AgentCapability.PERFORMANCE_OPTIMIZATION
                ],
                'coordination_type': RequestType.DEVELOPMENT
            },
            'creative_enhancement': {
                'required_capabilities': [
                    AgentCapability.CREATIVE_DIRECTION,
                    AgentCapability.MULTIMEDIA_CREATION,
                    AgentCapability.CONTENT_GENERATION
                ],
                'coordination_type': RequestType.CREATIVE
            },
            'financial_optimization': {
                'required_capabilities': [
                    AgentCapability.FINANCIAL_ANALYSIS,
                    AgentCapability.MARKET_INTELLIGENCE,
                    AgentCapability.RISK_ASSESSMENT
                ],
                'coordination_type': RequestType.FINANCIAL
            },
            'strategic_planning': {
                'required_capabilities': [
                    AgentCapability.TECHNICAL_STRATEGY,
                    AgentCapability.MARKET_INTELLIGENCE,
                    AgentCapability.PERFORMANCE_OPTIMIZATION
                ],
                'coordination_type': RequestType.STRATEGIC
            },
            'content_creation': {
                'required_capabilities': [
                    AgentCapability.CONTENT_GENERATION,
                    AgentCapability.CREATIVE_DIRECTION,
                    AgentCapability.MULTIMEDIA_CREATION
                ],
                'coordination_type': RequestType.MULTIMEDIA
            }
        }
        
        self.logger.info("Workspace Intelligence Service initialized")
    
    async def analyze_workspace(self) -> WorkspaceContext:
        """Analyze current workspace state and gather context"""
        
        try:
            # Gather workspace data
            from models import ReplitApp, AIAgent
            
            # Get active applications
            active_apps = []
            apps = ReplitApp.query.filter_by(is_active=True).all()
            
            for app in apps:
                app_agents = AIAgent.query.filter_by(app_id=app.id).all()
                
                app_context = {
                    'id': app.id,
                    'name': app.name,
                    'url': app.url,
                    'language': getattr(app, 'language', 'Unknown'),
                    'agent_count': len(app_agents),
                    'last_updated': app.updated_at.isoformat() if app.updated_at else None,
                    'agents': [{
                        'type': agent.agent_type,
                        'name': agent.agent_name,
                        'effectiveness': float(agent.effectiveness_score or 0),
                        'cost': float(agent.cost_estimate or 0)
                    } for agent in app_agents]
                }
                active_apps.append(app_context)
            
            # Calculate performance metrics
            total_agents = sum(len(app['agents']) for app in active_apps)
            total_cost = sum(sum(agent['cost'] for agent in app['agents']) for app in active_apps)
            avg_effectiveness = sum(sum(agent['effectiveness'] for agent in app['agents']) for app in active_apps) / total_agents if total_agents > 0 else 0
            
            performance_metrics = {
                'total_apps': len(active_apps),
                'total_agents': total_agents,
                'total_monthly_cost': total_cost,
                'average_effectiveness': avg_effectiveness,
                'cost_per_agent': total_cost / total_agents if total_agents > 0 else 0
            }
            
            # Generate optimization opportunities
            optimization_opportunities = await self._identify_optimization_opportunities(active_apps, performance_metrics)
            
            # Create workspace context
            self.workspace_context = WorkspaceContext(
                active_apps=active_apps,
                recent_activities=[],  # Would be populated from activity logs
                user_preferences={},   # Would be populated from user settings
                performance_metrics=performance_metrics,
                optimization_opportunities=optimization_opportunities
            )
            
            return self.workspace_context
            
        except Exception as e:
            self.logger.error(f"Workspace analysis failed: {str(e)}")
            # Return empty context
            return WorkspaceContext(
                active_apps=[],
                recent_activities=[],
                user_preferences={},
                performance_metrics={},
                optimization_opportunities=[]
            )
    
    async def _identify_optimization_opportunities(self, apps: List[Dict[str, Any]], 
                                                  metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify optimization opportunities across the workspace"""
        
        opportunities = []
        
        # Cost optimization opportunities
        if metrics.get('total_monthly_cost', 0) > 100:
            opportunities.append({
                'type': 'cost_optimization',
                'title': 'High AI Service Costs Detected',
                'description': f'Monthly costs of ${metrics["total_monthly_cost"]:.2f} could be optimized',
                'potential_savings': metrics["total_monthly_cost"] * 0.25,
                'priority': 'high',
                'affected_apps': [app['name'] for app in apps if sum(agent['cost'] for agent in app['agents']) > 10]
            })
        
        # Agent consolidation opportunities
        agent_types = {}
        for app in apps:
            for agent in app['agents']:
                agent_type = agent['type']
                if agent_type not in agent_types:
                    agent_types[agent_type] = []
                agent_types[agent_type].append({'app': app['name'], 'agent': agent})
        
        for agent_type, instances in agent_types.items():
            if len(instances) > 2:
                opportunities.append({
                    'type': 'agent_consolidation',
                    'title': f'Multiple {agent_type} Agents Detected',
                    'description': f'{len(instances)} instances of {agent_type} agents could be consolidated',
                    'potential_savings': len(instances) * 15,
                    'priority': 'medium',
                    'affected_apps': [instance['app'] for instance in instances]
                })
        
        # Performance enhancement opportunities
        low_performance_apps = [app for app in apps if app['agent_count'] > 0 and 
                               sum(agent['effectiveness'] for agent in app['agents']) / app['agent_count'] < 0.7]
        
        if low_performance_apps:
            opportunities.append({
                'type': 'performance_enhancement',
                'title': 'Low Agent Effectiveness Detected',
                'description': f'{len(low_performance_apps)} apps have agents with effectiveness below 70%',
                'potential_savings': 0,
                'priority': 'high',
                'affected_apps': [app['name'] for app in low_performance_apps]
            })
        
        return opportunities
    
    async def provide_intelligent_assistance(self, request: IntelligenceRequest) -> IntelligenceResponse:
        """
        Provide intelligent assistance using coordinated agent capabilities
        
        Args:
            request: Intelligence request describing what assistance is needed
            
        Returns:
            IntelligenceResponse: Comprehensive response with coordinated insights
        """
        
        try:
            # Ensure workspace context is current
            if not self.workspace_context:
                await self.analyze_workspace()
            
            # Determine appropriate intelligence pattern
            pattern = self._match_intelligence_pattern(request)
            
            if not pattern:
                return IntelligenceResponse(
                    primary_insight={'error': 'No suitable intelligence pattern found'},
                    recommendations=['Refine request description'],
                    implementation_plan=[],
                    affected_apps=[],
                    estimated_impact={},
                    confidence_score=0.0,
                    supporting_data=[]
                )
            
            # Create coordination request
            coordination_request = CoordinationRequest(
                request_id=f"intel_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                request_type=pattern['coordination_type'],
                description=request.description,
                context={
                    'workspace_context': self.workspace_context.__dict__,
                    'target_apps': request.target_apps,
                    'user_context': request.context
                },
                required_capabilities=pattern['required_capabilities'],
                priority=request.priority
            )
            
            # Execute coordinated request
            coordinated_response = await coordination_service.coordinate_request(coordination_request)
            
            # Transform coordinated response into intelligence response
            intelligence_response = await self._transform_to_intelligence_response(
                request, coordinated_response, pattern
            )
            
            return intelligence_response
            
        except Exception as e:
            self.logger.error(f"Intelligence assistance failed: {str(e)}")
            return IntelligenceResponse(
                primary_insight={'error': str(e)},
                recommendations=['Try again with a different request'],
                implementation_plan=[],
                affected_apps=[],
                estimated_impact={},
                confidence_score=0.0,
                supporting_data=[]
            )
    
    def _match_intelligence_pattern(self, request: IntelligenceRequest) -> Optional[Dict[str, Any]]:
        """Match request to appropriate intelligence pattern"""
        
        request_lower = request.description.lower()
        
        # Pattern matching based on keywords
        if any(keyword in request_lower for keyword in ['optimize', 'performance', 'code', 'speed']):
            return self.intelligence_patterns['code_optimization']
        elif any(keyword in request_lower for keyword in ['creative', 'design', 'brand', 'content']):
            return self.intelligence_patterns['creative_enhancement']
        elif any(keyword in request_lower for keyword in ['cost', 'financial', 'investment', 'money']):
            return self.intelligence_patterns['financial_optimization']
        elif any(keyword in request_lower for keyword in ['strategy', 'planning', 'roadmap', 'direction']):
            return self.intelligence_patterns['strategic_planning']
        elif any(keyword in request_lower for keyword in ['media', 'video', 'image', 'music', 'generate']):
            return self.intelligence_patterns['content_creation']
        
        # Default to strategic planning for unclear requests
        return self.intelligence_patterns['strategic_planning']
    
    async def _transform_to_intelligence_response(self, request: IntelligenceRequest,
                                                coordinated_response, pattern: Dict[str, Any]) -> IntelligenceResponse:
        """Transform coordinated agent response into intelligence response"""
        
        # Extract primary insight
        primary_insight = coordinated_response.primary_response
        
        # Generate implementation plan
        implementation_plan = self._generate_implementation_plan(request, coordinated_response, pattern)
        
        # Determine affected apps
        affected_apps = request.target_apps or []
        if not affected_apps and self.workspace_context:
            # Determine affected apps based on request type and context
            if pattern['coordination_type'] == RequestType.DEVELOPMENT:
                affected_apps = [app['name'] for app in self.workspace_context.active_apps 
                               if app['agent_count'] > 0]
            elif pattern['coordination_type'] == RequestType.CREATIVE:
                affected_apps = [app['name'] for app in self.workspace_context.active_apps 
                               if any('content' in agent['type'].lower() for agent in app['agents'])]
        
        # Calculate estimated impact
        estimated_impact = self._calculate_estimated_impact(coordinated_response, affected_apps)
        
        # Extract supporting data
        supporting_data = [{
            'agent': insight['agent'],
            'data': insight['insight'],
            'confidence': insight['confidence']
        } for insight in coordinated_response.supporting_insights]
        
        return IntelligenceResponse(
            primary_insight=primary_insight,
            recommendations=coordinated_response.recommendations,
            implementation_plan=implementation_plan,
            affected_apps=affected_apps,
            estimated_impact=estimated_impact,
            confidence_score=coordinated_response.overall_confidence,
            supporting_data=supporting_data
        )
    
    def _generate_implementation_plan(self, request: IntelligenceRequest,
                                    coordinated_response, pattern: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate step-by-step implementation plan"""
        
        plan = []
        
        if pattern['coordination_type'] == RequestType.DEVELOPMENT:
            plan.extend([
                {'step': '1', 'action': 'Review code analysis results', 'timeline': '1 day'},
                {'step': '2', 'action': 'Implement performance optimizations', 'timeline': '3-5 days'},
                {'step': '3', 'action': 'Test and validate improvements', 'timeline': '2 days'},
                {'step': '4', 'action': 'Deploy optimized version', 'timeline': '1 day'}
            ])
        elif pattern['coordination_type'] == RequestType.CREATIVE:
            plan.extend([
                {'step': '1', 'action': 'Develop creative strategy', 'timeline': '2 days'},
                {'step': '2', 'action': 'Create multimedia assets', 'timeline': '5-7 days'},
                {'step': '3', 'action': 'Review and refine content', 'timeline': '2 days'},
                {'step': '4', 'action': 'Implement across applications', 'timeline': '3 days'}
            ])
        elif pattern['coordination_type'] == RequestType.FINANCIAL:
            plan.extend([
                {'step': '1', 'action': 'Analyze financial recommendations', 'timeline': '1 day'},
                {'step': '2', 'action': 'Assess risk factors', 'timeline': '2 days'},
                {'step': '3', 'action': 'Develop implementation strategy', 'timeline': '3 days'},
                {'step': '4', 'action': 'Execute financial optimizations', 'timeline': '5 days'}
            ])
        
        return plan
    
    def _calculate_estimated_impact(self, coordinated_response, affected_apps: List[str]) -> Dict[str, Any]:
        """Calculate estimated impact of implementing recommendations"""
        
        return {
            'cost_savings': coordinated_response.total_cost * 2.5,  # Estimated 2.5x return
            'efficiency_gain': f"{min(coordinated_response.overall_confidence * 100, 95):.0f}%",
            'implementation_time': f"{len(affected_apps) * 2}-{len(affected_apps) * 4} days",
            'apps_affected': len(affected_apps),
            'confidence_level': coordinated_response.overall_confidence
        }
    
    async def get_workspace_insights(self) -> Dict[str, Any]:
        """Get comprehensive insights about workspace state and opportunities"""
        
        # Ensure current context
        workspace_context = await self.analyze_workspace()
        
        # Get coordination service status
        coordination_status = coordination_service.get_coordination_status()
        
        return {
            'workspace_overview': {
                'total_apps': len(workspace_context.active_apps),
                'total_agents': workspace_context.performance_metrics.get('total_agents', 0),
                'monthly_cost': workspace_context.performance_metrics.get('total_monthly_cost', 0),
                'avg_effectiveness': workspace_context.performance_metrics.get('average_effectiveness', 0)
            },
            'optimization_opportunities': workspace_context.optimization_opportunities,
            'agent_coordination': {
                'available_agents': coordination_status['registered_agents'],
                'capabilities': coordination_status['capabilities_available'],
                'status': coordination_status['service_status']
            },
            'top_apps': sorted(workspace_context.active_apps, 
                             key=lambda x: x['agent_count'], reverse=True)[:5],
            'recommendations': [
                'Regularly review agent performance metrics',
                'Consider consolidating similar AI services',
                'Leverage coordination service for complex tasks',
                'Monitor cost trends and optimization opportunities'
            ]
        }
    
    async def suggest_workspace_improvements(self) -> List[Dict[str, Any]]:
        """Suggest specific improvements for workspace optimization"""
        
        if not self.workspace_context:
            await self.analyze_workspace()
        
        suggestions = []
        
        # Based on current metrics and opportunities
        for opportunity in self.workspace_context.optimization_opportunities:
            if opportunity['type'] == 'cost_optimization':
                suggestions.append({
                    'category': 'Cost Efficiency',
                    'title': 'Implement Agent Cost Optimization',
                    'description': 'Reduce AI service costs through shared services and optimization',
                    'impact': 'High',
                    'effort': 'Medium',
                    'timeline': '1-2 weeks'
                })
            elif opportunity['type'] == 'agent_consolidation':
                suggestions.append({
                    'category': 'Resource Management',
                    'title': 'Consolidate Duplicate AI Agents',
                    'description': 'Share AI agents across applications to reduce redundancy',
                    'impact': 'Medium',
                    'effort': 'Low',
                    'timeline': '3-5 days'
                })
        
        # Always suggest coordination service usage
        suggestions.append({
            'category': 'Intelligence Enhancement',
            'title': 'Leverage Agent Coordination Service',
            'description': 'Use coordinated AI agents for complex, multi-faceted tasks',
            'impact': 'High',
            'effort': 'Low',
            'timeline': 'Immediate'
        })
        
        return suggestions


# Global workspace intelligence service instance
workspace_intelligence = WorkspaceIntelligenceService()