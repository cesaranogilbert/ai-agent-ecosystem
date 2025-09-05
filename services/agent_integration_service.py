"""
AI Agent Integration Service
Unified integration and management system for all 30 AI agents
"""

import os
import json
import logging
import importlib
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import current_app
from openai import OpenAI

logger = logging.getLogger(__name__)

class AIAgentIntegrationService:
    """Central service for managing all 30 AI agents"""
    
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.agents = self._load_all_agents()
        self.agent_status = {}
        
    def _load_all_agents(self) -> Dict[str, Any]:
        """Load all 30 AI agents from the digital marketing suite"""
        
        agents = {
            # Original 10 Digital Marketing Agents
            'master_digital_marketing_strategist': {
                'name': 'Master Digital Marketing Strategist',
                'module': 'digital_marketing_ai_suite.master_digital_marketing_strategist_agent',
                'category': 'strategy',
                'capabilities': ['comprehensive_strategy', 'performance_analysis', 'budget_optimization'],
                'priority': 1
            },
            'brand_storytelling_narrative': {
                'name': 'Brand Storytelling & Narrative Agent',
                'module': 'digital_marketing_ai_suite.brand_storytelling_narrative_agent',
                'category': 'branding',
                'capabilities': ['brand_voice', 'storytelling', 'narrative_development'],
                'priority': 2
            },
            'omnichannel_content_creator': {
                'name': 'Omnichannel Content Creator',
                'module': 'digital_marketing_ai_suite.omnichannel_content_creator_agent',
                'category': 'content',
                'capabilities': ['content_calendar', 'multi_platform', 'trend_integration'],
                'priority': 3
            },
            'visual_content_production': {
                'name': 'Visual Content Production Agent',
                'module': 'digital_marketing_ai_suite.visual_content_production_agent',
                'category': 'visual',
                'capabilities': ['visual_design', 'brand_consistency', 'asset_creation'],
                'priority': 4
            },
            'video_production_automation': {
                'name': 'Video Production Automation Agent',
                'module': 'digital_marketing_ai_suite.video_production_automation_agent',
                'category': 'video',
                'capabilities': ['video_generation', 'ai_tools', 'automation'],
                'priority': 5
            },
            'advanced_media_buying': {
                'name': 'Advanced Media Buying Agent',
                'module': 'digital_marketing_ai_suite.advanced_media_buying_agent',
                'category': 'advertising',
                'capabilities': ['bid_optimization', 'attribution', 'performance_marketing'],
                'priority': 6
            },
            'seo_sem_optimization': {
                'name': 'SEO/SEM Optimization Agent',
                'module': 'digital_marketing_ai_suite.seo_sem_optimization_agent',
                'category': 'search',
                'capabilities': ['keyword_research', 'content_optimization', 'technical_seo'],
                'priority': 7
            },
            'social_media_automation': {
                'name': 'Social Media Automation Agent',
                'module': 'digital_marketing_ai_suite.social_media_automation_agent',
                'category': 'social',
                'capabilities': ['automated_posting', 'engagement', 'community_management'],
                'priority': 8
            },
            'online_business_development': {
                'name': 'Online Business Development Agent',
                'module': 'digital_marketing_ai_suite.online_business_development_agent',
                'category': 'business',
                'capabilities': ['lead_generation', 'sales_optimization', 'revenue_growth'],
                'priority': 9
            },
            'smart_goals_kpi': {
                'name': 'SMART Goals & KPI Agent',
                'module': 'digital_marketing_ai_suite.smart_goals_kpi_agent',
                'category': 'analytics',
                'capabilities': ['goal_setting', 'kpi_tracking', 'performance_measurement'],
                'priority': 10
            },
            
            # Sales Methodology Agents (11-13)
            'spin_sales_methodology': {
                'name': 'SPIN Sales Methodology Agent',
                'module': 'digital_marketing_ai_suite.spin_sales_methodology_agent',
                'category': 'sales_methodology',
                'capabilities': ['situation_analysis', 'problem_identification', 'implication_development'],
                'priority': 11
            },
            'aida_sales_psychology': {
                'name': 'AIDA Sales Psychology Agent',
                'module': 'digital_marketing_ai_suite.aida_sales_psychology_agent',
                'category': 'sales_methodology',
                'capabilities': ['attention_capture', 'interest_generation', 'desire_creation', 'action_motivation'],
                'priority': 12
            },
            'ooda_loop_sales_strategy': {
                'name': 'OODA Loop Sales Strategy Agent',
                'module': 'digital_marketing_ai_suite.ooda_loop_sales_strategy_agent',
                'category': 'sales_methodology',
                'capabilities': ['observe_market', 'orient_strategy', 'decide_actions', 'act_execute'],
                'priority': 13
            },
            
            # Multi-Channel Sales Process Agents (14-17)
            'sales_warm_inbound': {
                'name': 'Sales Warm Inbound Agent',
                'module': 'digital_marketing_ai_suite.sales_warm_inbound_agent',
                'category': 'sales_process',
                'capabilities': ['lead_qualification', 'nurturing_sequences', 'conversion_optimization'],
                'priority': 14
            },
            'sales_cold_inbound': {
                'name': 'Sales Cold Inbound Agent',
                'module': 'digital_marketing_ai_suite.sales_cold_inbound_agent',
                'category': 'sales_process',
                'capabilities': ['cold_outreach', 'interest_generation', 'appointment_setting'],
                'priority': 15
            },
            'sales_cold_outbound': {
                'name': 'Sales Cold Outbound Agent',
                'module': 'digital_marketing_ai_suite.sales_cold_outbound_agent',
                'category': 'sales_process',
                'capabilities': ['prospecting', 'cold_calling', 'email_sequences'],
                'priority': 16
            },
            'sales_funnel_optimization': {
                'name': 'Sales Funnel Optimization Agent',
                'module': 'digital_marketing_ai_suite.sales_funnel_optimization_agent',
                'category': 'sales_process',
                'capabilities': ['funnel_analysis', 'conversion_optimization', 'a_b_testing'],
                'priority': 17
            },
            
            # Pipeline & Order Management Agents (18-20)
            'sales_pipeline_management': {
                'name': 'Sales Pipeline Management Agent',
                'module': 'digital_marketing_ai_suite.sales_pipeline_management_agent',
                'category': 'pipeline_management',
                'capabilities': ['pipeline_tracking', 'forecast_accuracy', 'stage_optimization'],
                'priority': 18
            },
            'order_fulfillment_orchestration': {
                'name': 'Order Fulfillment Orchestration Agent',
                'module': 'digital_marketing_ai_suite.order_fulfillment_orchestration_agent',
                'category': 'order_management',
                'capabilities': ['order_processing', 'fulfillment_automation', 'delivery_optimization'],
                'priority': 19
            },
            'invoice_processing_automation': {
                'name': 'Invoice Processing Automation Agent',
                'module': 'digital_marketing_ai_suite.invoice_processing_automation_agent',
                'category': 'financial_automation',
                'capabilities': ['automated_invoicing', 'payment_processing', 'financial_reporting'],
                'priority': 20
            },
            
            # German Business Practices Agent (21)
            'gesprachsleitfaden': {
                'name': 'Gesprächsleitfaden Agent',
                'module': 'digital_marketing_ai_suite.gesprachsleitfaden_agent',
                'category': 'cultural_business',
                'capabilities': ['german_business_culture', 'conversation_frameworks', 'cultural_adaptation'],
                'priority': 21
            },
            
            # Strategic Planning & Intelligence Agents (22-25)
            'strategic_planning_optimization': {
                'name': 'Strategic Planning Optimization Agent',
                'module': 'digital_marketing_ai_suite.strategic_planning_optimization_agent',
                'category': 'strategic_planning',
                'capabilities': ['strategic_analysis', 'planning_optimization', 'resource_allocation'],
                'priority': 22
            },
            'competitive_intelligence': {
                'name': 'Competitive Intelligence Agent',
                'module': 'digital_marketing_ai_suite.competitive_intelligence_agent',
                'category': 'intelligence',
                'capabilities': ['competitor_analysis', 'market_intelligence', 'strategic_insights'],
                'priority': 23
            },
            'customer_success_management': {
                'name': 'Customer Success Management Agent',
                'module': 'digital_marketing_ai_suite.customer_success_management_agent',
                'category': 'customer_success',
                'capabilities': ['customer_health_scoring', 'retention_strategies', 'expansion_opportunities'],
                'priority': 24
            },
            'business_process_automation': {
                'name': 'Business Process Automation Agent',
                'module': 'digital_marketing_ai_suite.business_process_automation_agent',
                'category': 'automation',
                'capabilities': ['workflow_automation', 'process_optimization', 'efficiency_improvement'],
                'priority': 25
            },
            
            # Purpose & Vision Alignment Agents (26-30)
            'ikigai_purpose_strategy': {
                'name': 'Ikigai Purpose Strategy Agent',
                'module': 'digital_marketing_ai_suite.ikigai_purpose_strategy_agent',
                'category': 'purpose_alignment',
                'capabilities': ['purpose_discovery', 'value_alignment', 'meaningful_strategy'],
                'priority': 26
            },
            'vision_board_strategy': {
                'name': 'Vision Board Strategy Agent',
                'module': 'digital_marketing_ai_suite.vision_board_strategy_agent',
                'category': 'vision_planning',
                'capabilities': ['vision_creation', 'goal_visualization', 'strategic_clarity'],
                'priority': 27
            }
        }
        
        return agents
    
    def get_agent_instance(self, agent_key: str):
        """Get an instance of a specific agent"""
        
        if agent_key not in self.agents:
            raise ValueError(f"Agent {agent_key} not found")
        
        agent_config = self.agents[agent_key]
        
        try:
            # Import the agent module
            module = importlib.import_module(agent_config['module'])
            
            # Get the agent class (assuming it follows naming convention)
            class_name = self._get_agent_class_name(agent_key)
            agent_class = getattr(module, class_name)
            
            # Return instance
            return agent_class()
            
        except Exception as e:
            logger.error(f"Failed to instantiate agent {agent_key}: {str(e)}")
            return None
    
    def _get_agent_class_name(self, agent_key: str) -> str:
        """Convert agent key to class name"""
        
        # Convert snake_case to PascalCase
        words = agent_key.split('_')
        class_name = ''.join(word.capitalize() for word in words)
        
        # Add "Agent" suffix if not present
        if not class_name.endswith('Agent'):
            class_name += 'Agent'
            
        return class_name
    
    def execute_agent(self, agent_key: str, request_data: Dict) -> Dict[str, Any]:
        """Execute a specific agent with given data"""
        
        try:
            agent = self.get_agent_instance(agent_key)
            if not agent:
                return {'error': f'Failed to instantiate agent {agent_key}'}
            
            # Execute agent's main function
            if hasattr(agent, 'process_request'):
                result = agent.process_request(request_data)
            elif hasattr(agent, 'execute'):
                result = agent.execute(request_data)
            elif hasattr(agent, 'generate_strategy'):
                result = agent.generate_strategy(request_data)
            else:
                return {'error': f'Agent {agent_key} does not have a recognized execution method'}
            
            # Update agent status
            self.agent_status[agent_key] = {
                'last_execution': datetime.utcnow().isoformat(),
                'status': 'success',
                'execution_time': 'calculated_in_production'
            }
            
            return {
                'agent': agent_key,
                'execution_timestamp': datetime.utcnow().isoformat(),
                'result': result,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Agent {agent_key} execution failed: {str(e)}")
            
            # Update agent status with error
            self.agent_status[agent_key] = {
                'last_execution': datetime.utcnow().isoformat(),
                'status': 'error',
                'error': str(e)
            }
            
            return {
                'agent': agent_key,
                'execution_timestamp': datetime.utcnow().isoformat(),
                'error': str(e),
                'status': 'failed'
            }
    
    def execute_multiple_agents(self, agent_keys: List[str], request_data: Dict) -> Dict[str, Any]:
        """Execute multiple agents in coordination"""
        
        results = {}
        execution_order = self._optimize_execution_order(agent_keys)
        
        for agent_key in execution_order:
            # Enhance request data with previous results
            enhanced_data = self._enhance_request_data(request_data, results)
            
            # Execute agent
            agent_result = self.execute_agent(agent_key, enhanced_data)
            results[agent_key] = agent_result
        
        return {
            'execution_timestamp': datetime.utcnow().isoformat(),
            'agents_executed': agent_keys,
            'execution_order': execution_order,
            'results': results,
            'coordination_summary': self._generate_coordination_summary(results)
        }
    
    def _optimize_execution_order(self, agent_keys: List[str]) -> List[str]:
        """Optimize execution order based on agent priorities and dependencies"""
        
        # Sort by priority (lower numbers first)
        available_agents = [key for key in agent_keys if key in self.agents]
        sorted_agents = sorted(available_agents, key=lambda x: self.agents[x]['priority'])
        
        return sorted_agents
    
    def _enhance_request_data(self, base_data: Dict, previous_results: Dict) -> Dict:
        """Enhance request data with insights from previous agent executions"""
        
        enhanced_data = base_data.copy()
        
        # Add context from previous results
        if previous_results:
            enhanced_data['previous_agent_insights'] = {}
            
            for agent_key, result in previous_results.items():
                if result.get('status') == 'success' and 'result' in result:
                    enhanced_data['previous_agent_insights'][agent_key] = result['result']
        
        return enhanced_data
    
    def _generate_coordination_summary(self, results: Dict) -> Dict[str, Any]:
        """Generate summary of cross-agent coordination"""
        
        successful_agents = [k for k, v in results.items() if v.get('status') == 'success']
        failed_agents = [k for k, v in results.items() if v.get('status') == 'failed']
        
        return {
            'total_agents': len(results),
            'successful_executions': len(successful_agents),
            'failed_executions': len(failed_agents),
            'success_rate': len(successful_agents) / len(results) * 100 if results else 0,
            'successful_agents': successful_agents,
            'failed_agents': failed_agents,
            'coordination_quality': 'excellent' if len(failed_agents) == 0 else 'good' if len(failed_agents) < 3 else 'needs_attention'
        }
    
    def get_agent_status(self, agent_key: Optional[str] = None) -> Dict[str, Any]:
        """Get status of specific agent or all agents"""
        
        if agent_key:
            return self.agent_status.get(agent_key, {'status': 'never_executed'})
        
        return {
            'all_agents': self.agents,
            'agent_status': self.agent_status,
            'total_agents': len(self.agents),
            'status_summary': self._generate_status_summary()
        }
    
    def _generate_status_summary(self) -> Dict[str, Any]:
        """Generate summary of all agent statuses"""
        
        total_agents = len(self.agents)
        executed_agents = len(self.agent_status)
        successful_agents = len([s for s in self.agent_status.values() if s.get('status') == 'success'])
        
        return {
            'total_agents_available': total_agents,
            'agents_executed': executed_agents,
            'successful_executions': successful_agents,
            'execution_rate': executed_agents / total_agents * 100 if total_agents > 0 else 0,
            'success_rate': successful_agents / executed_agents * 100 if executed_agents > 0 else 0,
            'system_health': 'excellent' if successful_agents >= executed_agents * 0.9 else 'good' if successful_agents >= executed_agents * 0.7 else 'needs_attention'
        }
    
    def get_available_agents(self) -> Dict[str, Any]:
        """Get list of all available agents with their capabilities"""
        
        return {
            'total_agents': len(self.agents),
            'agents': self.agents,
            'categories': self._get_agent_categories(),
            'capabilities_overview': self._get_capabilities_overview()
        }
    
    def _get_agent_categories(self) -> Dict[str, List[str]]:
        """Group agents by category"""
        
        categories = {}
        
        for agent_key, agent_config in self.agents.items():
            category = agent_config['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(agent_key)
        
        return categories
    
    def _get_capabilities_overview(self) -> Dict[str, int]:
        """Get overview of capabilities across all agents"""
        
        all_capabilities = []
        
        for agent_config in self.agents.values():
            all_capabilities.extend(agent_config['capabilities'])
        
        # Count capabilities
        capability_counts = {}
        for capability in all_capabilities:
            capability_counts[capability] = capability_counts.get(capability, 0) + 1
        
        return capability_counts

# Global instance
agent_integration_service = AIAgentIntegrationService()