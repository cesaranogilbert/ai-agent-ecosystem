#!/usr/bin/env python3
"""
Multi-Agent Collaboration Service
Enables specialized AI agents to work together like a virtual company
"""

import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class AgentRole(Enum):
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist" 
    VALIDATOR = "validator"
    EXECUTOR = "executor"
    RESEARCHER = "researcher"
    COMMUNICATOR = "communicator"

class CollaborationPattern(Enum):
    SEQUENTIAL = "sequential"  # Agents work one after another
    PARALLEL = "parallel"     # Agents work simultaneously
    HIERARCHICAL = "hierarchical"  # Manager-worker pattern
    NETWORK = "network"       # Peer-to-peer collaboration
    CONSENSUS = "consensus"   # Agents vote on decisions

class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

@dataclass
class CollaborationTask:
    """Represents a task in multi-agent collaboration"""
    id: str
    name: str
    description: str
    required_capabilities: List[str]
    assigned_agent: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 5  # 1-10 scale
    dependencies: List[str] = field(default_factory=list)
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentCapability:
    """Defines what an agent can do"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    performance_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass 
class CollaborativeAgent:
    """Enhanced agent definition for collaboration"""
    name: str
    role: AgentRole
    capabilities: List[AgentCapability]
    max_concurrent_tasks: int = 3
    current_tasks: Set[str] = field(default_factory=set)
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    collaboration_preferences: Dict[str, Any] = field(default_factory=dict)
    status: str = "available"  # available, busy, offline
    last_active: datetime = field(default_factory=datetime.now)

@dataclass
class CollaborationProject:
    """A project managed by multiple agents"""
    id: str
    name: str
    description: str
    objectives: List[str]
    tasks: List[CollaborationTask] = field(default_factory=list)
    participating_agents: List[str] = field(default_factory=list)
    collaboration_pattern: CollaborationPattern = CollaborationPattern.NETWORK
    status: str = "active"
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    budget: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class MultiAgentCollaborationService:
    """
    Advanced multi-agent collaboration enabling specialized agents to work together
    Supports various collaboration patterns and automatic task distribution
    """
    
    def __init__(self):
        self.agents: Dict[str, CollaborativeAgent] = {}
        self.projects: Dict[str, CollaborationProject] = {}
        self.task_queue: List[CollaborationTask] = []
        self.active_collaborations: Dict[str, Dict[str, Any]] = {}
        self.message_bus: Dict[str, List[Dict[str, Any]]] = {}
        
        # Initialize agents from your 38-agent ecosystem
        self._initialize_collaborative_agents()
        
        # Task scheduler
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    def _initialize_collaborative_agents(self):
        """Initialize collaborative agents from your existing 38-agent ecosystem"""
        
        # Sales & Acquisition Specialists
        acquisition_agents = [
            CollaborativeAgent(
                name="client_acquisition_specialist",
                role=AgentRole.SPECIALIST,
                capabilities=[
                    AgentCapability("lead_generation", "Generate qualified leads", {}, {}),
                    AgentCapability("market_analysis", "Analyze market opportunities", {}, {}),
                    AgentCapability("competitor_research", "Research competition", {}, {})
                ],
                collaboration_preferences={"works_well_with": ["high_ticket_closer", "low_ticket_closer"]}
            ),
            CollaborativeAgent(
                name="high_ticket_closer",
                role=AgentRole.EXECUTOR,
                capabilities=[
                    AgentCapability("deal_negotiation", "Negotiate high-value deals", {}, {}),
                    AgentCapability("stakeholder_mapping", "Map decision makers", {}, {}),
                    AgentCapability("proposal_creation", "Create winning proposals", {}, {})
                ],
                collaboration_preferences={"requires_input_from": ["client_acquisition_specialist"]}
            ),
            CollaborativeAgent(
                name="low_ticket_closer",
                role=AgentRole.EXECUTOR,
                capabilities=[
                    AgentCapability("volume_closing", "Handle high-volume sales", {}, {}),
                    AgentCapability("conversion_optimization", "Optimize conversion funnels", {}, {}),
                    AgentCapability("follow_up_automation", "Automate follow-up sequences", {}, {})
                ],
                max_concurrent_tasks=5
            )
        ]
        
        # Content & Marketing Team
        content_agents = [
            CollaborativeAgent(
                name="content_creator",
                role=AgentRole.SPECIALIST,
                capabilities=[
                    AgentCapability("content_writing", "Create engaging content", {}, {}),
                    AgentCapability("storytelling", "Craft compelling narratives", {}, {}),
                    AgentCapability("audience_targeting", "Target specific audiences", {}, {})
                ],
                collaboration_preferences={"works_well_with": ["seo_sem", "brand_storytelling"]}
            ),
            CollaborativeAgent(
                name="seo_sem",
                role=AgentRole.VALIDATOR,
                capabilities=[
                    AgentCapability("keyword_optimization", "Optimize for search", {}, {}),
                    AgentCapability("performance_tracking", "Track SEO performance", {}, {}),
                    AgentCapability("competitor_seo_analysis", "Analyze competitor SEO", {}, {})
                ]
            ),
            CollaborativeAgent(
                name="brand_storytelling",
                role=AgentRole.COMMUNICATOR,
                capabilities=[
                    AgentCapability("brand_messaging", "Develop brand voice", {}, {}),
                    AgentCapability("emotional_connection", "Create emotional bonds", {}, {}),
                    AgentCapability("narrative_consistency", "Maintain story consistency", {}, {})
                ]
            )
        ]
        
        # Research & Intelligence Team  
        research_agents = [
            CollaborativeAgent(
                name="wealth_generation_research",
                role=AgentRole.RESEARCHER,
                capabilities=[
                    AgentCapability("opportunity_scanning", "Scan for opportunities", {}, {}),
                    AgentCapability("market_trend_analysis", "Analyze market trends", {}, {}),
                    AgentCapability("risk_assessment", "Assess investment risks", {}, {})
                ],
                collaboration_preferences={"provides_data_to": ["arbitrage_opportunity", "master_strategist"]}
            ),
            CollaborativeAgent(
                name="arbitrage_opportunity",
                role=AgentRole.SPECIALIST,
                capabilities=[
                    AgentCapability("price_difference_detection", "Find arbitrage opportunities", {}, {}),
                    AgentCapability("execution_automation", "Automate arbitrage execution", {}, {}),
                    AgentCapability("profit_calculation", "Calculate profit potential", {}, {})
                ],
                collaboration_preferences={"requires_input_from": ["wealth_generation_research"]}
            )
        ]
        
        # Management & Operations Team
        management_agents = [
            CollaborativeAgent(
                name="project_management_suite",
                role=AgentRole.COORDINATOR,
                capabilities=[
                    AgentCapability("project_planning", "Create project plans", {}, {}),
                    AgentCapability("resource_allocation", "Allocate resources", {}, {}),
                    AgentCapability("progress_tracking", "Track project progress", {}, {}),
                    AgentCapability("risk_management", "Manage project risks", {}, {})
                ],
                collaboration_preferences={"coordinates_with": "all"}
            ),
            CollaborativeAgent(
                name="negotiation_mediator",
                role=AgentRole.VALIDATOR,
                capabilities=[
                    AgentCapability("conflict_resolution", "Resolve conflicts", {}, {}),
                    AgentCapability("deal_structuring", "Structure complex deals", {}, {}),
                    AgentCapability("stakeholder_communication", "Communicate with stakeholders", {}, {})
                ]
            ),
            CollaborativeAgent(
                name="master_strategist",
                role=AgentRole.COORDINATOR,
                capabilities=[
                    AgentCapability("strategic_planning", "Develop strategies", {}, {}),
                    AgentCapability("market_positioning", "Position in market", {}, {}),
                    AgentCapability("competitive_advantage", "Identify advantages", {}, {})
                ],
                collaboration_preferences={"coordinates_with": "all", "has_veto_power": True}
            )
        ]
        
        # Register all agents
        all_agents = acquisition_agents + content_agents + research_agents + management_agents
        
        for agent in all_agents:
            self.agents[agent.name] = agent
            self.message_bus[agent.name] = []
            
        logger.info(f"Initialized {len(all_agents)} collaborative agents")
    
    def create_collaboration_project(self, name: str, description: str, objectives: List[str],
                                   collaboration_pattern: CollaborationPattern = CollaborationPattern.NETWORK) -> str:
        """Create a new collaboration project"""
        project_id = f"collab_project_{uuid.uuid4().hex[:8]}"
        
        project = CollaborationProject(
            id=project_id,
            name=name,
            description=description,
            objectives=objectives,
            collaboration_pattern=collaboration_pattern
        )
        
        self.projects[project_id] = project
        logger.info(f"Created collaboration project: {name} ({project_id})")
        return project_id
    
    def add_task_to_project(self, project_id: str, task_name: str, task_description: str,
                           required_capabilities: List[str], priority: int = 5) -> str:
        """Add a task to a collaboration project"""
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        task = CollaborationTask(
            id=task_id,
            name=task_name,
            description=task_description,
            required_capabilities=required_capabilities,
            priority=priority
        )
        
        self.projects[project_id].tasks.append(task)
        self.task_queue.append(task)
        
        logger.info(f"Added task '{task_name}' to project {project_id}")
        return task_id
    
    def assign_agents_to_project(self, project_id: str, agent_names: List[str]) -> bool:
        """Assign agents to a collaboration project"""
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        # Verify all agents exist
        for agent_name in agent_names:
            if agent_name not in self.agents:
                raise ValueError(f"Agent {agent_name} not found")
        
        self.projects[project_id].participating_agents = agent_names
        
        # Initialize collaboration context
        self.active_collaborations[project_id] = {
            "agents": agent_names,
            "communication_log": [],
            "shared_context": {},
            "coordination_state": "initializing"
        }
        
        logger.info(f"Assigned {len(agent_names)} agents to project {project_id}")
        return True
    
    async def auto_assign_tasks(self, project_id: str) -> Dict[str, List[str]]:
        """Automatically assign tasks to best-suited agents"""
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        assignments = {}
        
        # Get available agents for this project
        available_agents = [
            self.agents[name] for name in project.participating_agents
            if self.agents[name].status == "available"
        ]
        
        for task in project.tasks:
            if task.status != TaskStatus.PENDING:
                continue
                
            # Find agents with required capabilities
            suitable_agents = []
            
            for agent in available_agents:
                agent_capabilities = [cap.name for cap in agent.capabilities]
                
                # Check if agent has required capabilities
                capability_match = sum(1 for cap in task.required_capabilities 
                                     if cap in agent_capabilities)
                
                if capability_match > 0:
                    # Calculate suitability score
                    capability_score = capability_match / len(task.required_capabilities)
                    workload_score = 1.0 - (len(agent.current_tasks) / agent.max_concurrent_tasks)
                    
                    # Consider collaboration preferences
                    preference_score = 1.0
                    if "works_well_with" in agent.collaboration_preferences:
                        project_agents = set(project.participating_agents)
                        preferred_agents = set(agent.collaboration_preferences["works_well_with"])
                        if project_agents.intersection(preferred_agents):
                            preference_score = 1.2
                    
                    total_score = capability_score * workload_score * preference_score
                    
                    suitable_agents.append((agent, total_score))
            
            if suitable_agents:
                # Sort by suitability score and assign to best agent
                suitable_agents.sort(key=lambda x: x[1], reverse=True)
                best_agent = suitable_agents[0][0]
                
                task.assigned_agent = best_agent.name
                task.status = TaskStatus.ASSIGNED
                best_agent.current_tasks.add(task.id)
                
                if best_agent.name not in assignments:
                    assignments[best_agent.name] = []
                assignments[best_agent.name].append(task.id)
        
        logger.info(f"Auto-assigned tasks in project {project_id}: {assignments}")
        return assignments
    
    async def execute_collaboration_project(self, project_id: str) -> Dict[str, Any]:
        """Execute a collaboration project with all assigned tasks"""
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        execution_context = {
            "project_id": project_id,
            "start_time": datetime.now(),
            "task_results": {},
            "agent_communications": [],
            "milestones": [],
            "status": "running"
        }
        
        try:
            # Auto-assign tasks if not already assigned
            await self.auto_assign_tasks(project_id)
            
            # Execute based on collaboration pattern
            if project.collaboration_pattern == CollaborationPattern.SEQUENTIAL:
                await self._execute_sequential_collaboration(project, execution_context)
            elif project.collaboration_pattern == CollaborationPattern.PARALLEL:
                await self._execute_parallel_collaboration(project, execution_context)
            elif project.collaboration_pattern == CollaborationPattern.HIERARCHICAL:
                await self._execute_hierarchical_collaboration(project, execution_context)
            else:
                await self._execute_network_collaboration(project, execution_context)
            
            execution_context["status"] = "completed"
            execution_context["end_time"] = datetime.now()
            
        except Exception as e:
            execution_context["status"] = "failed"
            execution_context["error"] = str(e)
            execution_context["end_time"] = datetime.now()
            logger.error(f"Collaboration project execution failed: {str(e)}")
        
        return execution_context
    
    async def _execute_network_collaboration(self, project: CollaborationProject, 
                                           execution_context: Dict[str, Any]):
        """Execute network-style collaboration (peer-to-peer)"""
        # Create task execution futures
        task_futures = []
        
        for task in project.tasks:
            if task.assigned_agent and task.status == TaskStatus.ASSIGNED:
                future = self._execute_collaborative_task(task, project, execution_context)
                task_futures.append(future)
        
        # Execute all tasks concurrently with coordination
        if task_futures:
            results = await asyncio.gather(*task_futures, return_exceptions=True)
            
            for i, result in enumerate(results):
                task_id = project.tasks[i].id
                if isinstance(result, Exception):
                    execution_context["task_results"][task_id] = {
                        "status": "failed",
                        "error": str(result)
                    }
                else:
                    execution_context["task_results"][task_id] = result
    
    async def _execute_sequential_collaboration(self, project: CollaborationProject,
                                              execution_context: Dict[str, Any]):
        """Execute sequential collaboration (one task after another)"""
        # Sort tasks by priority and dependencies
        sorted_tasks = sorted(project.tasks, key=lambda t: (-t.priority, len(t.dependencies)))
        
        for task in sorted_tasks:
            if task.assigned_agent and task.status == TaskStatus.ASSIGNED:
                result = await self._execute_collaborative_task(task, project, execution_context)
                execution_context["task_results"][task.id] = result
                
                # Share results with other agents
                await self._broadcast_task_result(task, result, project)
    
    async def _execute_parallel_collaboration(self, project: CollaborationProject,
                                            execution_context: Dict[str, Any]):
        """Execute parallel collaboration (all tasks simultaneously)"""
        tasks_by_agent = {}
        
        # Group tasks by assigned agent
        for task in project.tasks:
            if task.assigned_agent and task.status == TaskStatus.ASSIGNED:
                if task.assigned_agent not in tasks_by_agent:
                    tasks_by_agent[task.assigned_agent] = []
                tasks_by_agent[task.assigned_agent].append(task)
        
        # Execute agent workloads in parallel
        agent_futures = []
        for agent_name, agent_tasks in tasks_by_agent.items():
            future = self._execute_agent_workload(agent_name, agent_tasks, project, execution_context)
            agent_futures.append(future)
        
        if agent_futures:
            results = await asyncio.gather(*agent_futures, return_exceptions=True)
            
            # Merge results
            for result in results:
                if isinstance(result, dict):
                    execution_context["task_results"].update(result)
    
    async def _execute_hierarchical_collaboration(self, project: CollaborationProject,
                                                execution_context: Dict[str, Any]):
        """Execute hierarchical collaboration (manager-worker pattern)"""
        # Find coordinator agent
        coordinator = None
        for agent_name in project.participating_agents:
            agent = self.agents[agent_name]
            if agent.role == AgentRole.COORDINATOR:
                coordinator = agent
                break
        
        if coordinator:
            # Coordinator manages the entire project
            coordinator_result = await self._execute_coordinator_workflow(
                coordinator, project, execution_context
            )
            execution_context["coordinator_result"] = coordinator_result
        else:
            # Fall back to network collaboration
            await self._execute_network_collaboration(project, execution_context)
    
    async def _execute_collaborative_task(self, task: CollaborationTask, project: CollaborationProject,
                                        execution_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single collaborative task"""
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        try:
            agent = self.agents[task.assigned_agent]
            
            # Simulate task execution with agent coordination
            result = {
                "task_id": task.id,
                "agent": task.assigned_agent,
                "status": "completed",
                "output": f"Task '{task.name}' completed by {task.assigned_agent}",
                "capabilities_used": task.required_capabilities,
                "execution_time": 2.5,  # Simulated execution time
                "quality_score": 0.92,  # Simulated quality score
                "timestamp": datetime.now().isoformat()
            }
            
            # Update task status
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.output_data = result
            
            # Remove from agent's current tasks
            agent.current_tasks.discard(task.id)
            
            # Log agent performance
            agent.performance_history.append({
                "task_id": task.id,
                "completion_time": task.completed_at - task.started_at,
                "quality_score": result["quality_score"],
                "timestamp": task.completed_at
            })
            
            return result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
            
            return {
                "task_id": task.id,
                "agent": task.assigned_agent,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _broadcast_task_result(self, task: CollaborationTask, result: Dict[str, Any],
                                   project: CollaborationProject):
        """Broadcast task result to other agents in the project"""
        message = {
            "type": "task_completion",
            "task_id": task.id,
            "task_name": task.name,
            "completed_by": task.assigned_agent,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to all project participants
        for agent_name in project.participating_agents:
            if agent_name != task.assigned_agent:
                self.message_bus[agent_name].append(message)
    
    async def _execute_agent_workload(self, agent_name: str, tasks: List[CollaborationTask],
                                    project: CollaborationProject, execution_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all tasks assigned to a specific agent"""
        results = {}
        
        for task in tasks:
            result = await self._execute_collaborative_task(task, project, execution_context)
            results[task.id] = result
        
        return results
    
    async def _execute_coordinator_workflow(self, coordinator: CollaborativeAgent,
                                          project: CollaborationProject, execution_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute coordinator-managed workflow"""
        # Coordinator orchestrates the entire project
        return {
            "coordinator": coordinator.name,
            "managed_tasks": len(project.tasks),
            "coordination_style": "hierarchical",
            "status": "managed"
        }
    
    def get_agent_performance(self, agent_name: str) -> Dict[str, Any]:
        """Get performance metrics for an agent"""
        if agent_name not in self.agents:
            raise ValueError(f"Agent {agent_name} not found")
        
        agent = self.agents[agent_name]
        performance_data = {
            "agent": agent_name,
            "role": agent.role.value,
            "total_tasks_completed": len(agent.performance_history),
            "current_workload": len(agent.current_tasks),
            "max_capacity": agent.max_concurrent_tasks,
            "capacity_utilization": len(agent.current_tasks) / agent.max_concurrent_tasks,
            "status": agent.status,
            "last_active": agent.last_active.isoformat()
        }
        
        if agent.performance_history:
            avg_quality = sum(p["quality_score"] for p in agent.performance_history) / len(agent.performance_history)
            performance_data["average_quality_score"] = avg_quality
            performance_data["recent_performance"] = agent.performance_history[-5:]  # Last 5 tasks
        
        return performance_data
    
    def get_collaboration_insights(self, project_id: str) -> Dict[str, Any]:
        """Get insights about collaboration effectiveness"""
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.projects[project_id]
        
        # Calculate collaboration metrics
        completed_tasks = [t for t in project.tasks if t.status == TaskStatus.COMPLETED]
        total_tasks = len(project.tasks)
        
        insights = {
            "project_id": project_id,
            "project_name": project.name,
            "completion_rate": len(completed_tasks) / total_tasks if total_tasks > 0 else 0,
            "collaboration_pattern": project.collaboration_pattern.value,
            "participating_agents": len(project.participating_agents),
            "task_distribution": {},
            "agent_collaboration_scores": {}
        }
        
        # Task distribution analysis
        for task in project.tasks:
            agent = task.assigned_agent
            if agent:
                if agent not in insights["task_distribution"]:
                    insights["task_distribution"][agent] = 0
                insights["task_distribution"][agent] += 1
        
        # Agent collaboration effectiveness
        for agent_name in project.participating_agents:
            agent_tasks = [t for t in project.tasks if t.assigned_agent == agent_name]
            completed_agent_tasks = [t for t in agent_tasks if t.status == TaskStatus.COMPLETED]
            
            if agent_tasks:
                completion_rate = len(completed_agent_tasks) / len(agent_tasks)
                insights["agent_collaboration_scores"][agent_name] = completion_rate
        
        return insights
    
    def list_active_collaborations(self) -> List[Dict[str, Any]]:
        """List all active collaboration projects"""
        return [
            {
                "id": project.id,
                "name": project.name,
                "status": project.status,
                "agents": len(project.participating_agents),
                "tasks": len(project.tasks),
                "completed_tasks": len([t for t in project.tasks if t.status == TaskStatus.COMPLETED]),
                "pattern": project.collaboration_pattern.value,
                "created_at": project.created_at.isoformat()
            }
            for project in self.projects.values()
            if project.status == "active"
        ]

# Global multi-agent collaboration service instance
collaboration_service = MultiAgentCollaborationService()