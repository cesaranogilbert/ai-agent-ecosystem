#!/usr/bin/env python3
"""
Visual Workflow Orchestration Service
Drag-and-drop agent workflow builder for non-technical users
"""

import json
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import logging

logger = logging.getLogger(__name__)

class NodeType(Enum):
    AGENT = "agent"
    TRIGGER = "trigger" 
    ACTION = "action"
    CONDITION = "condition"
    DATA_TRANSFORM = "data_transform"
    OUTPUT = "output"
    MCP_TOOL = "mcp_tool"

class ConnectionType(Enum):
    SUCCESS = "success"
    ERROR = "error"
    CONDITIONAL = "conditional"
    DATA = "data"

@dataclass
class WorkflowNode:
    """Represents a node in the visual workflow"""
    id: str
    type: NodeType
    name: str
    description: str
    position: Dict[str, float]  # {x: float, y: float}
    config: Dict[str, Any] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    agent_name: Optional[str] = None
    mcp_tool: Optional[str] = None

@dataclass
class WorkflowConnection:
    """Represents a connection between workflow nodes"""
    id: str
    source_node: str
    target_node: str
    source_port: str
    target_port: str
    connection_type: ConnectionType
    conditions: Dict[str, Any] = field(default_factory=dict)
    data_mapping: Dict[str, str] = field(default_factory=dict)

@dataclass
class VisualWorkflow:
    """Complete visual workflow definition"""
    id: str
    name: str
    description: str
    nodes: List[WorkflowNode] = field(default_factory=list)
    connections: List[WorkflowConnection] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    status: str = "draft"

class VisualWorkflowService:
    """
    Visual workflow orchestration service enabling drag-and-drop agent coordination
    Converts visual workflows into executable agent sequences
    """
    
    def __init__(self):
        self.workflows: Dict[str, VisualWorkflow] = {}
        self.templates: Dict[str, VisualWorkflow] = {}
        self.execution_history: Dict[str, List[Dict]] = {}
        
        # Initialize pre-built workflow templates
        self._initialize_workflow_templates()
    
    def _initialize_workflow_templates(self):
        """Initialize common workflow templates for business scenarios"""
        
        # Template 1: Lead Generation to Closing
        lead_to_close_workflow = VisualWorkflow(
            id="lead_to_close_template",
            name="Lead Generation to Deal Closing",
            description="Complete sales pipeline from lead generation to deal closing",
            nodes=[
                WorkflowNode(
                    id="trigger_1", type=NodeType.TRIGGER, name="New Lead Trigger",
                    description="Triggers when new lead is identified",
                    position={"x": 100, "y": 100},
                    config={"trigger_type": "webhook", "source": "lead_form"}
                ),
                WorkflowNode(
                    id="agent_1", type=NodeType.AGENT, name="Lead Qualification",
                    description="Qualify and score incoming leads",
                    position={"x": 300, "y": 100},
                    agent_name="client_acquisition_specialist",
                    config={"qualification_criteria": {"budget": ">10000", "timeline": "<6months"}}
                ),
                WorkflowNode(
                    id="condition_1", type=NodeType.CONDITION, name="High Value Lead?",
                    description="Check if lead meets high-value criteria",
                    position={"x": 500, "y": 100},
                    config={"condition": "lead_score > 75"}
                ),
                WorkflowNode(
                    id="agent_2", type=NodeType.AGENT, name="High Ticket Closer",
                    description="Handle high-value deal closing",
                    position={"x": 700, "y": 50},
                    agent_name="high_ticket_closer",
                    config={"deal_threshold": 50000}
                ),
                WorkflowNode(
                    id="agent_3", type=NodeType.AGENT, name="Low Ticket Closer",
                    description="Handle standard deal closing",
                    position={"x": 700, "y": 150},
                    agent_name="low_ticket_closer",
                    config={"deal_threshold": 5000}
                ),
                WorkflowNode(
                    id="mcp_1", type=NodeType.MCP_TOOL, name="CRM Update",
                    description="Update CRM with deal outcome",
                    position={"x": 900, "y": 100},
                    mcp_tool="salesforce_connector",
                    config={"action": "update_opportunity"}
                )
            ],
            connections=[
                WorkflowConnection("conn_1", "trigger_1", "agent_1", "output", "input", ConnectionType.DATA),
                WorkflowConnection("conn_2", "agent_1", "condition_1", "output", "input", ConnectionType.DATA),
                WorkflowConnection("conn_3", "condition_1", "agent_2", "true", "input", ConnectionType.CONDITIONAL),
                WorkflowConnection("conn_4", "condition_1", "agent_3", "false", "input", ConnectionType.CONDITIONAL),
                WorkflowConnection("conn_5", "agent_2", "mcp_1", "output", "input", ConnectionType.SUCCESS),
                WorkflowConnection("conn_6", "agent_3", "mcp_1", "output", "input", ConnectionType.SUCCESS)
            ]
        )
        
        # Template 2: Content Creation Pipeline
        content_pipeline_workflow = VisualWorkflow(
            id="content_pipeline_template",
            name="AI Content Creation Pipeline",
            description="Automated content creation from research to publication",
            nodes=[
                WorkflowNode(
                    id="trigger_2", type=NodeType.TRIGGER, name="Content Request",
                    description="Triggers when content is requested",
                    position={"x": 100, "y": 200},
                    config={"trigger_type": "schedule", "frequency": "daily"}
                ),
                WorkflowNode(
                    id="agent_4", type=NodeType.AGENT, name="Market Research",
                    description="Research market trends and topics",
                    position={"x": 300, "y": 200},
                    agent_name="wealth_generation_research",
                    config={"research_depth": "comprehensive"}
                ),
                WorkflowNode(
                    id="agent_5", type=NodeType.AGENT, name="Content Creation",
                    description="Create engaging content based on research",
                    position={"x": 500, "y": 200},
                    agent_name="content_creator",
                    config={"content_type": "blog_post", "word_count": 1500}
                ),
                WorkflowNode(
                    id="agent_6", type=NodeType.AGENT, name="SEO Optimization",
                    description="Optimize content for search engines",
                    position={"x": 700, "y": 200},
                    agent_name="seo_sem",
                    config={"target_keywords": 5, "optimization_level": "advanced"}
                ),
                WorkflowNode(
                    id="mcp_2", type=NodeType.MCP_TOOL, name="Publish Content",
                    description="Publish to content management system",
                    position={"x": 900, "y": 200},
                    mcp_tool="google_drive_manager",
                    config={"action": "create_file", "folder": "published_content"}
                )
            ],
            connections=[
                WorkflowConnection("conn_7", "trigger_2", "agent_4", "output", "input", ConnectionType.DATA),
                WorkflowConnection("conn_8", "agent_4", "agent_5", "output", "input", ConnectionType.DATA),
                WorkflowConnection("conn_9", "agent_5", "agent_6", "output", "input", ConnectionType.DATA),
                WorkflowConnection("conn_10", "agent_6", "mcp_2", "output", "input", ConnectionType.SUCCESS)
            ]
        )
        
        # Template 3: Project Management Automation
        project_mgmt_workflow = VisualWorkflow(
            id="project_mgmt_template",
            name="Automated Project Management",
            description="End-to-end project management with AI coordination",
            nodes=[
                WorkflowNode(
                    id="trigger_3", type=NodeType.TRIGGER, name="New Project",
                    description="Triggers when new project is created",
                    position={"x": 100, "y": 300},
                    config={"trigger_type": "manual", "approval_required": True}
                ),
                WorkflowNode(
                    id="agent_7", type=NodeType.AGENT, name="Project Planning",
                    description="Create comprehensive project plan",
                    position={"x": 300, "y": 300},
                    agent_name="project_management_suite",
                    config={"methodology": "agile", "sprint_length": 14}
                ),
                WorkflowNode(
                    id="mcp_3", type=NodeType.MCP_TOOL, name="Create JIRA Project",
                    description="Set up project in JIRA",
                    position={"x": 500, "y": 300},
                    mcp_tool="jira_integration",
                    config={"action": "create_project", "template": "scrum"}
                ),
                WorkflowNode(
                    id="mcp_4", type=NodeType.MCP_TOOL, name="Schedule Kickoff",
                    description="Schedule project kickoff meeting",
                    position={"x": 700, "y": 300},
                    mcp_tool="zoom_manager",
                    config={"action": "create_meeting", "duration": 60}
                ),
                WorkflowNode(
                    id="mcp_5", type=NodeType.MCP_TOOL, name="Team Notification",
                    description="Notify team of new project",
                    position={"x": 900, "y": 300},
                    mcp_tool="slack_messenger",
                    config={"action": "send_message", "channel": "#projects"}
                )
            ],
            connections=[
                WorkflowConnection("conn_11", "trigger_3", "agent_7", "output", "input", ConnectionType.DATA),
                WorkflowConnection("conn_12", "agent_7", "mcp_3", "output", "input", ConnectionType.DATA),
                WorkflowConnection("conn_13", "mcp_3", "mcp_4", "output", "input", ConnectionType.SUCCESS),
                WorkflowConnection("conn_14", "mcp_4", "mcp_5", "output", "input", ConnectionType.SUCCESS)
            ]
        )
        
        # Store templates
        self.templates["lead_to_close"] = lead_to_close_workflow
        self.templates["content_pipeline"] = content_pipeline_workflow
        self.templates["project_management"] = project_mgmt_workflow
        
        logger.info(f"Initialized {len(self.templates)} workflow templates")
    
    def create_workflow(self, name: str, description: str = "") -> str:
        """Create a new empty workflow"""
        workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
        workflow = VisualWorkflow(
            id=workflow_id,
            name=name,
            description=description
        )
        self.workflows[workflow_id] = workflow
        logger.info(f"Created new workflow: {name} ({workflow_id})")
        return workflow_id
    
    def create_from_template(self, template_name: str, workflow_name: str) -> str:
        """Create a workflow from a pre-built template"""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found. Available: {list(self.templates.keys())}")
        
        template = self.templates[template_name]
        workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
        
        # Deep copy template with new ID
        workflow = VisualWorkflow(
            id=workflow_id,
            name=workflow_name,
            description=f"Created from template: {template.description}",
            nodes=[
                WorkflowNode(
                    id=f"{workflow_id}_{node.id}",
                    type=node.type,
                    name=node.name,
                    description=node.description,
                    position=node.position.copy(),
                    config=node.config.copy(),
                    inputs=node.inputs.copy(),
                    outputs=node.outputs.copy(),
                    metadata=node.metadata.copy(),
                    agent_name=node.agent_name,
                    mcp_tool=node.mcp_tool
                ) for node in template.nodes
            ],
            connections=[
                WorkflowConnection(
                    id=f"{workflow_id}_{conn.id}",
                    source_node=f"{workflow_id}_{conn.source_node}",
                    target_node=f"{workflow_id}_{conn.target_node}",
                    source_port=conn.source_port,
                    target_port=conn.target_port,
                    connection_type=conn.connection_type,
                    conditions=conn.conditions.copy(),
                    data_mapping=conn.data_mapping.copy()
                ) for conn in template.connections
            ],
            variables=template.variables.copy(),
            settings=template.settings.copy()
        )
        
        self.workflows[workflow_id] = workflow
        logger.info(f"Created workflow from template '{template_name}': {workflow_name} ({workflow_id})")
        return workflow_id
    
    def add_node(self, workflow_id: str, node_type: NodeType, name: str, 
                 position: Dict[str, float], config: Dict[str, Any] = None) -> str:
        """Add a node to the workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        node_id = f"node_{uuid.uuid4().hex[:8]}"
        node = WorkflowNode(
            id=node_id,
            type=node_type,
            name=name,
            description=f"{node_type.value.title()} node: {name}",
            position=position,
            config=config or {}
        )
        
        self.workflows[workflow_id].nodes.append(node)
        self.workflows[workflow_id].updated_at = datetime.now()
        
        logger.info(f"Added node '{name}' to workflow {workflow_id}")
        return node_id
    
    def connect_nodes(self, workflow_id: str, source_node: str, target_node: str,
                     connection_type: ConnectionType, source_port: str = "output",
                     target_port: str = "input") -> str:
        """Connect two nodes in the workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        connection_id = f"conn_{uuid.uuid4().hex[:8]}"
        connection = WorkflowConnection(
            id=connection_id,
            source_node=source_node,
            target_node=target_node,
            source_port=source_port,
            target_port=target_port,
            connection_type=connection_type
        )
        
        self.workflows[workflow_id].connections.append(connection)
        self.workflows[workflow_id].updated_at = datetime.now()
        
        logger.info(f"Connected nodes {source_node} -> {target_node} in workflow {workflow_id}")
        return connection_id
    
    def get_workflow(self, workflow_id: str) -> Optional[VisualWorkflow]:
        """Get workflow by ID"""
        return self.workflows.get(workflow_id)
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflows with summary information"""
        return [
            {
                "id": workflow.id,
                "name": workflow.name,
                "description": workflow.description,
                "nodes": len(workflow.nodes),
                "connections": len(workflow.connections),
                "status": workflow.status,
                "created_at": workflow.created_at.isoformat(),
                "updated_at": workflow.updated_at.isoformat()
            }
            for workflow in self.workflows.values()
        ]
    
    def get_templates(self) -> List[Dict[str, Any]]:
        """Get available workflow templates"""
        return [
            {
                "name": name,
                "id": template.id,
                "description": template.description,
                "nodes": len(template.nodes),
                "connections": len(template.connections),
                "use_cases": template.metadata.get("use_cases", [])
            }
            for name, template in self.templates.items()
        ]
    
    async def execute_workflow(self, workflow_id: str, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a visual workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        execution_id = f"exec_{uuid.uuid4().hex[:8]}"
        
        # Initialize execution context
        execution_context = {
            "workflow_id": workflow_id,
            "execution_id": execution_id,
            "input_data": input_data or {},
            "variables": workflow.variables.copy(),
            "node_outputs": {},
            "execution_log": [],
            "start_time": datetime.now(),
            "status": "running"
        }
        
        try:
            # Find trigger nodes to start execution
            trigger_nodes = [node for node in workflow.nodes if node.type == NodeType.TRIGGER]
            
            if not trigger_nodes:
                raise ValueError("Workflow has no trigger nodes")
            
            # Execute workflow starting from triggers
            for trigger_node in trigger_nodes:
                await self._execute_node_chain(workflow, trigger_node, execution_context)
            
            execution_context["status"] = "completed"
            execution_context["end_time"] = datetime.now()
            
        except Exception as e:
            execution_context["status"] = "failed"
            execution_context["error"] = str(e)
            execution_context["end_time"] = datetime.now()
            logger.error(f"Workflow execution failed: {str(e)}")
        
        # Store execution history
        if workflow_id not in self.execution_history:
            self.execution_history[workflow_id] = []
        self.execution_history[workflow_id].append(execution_context)
        
        return execution_context
    
    async def _execute_node_chain(self, workflow: VisualWorkflow, start_node: WorkflowNode, 
                                execution_context: Dict[str, Any]):
        """Execute a chain of connected nodes"""
        current_node = start_node
        visited_nodes = set()
        
        while current_node and current_node.id not in visited_nodes:
            visited_nodes.add(current_node.id)
            
            # Execute current node
            node_result = await self._execute_single_node(current_node, execution_context)
            execution_context["node_outputs"][current_node.id] = node_result
            
            # Log execution
            execution_context["execution_log"].append({
                "node_id": current_node.id,
                "node_name": current_node.name,
                "node_type": current_node.type.value,
                "result": node_result,
                "timestamp": datetime.now().isoformat()
            })
            
            # Find next node based on connections and results
            next_node = self._find_next_node(workflow, current_node, node_result, execution_context)
            current_node = next_node
    
    async def _execute_single_node(self, node: WorkflowNode, execution_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow node"""
        try:
            if node.type == NodeType.TRIGGER:
                return {"status": "triggered", "data": execution_context["input_data"]}
            
            elif node.type == NodeType.AGENT:
                # Import and execute agent
                from services.mcp_integration_service import execute_agent_tool
                
                # Simulate agent execution
                return {
                    "status": "success",
                    "agent": node.agent_name,
                    "output": f"Agent {node.agent_name} executed successfully",
                    "config": node.config
                }
            
            elif node.type == NodeType.MCP_TOOL:
                # Execute MCP tool
                from services.mcp_integration_service import mcp_service
                
                tool_params = {**node.config, **execution_context.get("variables", {})}
                result = await mcp_service.call_tool(node.mcp_tool, tool_params)
                return result
            
            elif node.type == NodeType.CONDITION:
                # Evaluate condition
                condition = node.config.get("condition", "true")
                # Simple condition evaluation (in production, use safe eval)
                result = True  # Simplified for demo
                return {"status": "evaluated", "condition": condition, "result": result}
            
            elif node.type == NodeType.DATA_TRANSFORM:
                # Transform data
                return {"status": "transformed", "config": node.config}
            
            else:
                return {"status": "executed", "type": node.type.value}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _find_next_node(self, workflow: VisualWorkflow, current_node: WorkflowNode,
                       node_result: Dict[str, Any], execution_context: Dict[str, Any]) -> Optional[WorkflowNode]:
        """Find the next node to execute based on connections and conditions"""
        
        # Find connections from current node
        outgoing_connections = [
            conn for conn in workflow.connections 
            if conn.source_node == current_node.id
        ]
        
        for connection in outgoing_connections:
            # Check connection conditions
            if connection.connection_type == ConnectionType.SUCCESS and node_result.get("status") == "success":
                return self._get_node_by_id(workflow, connection.target_node)
            elif connection.connection_type == ConnectionType.ERROR and node_result.get("status") == "error":
                return self._get_node_by_id(workflow, connection.target_node)
            elif connection.connection_type == ConnectionType.CONDITIONAL:
                if self._evaluate_connection_condition(connection, node_result, execution_context):
                    return self._get_node_by_id(workflow, connection.target_node)
            elif connection.connection_type == ConnectionType.DATA:
                return self._get_node_by_id(workflow, connection.target_node)
        
        return None
    
    def _get_node_by_id(self, workflow: VisualWorkflow, node_id: str) -> Optional[WorkflowNode]:
        """Get node by ID from workflow"""
        for node in workflow.nodes:
            if node.id == node_id:
                return node
        return None
    
    def _evaluate_connection_condition(self, connection: WorkflowConnection,
                                     node_result: Dict[str, Any], execution_context: Dict[str, Any]) -> bool:
        """Evaluate connection condition"""
        # Simple condition evaluation - in production use safe evaluation
        return True  # Simplified for demo
    
    def export_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Export workflow as JSON"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        return {
            "id": workflow.id,
            "name": workflow.name,
            "description": workflow.description,
            "nodes": [
                {
                    "id": node.id,
                    "type": node.type.value,
                    "name": node.name,
                    "description": node.description,
                    "position": node.position,
                    "config": node.config,
                    "agent_name": node.agent_name,
                    "mcp_tool": node.mcp_tool
                }
                for node in workflow.nodes
            ],
            "connections": [
                {
                    "id": conn.id,
                    "source_node": conn.source_node,
                    "target_node": conn.target_node,
                    "source_port": conn.source_port,
                    "target_port": conn.target_port,
                    "connection_type": conn.connection_type.value,
                    "conditions": conn.conditions,
                    "data_mapping": conn.data_mapping
                }
                for conn in workflow.connections
            ],
            "variables": workflow.variables,
            "settings": workflow.settings,
            "created_at": workflow.created_at.isoformat(),
            "updated_at": workflow.updated_at.isoformat(),
            "version": workflow.version
        }

# Global visual workflow service instance
visual_workflow_service = VisualWorkflowService()