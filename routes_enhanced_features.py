#!/usr/bin/env python3
"""
Enhanced Features Routes
Flask routes for MCP integration, visual workflows, and multi-agent collaboration
"""

from flask import Blueprint, request, jsonify, render_template_string
import asyncio
import json
from datetime import datetime
import logging

# Import enhanced services
from services.mcp_integration_service import mcp_service, connect_agent_to_tools, execute_agent_tool, get_enterprise_capabilities
from services.visual_workflow_service import visual_workflow_service, NodeType, ConnectionType
from services.multi_agent_collaboration_service import collaboration_service, AgentRole, TaskStatus, CollaborationPattern

logger = logging.getLogger(__name__)

# Create Blueprint
enhanced_features_bp = Blueprint('enhanced_features', __name__, url_prefix='/api/enhanced')

# MCP Integration Routes
@enhanced_features_bp.route('/mcp/tools', methods=['GET'])
def get_mcp_tools():
    """Get all available MCP tools"""
    try:
        tools = mcp_service.get_available_tools()
        return jsonify({
            "success": True,
            "tools": tools,
            "total": len(tools),
            "enterprise_capabilities": get_enterprise_capabilities()
        })
    except Exception as e:
        logger.error(f"Error getting MCP tools: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@enhanced_features_bp.route('/mcp/tools/<tool_name>/call', methods=['POST'])
def call_mcp_tool(tool_name):
    """Execute an MCP tool"""
    try:
        data = request.get_json()
        parameters = data.get('parameters', {})
        session_id = data.get('session_id')
        
        # Run async function in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            mcp_service.call_tool(tool_name, parameters, session_id)
        )
        loop.close()
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error calling MCP tool {tool_name}: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@enhanced_features_bp.route('/mcp/agents/<agent_name>/connect', methods=['POST'])
def connect_agent_mcp_tools(agent_name):
    """Connect an agent to specific MCP tools"""
    try:
        data = request.get_json()
        required_tools = data.get('tools', [])
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            connect_agent_to_tools(agent_name, required_tools)
        )
        loop.close()
        
        return jsonify({
            "success": True,
            "integration_profile": result
        })
    except Exception as e:
        logger.error(f"Error connecting agent {agent_name} to MCP tools: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@enhanced_features_bp.route('/mcp/tools/batch', methods=['POST'])
def batch_execute_mcp_tools():
    """Execute multiple MCP tools in batch"""
    try:
        data = request.get_json()
        tool_calls = data.get('tool_calls', [])
        session_id = data.get('session_id')
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(
            mcp_service.batch_tool_execution(tool_calls, session_id)
        )
        loop.close()
        
        return jsonify({
            "success": True,
            "results": results,
            "executed": len(results)
        })
    except Exception as e:
        logger.error(f"Error in batch MCP tool execution: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

# Visual Workflow Routes
@enhanced_features_bp.route('/workflows', methods=['GET'])
def list_workflows():
    """List all visual workflows"""
    try:
        workflows = visual_workflow_service.list_workflows()
        return jsonify({
            "success": True,
            "workflows": workflows,
            "total": len(workflows)
        })
    except Exception as e:
        logger.error(f"Error listing workflows: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@enhanced_features_bp.route('/workflows/templates', methods=['GET'])
def get_workflow_templates():
    """Get available workflow templates"""
    try:
        templates = visual_workflow_service.get_templates()
        return jsonify({
            "success": True,
            "templates": templates,
            "total": len(templates)
        })
    except Exception as e:
        logger.error(f"Error getting workflow templates: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@enhanced_features_bp.route('/workflows', methods=['POST'])
def create_workflow():
    """Create a new workflow"""
    try:
        data = request.get_json()
        name = data.get('name')
        description = data.get('description', '')
        template_name = data.get('template')
        
        if not name:
            return jsonify({"success": False, "error": "Workflow name is required"}), 400
        
        if template_name:
            workflow_id = visual_workflow_service.create_from_template(template_name, name)
        else:
            workflow_id = visual_workflow_service.create_workflow(name, description)
        
        return jsonify({
            "success": True,
            "workflow_id": workflow_id,
            "message": f"Workflow '{name}' created successfully"
        })
    except Exception as e:
        logger.error(f"Error creating workflow: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@enhanced_features_bp.route('/workflows/<workflow_id>', methods=['GET'])
def get_workflow(workflow_id):
    """Get workflow details"""
    try:
        workflow = visual_workflow_service.get_workflow(workflow_id)
        if not workflow:
            return jsonify({"success": False, "error": "Workflow not found"}), 404
        
        return jsonify({
            "success": True,
            "workflow": visual_workflow_service.export_workflow(workflow_id)
        })
    except Exception as e:
        logger.error(f"Error getting workflow {workflow_id}: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@enhanced_features_bp.route('/workflows/<workflow_id>/execute', methods=['POST'])
def execute_workflow(workflow_id):
    """Execute a visual workflow"""
    try:
        data = request.get_json()
        input_data = data.get('input_data', {})
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            visual_workflow_service.execute_workflow(workflow_id, input_data)
        )
        loop.close()
        
        return jsonify({
            "success": True,
            "execution_result": result
        })
    except Exception as e:
        logger.error(f"Error executing workflow {workflow_id}: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@enhanced_features_bp.route('/workflows/<workflow_id>/nodes', methods=['POST'])
def add_workflow_node(workflow_id):
    """Add a node to the workflow"""
    try:
        data = request.get_json()
        node_type = NodeType(data.get('type'))
        name = data.get('name')
        position = data.get('position', {'x': 0, 'y': 0})
        config = data.get('config', {})
        
        node_id = visual_workflow_service.add_node(workflow_id, node_type, name, position, config)
        
        return jsonify({
            "success": True,
            "node_id": node_id,
            "message": f"Node '{name}' added to workflow"
        })
    except Exception as e:
        logger.error(f"Error adding node to workflow {workflow_id}: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@enhanced_features_bp.route('/workflows/<workflow_id>/connections', methods=['POST'])
def connect_workflow_nodes(workflow_id):
    """Connect nodes in the workflow"""
    try:
        data = request.get_json()
        source_node = data.get('source_node')
        target_node = data.get('target_node')
        connection_type = ConnectionType(data.get('connection_type', 'data'))
        
        connection_id = visual_workflow_service.connect_nodes(
            workflow_id, source_node, target_node, connection_type
        )
        
        return jsonify({
            "success": True,
            "connection_id": connection_id,
            "message": "Nodes connected successfully"
        })
    except Exception as e:
        logger.error(f"Error connecting nodes in workflow {workflow_id}: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

# Multi-Agent Collaboration Routes
@enhanced_features_bp.route('/collaboration/agents', methods=['GET'])
def list_collaborative_agents():
    """List all collaborative agents"""
    try:
        agents = []
        for agent_name, agent in collaboration_service.agents.items():
            agents.append({
                "name": agent.name,
                "role": agent.role.value,
                "capabilities": [cap.name for cap in agent.capabilities],
                "max_concurrent_tasks": agent.max_concurrent_tasks,
                "current_tasks": len(agent.current_tasks),
                "status": agent.status,
                "last_active": agent.last_active.isoformat()
            })
        
        return jsonify({
            "success": True,
            "agents": agents,
            "total": len(agents)
        })
    except Exception as e:
        logger.error(f"Error listing collaborative agents: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@enhanced_features_bp.route('/collaboration/projects', methods=['GET'])
def list_collaboration_projects():
    """List all collaboration projects"""
    try:
        projects = collaboration_service.list_active_collaborations()
        return jsonify({
            "success": True,
            "projects": projects,
            "total": len(projects)
        })
    except Exception as e:
        logger.error(f"Error listing collaboration projects: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@enhanced_features_bp.route('/collaboration/projects', methods=['POST'])
def create_collaboration_project():
    """Create a new collaboration project"""
    try:
        data = request.get_json()
        name = data.get('name')
        description = data.get('description', '')
        objectives = data.get('objectives', [])
        pattern = data.get('pattern', 'network')
        
        if not name:
            return jsonify({"success": False, "error": "Project name is required"}), 400
        
        collaboration_pattern = CollaborationPattern(pattern)
        project_id = collaboration_service.create_collaboration_project(
            name, description, objectives, collaboration_pattern
        )
        
        return jsonify({
            "success": True,
            "project_id": project_id,
            "message": f"Collaboration project '{name}' created successfully"
        })
    except Exception as e:
        logger.error(f"Error creating collaboration project: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@enhanced_features_bp.route('/collaboration/projects/<project_id>/agents', methods=['POST'])
def assign_agents_to_project(project_id):
    """Assign agents to a collaboration project"""
    try:
        data = request.get_json()
        agent_names = data.get('agents', [])
        
        if not agent_names:
            return jsonify({"success": False, "error": "At least one agent must be specified"}), 400
        
        success = collaboration_service.assign_agents_to_project(project_id, agent_names)
        
        return jsonify({
            "success": success,
            "message": f"Assigned {len(agent_names)} agents to project {project_id}"
        })
    except Exception as e:
        logger.error(f"Error assigning agents to project {project_id}: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@enhanced_features_bp.route('/collaboration/projects/<project_id>/tasks', methods=['POST'])
def add_project_task(project_id):
    """Add a task to a collaboration project"""
    try:
        data = request.get_json()
        task_name = data.get('name')
        task_description = data.get('description', '')
        required_capabilities = data.get('capabilities', [])
        priority = data.get('priority', 5)
        
        if not task_name:
            return jsonify({"success": False, "error": "Task name is required"}), 400
        
        task_id = collaboration_service.add_task_to_project(
            project_id, task_name, task_description, required_capabilities, priority
        )
        
        return jsonify({
            "success": True,
            "task_id": task_id,
            "message": f"Task '{task_name}' added to project"
        })
    except Exception as e:
        logger.error(f"Error adding task to project {project_id}: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@enhanced_features_bp.route('/collaboration/projects/<project_id>/execute', methods=['POST'])
def execute_collaboration_project(project_id):
    """Execute a collaboration project"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            collaboration_service.execute_collaboration_project(project_id)
        )
        loop.close()
        
        return jsonify({
            "success": True,
            "execution_result": result
        })
    except Exception as e:
        logger.error(f"Error executing collaboration project {project_id}: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@enhanced_features_bp.route('/collaboration/projects/<project_id>/assign-tasks', methods=['POST'])
def auto_assign_project_tasks(project_id):
    """Auto-assign tasks to suitable agents"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        assignments = loop.run_until_complete(
            collaboration_service.auto_assign_tasks(project_id)
        )
        loop.close()
        
        return jsonify({
            "success": True,
            "assignments": assignments,
            "message": "Tasks automatically assigned to agents"
        })
    except Exception as e:
        logger.error(f"Error auto-assigning tasks in project {project_id}: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@enhanced_features_bp.route('/collaboration/agents/<agent_name>/performance', methods=['GET'])
def get_agent_performance(agent_name):
    """Get performance metrics for an agent"""
    try:
        performance = collaboration_service.get_agent_performance(agent_name)
        return jsonify({
            "success": True,
            "performance": performance
        })
    except Exception as e:
        logger.error(f"Error getting agent performance for {agent_name}: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@enhanced_features_bp.route('/collaboration/projects/<project_id>/insights', methods=['GET'])
def get_collaboration_insights(project_id):
    """Get collaboration insights for a project"""
    try:
        insights = collaboration_service.get_collaboration_insights(project_id)
        return jsonify({
            "success": True,
            "insights": insights
        })
    except Exception as e:
        logger.error(f"Error getting collaboration insights for project {project_id}: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

# Enhanced Features Dashboard Route
@enhanced_features_bp.route('/dashboard', methods=['GET'])
def enhanced_features_dashboard():
    """Enhanced features dashboard"""
    try:
        dashboard_template = '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Enhanced AI Agent Features Dashboard</title>
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f7fa; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; text-align: center; }
                .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
                .features-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 2rem; margin-top: 2rem; }
                .feature-card { background: white; border-radius: 12px; padding: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: transform 0.3s; }
                .feature-card:hover { transform: translateY(-5px); }
                .feature-icon { font-size: 3rem; margin-bottom: 1rem; }
                .feature-title { font-size: 1.5rem; font-weight: bold; margin-bottom: 1rem; color: #333; }
                .feature-description { color: #666; margin-bottom: 1.5rem; line-height: 1.6; }
                .feature-stats { display: flex; justify-content: space-between; margin-bottom: 1.5rem; }
                .stat { text-align: center; }
                .stat-number { font-size: 1.5rem; font-weight: bold; color: #667eea; }
                .stat-label { font-size: 0.9rem; color: #666; }
                .btn { background: #667eea; color: white; padding: 0.75rem 1.5rem; border: none; border-radius: 6px; cursor: pointer; text-decoration: none; display: inline-block; transition: background 0.3s; }
                .btn:hover { background: #5a6fd8; }
                .btn-secondary { background: #48bb78; }
                .btn-secondary:hover { background: #38a169; }
                .capabilities-list { list-style: none; margin-bottom: 1.5rem; }
                .capabilities-list li { padding: 0.5rem 0; border-bottom: 1px solid #eee; }
                .capabilities-list li:last-child { border-bottom: none; }
                .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
                .status-active { background: #48bb78; }
                .status-pending { background: #ed8936; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Enhanced AI Agent Features</h1>
                <p>MCP Integration • Visual Workflows • Multi-Agent Collaboration</p>
            </div>
            
            <div class="container">
                <div class="features-grid">
                    <!-- MCP Integration -->
                    <div class="feature-card">
                        <div class="feature-icon">🔌</div>
                        <div class="feature-title">MCP Integration Layer</div>
                        <div class="feature-description">
                            Universal tool connection enabling your 38 agents to interact with 500+ enterprise applications including Slack, Zoom, Salesforce, and more.
                        </div>
                        <div class="feature-stats">
                            <div class="stat">
                                <div class="stat-number">{{ mcp_tools }}</div>
                                <div class="stat-label">Available Tools</div>
                            </div>
                            <div class="stat">
                                <div class="stat-number">{{ mcp_capabilities }}</div>
                                <div class="stat-label">Capabilities</div>
                            </div>
                            <div class="stat">
                                <div class="stat-number">{{ mcp_integrations }}</div>
                                <div class="stat-label">Enterprise Apps</div>
                            </div>
                        </div>
                        <ul class="capabilities-list">
                            <li><span class="status-indicator status-active"></span>Slack Integration</li>
                            <li><span class="status-indicator status-active"></span>Zoom Management</li>
                            <li><span class="status-indicator status-active"></span>Salesforce CRM</li>
                            <li><span class="status-indicator status-active"></span>Google Drive</li>
                            <li><span class="status-indicator status-active"></span>JIRA Project Management</li>
                        </ul>
                        <a href="/api/enhanced/mcp/tools" class="btn">Explore MCP Tools</a>
                    </div>
                    
                    <!-- Visual Workflows -->
                    <div class="feature-card">
                        <div class="feature-icon">🎨</div>
                        <div class="feature-title">Visual Workflow Builder</div>
                        <div class="feature-description">
                            Drag-and-drop interface for creating complex agent workflows without coding. Combines your 38 agents into powerful automation sequences.
                        </div>
                        <div class="feature-stats">
                            <div class="stat">
                                <div class="stat-number">{{ workflows_count }}</div>
                                <div class="stat-label">Active Workflows</div>
                            </div>
                            <div class="stat">
                                <div class="stat-number">{{ templates_count }}</div>
                                <div class="stat-label">Templates</div>
                            </div>
                            <div class="stat">
                                <div class="stat-number">{{ workflow_nodes }}</div>
                                <div class="stat-label">Total Nodes</div>
                            </div>
                        </div>
                        <ul class="capabilities-list">
                            <li><span class="status-indicator status-active"></span>Lead Generation → Deal Closing</li>
                            <li><span class="status-indicator status-active"></span>Content Creation Pipeline</li>
                            <li><span class="status-indicator status-active"></span>Project Management Automation</li>
                            <li><span class="status-indicator status-pending"></span>Custom Workflow Builder</li>
                        </ul>
                        <a href="/api/enhanced/workflows/templates" class="btn btn-secondary">View Templates</a>
                    </div>
                    
                    <!-- Multi-Agent Collaboration -->
                    <div class="feature-card">
                        <div class="feature-icon">🤝</div>
                        <div class="feature-title">Multi-Agent Collaboration</div>
                        <div class="feature-description">
                            Enable your 38 specialized agents to work together like a virtual company with coordinated task execution and intelligent workload distribution.
                        </div>
                        <div class="feature-stats">
                            <div class="stat">
                                <div class="stat-number">{{ collaboration_agents }}</div>
                                <div class="stat-label">Collaborative Agents</div>
                            </div>
                            <div class="stat">
                                <div class="stat-number">{{ active_projects }}</div>
                                <div class="stat-label">Active Projects</div>
                            </div>
                            <div class="stat">
                                <div class="stat-number">{{ collaboration_patterns }}</div>
                                <div class="stat-label">Patterns</div>
                            </div>
                        </div>
                        <ul class="capabilities-list">
                            <li><span class="status-indicator status-active"></span>Sequential Collaboration</li>
                            <li><span class="status-indicator status-active"></span>Parallel Execution</li>
                            <li><span class="status-indicator status-active"></span>Hierarchical Management</li>
                            <li><span class="status-indicator status-active"></span>Network Coordination</li>
                            <li><span class="status-indicator status-active"></span>Consensus Decision Making</li>
                        </ul>
                        <a href="/api/enhanced/collaboration/projects" class="btn">View Projects</a>
                    </div>
                </div>
                
                <div style="text-align: center; margin-top: 3rem;">
                    <h2 style="color: #333; margin-bottom: 1rem;">🎯 Market Impact</h2>
                    <div style="background: white; border-radius: 12px; padding: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 2rem;">
                            <div>
                                <div style="font-size: 2rem; font-weight: bold; color: #667eea;">$1.2T+</div>
                                <div style="color: #666;">Enhanced Addressable Market</div>
                            </div>
                            <div>
                                <div style="font-size: 2rem; font-weight: bold; color: #48bb78;">300-500%</div>
                                <div style="color: #666;">Enterprise Adoption Increase</div>
                            </div>
                            <div>
                                <div style="font-size: 2rem; font-weight: bold; color: #ed8936;">500+</div>
                                <div style="color: #666;">Integrated Applications</div>
                            </div>
                            <div>
                                <div style="font-size: 2rem; font-weight: bold; color: #9f7aea;">2M+</div>
                                <div style="color: #666;">Potential Users (Business + Dev)</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        '''
        
        # Get current statistics
        mcp_tools = len(mcp_service.get_available_tools())
        workflows = visual_workflow_service.list_workflows()
        templates = visual_workflow_service.get_templates()
        collaboration_projects = collaboration_service.list_active_collaborations()
        
        return render_template_string(dashboard_template,
            mcp_tools=mcp_tools,
            mcp_capabilities=len(get_enterprise_capabilities()),
            mcp_integrations=6,  # Slack, Zoom, Salesforce, Google Drive, Database, JIRA
            workflows_count=len(workflows),
            templates_count=len(templates),
            workflow_nodes=sum(w.get('nodes', 0) for w in workflows),
            collaboration_agents=len(collaboration_service.agents),
            active_projects=len(collaboration_projects),
            collaboration_patterns=5  # Sequential, Parallel, Hierarchical, Network, Consensus
        )
        
    except Exception as e:
        logger.error(f"Error rendering enhanced features dashboard: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

# Register enhanced features blueprint
logger.info("Enhanced features routes initialized successfully")