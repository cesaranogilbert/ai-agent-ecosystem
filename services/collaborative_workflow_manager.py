"""
Collaborative Workflow Manager for Multi-Agent Systems
Orchestrates complex workflows across multiple AI agents
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import uuid
from enum import Enum

class WorkflowStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

class CollaborativeWorkflowManager:
    """
    Advanced workflow manager that orchestrates multi-agent collaboration
    with intelligent task distribution and real-time optimization
    """
    
    def __init__(self):
        self.workflows = {}
        self.task_queue = {}
        self.agent_availability = {}
        self.performance_metrics = {}
        self.collaboration_patterns = {}
        
        # Initialize workflow templates
        self._initialize_workflow_templates()
        
    def _initialize_workflow_templates(self):
        """Initialize pre-defined workflow templates for common scenarios"""
        self.workflow_templates = {
            'strategic_analysis': {
                'name': 'Comprehensive Strategic Analysis',
                'description': 'Multi-dimensional strategic analysis with executive reporting',
                'phases': [
                    {
                        'name': 'Strategic Intelligence Gathering',
                        'agents': ['csuite_strategic', 'board_analytics'],
                        'duration': timedelta(hours=2),
                        'dependencies': []
                    },
                    {
                        'name': 'Risk and Opportunity Assessment',
                        'agents': ['ma_due_diligence', 'cyber_threat'],
                        'duration': timedelta(hours=1.5),
                        'dependencies': ['Strategic Intelligence Gathering']
                    },
                    {
                        'name': 'Implementation Planning',
                        'agents': ['legacy_modernization', 'cultural_integration'],
                        'duration': timedelta(hours=1),
                        'dependencies': ['Risk and Opportunity Assessment']
                    },
                    {
                        'name': 'Executive Communication',
                        'agents': ['thought_leader', 'personalization'],
                        'duration': timedelta(minutes=45),
                        'dependencies': ['Implementation Planning']
                    }
                ],
                'estimated_duration': timedelta(hours=5.25),
                'required_inputs': ['company_profile', 'strategic_objectives', 'market_context']
            },
            'merger_acquisition': {
                'name': 'M&A Due Diligence and Integration',
                'description': 'Comprehensive merger and acquisition analysis and planning',
                'phases': [
                    {
                        'name': 'Due Diligence Analysis',
                        'agents': ['ma_due_diligence', 'csuite_strategic'],
                        'duration': timedelta(hours=3),
                        'dependencies': []
                    },
                    {
                        'name': 'Cultural Integration Assessment',
                        'agents': ['cultural_integration', 'sympathetic_writing'],
                        'duration': timedelta(hours=2),
                        'dependencies': ['Due Diligence Analysis']
                    },
                    {
                        'name': 'Technology Integration Planning',
                        'agents': ['legacy_modernization', 'cyber_threat'],
                        'duration': timedelta(hours=2.5),
                        'dependencies': ['Due Diligence Analysis']
                    },
                    {
                        'name': 'Board Presentation Preparation',
                        'agents': ['board_analytics', 'thought_leader'],
                        'duration': timedelta(hours=1.5),
                        'dependencies': ['Cultural Integration Assessment', 'Technology Integration Planning']
                    }
                ],
                'estimated_duration': timedelta(hours=9),
                'required_inputs': ['target_company_data', 'integration_objectives', 'financial_parameters']
            },
            'digital_transformation': {
                'name': 'Enterprise Digital Transformation',
                'description': 'Comprehensive digital transformation strategy and implementation',
                'phases': [
                    {
                        'name': 'Current State Assessment',
                        'agents': ['legacy_modernization', 'cyber_threat'],
                        'duration': timedelta(hours=2.5),
                        'dependencies': []
                    },
                    {
                        'name': 'Strategic Digital Vision',
                        'agents': ['csuite_strategic', 'thought_leader'],
                        'duration': timedelta(hours=2),
                        'dependencies': ['Current State Assessment']
                    },
                    {
                        'name': 'Change Management Strategy',
                        'agents': ['cultural_integration', 'sympathetic_writing'],
                        'duration': timedelta(hours=1.5),
                        'dependencies': ['Strategic Digital Vision']
                    },
                    {
                        'name': 'Implementation Roadmap',
                        'agents': ['board_analytics', 'dynamic_pricing'],
                        'duration': timedelta(hours=2),
                        'dependencies': ['Change Management Strategy']
                    }
                ],
                'estimated_duration': timedelta(hours=8),
                'required_inputs': ['current_technology_stack', 'business_objectives', 'transformation_budget']
            },
            'market_expansion': {
                'name': 'Global Market Expansion Strategy',
                'description': 'Strategic market expansion with cultural and pricing intelligence',
                'phases': [
                    {
                        'name': 'Market Intelligence Analysis',
                        'agents': ['csuite_strategic', 'dynamic_pricing'],
                        'duration': timedelta(hours=2),
                        'dependencies': []
                    },
                    {
                        'name': 'Cultural Market Assessment',
                        'agents': ['cultural_integration', 'sympathetic_writing'],
                        'duration': timedelta(hours=1.5),
                        'dependencies': ['Market Intelligence Analysis']
                    },
                    {
                        'name': 'Risk and Compliance Evaluation',
                        'agents': ['cyber_threat', 'ma_due_diligence'],
                        'duration': timedelta(hours=1.5),
                        'dependencies': ['Cultural Market Assessment']
                    },
                    {
                        'name': 'Go-to-Market Strategy',
                        'agents': ['board_analytics', 'personalization'],
                        'duration': timedelta(hours=2),
                        'dependencies': ['Risk and Compliance Evaluation']
                    }
                ],
                'estimated_duration': timedelta(hours=7),
                'required_inputs': ['target_markets', 'expansion_budget', 'competitive_landscape']
            }
        }
        
    def create_workflow(self, template_name: str, inputs: Dict[str, Any], priority: TaskPriority = TaskPriority.MEDIUM) -> str:
        """
        Create a new collaborative workflow from template
        """
        if template_name not in self.workflow_templates:
            raise ValueError(f"Template '{template_name}' not found")
            
        workflow_id = str(uuid.uuid4())
        template = self.workflow_templates[template_name]
        
        # Validate required inputs
        required_inputs = template.get('required_inputs', [])
        missing_inputs = [inp for inp in required_inputs if inp not in inputs]
        if missing_inputs:
            raise ValueError(f"Missing required inputs: {missing_inputs}")
        
        workflow = {
            'id': workflow_id,
            'template_name': template_name,
            'name': template['name'],
            'description': template['description'],
            'status': WorkflowStatus.PENDING,
            'priority': priority,
            'inputs': inputs,
            'phases': template['phases'].copy(),
            'current_phase': 0,
            'created_at': datetime.now(),
            'estimated_completion': datetime.now() + template['estimated_duration'],
            'actual_start': None,
            'actual_completion': None,
            'results': {},
            'metrics': {
                'total_phases': len(template['phases']),
                'completed_phases': 0,
                'active_agents': 0,
                'progress_percentage': 0
            }
        }
        
        self.workflows[workflow_id] = workflow
        
        # Initialize phase tasks
        self._initialize_phase_tasks(workflow_id)
        
        logging.info(f"Created workflow '{template_name}' with ID: {workflow_id}")
        return workflow_id
        
    def _initialize_phase_tasks(self, workflow_id: str):
        """Initialize tasks for all phases in the workflow"""
        workflow = self.workflows[workflow_id]
        
        for phase_index, phase in enumerate(workflow['phases']):
            phase_id = f"{workflow_id}_phase_{phase_index}"
            
            # Create task for each agent in the phase
            for agent_name in phase['agents']:
                task_id = f"{phase_id}_{agent_name}"
                
                task = {
                    'id': task_id,
                    'workflow_id': workflow_id,
                    'phase_index': phase_index,
                    'phase_name': phase['name'],
                    'agent': agent_name,
                    'status': WorkflowStatus.PENDING,
                    'priority': workflow['priority'],
                    'dependencies': phase['dependencies'],
                    'estimated_duration': phase['duration'],
                    'created_at': datetime.now(),
                    'started_at': None,
                    'completed_at': None,
                    'result': None,
                    'metadata': {
                        'phase_dependencies_met': False,
                        'agent_available': True,
                        'retry_count': 0
                    }
                }
                
                if workflow_id not in self.task_queue:
                    self.task_queue[workflow_id] = {}
                self.task_queue[workflow_id][task_id] = task
                
    def start_workflow(self, workflow_id: str) -> bool:
        """Start execution of a workflow"""
        if workflow_id not in self.workflows:
            return False
            
        workflow = self.workflows[workflow_id]
        workflow['status'] = WorkflowStatus.ACTIVE
        workflow['actual_start'] = datetime.now()
        workflow['current_phase'] = 0
        
        # Start first phase if dependencies are met
        self._check_and_start_next_phase(workflow_id)
        
        logging.info(f"Started workflow: {workflow_id}")
        return True
        
    def _check_and_start_next_phase(self, workflow_id: str):
        """Check if next phase can start and start it if possible"""
        workflow = self.workflows[workflow_id]
        
        if workflow['current_phase'] >= len(workflow['phases']):
            self._complete_workflow(workflow_id)
            return
            
        current_phase = workflow['phases'][workflow['current_phase']]
        phase_name = current_phase['name']
        
        # Check if phase dependencies are met
        if self._are_phase_dependencies_met(workflow_id, current_phase['dependencies']):
            self._start_phase(workflow_id, workflow['current_phase'])
        else:
            logging.info(f"Phase '{phase_name}' waiting for dependencies in workflow {workflow_id}")
            
    def _are_phase_dependencies_met(self, workflow_id: str, dependencies: List[str]) -> bool:
        """Check if all phase dependencies are completed"""
        if not dependencies:
            return True
            
        workflow = self.workflows[workflow_id]
        completed_phases = []
        
        for i in range(workflow['current_phase']):
            phase = workflow['phases'][i]
            phase_tasks = self._get_phase_tasks(workflow_id, i)
            
            if all(task['status'] == WorkflowStatus.COMPLETED for task in phase_tasks):
                completed_phases.append(phase['name'])
                
        return all(dep in completed_phases for dep in dependencies)
        
    def _start_phase(self, workflow_id: str, phase_index: int):
        """Start execution of a specific phase"""
        workflow = self.workflows[workflow_id]
        phase = workflow['phases'][phase_index]
        
        logging.info(f"Starting phase '{phase['name']}' in workflow {workflow_id}")
        
        # Start all tasks in this phase
        phase_tasks = self._get_phase_tasks(workflow_id, phase_index)
        
        for task in phase_tasks:
            self._start_task(task['id'])
            
        # Update workflow metrics
        workflow['metrics']['active_agents'] = len(phase['agents'])
        
    def _get_phase_tasks(self, workflow_id: str, phase_index: int) -> List[Dict[str, Any]]:
        """Get all tasks for a specific phase"""
        if workflow_id not in self.task_queue:
            return []
            
        phase_tasks = []
        for task in self.task_queue[workflow_id].values():
            if task['phase_index'] == phase_index:
                phase_tasks.append(task)
                
        return phase_tasks
        
    def _start_task(self, task_id: str):
        """Start execution of a specific task"""
        # Find the task across all workflows
        task = None
        workflow_id = None
        
        for wf_id, tasks in self.task_queue.items():
            if task_id in tasks:
                task = tasks[task_id]
                workflow_id = wf_id
                break
                
        if not task:
            return False
            
        task['status'] = WorkflowStatus.ACTIVE
        task['started_at'] = datetime.now()
        
        # Simulate task execution (in real implementation, this would call the actual agent)
        self._execute_task_simulation(task)
        
        logging.info(f"Started task: {task_id} for agent {task['agent']}")
        return True
        
    def _execute_task_simulation(self, task: Dict[str, Any]):
        """Simulate task execution (placeholder for actual agent integration)"""
        import time
        import random
        
        # Simulate processing time
        processing_time = random.uniform(0.1, 0.5)  # Quick simulation
        time.sleep(processing_time)
        
        # Generate simulated results
        agent_name = task['agent']
        phase_name = task['phase_name']
        
        result = {
            'agent': agent_name,
            'phase': phase_name,
            'analysis': f"Detailed analysis from {agent_name} for {phase_name}",
            'insights': [
                f"Key insight from {agent_name} perspective",
                f"Strategic recommendation based on {phase_name} analysis",
                f"Risk factors identified by {agent_name}"
            ],
            'confidence': random.uniform(0.8, 0.95),
            'processing_time': processing_time,
            'metadata': {
                'data_quality': random.uniform(0.85, 0.98),
                'completeness': random.uniform(0.90, 1.0)
            }
        }
        
        # Complete the task
        task['status'] = WorkflowStatus.COMPLETED
        task['completed_at'] = datetime.now()
        task['result'] = result
        
        # Check if phase is complete
        self._check_phase_completion(task['workflow_id'], task['phase_index'])
        
    def _check_phase_completion(self, workflow_id: str, phase_index: int):
        """Check if a phase is complete and advance workflow if needed"""
        phase_tasks = self._get_phase_tasks(workflow_id, phase_index)
        
        if all(task['status'] == WorkflowStatus.COMPLETED for task in phase_tasks):
            self._complete_phase(workflow_id, phase_index)
            
    def _complete_phase(self, workflow_id: str, phase_index: int):
        """Complete a phase and advance to next phase"""
        workflow = self.workflows[workflow_id]
        phase = workflow['phases'][phase_index]
        
        logging.info(f"Completed phase '{phase['name']}' in workflow {workflow_id}")
        
        # Collect phase results
        phase_tasks = self._get_phase_tasks(workflow_id, phase_index)
        phase_results = {}
        
        for task in phase_tasks:
            if task['result']:
                phase_results[task['agent']] = task['result']
                
        # Store phase results
        workflow['results'][f"phase_{phase_index}"] = {
            'phase_name': phase['name'],
            'completion_time': datetime.now(),
            'agent_results': phase_results,
            'phase_summary': f"Completed {phase['name']} with {len(phase_results)} agent contributions"
        }
        
        # Update workflow metrics
        workflow['metrics']['completed_phases'] += 1
        workflow['metrics']['progress_percentage'] = (workflow['metrics']['completed_phases'] / workflow['metrics']['total_phases']) * 100
        workflow['metrics']['active_agents'] = 0
        
        # Advance to next phase
        workflow['current_phase'] += 1
        self._check_and_start_next_phase(workflow_id)
        
    def _complete_workflow(self, workflow_id: str):
        """Complete the entire workflow"""
        workflow = self.workflows[workflow_id]
        workflow['status'] = WorkflowStatus.COMPLETED
        workflow['actual_completion'] = datetime.now()
        workflow['metrics']['progress_percentage'] = 100
        
        # Generate final workflow summary
        workflow['final_summary'] = self._generate_workflow_summary(workflow_id)
        
        logging.info(f"Completed workflow: {workflow_id}")
        
    def _generate_workflow_summary(self, workflow_id: str) -> Dict[str, Any]:
        """Generate comprehensive summary of completed workflow"""
        workflow = self.workflows[workflow_id]
        
        # Calculate execution metrics
        total_duration = workflow['actual_completion'] - workflow['actual_start']
        estimated_duration = workflow['estimated_completion'] - workflow['created_at']
        
        # Collect all insights and recommendations
        all_insights = []
        all_recommendations = []
        agent_contributions = {}
        
        for phase_key, phase_result in workflow['results'].items():
            if 'agent_results' in phase_result:
                for agent, result in phase_result['agent_results'].items():
                    if agent not in agent_contributions:
                        agent_contributions[agent] = 0
                    agent_contributions[agent] += 1
                    
                    if 'insights' in result:
                        all_insights.extend(result['insights'])
                        
        summary = {
            'workflow_id': workflow_id,
            'template_name': workflow['template_name'],
            'execution_metrics': {
                'estimated_duration': str(estimated_duration),
                'actual_duration': str(total_duration),
                'efficiency_ratio': estimated_duration.total_seconds() / max(1, total_duration.total_seconds()),
                'phases_completed': workflow['metrics']['completed_phases'],
                'total_phases': workflow['metrics']['total_phases']
            },
            'collaboration_metrics': {
                'unique_agents': len(agent_contributions),
                'total_agent_tasks': sum(agent_contributions.values()),
                'most_active_agent': max(agent_contributions.items(), key=lambda x: x[1])[0] if agent_contributions else None
            },
            'output_metrics': {
                'total_insights': len(all_insights),
                'insights_per_phase': len(all_insights) / max(1, workflow['metrics']['total_phases']),
                'quality_indicators': {
                    'comprehensive_analysis': True,
                    'multi_agent_collaboration': len(agent_contributions) > 1,
                    'cross_functional_insights': len(agent_contributions) >= 3
                }
            },
            'key_outcomes': all_insights[:10],  # Top 10 insights
            'agent_contributions': agent_contributions,
            'completion_timestamp': workflow['actual_completion'].isoformat()
        }
        
        return summary
        
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a workflow"""
        if workflow_id not in self.workflows:
            return None
            
        workflow = self.workflows[workflow_id]
        
        # Get active tasks
        active_tasks = []
        if workflow_id in self.task_queue:
            for task in self.task_queue[workflow_id].values():
                if task['status'] == WorkflowStatus.ACTIVE:
                    active_tasks.append({
                        'task_id': task['id'],
                        'agent': task['agent'],
                        'phase': task['phase_name'],
                        'started_at': task['started_at'].isoformat() if task['started_at'] else None
                    })
        
        status = {
            'workflow_id': workflow_id,
            'name': workflow['name'],
            'status': workflow['status'].value,
            'progress_percentage': workflow['metrics']['progress_percentage'],
            'current_phase': workflow['current_phase'],
            'total_phases': workflow['metrics']['total_phases'],
            'active_tasks': active_tasks,
            'created_at': workflow['created_at'].isoformat(),
            'estimated_completion': workflow['estimated_completion'].isoformat(),
            'actual_completion': workflow['actual_completion'].isoformat() if workflow.get('actual_completion') else None
        }
        
        if workflow['status'] == WorkflowStatus.COMPLETED and 'final_summary' in workflow:
            status['final_summary'] = workflow['final_summary']
            
        return status
        
    def get_all_workflows(self) -> List[Dict[str, Any]]:
        """Get status of all workflows"""
        return [self.get_workflow_status(wf_id) for wf_id in self.workflows.keys()]
        
    def pause_workflow(self, workflow_id: str) -> bool:
        """Pause a workflow"""
        if workflow_id not in self.workflows:
            return False
            
        workflow = self.workflows[workflow_id]
        if workflow['status'] == WorkflowStatus.ACTIVE:
            workflow['status'] = WorkflowStatus.PAUSED
            logging.info(f"Paused workflow: {workflow_id}")
            return True
            
        return False
        
    def resume_workflow(self, workflow_id: str) -> bool:
        """Resume a paused workflow"""
        if workflow_id not in self.workflows:
            return False
            
        workflow = self.workflows[workflow_id]
        if workflow['status'] == WorkflowStatus.PAUSED:
            workflow['status'] = WorkflowStatus.ACTIVE
            self._check_and_start_next_phase(workflow_id)
            logging.info(f"Resumed workflow: {workflow_id}")
            return True
            
        return False
        
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a workflow"""
        if workflow_id not in self.workflows:
            return False
            
        workflow = self.workflows[workflow_id]
        workflow['status'] = WorkflowStatus.CANCELLED
        logging.info(f"Cancelled workflow: {workflow_id}")
        return True

def create_collaborative_workflow_manager():
    """Factory function to create workflow manager"""
    return CollaborativeWorkflowManager()

def test_collaborative_workflow_manager():
    """Test the collaborative workflow manager"""
    print("🧪 Testing Collaborative Workflow Manager")
    print("=" * 45)
    
    try:
        manager = create_collaborative_workflow_manager()
        
        # Test workflow creation
        inputs = {
            'company_profile': 'Fortune 500 Technology Company',
            'strategic_objectives': 'Market expansion and digital transformation',
            'market_context': 'Competitive technology landscape'
        }
        
        workflow_id = manager.create_workflow('strategic_analysis', inputs, TaskPriority.HIGH)
        print(f"✅ Created workflow: {workflow_id}")
        
        # Start workflow
        success = manager.start_workflow(workflow_id)
        print(f"✅ Started workflow: {success}")
        
        # Check status
        status = manager.get_workflow_status(workflow_id)
        print(f"✅ Workflow status: {status['status']} ({status['progress_percentage']:.1f}% complete)")
        
        # Wait for completion (simulated)
        import time
        time.sleep(2)  # Allow time for simulated execution
        
        final_status = manager.get_workflow_status(workflow_id)
        print(f"✅ Final status: {final_status['status']} ({final_status['progress_percentage']:.1f}% complete)")
        
        return {
            'manager_initialized': True,
            'workflow_created': workflow_id,
            'workflow_completed': final_status['status'] == 'completed',
            'total_phases': final_status['total_phases']
        }
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return {'error': str(e)}

if __name__ == "__main__":
    test_collaborative_workflow_manager()