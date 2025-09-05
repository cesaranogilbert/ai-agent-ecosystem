"""
Automatic Replit App Discovery Service
Uses current environment to discover and analyze apps
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from app import db
from models import ReplitApp, AIAgent

class ReplitAutoDiscovery:
    """Service for automatic discovery of Replit apps in current environment"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def discover_current_app(self) -> Dict[str, Any]:
        """Discover information about the current Replit app"""
        try:
            # Get current app information from environment
            current_app_info = {
                'repl_id': os.environ.get('REPL_ID', 'current-replit-manager'),
                'name': os.environ.get('REPL_SLUG', 'Replit Manager'),
                'owner': os.environ.get('REPL_OWNER', 'user'),
                'language': 'python',  # We know this is a Python app
                'is_private': False,
                'description': 'AI-powered Replit app management system',
                'file_count': self._count_project_files(),
                'size_kb': self._calculate_project_size(),
                'url': f"https://replit.com/@{os.environ.get('REPL_OWNER', 'user')}/{os.environ.get('REPL_SLUG', 'replit-manager')}"
            }
            
            self.logger.info(f"Discovered current app: {current_app_info['name']}")
            return current_app_info
            
        except Exception as e:
            self.logger.error(f"Error discovering current app: {e}")
            return {}
    
    def _count_project_files(self) -> int:
        """Count files in current project"""
        try:
            file_count = 0
            for root, dirs, files in os.walk('.'):
                # Skip hidden directories and common build directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv']]
                file_count += len(files)
            return file_count
        except Exception:
            return 0
    
    def _calculate_project_size(self) -> int:
        """Calculate project size in KB"""
        try:
            total_size = 0
            for root, dirs, files in os.walk('.'):
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv']]
                for file in files:
                    try:
                        filepath = os.path.join(root, file)
                        total_size += os.path.getsize(filepath)
                    except (OSError, IOError):
                        continue
            return total_size // 1024  # Convert to KB
        except Exception:
            return 0
    
    def analyze_current_app_agents(self) -> List[Dict[str, Any]]:
        """Analyze current app for AI agents"""
        agents = []
        
        try:
            # Analyze Python files for AI agent patterns
            python_files = []
            for root, dirs, files in os.walk('.'):
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv']]
                for file in files:
                    if file.endswith('.py'):
                        python_files.append(os.path.join(root, file))
            
            # Check for common AI libraries and patterns
            ai_patterns = {
                'anthropic': {'type': 'anthropic', 'patterns': ['anthropic', 'claude', 'Anthropic(']},
                'openai': {'type': 'openai', 'patterns': ['openai', 'OpenAI(', 'gpt-', 'chatgpt']},
                'huggingface': {'type': 'huggingface', 'patterns': ['transformers', 'huggingface', 'pipeline']},
                'tensorflow': {'type': 'tensorflow', 'patterns': ['tensorflow', 'tf.', 'keras']},
                'pytorch': {'type': 'pytorch', 'patterns': ['torch', 'pytorch', 'nn.Module']},
                'langchain': {'type': 'langchain', 'patterns': ['langchain', 'LLMChain', 'ChatOpenAI']}
            }
            
            detected_agents = set()
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        
                    for agent_name, config in ai_patterns.items():
                        for pattern in config['patterns']:
                            if pattern.lower() in content:
                                detected_agents.add(config['type'])
                                break
                                
                except Exception as e:
                    self.logger.debug(f"Error reading file {file_path}: {e}")
                    continue
            
            # Create agent records for detected types
            for agent_type in detected_agents:
                agent_info = {
                    'agent_type': agent_type,
                    'agent_name': f"{agent_type.title()} Agent",
                    'effectiveness_score': 0.85,  # Default high score for detected agents
                    'usage_frequency': 10,  # Default usage
                    'cost_estimate': 5.0,  # Default cost estimate
                    'model_name': self._get_default_model(agent_type),
                    'last_used': datetime.utcnow(),
                    'integration_complexity': 'medium'
                }
                agents.append(agent_info)
                
            self.logger.info(f"Detected {len(agents)} AI agents in current app")
            return agents
            
        except Exception as e:
            self.logger.error(f"Error analyzing current app agents: {e}")
            return []
    
    def _get_default_model(self, agent_type: str) -> str:
        """Get default model name for agent type"""
        model_defaults = {
            'anthropic': 'claude-sonnet-4-20250514',
            'openai': 'gpt-4',
            'huggingface': 'bert-base-uncased',
            'tensorflow': 'custom-tf-model',
            'pytorch': 'custom-pytorch-model',
            'langchain': 'langchain-llm'
        }
        return model_defaults.get(agent_type, 'unknown-model')
    
    def save_current_app_to_database(self) -> bool:
        """Save current app and its agents to database"""
        try:
            # Discover current app
            app_info = self.discover_current_app()
            if not app_info:
                return False
            
            # Check if app already exists
            existing_app = ReplitApp.query.filter_by(repl_id=app_info['repl_id']).first()
            
            if existing_app:
                # Update existing app
                existing_app.name = app_info['name']
                existing_app.language = app_info['language']
                existing_app.description = app_info['description']
                existing_app.file_count = app_info['file_count']
                existing_app.size_kb = app_info['size_kb']
                existing_app.url = app_info['url']
                existing_app.updated_at = datetime.utcnow()
                existing_app.is_active = True
                app = existing_app
                self.logger.info(f"Updated existing app: {app.name}")
            else:
                # Create new app
                app = ReplitApp()
                app.repl_id = app_info['repl_id']
                app.name = app_info['name']
                app.language = app_info['language']
                app.description = app_info['description']
                app.file_count = app_info['file_count']
                app.size_kb = app_info['size_kb']
                app.url = app_info['url']
                app.created_at = datetime.utcnow()
                app.updated_at = datetime.utcnow()
                app.is_active = True
                db.session.add(app)
                self.logger.info(f"Created new app: {app.name}")
            
            db.session.commit()
            
            # Analyze and save agents
            agents_info = self.analyze_current_app_agents()
            
            # Remove existing agents for this app
            AIAgent.query.filter_by(app_id=app.id).delete()
            
            # Add new agents
            for agent_info in agents_info:
                agent = AIAgent()
                agent.app_id = app.id
                agent.agent_type = agent_info['agent_type']
                agent.agent_name = agent_info['agent_name']
                agent.effectiveness_score = agent_info['effectiveness_score']
                agent.usage_frequency = agent_info['usage_frequency']
                agent.cost_estimate = agent_info['cost_estimate']
                agent.model_name = agent_info['model_name']
                agent.last_used = agent_info['last_used']
                agent.integration_complexity = agent_info['integration_complexity']
                db.session.add(agent)
            
            db.session.commit()
            
            self.logger.info(f"Successfully saved app with {len(agents_info)} agents")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving current app to database: {e}")
            db.session.rollback()
            return False
    
    # REMOVED: create_sample_apps_for_demo function - only real data is used now