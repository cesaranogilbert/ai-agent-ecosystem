"""
AI Agents Routes
Unified routing system for all 30 AI agents
"""

import logging
from flask import Blueprint, request, jsonify, render_template
from services.agent_integration_service import agent_integration_service

logger = logging.getLogger(__name__)

# Create AI Agents Blueprint
ai_agents_bp = Blueprint('ai_agents', __name__, url_prefix='/ai-agents')

@ai_agents_bp.route('/')
def agents_dashboard():
    """Main dashboard for AI agents"""
    try:
        agents_info = agent_integration_service.get_available_agents()
        status_info = agent_integration_service.get_agent_status()
        
        return render_template('ai_agents_dashboard.html', 
                             agents_info=agents_info,
                             status_info=status_info)
    except Exception as e:
        logger.error(f"Agents dashboard error: {e}")
        return render_template('ai_agents_dashboard.html', 
                             agents_info={'total_agents': 30, 'agents': {}},
                             status_info={'status_summary': {'system_health': 'initializing'}})

@ai_agents_bp.route('/api/execute/<agent_key>', methods=['POST'])
def execute_single_agent(agent_key):
    """Execute a single AI agent"""
    try:
        request_data = request.get_json() or {}
        
        result = agent_integration_service.execute_agent(agent_key, request_data)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Agent execution error for {agent_key}: {e}")
        return jsonify({
            'error': str(e),
            'agent': agent_key,
            'status': 'failed'
        }), 500

@ai_agents_bp.route('/api/execute-multiple', methods=['POST'])
def execute_multiple_agents():
    """Execute multiple AI agents in coordination"""
    try:
        request_data = request.get_json() or {}
        agent_keys = request_data.get('agents', [])
        
        if not agent_keys:
            return jsonify({'error': 'No agents specified'}), 400
        
        result = agent_integration_service.execute_multiple_agents(agent_keys, request_data)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Multiple agents execution error: {e}")
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 500

@ai_agents_bp.route('/api/agents')
def get_available_agents():
    """Get list of all available agents"""
    try:
        agents_info = agent_integration_service.get_available_agents()
        return jsonify(agents_info)
        
    except Exception as e:
        logger.error(f"Get agents error: {e}")
        return jsonify({'error': str(e)}), 500

@ai_agents_bp.route('/api/status')
def get_agents_status():
    """Get status of all agents"""
    try:
        status_info = agent_integration_service.get_agent_status()
        return jsonify(status_info)
        
    except Exception as e:
        logger.error(f"Get status error: {e}")
        return jsonify({'error': str(e)}), 500

@ai_agents_bp.route('/api/status/<agent_key>')
def get_agent_status(agent_key):
    """Get status of specific agent"""
    try:
        status_info = agent_integration_service.get_agent_status(agent_key)
        return jsonify({
            'agent': agent_key,
            'status': status_info
        })
        
    except Exception as e:
        logger.error(f"Get agent status error for {agent_key}: {e}")
        return jsonify({'error': str(e)}), 500

# Individual Agent Routes

@ai_agents_bp.route('/master-strategist', methods=['GET', 'POST'])
def master_strategist():
    """Master Digital Marketing Strategist Agent"""
    if request.method == 'POST':
        request_data = request.get_json() or {}
        result = agent_integration_service.execute_agent('master_digital_marketing_strategist', request_data)
        return jsonify(result)
    
    return render_template('agent_interface.html', 
                         agent_name='Master Digital Marketing Strategist',
                         agent_key='master_digital_marketing_strategist')

@ai_agents_bp.route('/brand-storytelling', methods=['GET', 'POST'])
def brand_storytelling():
    """Brand Storytelling & Narrative Agent"""
    if request.method == 'POST':
        request_data = request.get_json() or {}
        result = agent_integration_service.execute_agent('brand_storytelling_narrative', request_data)
        return jsonify(result)
    
    return render_template('agent_interface.html', 
                         agent_name='Brand Storytelling & Narrative',
                         agent_key='brand_storytelling_narrative')

@ai_agents_bp.route('/content-creator', methods=['GET', 'POST'])
def content_creator():
    """Omnichannel Content Creator Agent"""
    if request.method == 'POST':
        request_data = request.get_json() or {}
        result = agent_integration_service.execute_agent('omnichannel_content_creator', request_data)
        return jsonify(result)
    
    return render_template('agent_interface.html', 
                         agent_name='Omnichannel Content Creator',
                         agent_key='omnichannel_content_creator')

@ai_agents_bp.route('/visual-production', methods=['GET', 'POST'])
def visual_production():
    """Visual Content Production Agent"""
    if request.method == 'POST':
        request_data = request.get_json() or {}
        result = agent_integration_service.execute_agent('visual_content_production', request_data)
        return jsonify(result)
    
    return render_template('agent_interface.html', 
                         agent_name='Visual Content Production',
                         agent_key='visual_content_production')

@ai_agents_bp.route('/video-production', methods=['GET', 'POST'])
def video_production():
    """Video Production Automation Agent"""
    if request.method == 'POST':
        request_data = request.get_json() or {}
        result = agent_integration_service.execute_agent('video_production_automation', request_data)
        return jsonify(result)
    
    return render_template('agent_interface.html', 
                         agent_name='Video Production Automation',
                         agent_key='video_production_automation')

@ai_agents_bp.route('/media-buying', methods=['GET', 'POST'])
def media_buying():
    """Advanced Media Buying Agent"""
    if request.method == 'POST':
        request_data = request.get_json() or {}
        result = agent_integration_service.execute_agent('advanced_media_buying', request_data)
        return jsonify(result)
    
    return render_template('agent_interface.html', 
                         agent_name='Advanced Media Buying',
                         agent_key='advanced_media_buying')

@ai_agents_bp.route('/seo-sem', methods=['GET', 'POST'])
def seo_sem():
    """SEO/SEM Optimization Agent"""
    if request.method == 'POST':
        request_data = request.get_json() or {}
        result = agent_integration_service.execute_agent('seo_sem_optimization', request_data)
        return jsonify(result)
    
    return render_template('agent_interface.html', 
                         agent_name='SEO/SEM Optimization',
                         agent_key='seo_sem_optimization')

@ai_agents_bp.route('/social-automation', methods=['GET', 'POST'])
def social_automation():
    """Social Media Automation Agent"""
    if request.method == 'POST':
        request_data = request.get_json() or {}
        result = agent_integration_service.execute_agent('social_media_automation', request_data)
        return jsonify(result)
    
    return render_template('agent_interface.html', 
                         agent_name='Social Media Automation',
                         agent_key='social_media_automation')

@ai_agents_bp.route('/business-development', methods=['GET', 'POST'])
def business_development():
    """Online Business Development Agent"""
    if request.method == 'POST':
        request_data = request.get_json() or {}
        result = agent_integration_service.execute_agent('online_business_development', request_data)
        return jsonify(result)
    
    return render_template('agent_interface.html', 
                         agent_name='Online Business Development',
                         agent_key='online_business_development')

@ai_agents_bp.route('/smart-goals', methods=['GET', 'POST'])
def smart_goals():
    """SMART Goals & KPI Agent"""
    if request.method == 'POST':
        request_data = request.get_json() or {}
        result = agent_integration_service.execute_agent('smart_goals_kpi', request_data)
        return jsonify(result)
    
    return render_template('agent_interface.html', 
                         agent_name='SMART Goals & KPI',
                         agent_key='smart_goals_kpi')

# Sales Methodology Agents

@ai_agents_bp.route('/spin-sales', methods=['GET', 'POST'])
def spin_sales():
    """SPIN Sales Methodology Agent"""
    if request.method == 'POST':
        request_data = request.get_json() or {}
        result = agent_integration_service.execute_agent('spin_sales_methodology', request_data)
        return jsonify(result)
    
    return render_template('agent_interface.html', 
                         agent_name='SPIN Sales Methodology',
                         agent_key='spin_sales_methodology')

@ai_agents_bp.route('/aida-sales', methods=['GET', 'POST'])
def aida_sales():
    """AIDA Sales Psychology Agent"""
    if request.method == 'POST':
        request_data = request.get_json() or {}
        result = agent_integration_service.execute_agent('aida_sales_psychology', request_data)
        return jsonify(result)
    
    return render_template('agent_interface.html', 
                         agent_name='AIDA Sales Psychology',
                         agent_key='aida_sales_psychology')

@ai_agents_bp.route('/ooda-sales', methods=['GET', 'POST'])
def ooda_sales():
    """OODA Loop Sales Strategy Agent"""
    if request.method == 'POST':
        request_data = request.get_json() or {}
        result = agent_integration_service.execute_agent('ooda_loop_sales_strategy', request_data)
        return jsonify(result)
    
    return render_template('agent_interface.html', 
                         agent_name='OODA Loop Sales Strategy',
                         agent_key='ooda_loop_sales_strategy')

# Sales Process Agents

@ai_agents_bp.route('/warm-inbound', methods=['GET', 'POST'])
def warm_inbound():
    """Sales Warm Inbound Agent"""
    if request.method == 'POST':
        request_data = request.get_json() or {}
        result = agent_integration_service.execute_agent('sales_warm_inbound', request_data)
        return jsonify(result)
    
    return render_template('agent_interface.html', 
                         agent_name='Sales Warm Inbound',
                         agent_key='sales_warm_inbound')

@ai_agents_bp.route('/cold-inbound', methods=['GET', 'POST'])
def cold_inbound():
    """Sales Cold Inbound Agent"""
    if request.method == 'POST':
        request_data = request.get_json() or {}
        result = agent_integration_service.execute_agent('sales_cold_inbound', request_data)
        return jsonify(result)
    
    return render_template('agent_interface.html', 
                         agent_name='Sales Cold Inbound',
                         agent_key='sales_cold_inbound')

@ai_agents_bp.route('/cold-outbound', methods=['GET', 'POST'])
def cold_outbound():
    """Sales Cold Outbound Agent"""
    if request.method == 'POST':
        request_data = request.get_json() or {}
        result = agent_integration_service.execute_agent('sales_cold_outbound', request_data)
        return jsonify(result)
    
    return render_template('agent_interface.html', 
                         agent_name='Sales Cold Outbound',
                         agent_key='sales_cold_outbound')

@ai_agents_bp.route('/funnel-optimization', methods=['GET', 'POST'])
def funnel_optimization():
    """Sales Funnel Optimization Agent"""
    if request.method == 'POST':
        request_data = request.get_json() or {}
        result = agent_integration_service.execute_agent('sales_funnel_optimization', request_data)
        return jsonify(result)
    
    return render_template('agent_interface.html', 
                         agent_name='Sales Funnel Optimization',
                         agent_key='sales_funnel_optimization')

# Pipeline & Order Management

@ai_agents_bp.route('/pipeline-management', methods=['GET', 'POST'])
def pipeline_management():
    """Sales Pipeline Management Agent"""
    if request.method == 'POST':
        request_data = request.get_json() or {}
        result = agent_integration_service.execute_agent('sales_pipeline_management', request_data)
        return jsonify(result)
    
    return render_template('agent_interface.html', 
                         agent_name='Sales Pipeline Management',
                         agent_key='sales_pipeline_management')

@ai_agents_bp.route('/order-fulfillment', methods=['GET', 'POST'])
def order_fulfillment():
    """Order Fulfillment Orchestration Agent"""
    if request.method == 'POST':
        request_data = request.get_json() or {}
        result = agent_integration_service.execute_agent('order_fulfillment_orchestration', request_data)
        return jsonify(result)
    
    return render_template('agent_interface.html', 
                         agent_name='Order Fulfillment Orchestration',
                         agent_key='order_fulfillment_orchestration')

@ai_agents_bp.route('/invoice-processing', methods=['GET', 'POST'])
def invoice_processing():
    """Invoice Processing Automation Agent"""
    if request.method == 'POST':
        request_data = request.get_json() or {}
        result = agent_integration_service.execute_agent('invoice_processing_automation', request_data)
        return jsonify(result)
    
    return render_template('agent_interface.html', 
                         agent_name='Invoice Processing Automation',
                         agent_key='invoice_processing_automation')

# German Business Practices

@ai_agents_bp.route('/gesprachsleitfaden', methods=['GET', 'POST'])
def gesprachsleitfaden():
    """Gesprächsleitfaden Agent"""
    if request.method == 'POST':
        request_data = request.get_json() or {}
        result = agent_integration_service.execute_agent('gesprachsleitfaden', request_data)
        return jsonify(result)
    
    return render_template('agent_interface.html', 
                         agent_name='Gesprächsleitfaden',
                         agent_key='gesprachsleitfaden')

# Strategic & Intelligence Agents

@ai_agents_bp.route('/strategic-planning', methods=['GET', 'POST'])
def strategic_planning():
    """Strategic Planning Optimization Agent"""
    if request.method == 'POST':
        request_data = request.get_json() or {}
        result = agent_integration_service.execute_agent('strategic_planning_optimization', request_data)
        return jsonify(result)
    
    return render_template('agent_interface.html', 
                         agent_name='Strategic Planning Optimization',
                         agent_key='strategic_planning_optimization')

@ai_agents_bp.route('/competitive-intelligence', methods=['GET', 'POST'])
def competitive_intelligence():
    """Competitive Intelligence Agent"""
    if request.method == 'POST':
        request_data = request.get_json() or {}
        result = agent_integration_service.execute_agent('competitive_intelligence', request_data)
        return jsonify(result)
    
    return render_template('agent_interface.html', 
                         agent_name='Competitive Intelligence',
                         agent_key='competitive_intelligence')

@ai_agents_bp.route('/customer-success', methods=['GET', 'POST'])
def customer_success():
    """Customer Success Management Agent"""
    if request.method == 'POST':
        request_data = request.get_json() or {}
        result = agent_integration_service.execute_agent('customer_success_management', request_data)
        return jsonify(result)
    
    return render_template('agent_interface.html', 
                         agent_name='Customer Success Management',
                         agent_key='customer_success_management')

@ai_agents_bp.route('/process-automation', methods=['GET', 'POST'])
def process_automation():
    """Business Process Automation Agent"""
    if request.method == 'POST':
        request_data = request.get_json() or {}
        result = agent_integration_service.execute_agent('business_process_automation', request_data)
        return jsonify(result)
    
    return render_template('agent_interface.html', 
                         agent_name='Business Process Automation',
                         agent_key='business_process_automation')

# Purpose & Vision Agents

@ai_agents_bp.route('/ikigai-purpose', methods=['GET', 'POST'])
def ikigai_purpose():
    """Ikigai Purpose Strategy Agent"""
    if request.method == 'POST':
        request_data = request.get_json() or {}
        result = agent_integration_service.execute_agent('ikigai_purpose_strategy', request_data)
        return jsonify(result)
    
    return render_template('agent_interface.html', 
                         agent_name='Ikigai Purpose Strategy',
                         agent_key='ikigai_purpose_strategy')

@ai_agents_bp.route('/vision-board', methods=['GET', 'POST'])
def vision_board():
    """Vision Board Strategy Agent"""
    if request.method == 'POST':
        request_data = request.get_json() or {}
        result = agent_integration_service.execute_agent('vision_board_strategy', request_data)
        return jsonify(result)
    
    return render_template('agent_interface.html', 
                         agent_name='Vision Board Strategy',
                         agent_key='vision_board_strategy')

# Comprehensive Campaign Orchestration

@ai_agents_bp.route('/orchestrate-campaign', methods=['POST'])
def orchestrate_campaign():
    """Orchestrate comprehensive marketing campaign across multiple agents"""
    try:
        request_data = request.get_json() or {}
        
        # Default to all agents if none specified
        agent_keys = request_data.get('agents', list(agent_integration_service.agents.keys()))
        
        result = agent_integration_service.execute_multiple_agents(agent_keys, request_data)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Campaign orchestration error: {e}")
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 500

@ai_agents_bp.route('/test-all-agents', methods=['POST'])
def test_all_agents():
    """Test all 30 agents functionality"""
    try:
        test_data = {
            'test_execution': True,
            'campaign_id': 'TEST_ALL_AGENTS',
            'business_id': 'TEST_BUSINESS'
        }
        
        all_agent_keys = list(agent_integration_service.agents.keys())
        result = agent_integration_service.execute_multiple_agents(all_agent_keys, test_data)
        
        return jsonify({
            'test_execution': True,
            'total_agents_tested': len(all_agent_keys),
            'execution_results': result
        })
        
    except Exception as e:
        logger.error(f"All agents test error: {e}")
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 500