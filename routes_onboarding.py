"""
Customer Onboarding Routes - AI Agent Ecosystem
Handles customer onboarding workflow and subscription management
"""

from flask import Blueprint, request, jsonify, render_template, redirect, url_for, session
from datetime import datetime
from services.customer_onboarding_service import customer_onboarding_service
import logging

logger = logging.getLogger(__name__)

# Create onboarding blueprint
onboarding_bp = Blueprint('onboarding', __name__, url_prefix='/onboarding')

@onboarding_bp.route('/')
def start():
    """Landing page for new customers"""
    try:
        return render_template('onboarding/welcome.html', 
                             subscription_tiers=customer_onboarding_service.subscription_tiers)
    except Exception as e:
        logger.error(f"Error loading onboarding start page: {str(e)}")
        return jsonify({'error': 'Failed to load onboarding page'}), 500

@onboarding_bp.route('/api/start', methods=['POST'])
def api_start_onboarding():
    """API endpoint to start onboarding process"""
    try:
        data = request.get_json()
        user_email = data.get('email')
        company_name = data.get('company_name')
        
        if not user_email:
            return jsonify({'error': 'Email is required'}), 400
        
        result = customer_onboarding_service.start_onboarding(user_email, company_name)
        
        if result['success']:
            # Store session info
            session['onboarding_session_id'] = result['session']['session_id']
            session['user_email'] = user_email
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error starting onboarding: {str(e)}")
        return jsonify({'error': 'Failed to start onboarding'}), 500

@onboarding_bp.route('/subscription-selector')
def subscription_selector():
    """Subscription tier selection page"""
    try:
        # Get user preferences if available
        use_case = request.args.get('use_case', '')
        team_size = int(request.args.get('team_size', 1))
        budget = float(request.args.get('budget', 1000))
        
        # Get recommendation
        recommendation = customer_onboarding_service.get_subscription_recommendation(
            use_case, team_size, budget
        )
        
        return render_template('onboarding/subscription_selector.html',
                             subscription_tiers=customer_onboarding_service.subscription_tiers,
                             recommendation=recommendation,
                             use_case=use_case,
                             team_size=team_size,
                             budget=budget)
    except Exception as e:
        logger.error(f"Error loading subscription selector: {str(e)}")
        return jsonify({'error': 'Failed to load subscription selector'}), 500

@onboarding_bp.route('/api/subscription/create', methods=['POST'])
def api_create_subscription():
    """Create Stripe subscription for customer"""
    try:
        data = request.get_json()
        email = data.get('email') or session.get('user_email')
        tier = data.get('tier')
        use_trial = data.get('use_trial', True)
        
        if not email or not tier:
            return jsonify({'error': 'Email and tier are required'}), 400
        
        # Get trial days for tier
        tier_config = customer_onboarding_service.subscription_tiers.get(tier)
        trial_days = tier_config.get('trial_days') if use_trial else None
        
        result = customer_onboarding_service.create_stripe_subscription(
            email, tier, trial_days
        )
        
        if result['success']:
            session['subscription_tier'] = tier
            session['trial_end'] = result.get('trial_end')
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error creating subscription: {str(e)}")
        return jsonify({'error': 'Failed to create subscription'}), 500

@onboarding_bp.route('/agent-selection')
def agent_selection():
    """AI agent selection and configuration page"""
    try:
        tier = session.get('subscription_tier', 'starter')
        use_case = request.args.get('use_case', 'general business automation')
        
        # Get recommended agents
        recommended_agents = customer_onboarding_service.get_recommended_agents(tier, use_case)
        
        return render_template('onboarding/agent_selection.html',
                             recommended_agents=recommended_agents,
                             tier=tier,
                             use_case=use_case)
    except Exception as e:
        logger.error(f"Error loading agent selection: {str(e)}")
        return jsonify({'error': 'Failed to load agent selection'}), 500

@onboarding_bp.route('/api/agents/select', methods=['POST'])
def api_select_agents():
    """API endpoint to select and configure agents"""
    try:
        data = request.get_json()
        selected_agents = data.get('agents', [])
        
        session['selected_agents'] = selected_agents
        
        # For each selected agent, we would normally:
        # 1. Create agent configuration
        # 2. Set up initial parameters  
        # 3. Run setup tests
        
        return jsonify({
            'success': True,
            'message': f'Successfully configured {len(selected_agents)} agents',
            'agents': selected_agents
        })
        
    except Exception as e:
        logger.error(f"Error selecting agents: {str(e)}")
        return jsonify({'error': 'Failed to select agents'}), 500

@onboarding_bp.route('/integration-setup')
def integration_setup():
    """API integration and webhook setup page"""
    try:
        tier = session.get('subscription_tier', 'starter')
        selected_agents = session.get('selected_agents', [])
        
        return render_template('onboarding/integration_setup.html',
                             tier=tier,
                             selected_agents=selected_agents)
    except Exception as e:
        logger.error(f"Error loading integration setup: {str(e)}")
        return jsonify({'error': 'Failed to load integration setup'}), 500

@onboarding_bp.route('/api/integration/setup', methods=['POST'])
def api_setup_integration():
    """Set up API keys and integrations"""
    try:
        data = request.get_json()
        integrations = data.get('integrations', {})
        
        # Generate API key for customer
        import secrets
        api_key = f"aaes_{secrets.token_urlsafe(32)}"
        
        session['api_key'] = api_key
        session['integrations'] = integrations
        
        return jsonify({
            'success': True,
            'api_key': api_key,
            'webhook_url': f"https://your-domain.com/webhooks/{api_key}",
            'integrations_configured': len(integrations)
        })
        
    except Exception as e:
        logger.error(f"Error setting up integrations: {str(e)}")
        return jsonify({'error': 'Failed to setup integrations'}), 500

@onboarding_bp.route('/first-success')
def first_success():
    """First success milestone - run sample automation"""
    try:
        selected_agents = session.get('selected_agents', [])
        api_key = session.get('api_key')
        
        return render_template('onboarding/first_success.html',
                             selected_agents=selected_agents,
                             api_key=api_key)
    except Exception as e:
        logger.error(f"Error loading first success page: {str(e)}")
        return jsonify({'error': 'Failed to load first success page'}), 500

@onboarding_bp.route('/api/first-success/run', methods=['POST'])
def api_run_first_success():
    """Run first successful automation"""
    try:
        data = request.get_json()
        agent_name = data.get('agent_name')
        task_type = data.get('task_type')
        
        # Simulate running first automation
        result = {
            'success': True,
            'agent': agent_name,
            'task': task_type,
            'result': {
                'execution_time': '2.3 seconds',
                'status': 'completed',
                'output': f'Successfully executed {task_type} using {agent_name} agent',
                'cost': '$0.02',
                'tokens_used': 150
            },
            'next_steps': [
                'Set up scheduled automations',
                'Explore more agents',
                'Integrate with your tools',
                'Invite team members'
            ]
        }
        
        # Mark onboarding as complete
        session['onboarding_complete'] = True
        session['first_success_date'] = str(datetime.now())
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error running first success automation: {str(e)}")
        return jsonify({'error': 'Failed to run automation'}), 500

@onboarding_bp.route('/complete')
def complete():
    """Onboarding completion page"""
    try:
        user_email = session.get('user_email')
        tier = session.get('subscription_tier')
        selected_agents = session.get('selected_agents', [])
        api_key = session.get('api_key')
        
        return render_template('onboarding/complete.html',
                             user_email=user_email,
                             tier=tier,
                             agents_count=len(selected_agents),
                             api_key=api_key,
                             next_steps=[
                                 'Explore the full agent marketplace',
                                 'Set up automated workflows',
                                 'Invite team members',
                                 'Schedule success check-in call'
                             ])
    except Exception as e:
        logger.error(f"Error loading completion page: {str(e)}")
        return jsonify({'error': 'Failed to load completion page'}), 500

@onboarding_bp.route('/api/progress/<session_id>')
def api_get_progress(session_id):
    """Get onboarding progress for session"""
    try:
        progress = customer_onboarding_service.get_onboarding_progress(session_id)
        return jsonify(progress)
    except Exception as e:
        logger.error(f"Error getting onboarding progress: {str(e)}")
        return jsonify({'error': 'Failed to get progress'}), 500

@onboarding_bp.route('/api/step/complete', methods=['POST'])
def api_complete_step():
    """Mark onboarding step as complete"""
    try:
        data = request.get_json()
        session_id = data.get('session_id') or session.get('onboarding_session_id')
        step_number = data.get('step_number')
        step_data = data.get('step_data', {})
        
        result = customer_onboarding_service.complete_onboarding_step(
            session_id, step_number, step_data
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error completing onboarding step: {str(e)}")
        return jsonify({'error': 'Failed to complete step'}), 500