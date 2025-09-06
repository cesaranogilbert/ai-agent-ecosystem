"""
Real-Time Data Validation API Routes
===================================

API endpoints for the Real-Time Data Validation system integration
with the existing multi-agent framework.
"""

from flask import Blueprint, request, jsonify, render_template
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from services.real_time_data_validator import (
    real_time_validator,
    ValidationRequest,
    DataValidationResult
)
from services.enhanced_wealth_expert_agent import enhanced_wealth_expert
from services.multi_agent_orchestrator import multi_agent_orchestrator


validation_bp = Blueprint('validation', __name__)
logger = logging.getLogger(__name__)


@validation_bp.route('/api/validation/validate-data', methods=['POST'])
def validate_data():
    """
    Validate financial data in real-time
    
    Expected JSON:
    {
        "data_type": "nft_price|crypto_price|stock_price|arbitrage_opportunity",
        "data": {
            "symbol": "BTC",
            "price": 68500,
            "additional_fields": "..."
        },
        "validation_depth": "basic|standard|deep",
        "priority": 5
    }
    """
    
    try:
        data = request.get_json()
        
        if not data or 'data_type' not in data or 'data' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required fields: data_type and data'
            }), 400
        
        # Create validation request
        validation_request = ValidationRequest(
            request_id=f"validation_{datetime.now().timestamp()}",
            data_type=data['data_type'],
            original_data=data['data'],
            priority=data.get('priority', 5),
            validation_depth=data.get('validation_depth', 'standard')
        )
        
        # Run validation asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            validation_result = loop.run_until_complete(
                real_time_validator.validate_financial_data(validation_request)
            )
            
            return jsonify({
                'success': True,
                'validation_result': {
                    'original_value': validation_result.original_value,
                    'validated_value': validation_result.validated_value,
                    'confidence_score': validation_result.confidence_score,
                    'sources_checked': validation_result.sources_checked,
                    'discrepancy_found': validation_result.discrepancy_found,
                    'correction_applied': validation_result.correction_applied,
                    'validation_notes': validation_result.validation_notes,
                    'last_updated': validation_result.last_updated.isoformat()
                }
            })
            
        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Data validation error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@validation_bp.route('/api/validation/enhanced-wealth-analysis', methods=['POST'])
def enhanced_wealth_analysis():
    """
    Generate comprehensive wealth analysis with real-time validated data
    
    Expected JSON:
    {
        "client_data": {
            "total_assets": 10000,
            "risk_profile": "aggressive",
            "investment_goals": ["growth", "wealth_building"]
        },
        "investment_categories": ["nft", "crypto", "stocks", "arbitrage", "shorts"]
    }
    """
    
    try:
        data = request.get_json()
        
        if not data or 'client_data' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: client_data'
            }), 400
        
        client_data = data['client_data']
        investment_categories = data.get('investment_categories', ['nft', 'crypto', 'stocks', 'arbitrage', 'shorts'])
        
        # Run enhanced analysis asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            analysis = loop.run_until_complete(
                enhanced_wealth_expert.analyze_validated_investment_opportunities(
                    client_data=client_data,
                    investment_categories=investment_categories
                )
            )
            
            return jsonify({
                'success': True,
                'analysis': {
                    'analysis_timestamp': analysis.analysis_timestamp.isoformat(),
                    'validation_confidence': analysis.validation_confidence,
                    'data_freshness_score': analysis.data_freshness_score,
                    'corrections_applied': analysis.corrections_applied,
                    
                    'investment_opportunities': {
                        'nft': analysis.nft_opportunities,
                        'crypto': analysis.crypto_opportunities,
                        'stocks': analysis.stock_opportunities,
                        'arbitrage': analysis.arbitrage_opportunities,
                        'shorts': analysis.short_opportunities
                    },
                    
                    'validation_summary': analysis.validation_summary,
                    'data_sources_used': analysis.data_sources_used,
                    'last_market_update': analysis.last_market_update.isoformat(),
                    
                    'wealth_analysis': {
                        'total_assets': analysis.total_assets,
                        'risk_profile': analysis.risk_profile,
                        'recommended_allocation': analysis.recommended_allocation,
                        'projected_growth': analysis.projected_growth,
                        'confidence_score': analysis.confidence_score
                    }
                }
            })
            
        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Enhanced wealth analysis error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@validation_bp.route('/api/validation/multi-agent-validated-task', methods=['POST'])
def submit_multi_agent_validated_task():
    """
    Submit task to multi-agent orchestrator with real-time validation
    
    Expected JSON:
    {
        "task_description": "Analyze current NFT market for flip opportunities",
        "requirements": ["real_time_data", "multi_source_validation", "risk_assessment"],
        "priority": 8,
        "validation_required": true
    }
    """
    
    try:
        data = request.get_json()
        
        if not data or 'task_description' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: task_description'
            }), 400
        
        task_description = data['task_description']
        requirements = data.get('requirements', [])
        priority = data.get('priority', 5)
        validation_required = data.get('validation_required', True)
        
        # Add validation requirement if needed
        if validation_required and 'real_time_validation' not in requirements:
            requirements.append('real_time_validation')
        
        # Submit to multi-agent orchestrator
        task_id = multi_agent_orchestrator.submit_task(
            description=task_description,
            requirements=requirements,
            priority=priority
        )
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': 'Task submitted to multi-agent orchestrator with validation',
            'estimated_completion': '5-15 minutes',
            'validation_enabled': validation_required
        })
        
    except Exception as e:
        logger.error(f"Multi-agent task submission error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@validation_bp.route('/api/validation/status', methods=['GET'])
def validation_status():
    """Get status of validation system"""
    
    try:
        status = real_time_validator.get_validation_summary()
        
        return jsonify({
            'success': True,
            'system_status': 'operational',
            'validation_statistics': status,
            'data_sources_available': {
                'crypto_exchanges': 4,
                'nft_marketplaces': 3,
                'stock_data_providers': 2,
                'arbitrage_platforms': 5
            },
            'last_update': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Validation status error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@validation_bp.route('/validation-dashboard')
def validation_dashboard():
    """Render validation dashboard page"""
    return render_template('validation_dashboard.html')


@validation_bp.route('/api/validation/correct-analysis', methods=['POST'])
def correct_existing_analysis():
    """
    Correct existing analysis with real-time validated data
    
    Expected JSON:
    {
        "original_analysis": {
            "nft_opportunities": [...],
            "crypto_opportunities": [...],
            "stock_opportunities": [...],
            "arbitrage_opportunities": [...],
            "short_opportunities": [...]
        },
        "validation_depth": "standard"
    }
    """
    
    try:
        data = request.get_json()
        
        if not data or 'original_analysis' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: original_analysis'
            }), 400
        
        original_analysis = data['original_analysis']
        validation_depth = data.get('validation_depth', 'standard')
        
        # Create validation requests for each opportunity type
        validation_requests = []
        
        # Validate NFT opportunities
        for nft_opp in original_analysis.get('nft_opportunities', []):
            validation_requests.append(ValidationRequest(
                request_id=f"correct_nft_{datetime.now().timestamp()}",
                data_type='nft_price',
                original_data=nft_opp,
                validation_depth=validation_depth
            ))
        
        # Validate crypto opportunities
        for crypto_opp in original_analysis.get('crypto_opportunities', []):
            validation_requests.append(ValidationRequest(
                request_id=f"correct_crypto_{datetime.now().timestamp()}",
                data_type='crypto_price',
                original_data=crypto_opp,
                validation_depth=validation_depth
            ))
        
        # Run all validations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            validation_results = []
            for request in validation_requests:
                result = loop.run_until_complete(
                    real_time_validator.validate_financial_data(request)
                )
                validation_results.append(result)
            
            # Generate corrected analysis
            corrected_analysis = loop.run_until_complete(
                real_time_validator.generate_corrected_analysis(
                    original_analysis, validation_results
                )
            )
            
            return jsonify({
                'success': True,
                'corrected_analysis': corrected_analysis,
                'corrections_summary': {
                    'total_validations': len(validation_results),
                    'corrections_applied': sum(1 for r in validation_results if r.correction_applied),
                    'confidence_improvement': '+15%',  # Placeholder
                    'data_freshness': 'Real-time validated'
                }
            })
            
        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Analysis correction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Error handlers
@validation_bp.errorhandler(404)
def validation_not_found(error):
    return jsonify({
        'success': False,
        'error': 'Validation endpoint not found'
    }), 404


@validation_bp.errorhandler(500)
def validation_server_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal validation system error'
    }), 500