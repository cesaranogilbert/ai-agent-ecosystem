"""
Content Generation Routes
Routes for accessing the four Content Generation AI Agents
"""

from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
import json
import logging
from datetime import datetime

# Import the content generation agents
from services.digital_course_creator import digital_course_creator
from services.lead_magnet_generator import lead_magnet_generator
from services.ebook_publishing_specialist import ebook_publishing_specialist
from services.webinar_content_engine import webinar_content_engine

content_generation_bp = Blueprint('content_generation', __name__)

@content_generation_bp.route('/content-generation')
def content_generation_dashboard():
    """Content Generation Dashboard"""
    try:
        # Get some sample statistics
        stats = {
            'available_agents': 4,
            'content_types': 12,
            'estimated_value': '$500K-$5.4M',
            'platforms_supported': 8,
            'expertise_areas': ['Tax Optimization', 'M&A Analysis', 'Executive Coaching', 'Strategic Planning']
        }
        
        return render_template('content_generation_dashboard.html', 
                             stats=stats,
                             current_time=datetime.now())
    except Exception as e:
        logging.error(f"Content generation dashboard error: {e}")
        flash(f"Error loading dashboard: {str(e)}", 'error')
        return redirect(url_for('landing_page'))

@content_generation_bp.route('/digital-course-creator')
def digital_course_creator_page():
    """Digital Course Creator interface"""
    try:
        return render_template('digital_course_creator.html')
    except Exception as e:
        logging.error(f"Digital course creator page error: {e}")
        flash(f"Error loading page: {str(e)}", 'error')
        return redirect(url_for('content_generation.content_generation_dashboard'))

@content_generation_bp.route('/create-course', methods=['POST'])
def create_course():
    """Create a complete online course"""
    try:
        # Get form data
        course_data = {
            'title': request.form.get('title', 'Professional Course'),
            'expertise_area': request.form.get('expertise_area', 'business'),
            'target_audience': request.form.get('target_audience', 'professionals'),
            'duration_hours': int(request.form.get('duration_hours', 8)),
            'pricing_tier': request.form.get('pricing_tier', 'premium')
        }
        
        # Create the course
        result = digital_course_creator.create_complete_course(course_data)
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'course': result,
                'message': 'Course created successfully!'
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'message': 'Course creation failed'
            })
            
    except Exception as e:
        logging.error(f"Course creation error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Course creation failed'
        })

@content_generation_bp.route('/lead-magnet-generator')
def lead_magnet_generator_page():
    """Lead Magnet Generator interface"""
    try:
        return render_template('lead_magnet_generator.html')
    except Exception as e:
        logging.error(f"Lead magnet generator page error: {e}")
        flash(f"Error loading page: {str(e)}", 'error')
        return redirect(url_for('content_generation.content_generation_dashboard'))

@content_generation_bp.route('/create-lead-magnet', methods=['POST'])
def create_lead_magnet():
    """Create a complete lead magnet suite"""
    try:
        # Get form data
        magnet_data = {
            'title': request.form.get('title', 'Professional Guide'),
            'expertise_area': request.form.get('expertise_area', 'business'),
            'target_audience': request.form.get('target_audience', 'professionals'),
            'type': request.form.get('type', 'pdf_guide'),
            'conversion_goal': request.form.get('conversion_goal', 'course_enrollment')
        }
        
        # Create the lead magnet
        result = lead_magnet_generator.create_lead_magnet_suite(magnet_data)
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'magnet': result,
                'message': 'Lead magnet created successfully!'
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'message': 'Lead magnet creation failed'
            })
            
    except Exception as e:
        logging.error(f"Lead magnet creation error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Lead magnet creation failed'
        })

@content_generation_bp.route('/ebook-publisher')
def ebook_publisher_page():
    """eBook Publishing Specialist interface"""
    try:
        return render_template('ebook_publisher.html')
    except Exception as e:
        logging.error(f"eBook publisher page error: {e}")
        flash(f"Error loading page: {str(e)}", 'error')
        return redirect(url_for('content_generation.content_generation_dashboard'))

@content_generation_bp.route('/create-ebook', methods=['POST'])
def create_ebook():
    """Create a complete eBook ready for publishing"""
    try:
        # Get form data
        book_data = {
            'title': request.form.get('title', 'Professional Business Guide'),
            'expertise_area': request.form.get('expertise_area', 'business'),
            'target_audience': request.form.get('target_audience', 'professionals'),
            'length': request.form.get('length', 'standard'),
            'pricing': request.form.get('pricing', 'premium')
        }
        
        # Create the eBook
        result = ebook_publishing_specialist.create_complete_ebook(book_data)
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'ebook': result,
                'message': 'eBook created successfully!'
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'message': 'eBook creation failed'
            })
            
    except Exception as e:
        logging.error(f"eBook creation error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'eBook creation failed'
        })

@content_generation_bp.route('/webinar-engine')
def webinar_engine_page():
    """Webinar Content Engine interface"""
    try:
        return render_template('webinar_engine.html')
    except Exception as e:
        logging.error(f"Webinar engine page error: {e}")
        flash(f"Error loading page: {str(e)}", 'error')
        return redirect(url_for('content_generation.content_generation_dashboard'))

@content_generation_bp.route('/create-webinar', methods=['POST'])
def create_webinar():
    """Create a complete webinar system"""
    try:
        # Get form data
        webinar_data = {
            'title': request.form.get('title', 'Professional Webinar'),
            'expertise_area': request.form.get('expertise_area', 'business'),
            'target_audience': request.form.get('target_audience', 'professionals'),
            'length': int(request.form.get('length', 60)),
            'conversion_goal': request.form.get('conversion_goal', 'course_enrollment'),
            'pricing_tier': request.form.get('pricing_tier', 'premium')
        }
        
        # Create the webinar
        result = webinar_content_engine.create_complete_webinar_system(webinar_data)
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'webinar': result,
                'message': 'Webinar system created successfully!'
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'message': 'Webinar creation failed'
            })
            
    except Exception as e:
        logging.error(f"Webinar creation error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Webinar creation failed'
        })

@content_generation_bp.route('/quick-generate')
def quick_generate_page():
    """Quick Generate interface for rapid content creation"""
    try:
        return render_template('quick_generate.html')
    except Exception as e:
        logging.error(f"Quick generate page error: {e}")
        flash(f"Error loading page: {str(e)}", 'error')
        return redirect(url_for('content_generation.content_generation_dashboard'))

@content_generation_bp.route('/quick-generate-content', methods=['POST'])
def quick_generate_content():
    """Quick generate content for immediate use"""
    try:
        # Get form data
        content_type = request.form.get('content_type')
        expertise_area = request.form.get('expertise_area', 'business')
        
        if content_type == 'course':
            course_data = {
                'title': f'{expertise_area.title()} Excellence Course',
                'expertise_area': expertise_area,
                'target_audience': 'professionals',
                'duration_hours': 6,
                'pricing_tier': 'premium'
            }
            result = digital_course_creator.create_complete_course(course_data)
            
        elif content_type == 'lead_magnet':
            magnet_data = {
                'title': f'{expertise_area.title()} Professional Guide',
                'expertise_area': expertise_area,
                'target_audience': 'professionals',
                'type': 'pdf_guide',
                'conversion_goal': 'course_enrollment'
            }
            result = lead_magnet_generator.create_lead_magnet_suite(magnet_data)
            
        elif content_type == 'ebook':
            book_data = {
                'title': f'{expertise_area.title()} Excellence: Professional Guide',
                'expertise_area': expertise_area,
                'target_audience': 'professionals',
                'length': 'standard',
                'pricing': 'premium'
            }
            result = ebook_publishing_specialist.create_complete_ebook(book_data)
            
        elif content_type == 'webinar':
            webinar_data = {
                'title': f'{expertise_area.title()} Excellence Masterclass',
                'expertise_area': expertise_area,
                'target_audience': 'professionals',
                'length': 60,
                'conversion_goal': 'course_enrollment',
                'pricing_tier': 'premium'
            }
            result = webinar_content_engine.create_complete_webinar_system(webinar_data)
            
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid content type',
                'message': 'Content generation failed'
            })
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'content': result,
                'content_type': content_type,
                'message': f'{content_type.title()} generated successfully!'
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'message': f'{content_type.title()} generation failed'
            })
            
    except Exception as e:
        logging.error(f"Quick content generation error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Content generation failed'
        })

@content_generation_bp.route('/content-analytics')
def content_analytics():
    """Content generation analytics and performance tracking"""
    try:
        # Sample analytics data
        analytics_data = {
            'total_content_generated': 47,
            'courses_created': 12,
            'lead_magnets_created': 18,
            'ebooks_created': 8,
            'webinars_created': 9,
            'estimated_value_generated': '$2.4M',
            'average_creation_time': '15 minutes',
            'success_rate': '100%',
            'top_expertise_areas': [
                {'area': 'Tax Optimization', 'count': 15, 'value': '$800K'},
                {'area': 'M&A Analysis', 'count': 12, 'value': '$600K'},
                {'area': 'Executive Coaching', 'count': 10, 'value': '$500K'},
                {'area': 'Strategic Planning', 'count': 10, 'value': '$500K'}
            ],
            'recent_creations': [
                {'type': 'Course', 'title': 'Swiss Tax Optimization Excellence', 'created': '2024-01-15', 'value': '$25K'},
                {'type': 'eBook', 'title': 'M&A Due Diligence Mastery', 'created': '2024-01-14', 'value': '$47'},
                {'type': 'Webinar', 'title': 'Executive Performance Transformation', 'created': '2024-01-13', 'value': '$15K'},
                {'type': 'Lead Magnet', 'title': 'Strategic Planning Quick Guide', 'created': '2024-01-12', 'value': 'Free'}
            ]
        }
        
        return render_template('content_analytics.html', 
                             analytics=analytics_data,
                             current_time=datetime.now())
    except Exception as e:
        logging.error(f"Content analytics error: {e}")
        flash(f"Error loading analytics: {str(e)}", 'error')
        return redirect(url_for('content_generation.content_generation_dashboard'))

@content_generation_bp.route('/export-content/<content_type>/<content_id>')
def export_content(content_type, content_id):
    """Export generated content in various formats"""
    try:
        # This would typically fetch the content from database
        # For now, return a sample export
        export_data = {
            'content_type': content_type,
            'content_id': content_id,
            'export_formats': ['PDF', 'DOCX', 'HTML', 'JSON'],
            'export_url': f'/download/{content_type}/{content_id}',
            'message': f'{content_type.title()} ready for export'
        }
        
        return jsonify({
            'success': True,
            'export': export_data,
            'message': 'Export prepared successfully'
        })
        
    except Exception as e:
        logging.error(f"Content export error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Export preparation failed'
        })

@content_generation_bp.route('/test-agents')
def test_agents():
    """Test all content generation agents"""
    try:
        test_results = {}
        
        # Test Digital Course Creator
        try:
            course_result = digital_course_creator.create_complete_course({
                'title': 'Test Course',
                'expertise_area': 'tax_optimization',
                'target_audience': 'professionals',
                'duration_hours': 6,
                'pricing_tier': 'premium'
            })
            test_results['digital_course_creator'] = {
                'status': 'success' if course_result.get('success') else 'failed',
                'result': course_result
            }
        except Exception as e:
            test_results['digital_course_creator'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Test Lead Magnet Generator
        try:
            magnet_result = lead_magnet_generator.create_lead_magnet_suite({
                'title': 'Test Lead Magnet',
                'expertise_area': 'ma_analysis',
                'target_audience': 'professionals',
                'type': 'pdf_guide',
                'conversion_goal': 'course_enrollment'
            })
            test_results['lead_magnet_generator'] = {
                'status': 'success' if magnet_result.get('success') else 'failed',
                'result': magnet_result
            }
        except Exception as e:
            test_results['lead_magnet_generator'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Test eBook Publishing Specialist
        try:
            ebook_result = ebook_publishing_specialist.create_complete_ebook({
                'title': 'Test eBook',
                'expertise_area': 'executive_coaching',
                'target_audience': 'professionals',
                'length': 'standard',
                'pricing': 'premium'
            })
            test_results['ebook_publishing_specialist'] = {
                'status': 'success' if ebook_result.get('success') else 'failed',
                'result': ebook_result
            }
        except Exception as e:
            test_results['ebook_publishing_specialist'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Test Webinar Content Engine
        try:
            webinar_result = webinar_content_engine.create_complete_webinar_system({
                'title': 'Test Webinar',
                'expertise_area': 'strategic_planning',
                'target_audience': 'professionals',
                'length': 60,
                'conversion_goal': 'course_enrollment',
                'pricing_tier': 'premium'
            })
            test_results['webinar_content_engine'] = {
                'status': 'success' if webinar_result.get('success') else 'failed',
                'result': webinar_result
            }
        except Exception as e:
            test_results['webinar_content_engine'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Calculate overall test results
        total_tests = len(test_results)
        successful_tests = sum(1 for result in test_results.values() if result['status'] == 'success')
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        return jsonify({
            'success': True,
            'test_summary': {
                'total_agents': total_tests,
                'successful_tests': successful_tests,
                'success_rate': f'{success_rate:.1f}%',
                'all_tests_passed': success_rate == 100.0
            },
            'detailed_results': test_results,
            'message': f'Content generation agent testing completed: {successful_tests}/{total_tests} agents working correctly'
        })
        
    except Exception as e:
        logging.error(f"Agent testing error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Agent testing failed'
        })