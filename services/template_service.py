"""
Template Service - Manages template generation, marketplace, and monetization
"""
import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from models import (AppTemplate, TemplateCategory, TemplatePurchase, 
                   TemplateReview, TemplateTag, db)

class TemplateService:
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def initialize_template_categories(self):
        """Initialize default template categories"""
        categories = [
            {
                'name': 'AI & Machine Learning',
                'slug': 'ai-ml',
                'description': 'Templates featuring AI integrations and machine learning capabilities',
                'icon': 'fas fa-robot',
                'color': '#6f42c1',
                'sort_order': 1
            },
            {
                'name': 'Business & Analytics',
                'slug': 'business-analytics', 
                'description': 'Dashboard and analytics templates for business intelligence',
                'icon': 'fas fa-chart-line',
                'color': '#007bff',
                'sort_order': 2
            },
            {
                'name': 'E-commerce & Marketplace',
                'slug': 'ecommerce',
                'description': 'Online store and marketplace templates',
                'icon': 'fas fa-shopping-cart',
                'color': '#28a745',
                'sort_order': 3
            },
            {
                'name': 'SaaS & Web Apps',
                'slug': 'saas-webapp',
                'description': 'Complete SaaS application templates',
                'icon': 'fas fa-cloud',
                'color': '#17a2b8', 
                'sort_order': 4
            },
            {
                'name': 'API & Integration',
                'slug': 'api-integration',
                'description': 'API development and integration templates',
                'icon': 'fas fa-plug',
                'color': '#ffc107',
                'sort_order': 5
            },
            {
                'name': 'Automation & DevOps',
                'slug': 'automation-devops',
                'description': 'Automation tools and DevOps templates',
                'icon': 'fas fa-cog',
                'color': '#fd7e14',
                'sort_order': 6
            }
        ]
        
        for cat_data in categories:
            existing = TemplateCategory.query.filter_by(slug=cat_data['slug']).first()
            if not existing:
                category = TemplateCategory()
                category.name = cat_data['name']
                category.slug = cat_data['slug']
                category.description = cat_data['description']
                category.icon = cat_data['icon']
                category.color = cat_data['color']
                category.sort_order = cat_data['sort_order']
                db.session.add(category)
        
        db.session.commit()
        self.logger.info("Template categories initialized")

    def initialize_template_tags(self):
        """Initialize commonly used tags"""
        tags = [
            ('AI', 'ai', '#6f42c1'),
            ('Python', 'python', '#3776ab'),
            ('Flask', 'flask', '#000000'),
            ('React', 'react', '#61dafb'),
            ('API', 'api', '#ff6b6b'),
            ('Database', 'database', '#336791'),
            ('Authentication', 'auth', '#4ecdc4'),
            ('Dashboard', 'dashboard', '#45b7d1'),
            ('E-commerce', 'ecommerce', '#96ceb4'),
            ('SaaS', 'saas', '#ffeaa7'),
            ('Bootstrap', 'bootstrap', '#7952b3'),
            ('OpenAI', 'openai', '#10a37f'),
            ('Anthropic', 'anthropic', '#d4a574'),
            ('Starter', 'starter', '#74b9ff'),
            ('Premium', 'premium', '#fdcb6e')
        ]
        
        for name, slug, color in tags:
            existing = TemplateTag.query.filter_by(slug=slug).first()
            if not existing:
                tag = TemplateTag()
                tag.name = name
                tag.slug = slug
                tag.color = color
                db.session.add(tag)
        
        db.session.commit()
        self.logger.info("Template tags initialized")

    def create_premium_templates(self):
        """Create initial premium templates based on high-value app suggestions"""
        
        # Get or create categories
        ai_ml_cat = TemplateCategory.query.filter_by(slug='ai-ml').first()
        business_cat = TemplateCategory.query.filter_by(slug='business-analytics').first()
        saas_cat = TemplateCategory.query.filter_by(slug='saas-webapp').first()
        ecommerce_cat = TemplateCategory.query.filter_by(slug='ecommerce').first()
        
        if not ai_ml_cat or not business_cat or not saas_cat or not ecommerce_cat:
            self.logger.error("Required categories not found. Please initialize categories first.")
            return
        
        templates = [
            {
                'title': 'AI-Powered Multi-App Dashboard Template',
                'slug': 'ai-multi-app-dashboard',
                'description': 'Complete dashboard template that aggregates data from multiple applications using AI-powered analytics and insights.',
                'long_description': '''A comprehensive dashboard solution that demonstrates how to build unified visibility across multiple applications. Features real-time data aggregation, AI-powered insights, custom KPI tracking, and automated alerting systems.

Perfect for developers who want to create centralized monitoring solutions or offer dashboard-as-a-service to clients.''',
                'category_id': business_cat.id,
                'price': 49.99,
                'is_premium': True,
                'is_featured': True,
                'tech_stack': ['Python', 'Flask', 'PostgreSQL', 'Chart.js', 'Bootstrap', 'OpenAI API'],
                'complexity_level': 'intermediate',
                'estimated_dev_time': '4-6 hours',
                'estimated_value': 2500.0,
                'potential_revenue': '$500-2000/month',
                'target_market': 'SaaS companies, agencies, enterprise clients',
                'demo_url': 'https://devopt-dashboard-demo.replit.app',
                'tags': ['ai', 'dashboard', 'python', 'flask', 'premium']
            },
            {
                'title': 'Intelligent API Gateway & Rate Limiter',
                'slug': 'intelligent-api-gateway',
                'description': 'Smart API gateway template with intelligent routing, cost optimization, and automatic rate limiting for AI services.',
                'long_description': '''Production-ready API gateway that optimizes AI service calls with intelligent routing, automatic rate limiting, response caching, and cost optimization features. Reduces AI API costs by 30-50% while improving reliability.

Includes real-time usage analytics, quota management, and support for multiple AI providers (OpenAI, Anthropic, HuggingFace).''',
                'category_id': ai_ml_cat.id,
                'price': 39.99,
                'is_premium': True,
                'is_featured': True,
                'tech_stack': ['Node.js', 'Redis', 'PostgreSQL', 'Express', 'Docker'],
                'complexity_level': 'advanced',
                'estimated_dev_time': '6-8 hours',
                'estimated_value': 1800.0,
                'potential_revenue': '$300-1500/month',
                'target_market': 'AI-powered applications, development agencies',
                'tags': ['api', 'ai', 'nodejs', 'redis', 'premium']
            },
            {
                'title': 'AI Assistant Marketplace Platform',
                'slug': 'ai-assistant-marketplace',
                'description': 'Complete marketplace platform for creating, sharing, and monetizing AI assistants with enterprise-grade features.',
                'long_description': '''Full-featured marketplace platform that allows users to create, share, and monetize AI assistants. Includes user management, payment processing, template system, and enterprise integration APIs.

Features assistant creation tools, community marketplace, performance analytics, and revenue sharing capabilities.''',
                'category_id': ecommerce_cat.id,
                'price': 75.00,
                'is_premium': True,
                'is_featured': True,
                'tech_stack': ['Python', 'Flask', 'PostgreSQL', 'Stripe', 'Redis', 'Celery'],
                'complexity_level': 'advanced',
                'estimated_dev_time': '12-16 hours',
                'estimated_value': 3500.0,
                'potential_revenue': '$1000-5000/month',
                'target_market': 'AI entrepreneurs, enterprise clients, development agencies',
                'tags': ['ai', 'marketplace', 'ecommerce', 'python', 'premium']
            },
            {
                'title': 'Document Intelligence & Data Extraction Service',
                'slug': 'document-intelligence-service',
                'description': 'B2B document processing service template with AI-powered extraction, contract analysis, and compliance reporting.',
                'long_description': '''Enterprise-grade document processing service that automates document analysis, data extraction, and compliance reporting. Perfect for creating B2B services targeting the $12B document processing market.

Includes bulk processing, contract analysis, risk assessment, custom data pipelines, and enterprise integrations.''',
                'category_id': business_cat.id,
                'price': 65.00,
                'is_premium': True,
                'is_featured': True,
                'tech_stack': ['Python', 'FastAPI', 'PostgreSQL', 'OpenAI', 'PDF processing', 'AWS S3'],
                'complexity_level': 'advanced',
                'estimated_dev_time': '10-14 hours',
                'estimated_value': 8500.0,
                'potential_revenue': '$2000-10000/month',
                'target_market': 'Enterprises, legal firms, financial services',
                'tags': ['ai', 'b2b', 'document', 'python', 'enterprise', 'premium']
            },
            {
                'title': 'SaaS Starter Kit with AI Integration',
                'slug': 'saas-starter-ai',
                'description': 'Complete SaaS application template with user management, subscriptions, AI features, and payment processing.',
                'long_description': '''Full-stack SaaS template with everything needed to launch an AI-powered service. Includes user authentication, subscription billing, AI integrations, admin dashboard, and deployment configurations.

Perfect for launching AI-powered SaaS products quickly with proven patterns and best practices.''',
                'category_id': saas_cat.id,
                'price': 55.00,
                'is_premium': True,
                'is_featured': True,
                'tech_stack': ['Python', 'Flask', 'PostgreSQL', 'Stripe', 'Bootstrap', 'OpenAI'],
                'complexity_level': 'intermediate',
                'estimated_dev_time': '8-12 hours',
                'estimated_value': 5000.0,
                'potential_revenue': '$1000-8000/month',
                'target_market': 'Startup founders, indie developers, agencies',
                'tags': ['saas', 'ai', 'python', 'flask', 'stripe', 'premium']
            }
        ]
        
        for template_data in templates:
            existing = AppTemplate.query.filter_by(slug=template_data['slug']).first()
            if not existing:
                template = AppTemplate()
                template.title = template_data['title']
                template.slug = template_data['slug']
                template.description = template_data['description']
                template.long_description = template_data['long_description']
                template.category_id = template_data['category_id']
                template.price = template_data['price']
                template.is_premium = template_data['is_premium']
                template.is_featured = template_data['is_featured']
                template.tech_stack = template_data['tech_stack']
                template.complexity_level = template_data['complexity_level']
                template.estimated_dev_time = template_data['estimated_dev_time']
                template.estimated_value = template_data['estimated_value']
                template.potential_revenue = template_data['potential_revenue']
                template.target_market = template_data['target_market']
                template.demo_url = template_data.get('demo_url')
                template.is_approved = True
                template.rating_avg = 4.8  # High initial rating for premium templates
                template.rating_count = 15
                db.session.add(template)
                db.session.flush()  # Get the ID
                
                # Add tags
                for tag_slug in template_data['tags']:
                    tag = TemplateTag.query.filter_by(slug=tag_slug).first()
                    if tag:
                        template.tags.append(tag)
        
        db.session.commit()
        self.logger.info("Premium templates created")

    def get_featured_templates(self, limit: int = 6) -> List[Dict]:
        """Get featured templates for homepage"""
        templates = AppTemplate.query.filter_by(
            is_featured=True, 
            is_active=True,
            is_approved=True
        ).order_by(AppTemplate.rating_avg.desc()).limit(limit).all()
        
        return [self._serialize_template(template) for template in templates]

    def get_templates_by_category(self, category_slug: str, page: int = 1, per_page: int = 12) -> Dict:
        """Get templates by category with pagination"""
        category = TemplateCategory.query.filter_by(slug=category_slug).first()
        if not category:
            return {'templates': [], 'total': 0, 'page': page, 'per_page': per_page}
        
        pagination = AppTemplate.query.filter_by(
            category_id=category.id,
            is_active=True,
            is_approved=True
        ).order_by(AppTemplate.is_featured.desc(), AppTemplate.rating_avg.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return {
            'templates': [self._serialize_template(t) for t in pagination.items],
            'total': pagination.total,
            'page': page,
            'per_page': per_page,
            'pages': pagination.pages
        }

    def search_templates(self, query: str, category: str = None, tags: List[str] = None, 
                        price_range: str = None, page: int = 1, per_page: int = 12) -> Dict:
        """Search templates with filters"""
        base_query = AppTemplate.query.filter(
            AppTemplate.is_active == True,
            AppTemplate.is_approved == True
        )
        
        if query:
            from sqlalchemy import or_
            base_query = base_query.filter(
                or_(
                    AppTemplate.title.contains(query),
                    AppTemplate.description.contains(query),
                    AppTemplate.long_description.contains(query)
                )
            )
        
        if category:
            cat = TemplateCategory.query.filter_by(slug=category).first()
            if cat and cat.id:
                base_query = base_query.filter(AppTemplate.category_id == cat.id)
        
        if price_range:
            if price_range == 'free':
                base_query = base_query.filter(AppTemplate.price == 0)
            elif price_range == 'paid':
                base_query = base_query.filter(AppTemplate.price > 0)
        
        pagination = base_query.order_by(
            AppTemplate.is_featured.desc(),
            AppTemplate.rating_avg.desc()
        ).paginate(page=page, per_page=per_page, error_out=False)
        
        return {
            'templates': [self._serialize_template(t) for t in pagination.items],
            'total': pagination.total,
            'page': page,
            'per_page': per_page,
            'pages': pagination.pages
        }

    def get_template_details(self, slug: str) -> Optional[Dict]:
        """Get detailed template information"""
        template = AppTemplate.query.filter_by(
            slug=slug, 
            is_active=True,
            is_approved=True
        ).first()
        
        if not template:
            return None
        
        # Increment view count
        template.view_count += 1
        db.session.commit()
        
        return self._serialize_template(template, include_reviews=True)

    def _serialize_template(self, template: AppTemplate, include_reviews: bool = False) -> Dict:
        """Convert template to dictionary"""
        data = {
            'id': template.id,
            'title': template.title,
            'slug': template.slug,
            'description': template.description,
            'long_description': template.long_description,
            'price': template.price,
            'is_premium': template.is_premium,
            'is_featured': template.is_featured,
            'tech_stack': template.tech_stack or [],
            'complexity_level': template.complexity_level,
            'estimated_dev_time': template.estimated_dev_time,
            'estimated_value': template.estimated_value,
            'potential_revenue': template.potential_revenue,
            'target_market': template.target_market,
            'demo_url': template.demo_url,
            'github_url': template.github_url,
            'documentation_url': template.documentation_url,
            'download_count': template.download_count,
            'rating_avg': round(template.rating_avg, 1),
            'rating_count': template.rating_count,
            'view_count': template.view_count,
            'created_at': template.created_at.isoformat() if template.created_at else None,
            'category': {
                'name': template.category.name if template.category else 'Unknown',
                'slug': template.category.slug if template.category else '',
                'color': template.category.color if template.category else '#007bff',
                'icon': template.category.icon if template.category else 'fas fa-folder'
            },
            'tags': [{'name': tag.name, 'slug': tag.slug, 'color': tag.color} for tag in template.tags] if hasattr(template, 'tags') else []
        }
        
        if include_reviews:
            data['reviews'] = [
                {
                    'reviewer_name': review.reviewer_name,
                    'rating': review.rating,
                    'title': review.title,
                    'review_text': review.review_text,
                    'implementation_time': review.implementation_time,
                    'would_recommend': review.would_recommend,
                    'created_at': review.created_at.isoformat() if review.created_at else None
                }
                for review in template.reviews if hasattr(template, 'reviews') and review.is_approved
            ]
        
        return data

    def initialize_template_system(self):
        """Initialize the complete template system"""
        try:
            self.logger.info("Initializing template system...")
            
            # Create tables if they don't exist
            db.create_all()
            
            # Initialize categories
            self.initialize_template_categories()
            
            # Initialize tags  
            self.initialize_template_tags()
            
            # Create premium templates
            self.create_premium_templates()
            
            self.logger.info("Template system initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing template system: {str(e)}")
            db.session.rollback()
            return False