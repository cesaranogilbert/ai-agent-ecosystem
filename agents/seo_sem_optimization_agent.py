"""
SEO/SEM Optimization Agent - AI-Powered Search Marketing & Organic Optimization
Keyword Research, Content Optimization, Technical SEO & Performance Marketing Integration
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
import pandas as pd
import numpy as np
from openai import OpenAI
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "seo-sem-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///seo_sem.db")

db.init_app(app)

# SEO/SEM Enums
class SearchEngineType(Enum):
    GOOGLE = "google"
    BING = "bing"
    YAHOO = "yahoo"
    DUCKDUCKGO = "duckduckgo"

class KeywordIntent(Enum):
    INFORMATIONAL = "informational"
    NAVIGATIONAL = "navigational"
    TRANSACTIONAL = "transactional"
    COMMERCIAL = "commercial"

class ContentType(Enum):
    BLOG_POST = "blog_post"
    LANDING_PAGE = "landing_page"
    PRODUCT_PAGE = "product_page"
    CATEGORY_PAGE = "category_page"
    SERVICE_PAGE = "service_page"

# Data Models
class SEOProject(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.String(100), unique=True, nullable=False)
    project_name = db.Column(db.String(200), nullable=False)
    
    # Website Details
    domain = db.Column(db.String(200), nullable=False)
    industry = db.Column(db.String(100))
    target_markets = db.Column(db.JSON)
    business_model = db.Column(db.String(50))
    
    # SEO Configuration
    primary_keywords = db.Column(db.JSON)
    target_pages = db.Column(db.JSON)
    competitor_domains = db.Column(db.JSON)
    
    # Performance Baselines
    baseline_organic_traffic = db.Column(db.Integer, default=0)
    baseline_keyword_rankings = db.Column(db.JSON)
    baseline_domain_authority = db.Column(db.Float, default=0.0)
    
    # Goals and Targets
    traffic_goals = db.Column(db.JSON)
    ranking_goals = db.Column(db.JSON)
    conversion_goals = db.Column(db.JSON)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class KeywordResearch(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    keyword_id = db.Column(db.String(100), unique=True, nullable=False)
    project_id = db.Column(db.String(100), db.ForeignKey('seo_project.project_id'), nullable=False)
    
    # Keyword Details
    keyword = db.Column(db.String(300), nullable=False)
    keyword_intent = db.Column(db.Enum(KeywordIntent), nullable=False)
    keyword_variations = db.Column(db.JSON)
    
    # Search Volume Data
    monthly_search_volume = db.Column(db.Integer, default=0)
    search_volume_trend = db.Column(db.JSON)
    seasonal_patterns = db.Column(db.JSON)
    
    # Competition Analysis
    keyword_difficulty = db.Column(db.Float, default=0.0)
    competition_level = db.Column(db.String(20))
    top_competitors = db.Column(db.JSON)
    
    # Opportunity Analysis
    opportunity_score = db.Column(db.Float, default=0.0)
    ranking_difficulty = db.Column(db.Float, default=0.0)
    traffic_potential = db.Column(db.Integer, default=0)
    conversion_potential = db.Column(db.Float, default=0.0)
    
    # Current Performance
    current_ranking = db.Column(db.Integer, default=0)
    current_traffic = db.Column(db.Integer, default=0)
    click_through_rate = db.Column(db.Float, default=0.0)
    
    # AI Insights
    content_recommendations = db.Column(db.JSON)
    optimization_priority = db.Column(db.String(20), default='medium')
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ContentOptimization(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    optimization_id = db.Column(db.String(100), unique=True, nullable=False)
    project_id = db.Column(db.String(100), db.ForeignKey('seo_project.project_id'), nullable=False)
    
    # Page Details
    page_url = db.Column(db.String(500), nullable=False)
    page_title = db.Column(db.String(200))
    content_type = db.Column(db.Enum(ContentType), nullable=False)
    target_keywords = db.Column(db.JSON)
    
    # Current SEO Status
    title_tag = db.Column(db.String(200))
    meta_description = db.Column(db.Text)
    h1_tag = db.Column(db.String(200))
    content_length = db.Column(db.Integer, default=0)
    
    # SEO Scores
    on_page_score = db.Column(db.Float, default=0.0)
    content_quality_score = db.Column(db.Float, default=0.0)
    technical_score = db.Column(db.Float, default=0.0)
    user_experience_score = db.Column(db.Float, default=0.0)
    
    # Optimization Recommendations
    title_recommendations = db.Column(db.JSON)
    content_recommendations = db.Column(db.JSON)
    technical_recommendations = db.Column(db.JSON)
    internal_linking_recommendations = db.Column(db.JSON)
    
    # Performance Tracking
    ranking_improvements = db.Column(db.JSON)
    traffic_improvements = db.Column(db.JSON)
    conversion_improvements = db.Column(db.JSON)
    
    # AI Analysis
    ai_content_suggestions = db.Column(db.JSON)
    semantic_keyword_opportunities = db.Column(db.JSON)
    competitor_gap_analysis = db.Column(db.JSON)
    
    status = db.Column(db.String(50), default='pending')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_optimized = db.Column(db.DateTime)

class TechnicalSEO(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    audit_id = db.Column(db.String(100), unique=True, nullable=False)
    project_id = db.Column(db.String(100), db.ForeignKey('seo_project.project_id'), nullable=False)
    
    # Technical Audit Results
    audit_date = db.Column(db.Date, nullable=False)
    
    # Site Performance
    page_speed_score = db.Column(db.Float, default=0.0)
    core_web_vitals = db.Column(db.JSON)
    mobile_usability_score = db.Column(db.Float, default=0.0)
    
    # Crawlability
    crawl_errors = db.Column(db.JSON)
    indexability_issues = db.Column(db.JSON)
    sitemap_status = db.Column(db.JSON)
    robots_txt_status = db.Column(db.JSON)
    
    # Site Architecture
    internal_linking_analysis = db.Column(db.JSON)
    url_structure_analysis = db.Column(db.JSON)
    duplicate_content_issues = db.Column(db.JSON)
    
    # Schema and Markup
    structured_data_analysis = db.Column(db.JSON)
    schema_opportunities = db.Column(db.JSON)
    markup_errors = db.Column(db.JSON)
    
    # Security and Accessibility
    security_issues = db.Column(db.JSON)
    accessibility_score = db.Column(db.Float, default=0.0)
    https_implementation = db.Column(db.JSON)
    
    # Overall Scores
    technical_health_score = db.Column(db.Float, default=0.0)
    priority_issues = db.Column(db.JSON)
    recommendations = db.Column(db.JSON)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# SEO/SEM Optimization Engine
class SEOSEMOptimizationEngine:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    def generate_comprehensive_seo_strategy(self, project_id: str) -> Dict[str, Any]:
        """Generate comprehensive SEO/SEM optimization strategy"""
        
        project = SEOProject.query.filter_by(project_id=project_id).first()
        if not project:
            return {'error': 'Project not found'}
        
        # Keyword research and analysis
        keyword_strategy = self._develop_keyword_strategy(project)
        
        # Content optimization strategy
        content_strategy = self._develop_content_optimization_strategy(project)
        
        # Technical SEO audit and recommendations
        technical_strategy = self._develop_technical_seo_strategy(project)
        
        # Competitive analysis
        competitive_analysis = self._perform_competitive_analysis(project)
        
        # Link building strategy
        link_building_strategy = self._develop_link_building_strategy(project)
        
        # Performance tracking framework
        performance_framework = self._establish_performance_tracking(project)
        
        return {
            'project_id': project_id,
            'strategy_date': datetime.utcnow().isoformat(),
            'keyword_strategy': keyword_strategy,
            'content_optimization_strategy': content_strategy,
            'technical_seo_strategy': technical_strategy,
            'competitive_analysis': competitive_analysis,
            'link_building_strategy': link_building_strategy,
            'performance_tracking_framework': performance_framework,
            'implementation_roadmap': self._create_implementation_roadmap(keyword_strategy, content_strategy, technical_strategy),
            'projected_results': self._project_seo_results(project, keyword_strategy)
        }
    
    def _develop_keyword_strategy(self, project: SEOProject) -> Dict[str, Any]:
        """Develop comprehensive keyword research and targeting strategy"""
        
        # Analyze current keyword performance
        current_keywords = KeywordResearch.query.filter_by(project_id=project.project_id).all()
        
        # Keyword gap analysis
        keyword_gaps = self._identify_keyword_gaps(project, current_keywords)
        
        # Intent-based keyword grouping
        intent_groups = self._group_keywords_by_intent(current_keywords, keyword_gaps)
        
        # Opportunity prioritization
        keyword_opportunities = self._prioritize_keyword_opportunities(intent_groups)
        
        # Content mapping strategy
        content_mapping = self._map_keywords_to_content(keyword_opportunities)
        
        return {
            'keyword_analysis': {
                'total_target_keywords': len(current_keywords) + len(keyword_gaps),
                'high_opportunity_keywords': len([k for k in current_keywords if k.opportunity_score > 70]),
                'content_gap_keywords': len(keyword_gaps),
                'competitive_keywords': self._identify_competitive_keywords(current_keywords)
            },
            'intent_based_strategy': intent_groups,
            'keyword_opportunities': keyword_opportunities,
            'content_mapping_strategy': content_mapping,
            'keyword_expansion_plan': self._plan_keyword_expansion(project, intent_groups),
            'local_seo_keywords': self._identify_local_seo_opportunities(project) if self._is_local_business(project) else None
        }
    
    def _identify_keyword_gaps(self, project: SEOProject, current_keywords: List[KeywordResearch]) -> List[Dict[str, Any]]:
        """Identify keyword gaps and new opportunities"""
        
        # AI-powered keyword discovery
        keyword_expansion_prompt = f"""
        Generate 20 high-value keyword opportunities for a {project.industry} business targeting {project.target_markets}.
        Focus on keywords with commercial intent and reasonable competition levels.
        Current keywords: {[k.keyword for k in current_keywords[:10]]}
        
        For each keyword, provide:
        - Search intent (informational, commercial, transactional)
        - Estimated difficulty (1-100)
        - Content type recommendation
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert SEO strategist specializing in keyword research and competitive analysis."},
                {"role": "user", "content": keyword_expansion_prompt}
            ],
            max_tokens=1500
        )
        
        # Parse AI response and structure keyword gaps
        ai_keywords = self._parse_keyword_suggestions(response.choices[0].message.content)
        
        # Add competitive intelligence
        keyword_gaps = []
        for keyword_data in ai_keywords:
            gap = {
                'keyword': keyword_data['keyword'],
                'search_intent': keyword_data['intent'],
                'estimated_difficulty': keyword_data['difficulty'],
                'content_type': keyword_data['content_type'],
                'opportunity_score': self._calculate_keyword_opportunity_score(keyword_data),
                'source': 'ai_analysis'
            }
            keyword_gaps.append(gap)
        
        return keyword_gaps
    
    def _parse_keyword_suggestions(self, ai_response: str) -> List[Dict[str, Any]]:
        """Parse AI keyword suggestions into structured data"""
        
        # Simplified parsing - in production would use more sophisticated NLP
        keywords = []
        lines = ai_response.split('\n')
        
        for line in lines:
            if '-' in line and any(intent in line.lower() for intent in ['informational', 'commercial', 'transactional']):
                # Extract keyword and metadata
                keyword_data = {
                    'keyword': line.split('-')[0].strip(),
                    'intent': 'commercial',  # Default
                    'difficulty': 50,  # Default
                    'content_type': 'blog_post'  # Default
                }
                keywords.append(keyword_data)
        
        return keywords[:20]  # Limit to 20 keywords
    
    def _calculate_keyword_opportunity_score(self, keyword_data: Dict) -> float:
        """Calculate opportunity score for keyword"""
        
        # Simplified scoring algorithm
        base_score = 50
        
        # Adjust for difficulty
        difficulty = keyword_data.get('difficulty', 50)
        difficulty_score = max(0, 100 - difficulty) * 0.4
        
        # Adjust for intent
        intent_scores = {
            'transactional': 30,
            'commercial': 25,
            'informational': 15,
            'navigational': 10
        }
        intent_score = intent_scores.get(keyword_data.get('intent', 'informational'), 15)
        
        return min(100, base_score + difficulty_score + intent_score)
    
    def _group_keywords_by_intent(self, current_keywords: List[KeywordResearch], keyword_gaps: List[Dict]) -> Dict[str, Any]:
        """Group keywords by search intent for strategic targeting"""
        
        intent_groups = {
            'informational': {
                'keywords': [],
                'content_strategy': 'educational_blog_posts_and_guides',
                'funnel_stage': 'awareness',
                'content_types': ['blog_posts', 'guides', 'tutorials'],
                'success_metrics': ['organic_traffic', 'time_on_page', 'pages_per_session']
            },
            'commercial': {
                'keywords': [],
                'content_strategy': 'comparison_and_evaluation_content',
                'funnel_stage': 'consideration',
                'content_types': ['comparison_pages', 'reviews', 'case_studies'],
                'success_metrics': ['qualified_traffic', 'email_signups', 'content_downloads']
            },
            'transactional': {
                'keywords': [],
                'content_strategy': 'product_and_service_pages',
                'funnel_stage': 'decision',
                'content_types': ['product_pages', 'service_pages', 'landing_pages'],
                'success_metrics': ['conversions', 'revenue', 'qualified_leads']
            },
            'navigational': {
                'keywords': [],
                'content_strategy': 'brand_and_navigation_optimization',
                'funnel_stage': 'retention',
                'content_types': ['brand_pages', 'about_pages', 'contact_pages'],
                'success_metrics': ['brand_traffic', 'direct_traffic', 'brand_awareness']
            }
        }
        
        # Categorize current keywords
        for keyword in current_keywords:
            intent = keyword.keyword_intent.value
            if intent in intent_groups:
                intent_groups[intent]['keywords'].append({
                    'keyword': keyword.keyword,
                    'search_volume': keyword.monthly_search_volume,
                    'difficulty': keyword.keyword_difficulty,
                    'current_ranking': keyword.current_ranking,
                    'opportunity_score': keyword.opportunity_score
                })
        
        # Add keyword gaps
        for gap in keyword_gaps:
            intent = gap.get('search_intent', 'informational')
            if intent in intent_groups:
                intent_groups[intent]['keywords'].append(gap)
        
        # Calculate group metrics
        for intent, group in intent_groups.items():
            keywords = group['keywords']
            group['keyword_count'] = len(keywords)
            group['avg_opportunity_score'] = np.mean([k.get('opportunity_score', 50) for k in keywords]) if keywords else 0
            group['total_search_volume'] = sum([k.get('search_volume', 0) for k in keywords])
            group['priority_level'] = self._calculate_intent_group_priority(group)
        
        return intent_groups
    
    def _calculate_intent_group_priority(self, group: Dict) -> str:
        """Calculate priority level for intent group"""
        
        avg_opportunity = group['avg_opportunity_score']
        total_volume = group['total_search_volume']
        keyword_count = group['keyword_count']
        
        priority_score = (avg_opportunity * 0.4 + 
                         min(total_volume / 10000, 100) * 0.3 + 
                         min(keyword_count / 20, 100) * 0.3)
        
        if priority_score >= 70:
            return 'high'
        elif priority_score >= 50:
            return 'medium'
        else:
            return 'low'
    
    def _prioritize_keyword_opportunities(self, intent_groups: Dict) -> List[Dict[str, Any]]:
        """Prioritize keyword opportunities across all intent groups"""
        
        all_opportunities = []
        
        for intent, group in intent_groups.items():
            for keyword in group['keywords']:
                opportunity = {
                    'keyword': keyword['keyword'],
                    'intent': intent,
                    'opportunity_score': keyword.get('opportunity_score', 50),
                    'search_volume': keyword.get('search_volume', 0),
                    'difficulty': keyword.get('difficulty', 50),
                    'current_ranking': keyword.get('current_ranking', 0),
                    'implementation_priority': self._calculate_implementation_priority(keyword, intent),
                    'expected_timeline': self._estimate_ranking_timeline(keyword),
                    'resource_requirements': self._estimate_resource_requirements(keyword, intent)
                }
                all_opportunities.append(opportunity)
        
        # Sort by implementation priority and opportunity score
        prioritized = sorted(all_opportunities, 
                           key=lambda x: (x['implementation_priority'], x['opportunity_score']), 
                           reverse=True)
        
        return prioritized[:50]  # Top 50 opportunities
    
    def _calculate_implementation_priority(self, keyword: Dict, intent: str) -> float:
        """Calculate implementation priority for keyword"""
        
        opportunity_score = keyword.get('opportunity_score', 50)
        difficulty = keyword.get('difficulty', 50)
        current_ranking = keyword.get('current_ranking', 0)
        
        # Higher priority for high opportunity, low difficulty, and current rankings
        priority_score = (opportunity_score * 0.4 + 
                         (100 - difficulty) * 0.3 + 
                         (100 - current_ranking) * 0.3 if current_ranking > 0 else opportunity_score * 0.7)
        
        # Intent-based adjustments
        intent_multipliers = {
            'transactional': 1.2,
            'commercial': 1.1,
            'informational': 1.0,
            'navigational': 0.9
        }
        
        return priority_score * intent_multipliers.get(intent, 1.0)
    
    def _estimate_ranking_timeline(self, keyword: Dict) -> str:
        """Estimate timeline to achieve target ranking"""
        
        difficulty = keyword.get('difficulty', 50)
        current_ranking = keyword.get('current_ranking', 0)
        
        if difficulty < 30:
            return '1-3 months'
        elif difficulty < 50:
            return '3-6 months'
        elif difficulty < 70:
            return '6-12 months'
        else:
            return '12+ months'
    
    def _estimate_resource_requirements(self, keyword: Dict, intent: str) -> Dict[str, str]:
        """Estimate resource requirements for keyword optimization"""
        
        difficulty = keyword.get('difficulty', 50)
        
        resource_levels = {
            'content_creation': 'medium',
            'technical_optimization': 'low',
            'link_building': 'medium',
            'ongoing_optimization': 'low'
        }
        
        if difficulty > 70:
            resource_levels['content_creation'] = 'high'
            resource_levels['link_building'] = 'high'
            resource_levels['ongoing_optimization'] = 'medium'
        elif difficulty < 30:
            resource_levels['content_creation'] = 'low'
            resource_levels['link_building'] = 'low'
        
        return resource_levels
    
    def _map_keywords_to_content(self, opportunities: List[Dict]) -> Dict[str, Any]:
        """Map keywords to content creation and optimization strategy"""
        
        content_mapping = {
            'new_content_needed': [],
            'existing_content_optimization': [],
            'content_consolidation': [],
            'content_gap_analysis': []
        }
        
        for opportunity in opportunities:
            keyword = opportunity['keyword']
            intent = opportunity['intent']
            
            # Determine content action needed
            if opportunity['current_ranking'] == 0:
                # New content needed
                content_mapping['new_content_needed'].append({
                    'keyword': keyword,
                    'intent': intent,
                    'content_type': self._recommend_content_type(intent),
                    'target_word_count': self._recommend_word_count(intent),
                    'priority': opportunity['implementation_priority']
                })
            elif opportunity['current_ranking'] > 10:
                # Existing content optimization
                content_mapping['existing_content_optimization'].append({
                    'keyword': keyword,
                    'current_ranking': opportunity['current_ranking'],
                    'optimization_type': 'content_enhancement',
                    'priority': opportunity['implementation_priority']
                })
        
        # Content creation schedule
        content_schedule = self._create_content_creation_schedule(content_mapping)
        
        return {
            'content_actions': content_mapping,
            'content_creation_schedule': content_schedule,
            'content_optimization_framework': self._design_content_optimization_framework(),
            'content_performance_tracking': self._design_content_performance_tracking()
        }
    
    def _recommend_content_type(self, intent: str) -> str:
        """Recommend content type based on search intent"""
        
        content_type_map = {
            'informational': 'comprehensive_guide',
            'commercial': 'comparison_page',
            'transactional': 'product_page',
            'navigational': 'optimized_landing_page'
        }
        
        return content_type_map.get(intent, 'blog_post')
    
    def _recommend_word_count(self, intent: str) -> int:
        """Recommend word count based on search intent"""
        
        word_count_map = {
            'informational': 2500,
            'commercial': 1800,
            'transactional': 1200,
            'navigational': 800
        }
        
        return word_count_map.get(intent, 1500)
    
    def _create_content_creation_schedule(self, content_mapping: Dict) -> Dict[str, Any]:
        """Create content creation schedule based on priority"""
        
        new_content = content_mapping['new_content_needed']
        
        # Sort by priority
        prioritized_content = sorted(new_content, key=lambda x: x['priority'], reverse=True)
        
        # Create monthly schedule
        monthly_schedule = {}
        content_per_month = 8  # 2 pieces per week
        
        for i, content in enumerate(prioritized_content):
            month = (i // content_per_month) + 1
            month_key = f'month_{month}'
            
            if month_key not in monthly_schedule:
                monthly_schedule[month_key] = []
            
            monthly_schedule[month_key].append({
                'keyword': content['keyword'],
                'content_type': content['content_type'],
                'target_word_count': content['target_word_count'],
                'priority': content['priority']
            })
        
        return {
            'monthly_content_schedule': monthly_schedule,
            'total_content_pieces': len(new_content),
            'estimated_timeline': f'{len(monthly_schedule)} months',
            'resource_allocation': 'content_team_optimization_required'
        }
    
    def _design_content_optimization_framework(self) -> Dict[str, Any]:
        """Design framework for ongoing content optimization"""
        
        return {
            'optimization_checklist': [
                'keyword_density_optimization',
                'semantic_keyword_integration',
                'title_tag_optimization',
                'meta_description_enhancement',
                'header_structure_improvement',
                'internal_linking_optimization',
                'content_freshness_updates',
                'user_experience_improvements'
            ],
            'optimization_frequency': {
                'high_priority_pages': 'monthly',
                'medium_priority_pages': 'quarterly',
                'low_priority_pages': 'semi_annually'
            },
            'performance_triggers': {
                'ranking_decline': 'immediate_optimization',
                'traffic_decline': 'content_refresh',
                'low_engagement': 'user_experience_focus'
            },
            'ai_optimization_tools': [
                'content_gap_analysis',
                'semantic_keyword_discovery',
                'competitor_content_analysis',
                'user_intent_optimization'
            ]
        }
    
    def _design_content_performance_tracking(self) -> Dict[str, Any]:
        """Design content performance tracking framework"""
        
        return {
            'content_kpis': [
                'organic_traffic_growth',
                'keyword_ranking_improvements',
                'time_on_page',
                'bounce_rate',
                'conversion_rate',
                'social_shares',
                'backlink_acquisition'
            ],
            'tracking_frequency': 'weekly',
            'reporting_dashboards': {
                'content_performance_overview': 'executive_summary',
                'keyword_ranking_tracker': 'seo_team',
                'content_engagement_metrics': 'content_team',
                'conversion_attribution': 'marketing_team'
            },
            'optimization_alerts': {
                'ranking_drops': 'immediate_notification',
                'traffic_declines': 'daily_monitoring',
                'conversion_issues': 'real_time_alerts'
            }
        }
    
    def _plan_keyword_expansion(self, project: SEOProject, intent_groups: Dict) -> Dict[str, Any]:
        """Plan systematic keyword expansion strategy"""
        
        return {
            'expansion_methodology': {
                'competitor_keyword_gap_analysis': 'monthly',
                'search_query_mining': 'weekly',
                'ai_powered_keyword_discovery': 'bi_weekly',
                'seasonal_keyword_research': 'quarterly'
            },
            'expansion_targets': {
                'quarterly_new_keywords': 50,
                'long_tail_keyword_focus': '70%',
                'voice_search_optimization': '20%',
                'local_search_integration': '10%' if self._is_local_business(project) else '0%'
            },
            'automation_framework': {
                'keyword_discovery': 'ai_powered_suggestions',
                'opportunity_scoring': 'automated_algorithm',
                'content_mapping': 'intelligent_assignment',
                'performance_monitoring': 'real_time_tracking'
            }
        }
    
    def _identify_local_seo_opportunities(self, project: SEOProject) -> Dict[str, Any]:
        """Identify local SEO opportunities for local businesses"""
        
        return {
            'local_keyword_strategy': {
                'city_based_keywords': f"{project.industry} + {project.target_markets}",
                'near_me_keywords': f"{project.industry} near me variations",
                'service_area_keywords': 'neighborhood and district targeting',
                'local_intent_modifiers': ['best', 'top', 'reviews', 'directions']
            },
            'local_content_strategy': {
                'location_pages': 'dedicated_pages_for_each_service_area',
                'local_blog_content': 'community_focused_content',
                'local_landing_pages': 'location_specific_service_pages',
                'google_my_business_optimization': 'complete_profile_optimization'
            },
            'local_seo_tactics': {
                'citation_building': 'consistent_nap_across_directories',
                'review_management': 'proactive_review_acquisition',
                'local_link_building': 'community_and_business_partnerships',
                'local_schema_markup': 'structured_data_for_local_business'
            }
        }
    
    def _is_local_business(self, project: SEOProject) -> bool:
        """Determine if project is for a local business"""
        
        local_indicators = ['restaurant', 'dental', 'medical', 'legal', 'retail', 'service']
        return any(indicator in project.industry.lower() for indicator in local_indicators) if project.industry else False
    
    def _identify_competitive_keywords(self, keywords: List[KeywordResearch]) -> List[Dict[str, Any]]:
        """Identify highly competitive keywords requiring special attention"""
        
        competitive_keywords = []
        
        for keyword in keywords:
            if keyword.keyword_difficulty > 70:
                competitive_keywords.append({
                    'keyword': keyword.keyword,
                    'difficulty': keyword.keyword_difficulty,
                    'search_volume': keyword.monthly_search_volume,
                    'current_ranking': keyword.current_ranking,
                    'strategy': 'long_term_authority_building'
                })
        
        return competitive_keywords[:10]  # Top 10 most competitive

# Initialize SEO/SEM engine
seo_engine = SEOSEMOptimizationEngine()

# Routes
@app.route('/seo-sem-optimization')
def seo_dashboard():
    """SEO/SEM Optimization dashboard"""
    
    recent_projects = SEOProject.query.order_by(SEOProject.created_at.desc()).limit(10).all()
    
    return render_template('seo/dashboard.html',
                         projects=recent_projects)

@app.route('/seo-sem-optimization/api/comprehensive-strategy', methods=['POST'])
def create_seo_strategy():
    """API endpoint for comprehensive SEO strategy"""
    
    data = request.get_json()
    project_id = data.get('project_id')
    
    if not project_id:
        return jsonify({'error': 'Project ID required'}), 400
    
    strategy = seo_engine.generate_comprehensive_seo_strategy(project_id)
    return jsonify(strategy)

# Initialize database
with app.app_context():
    db.create_all()
    
    # Create sample data
    if SEOProject.query.count() == 0:
        sample_project = SEOProject(
            project_id='SEO_DEMO_001',
            project_name='Demo SEO Project',
            domain='demo-website.com',
            industry='technology',
            target_markets=['United States', 'Canada']
        )
        
        db.session.add(sample_project)
        db.session.commit()
        logger.info("Sample SEO/SEM data created")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5036, debug=True)