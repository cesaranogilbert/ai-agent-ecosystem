"""
Omnichannel Content Creator Agent - AI-Powered Multi-Format Content Generation
Content Calendar Planning, Cross-Platform Optimization & Trend Integration
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
app.secret_key = os.environ.get("SESSION_SECRET", "content-creator-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///content_creator.db")

db.init_app(app)

# Content Creation Enums
class ContentFormat(Enum):
    BLOG_POST = "blog_post"
    SOCIAL_POST = "social_post"
    VIDEO_SCRIPT = "video_script"
    EMAIL_NEWSLETTER = "email_newsletter"
    INFOGRAPHIC = "infographic"
    PODCAST_SCRIPT = "podcast_script"
    CASE_STUDY = "case_study"
    EBOOK = "ebook"

class ContentPurpose(Enum):
    AWARENESS = "awareness"
    ENGAGEMENT = "engagement"
    EDUCATION = "education"
    CONVERSION = "conversion"
    RETENTION = "retention"
    ADVOCACY = "advocacy"

class Platform(Enum):
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"
    LINKEDIN = "linkedin"
    TWITTER = "twitter"
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    WEBSITE_BLOG = "website_blog"
    EMAIL = "email"

# Data Models
class ContentBrand(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    brand_id = db.Column(db.String(100), unique=True, nullable=False)
    brand_name = db.Column(db.String(200), nullable=False)
    
    # Brand Guidelines
    brand_voice = db.Column(db.JSON)
    tone_attributes = db.Column(db.JSON)
    messaging_framework = db.Column(db.JSON)
    visual_guidelines = db.Column(db.JSON)
    
    # Content Strategy
    content_pillars = db.Column(db.JSON)
    target_keywords = db.Column(db.JSON)
    content_themes = db.Column(db.JSON)
    publishing_frequency = db.Column(db.JSON)
    
    # Performance Goals
    engagement_targets = db.Column(db.JSON)
    conversion_goals = db.Column(db.JSON)
    brand_awareness_kpis = db.Column(db.JSON)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ContentPiece(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.String(100), unique=True, nullable=False)
    brand_id = db.Column(db.String(100), db.ForeignKey('content_brand.brand_id'), nullable=False)
    
    # Content Details
    title = db.Column(db.String(300), nullable=False)
    content_format = db.Column(db.Enum(ContentFormat), nullable=False)
    content_purpose = db.Column(db.Enum(ContentPurpose), nullable=False)
    target_platform = db.Column(db.Enum(Platform), nullable=False)
    
    # Content Body
    primary_content = db.Column(db.Text)
    secondary_content = db.Column(db.Text)  # Captions, descriptions, etc.
    call_to_action = db.Column(db.Text)
    hashtags = db.Column(db.JSON)
    
    # SEO and Optimization
    target_keywords = db.Column(db.JSON)
    meta_description = db.Column(db.Text)
    seo_title = db.Column(db.String(200))
    
    # Scheduling and Distribution
    scheduled_date = db.Column(db.DateTime)
    optimal_posting_time = db.Column(db.Time)
    cross_platform_versions = db.Column(db.JSON)
    
    # Performance Tracking
    engagement_prediction = db.Column(db.Float, default=0.0)
    actual_engagement = db.Column(db.Float, default=0.0)
    reach_estimate = db.Column(db.Integer, default=0)
    conversion_potential = db.Column(db.Float, default=0.0)
    
    # Quality Metrics
    brand_consistency_score = db.Column(db.Float, default=85.0)
    content_quality_score = db.Column(db.Float, default=80.0)
    seo_optimization_score = db.Column(db.Float, default=75.0)
    
    # AI Enhancement
    content_optimization_suggestions = db.Column(db.JSON)
    trend_integration_score = db.Column(db.Float, default=0.0)
    viral_potential = db.Column(db.Float, default=25.0)
    
    status = db.Column(db.String(50), default='draft')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ContentCalendar(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    calendar_id = db.Column(db.String(100), unique=True, nullable=False)
    brand_id = db.Column(db.String(100), db.ForeignKey('content_brand.brand_id'), nullable=False)
    
    # Calendar Details
    calendar_name = db.Column(db.String(200), nullable=False)
    calendar_period = db.Column(db.String(50))  # monthly, quarterly, yearly
    start_date = db.Column(db.Date, nullable=False)
    end_date = db.Column(db.Date, nullable=False)
    
    # Content Planning
    content_schedule = db.Column(db.JSON)  # Detailed content schedule
    platform_distribution = db.Column(db.JSON)  # Content per platform
    theme_calendar = db.Column(db.JSON)  # Monthly/weekly themes
    
    # Campaign Integration
    campaign_alignments = db.Column(db.JSON)
    seasonal_content = db.Column(db.JSON)
    promotional_content = db.Column(db.JSON)
    
    # Performance Planning
    engagement_forecasts = db.Column(db.JSON)
    conversion_projections = db.Column(db.JSON)
    reach_targets = db.Column(db.JSON)
    
    # AI Optimization
    optimal_posting_schedule = db.Column(db.JSON)
    content_gap_analysis = db.Column(db.JSON)
    trend_integration_calendar = db.Column(db.JSON)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class TrendAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    trend_id = db.Column(db.String(100), unique=True, nullable=False)
    
    # Trend Information
    trend_name = db.Column(db.String(200), nullable=False)
    trend_category = db.Column(db.String(100))
    trend_platform = db.Column(db.Enum(Platform))
    
    # Trend Metrics
    trend_velocity = db.Column(db.Float, default=0.0)  # Rate of growth
    engagement_rate = db.Column(db.Float, default=0.0)
    participation_volume = db.Column(db.Integer, default=0)
    
    # Trend Analysis
    demographic_appeal = db.Column(db.JSON)
    geographic_distribution = db.Column(db.JSON)
    content_themes = db.Column(db.JSON)
    
    # Opportunity Assessment
    brand_fit_score = db.Column(db.Float, default=50.0)
    difficulty_level = db.Column(db.Float, default=50.0)
    saturation_level = db.Column(db.Float, default=30.0)
    roi_potential = db.Column(db.Float, default=25.0)
    
    # Timeline
    peak_prediction = db.Column(db.Date)
    decline_prediction = db.Column(db.Date)
    optimal_entry_window = db.Column(db.JSON)
    
    # Content Recommendations
    content_format_recommendations = db.Column(db.JSON)
    messaging_suggestions = db.Column(db.JSON)
    hashtag_recommendations = db.Column(db.JSON)
    
    discovered_at = db.Column(db.DateTime, default=datetime.utcnow)

# Omnichannel Content Creator Engine
class OmnichannelContentEngine:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    def generate_comprehensive_content_strategy(self, brand_id: str, period_days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive omnichannel content strategy"""
        
        brand = ContentBrand.query.filter_by(brand_id=brand_id).first()
        if not brand:
            return {'error': 'Brand not found'}
        
        # Analyze current content performance
        content_analysis = self._analyze_current_content_performance(brand_id)
        
        # Identify trending opportunities
        trend_opportunities = self._identify_trend_opportunities(brand)
        
        # Generate content calendar
        content_calendar = self._generate_strategic_content_calendar(brand, period_days, trend_opportunities)
        
        # Create platform-specific content
        platform_content = self._generate_platform_specific_content(brand, content_calendar)
        
        # Optimize for engagement
        engagement_optimization = self._optimize_for_engagement(brand, platform_content)
        
        return {
            'brand_id': brand_id,
            'strategy_period': f'{period_days} days',
            'content_analysis': content_analysis,
            'trend_opportunities': trend_opportunities,
            'content_calendar': content_calendar,
            'platform_content': platform_content,
            'engagement_optimization': engagement_optimization,
            'performance_projections': self._project_content_performance(platform_content),
            'generated_at': datetime.utcnow().isoformat()
        }
    
    def _analyze_current_content_performance(self, brand_id: str) -> Dict[str, Any]:
        """Analyze current content performance across platforms"""
        
        # Get recent content
        recent_content = ContentPiece.query.filter_by(brand_id=brand_id)\
                                          .filter(ContentPiece.created_at >= datetime.utcnow() - timedelta(days=90))\
                                          .all()
        
        if not recent_content:
            return {'status': 'insufficient_data'}
        
        # Analyze performance by platform
        platform_performance = {}
        for platform in Platform:
            platform_content = [c for c in recent_content if c.target_platform == platform]
            
            if platform_content:
                avg_engagement = np.mean([c.actual_engagement for c in platform_content if c.actual_engagement > 0])
                avg_quality = np.mean([c.content_quality_score for c in platform_content])
                
                platform_performance[platform.value] = {
                    'content_count': len(platform_content),
                    'average_engagement': avg_engagement,
                    'average_quality': avg_quality,
                    'top_performing_format': self._identify_top_format(platform_content),
                    'optimization_potential': max(0, 90 - avg_quality)
                }
        
        # Content format analysis
        format_performance = self._analyze_format_performance(recent_content)
        
        # Brand consistency analysis
        consistency_analysis = self._analyze_brand_consistency(recent_content)
        
        return {
            'analysis_period': '90 days',
            'total_content_pieces': len(recent_content),
            'platform_performance': platform_performance,
            'format_performance': format_performance,
            'brand_consistency': consistency_analysis,
            'key_insights': self._generate_content_insights(platform_performance, format_performance)
        }
    
    def _identify_top_format(self, content_list: List[ContentPiece]) -> str:
        """Identify top performing content format"""
        
        format_performance = {}
        for content in content_list:
            format_key = content.content_format.value
            if format_key not in format_performance:
                format_performance[format_key] = []
            
            if content.actual_engagement > 0:
                format_performance[format_key].append(content.actual_engagement)
        
        # Find format with highest average engagement
        best_format = 'social_post'  # default
        best_avg = 0
        
        for format_type, engagements in format_performance.items():
            if engagements:
                avg_engagement = np.mean(engagements)
                if avg_engagement > best_avg:
                    best_avg = avg_engagement
                    best_format = format_type
        
        return best_format
    
    def _analyze_format_performance(self, content_list: List[ContentPiece]) -> Dict[str, Any]:
        """Analyze performance by content format"""
        
        format_stats = {}
        
        for format_type in ContentFormat:
            format_content = [c for c in content_list if c.content_format == format_type]
            
            if format_content:
                avg_engagement = np.mean([c.actual_engagement for c in format_content if c.actual_engagement > 0])
                avg_quality = np.mean([c.content_quality_score for c in format_content])
                avg_viral_potential = np.mean([c.viral_potential for c in format_content])
                
                format_stats[format_type.value] = {
                    'content_count': len(format_content),
                    'average_engagement': avg_engagement,
                    'average_quality': avg_quality,
                    'viral_potential': avg_viral_potential,
                    'recommendation': self._get_format_recommendation(avg_engagement, avg_quality)
                }
        
        return format_stats
    
    def _get_format_recommendation(self, engagement: float, quality: float) -> str:
        """Get recommendation for content format"""
        
        if engagement > 5.0 and quality > 85:
            return 'scale_up'
        elif quality > 80:
            return 'optimize_distribution'
        elif engagement > 3.0:
            return 'improve_quality'
        else:
            return 'major_optimization_needed'
    
    def _analyze_brand_consistency(self, content_list: List[ContentPiece]) -> Dict[str, Any]:
        """Analyze brand consistency across content"""
        
        consistency_scores = [c.brand_consistency_score for c in content_list]
        
        return {
            'average_consistency': np.mean(consistency_scores),
            'consistency_variance': np.var(consistency_scores),
            'low_consistency_count': len([s for s in consistency_scores if s < 75]),
            'consistency_trend': self._calculate_consistency_trend(content_list)
        }
    
    def _calculate_consistency_trend(self, content_list: List[ContentPiece]) -> str:
        """Calculate consistency trend over time"""
        
        # Sort by creation date
        sorted_content = sorted(content_list, key=lambda x: x.created_at)
        
        if len(sorted_content) < 3:
            return 'insufficient_data'
        
        # Compare first third vs last third
        first_third = sorted_content[:len(sorted_content)//3]
        last_third = sorted_content[-len(sorted_content)//3:]
        
        first_avg = np.mean([c.brand_consistency_score for c in first_third])
        last_avg = np.mean([c.brand_consistency_score for c in last_third])
        
        if last_avg > first_avg + 5:
            return 'improving'
        elif last_avg < first_avg - 5:
            return 'declining'
        else:
            return 'stable'
    
    def _generate_content_insights(self, platform_perf: Dict, format_perf: Dict) -> List[str]:
        """Generate actionable content insights"""
        
        insights = []
        
        # Platform insights
        if platform_perf:
            best_platform = max(platform_perf.items(), key=lambda x: x[1].get('average_engagement', 0))
            insights.append(f"{best_platform[0].title()} is your top performing platform")
            
            high_potential_platforms = [p for p, data in platform_perf.items() 
                                      if data.get('optimization_potential', 0) > 30]
            if high_potential_platforms:
                insights.append(f"High optimization potential on {', '.join(high_potential_platforms)}")
        
        # Format insights
        if format_perf:
            top_format = max(format_perf.items(), key=lambda x: x[1].get('average_engagement', 0))
            insights.append(f"{top_format[0].replace('_', ' ').title()} format drives highest engagement")
        
        return insights
    
    def _identify_trend_opportunities(self, brand: ContentBrand) -> Dict[str, Any]:
        """Identify trending opportunities for brand"""
        
        # Get recent trends
        recent_trends = TrendAnalysis.query.filter(
            TrendAnalysis.discovered_at >= datetime.utcnow() - timedelta(days=7)
        ).order_by(TrendAnalysis.roi_potential.desc()).limit(20).all()
        
        # Filter trends by brand fit
        suitable_trends = []
        for trend in recent_trends:
            if trend.brand_fit_score > 60 and trend.saturation_level < 70:
                suitable_trends.append({
                    'trend_name': trend.trend_name,
                    'platform': trend.trend_platform.value if trend.trend_platform else 'multi_platform',
                    'roi_potential': trend.roi_potential,
                    'difficulty_level': trend.difficulty_level,
                    'peak_prediction': trend.peak_prediction.isoformat() if trend.peak_prediction else None,
                    'content_recommendations': trend.content_format_recommendations or [],
                    'hashtag_suggestions': trend.hashtag_recommendations or []
                })
        
        # Categorize opportunities
        high_priority = [t for t in suitable_trends if t['roi_potential'] > 40 and t['difficulty_level'] < 50]
        medium_priority = [t for t in suitable_trends if t['roi_potential'] > 25 and t['difficulty_level'] < 70]
        
        return {
            'total_trends_analyzed': len(recent_trends),
            'suitable_trends_count': len(suitable_trends),
            'high_priority_opportunities': high_priority[:5],
            'medium_priority_opportunities': medium_priority[:8],
            'trend_integration_strategy': self._develop_trend_integration_strategy(high_priority, brand)
        }
    
    def _develop_trend_integration_strategy(self, trends: List[Dict], brand: ContentBrand) -> Dict[str, Any]:
        """Develop strategy for integrating trends"""
        
        if not trends:
            return {'status': 'no_suitable_trends'}
        
        strategy = {
            'integration_approach': 'selective_authentic',
            'content_allocation': '20% trend-driven, 80% brand-core',
            'platform_priorities': [],
            'content_format_focus': [],
            'timing_strategy': 'early_adopter',
            'brand_safety_measures': [
                'Brand voice consistency check',
                'Value alignment verification',
                'Audience appropriateness review'
            ]
        }
        
        # Analyze platform distribution of trends
        platform_counts = {}
        for trend in trends:
            platform = trend['platform']
            platform_counts[platform] = platform_counts.get(platform, 0) + 1
        
        strategy['platform_priorities'] = sorted(platform_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Analyze content format recommendations
        format_recommendations = []
        for trend in trends:
            format_recommendations.extend(trend['content_recommendations'])
        
        format_counts = {}
        for format_rec in format_recommendations:
            format_counts[format_rec] = format_counts.get(format_rec, 0) + 1
        
        strategy['content_format_focus'] = sorted(format_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return strategy
    
    def _generate_strategic_content_calendar(self, brand: ContentBrand, period_days: int, trends: Dict) -> Dict[str, Any]:
        """Generate strategic content calendar"""
        
        start_date = datetime.now().date()
        end_date = start_date + timedelta(days=period_days)
        
        # Get content pillars and themes
        content_pillars = brand.content_pillars or ['brand_awareness', 'education', 'engagement']
        content_themes = brand.content_themes or ['innovation', 'customer_success', 'behind_scenes']
        
        # Generate daily content schedule
        daily_schedule = {}
        
        for day_offset in range(period_days):
            current_date = start_date + timedelta(days=day_offset)
            
            # Determine daily theme and pillar
            pillar_index = day_offset % len(content_pillars)
            theme_index = (day_offset // 7) % len(content_themes)  # Weekly theme rotation
            
            daily_content = {
                'date': current_date.isoformat(),
                'primary_pillar': content_pillars[pillar_index],
                'theme': content_themes[theme_index],
                'content_pieces': self._plan_daily_content(brand, current_date, content_pillars[pillar_index], trends),
                'posting_schedule': self._optimize_posting_times(brand, current_date),
                'cross_platform_coordination': self._plan_cross_platform_coordination(current_date)
            }
            
            daily_schedule[current_date.isoformat()] = daily_content
        
        # Generate weekly and monthly views
        weekly_overview = self._generate_weekly_overview(daily_schedule, period_days)
        monthly_themes = self._generate_monthly_themes(period_days)
        
        return {
            'calendar_period': f'{start_date} to {end_date}',
            'daily_schedule': daily_schedule,
            'weekly_overview': weekly_overview,
            'monthly_themes': monthly_themes,
            'content_distribution': self._calculate_content_distribution(daily_schedule),
            'performance_targets': self._set_performance_targets(brand, period_days)
        }
    
    def _plan_daily_content(self, brand: ContentBrand, date: datetime.date, pillar: str, trends: Dict) -> List[Dict[str, Any]]:
        """Plan content pieces for a specific day"""
        
        publishing_freq = brand.publishing_frequency or {}
        base_frequency = publishing_freq.get('daily_posts', 2)
        
        content_pieces = []
        
        # Primary content piece
        primary_piece = {
            'content_type': 'primary',
            'format': self._select_optimal_format(pillar, date),
            'platform': self._select_primary_platform(brand, date),
            'pillar': pillar,
            'estimated_engagement': self._estimate_engagement(pillar, date),
            'production_requirements': self._assess_production_needs(pillar)
        }
        content_pieces.append(primary_piece)
        
        # Secondary content pieces
        for i in range(base_frequency - 1):
            secondary_piece = {
                'content_type': 'secondary',
                'format': self._select_secondary_format(i),
                'platform': self._select_secondary_platform(brand, date, i),
                'pillar': pillar,
                'estimated_engagement': self._estimate_engagement(pillar, date) * 0.7,
                'production_requirements': self._assess_production_needs('secondary')
            }
            content_pieces.append(secondary_piece)
        
        # Trend integration (20% of content)
        if self._should_integrate_trend(date, trends):
            trend_piece = self._plan_trend_content(trends, pillar, date)
            content_pieces.append(trend_piece)
        
        return content_pieces
    
    def _select_optimal_format(self, pillar: str, date: datetime.date) -> str:
        """Select optimal content format for pillar and date"""
        
        # Format preferences by pillar
        pillar_formats = {
            'brand_awareness': ['video_script', 'infographic', 'social_post'],
            'education': ['blog_post', 'infographic', 'podcast_script'],
            'engagement': ['social_post', 'video_script', 'email_newsletter'],
            'conversion': ['case_study', 'ebook', 'email_newsletter'],
            'retention': ['email_newsletter', 'blog_post', 'social_post']
        }
        
        # Day of week considerations
        weekday_formats = {
            0: 'blog_post',      # Monday - detailed content
            1: 'social_post',    # Tuesday - engagement
            2: 'infographic',    # Wednesday - visual content
            3: 'video_script',   # Thursday - video content
            4: 'social_post',    # Friday - casual engagement
            5: 'email_newsletter', # Saturday - newsletter
            6: 'social_post'     # Sunday - light content
        }
        
        preferred_formats = pillar_formats.get(pillar, ['social_post'])
        weekday_preference = weekday_formats.get(date.weekday(), 'social_post')
        
        # Combine preferences
        if weekday_preference in preferred_formats:
            return weekday_preference
        else:
            return preferred_formats[0]
    
    def _select_primary_platform(self, brand: ContentBrand, date: datetime.date) -> str:
        """Select primary platform for content"""
        
        # Platform rotation strategy
        platforms = ['instagram', 'linkedin', 'facebook', 'twitter', 'youtube']
        
        # Different platform focus by day
        weekday_platforms = {
            0: 'linkedin',     # Monday - professional
            1: 'instagram',    # Tuesday - visual
            2: 'facebook',     # Wednesday - community
            3: 'youtube',      # Thursday - video
            4: 'instagram',    # Friday - engagement
            5: 'facebook',     # Saturday - community
            6: 'instagram'     # Sunday - lifestyle
        }
        
        return weekday_platforms.get(date.weekday(), 'instagram')
    
    def _select_secondary_format(self, index: int) -> str:
        """Select format for secondary content pieces"""
        
        secondary_formats = ['social_post', 'infographic', 'video_script']
        return secondary_formats[index % len(secondary_formats)]
    
    def _select_secondary_platform(self, brand: ContentBrand, date: datetime.date, index: int) -> str:
        """Select platform for secondary content"""
        
        secondary_platforms = ['twitter', 'linkedin', 'facebook']
        return secondary_platforms[index % len(secondary_platforms)]
    
    def _estimate_engagement(self, pillar: str, date: datetime.date) -> float:
        """Estimate engagement for content piece"""
        
        # Base engagement by pillar
        pillar_engagement = {
            'brand_awareness': 3.5,
            'education': 4.2,
            'engagement': 5.8,
            'conversion': 2.9,
            'retention': 4.1
        }
        
        base_engagement = pillar_engagement.get(pillar, 4.0)
        
        # Day of week multipliers
        weekday_multipliers = {
            0: 1.1,  # Monday
            1: 1.2,  # Tuesday
            2: 1.3,  # Wednesday - peak
            3: 1.2,  # Thursday
            4: 1.1,  # Friday
            5: 0.8,  # Saturday
            6: 0.9   # Sunday
        }
        
        multiplier = weekday_multipliers.get(date.weekday(), 1.0)
        return base_engagement * multiplier
    
    def _assess_production_needs(self, content_type: str) -> Dict[str, Any]:
        """Assess production requirements for content"""
        
        if content_type == 'secondary':
            return {
                'complexity': 'low',
                'time_required': '30 minutes',
                'resources_needed': ['copywriter'],
                'design_support': False
            }
        
        production_requirements = {
            'blog_post': {
                'complexity': 'medium',
                'time_required': '2-3 hours',
                'resources_needed': ['copywriter', 'seo_specialist'],
                'design_support': True
            },
            'video_script': {
                'complexity': 'high',
                'time_required': '4-6 hours',
                'resources_needed': ['copywriter', 'video_producer', 'designer'],
                'design_support': True
            },
            'infographic': {
                'complexity': 'medium',
                'time_required': '3-4 hours',
                'resources_needed': ['designer', 'copywriter'],
                'design_support': True
            },
            'social_post': {
                'complexity': 'low',
                'time_required': '45 minutes',
                'resources_needed': ['copywriter'],
                'design_support': True
            }
        }
        
        return production_requirements.get(content_type, production_requirements['social_post'])
    
    def _should_integrate_trend(self, date: datetime.date, trends: Dict) -> bool:
        """Determine if trend should be integrated on this date"""
        
        # 20% of content should be trend-driven
        # Use date hash to create consistent randomness
        date_hash = hash(date.isoformat()) % 100
        
        # Integrate trends on specific days (roughly 20% of time)
        trend_days = [1, 3, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27]  # ~20% of month
        
        return date.day in trend_days and len(trends.get('high_priority_opportunities', [])) > 0
    
    def _plan_trend_content(self, trends: Dict, pillar: str, date: datetime.date) -> Dict[str, Any]:
        """Plan trend-integrated content"""
        
        high_priority_trends = trends.get('high_priority_opportunities', [])
        
        if not high_priority_trends:
            return self._create_default_content(pillar)
        
        # Select trend based on date
        trend_index = date.day % len(high_priority_trends)
        selected_trend = high_priority_trends[trend_index]
        
        return {
            'content_type': 'trend_integration',
            'trend_name': selected_trend['trend_name'],
            'format': selected_trend['content_recommendations'][0] if selected_trend['content_recommendations'] else 'social_post',
            'platform': selected_trend['platform'],
            'pillar': pillar,
            'hashtags': selected_trend['hashtag_suggestions'][:5],
            'estimated_engagement': self._estimate_engagement(pillar, date) * 1.4,  # Trend boost
            'production_requirements': self._assess_production_needs('trend')
        }
    
    def _create_default_content(self, pillar: str) -> Dict[str, Any]:
        """Create default content when no trends available"""
        
        return {
            'content_type': 'default',
            'format': 'social_post',
            'platform': 'instagram',
            'pillar': pillar,
            'estimated_engagement': 3.5,
            'production_requirements': self._assess_production_needs('social_post')
        }
    
    def _optimize_posting_times(self, brand: ContentBrand, date: datetime.date) -> Dict[str, Any]:
        """Optimize posting times for maximum engagement"""
        
        # Platform-specific optimal times
        optimal_times = {
            'instagram': {
                'weekday': ['11:00', '14:00', '17:00'],
                'weekend': ['10:00', '13:00', '16:00']
            },
            'linkedin': {
                'weekday': ['08:00', '12:00', '17:00'],
                'weekend': ['09:00', '14:00']
            },
            'facebook': {
                'weekday': ['09:00', '13:00', '15:00'],
                'weekend': ['12:00', '14:00', '16:00']
            },
            'twitter': {
                'weekday': ['08:00', '12:00', '17:00', '19:00'],
                'weekend': ['10:00', '14:00', '18:00']
            },
            'youtube': {
                'weekday': ['14:00', '18:00', '20:00'],
                'weekend': ['11:00', '15:00', '19:00']
            }
        }
        
        is_weekend = date.weekday() >= 5
        time_category = 'weekend' if is_weekend else 'weekday'
        
        schedule = {}
        for platform, times in optimal_times.items():
            platform_times = times.get(time_category, times['weekday'])
            schedule[platform] = {
                'optimal_times': platform_times,
                'recommended_time': platform_times[0],  # Primary recommendation
                'engagement_multiplier': 1.3 if not is_weekend else 0.9
            }
        
        return schedule
    
    def _plan_cross_platform_coordination(self, date: datetime.date) -> Dict[str, Any]:
        """Plan cross-platform content coordination"""
        
        return {
            'coordination_strategy': 'sequential_release',
            'platform_sequence': ['linkedin', 'instagram', 'facebook', 'twitter'],
            'time_intervals': '2 hours between platforms',
            'content_variations': {
                'messaging': 'platform_optimized',
                'visuals': 'format_adapted',
                'hashtags': 'platform_specific'
            },
            'cross_promotion': {
                'enabled': True,
                'delay': '4 hours',
                'platforms': ['instagram_stories', 'linkedin_activity']
            }
        }
    
    def _generate_weekly_overview(self, daily_schedule: Dict, period_days: int) -> Dict[str, Any]:
        """Generate weekly content overview"""
        
        weekly_summaries = {}
        
        # Group by weeks
        for week_start in range(0, period_days, 7):
            week_end = min(week_start + 7, period_days)
            week_key = f"week_{week_start//7 + 1}"
            
            week_dates = []
            week_content_count = 0
            week_platforms = set()
            week_formats = set()
            
            for day_offset in range(week_start, week_end):
                date = (datetime.now().date() + timedelta(days=day_offset)).isoformat()
                
                if date in daily_schedule:
                    week_dates.append(date)
                    day_content = daily_schedule[date]['content_pieces']
                    week_content_count += len(day_content)
                    
                    for piece in day_content:
                        week_platforms.add(piece['platform'])
                        week_formats.add(piece['format'])
            
            weekly_summaries[week_key] = {
                'date_range': f"{week_dates[0]} to {week_dates[-1]}" if week_dates else 'No dates',
                'total_content_pieces': week_content_count,
                'platforms_used': list(week_platforms),
                'formats_used': list(week_formats),
                'weekly_theme': self._determine_weekly_theme(week_start),
                'content_distribution': self._calculate_weekly_distribution(daily_schedule, week_dates)
            }
        
        return weekly_summaries
    
    def _determine_weekly_theme(self, week_start: int) -> str:
        """Determine theme for the week"""
        
        themes = ['Innovation Week', 'Customer Stories', 'Behind the Scenes', 'Industry Insights']
        return themes[week_start // 7 % len(themes)]
    
    def _calculate_weekly_distribution(self, daily_schedule: Dict, week_dates: List[str]) -> Dict[str, int]:
        """Calculate content distribution for the week"""
        
        distribution = {}
        
        for date in week_dates:
            if date in daily_schedule:
                for piece in daily_schedule[date]['content_pieces']:
                    platform = piece['platform']
                    distribution[platform] = distribution.get(platform, 0) + 1
        
        return distribution
    
    def _generate_monthly_themes(self, period_days: int) -> Dict[str, Any]:
        """Generate monthly theme calendar"""
        
        monthly_themes = {
            'month_1': {
                'primary_theme': 'Innovation & Growth',
                'sub_themes': ['Product Innovation', 'Market Expansion', 'Customer Success'],
                'content_focus': 'thought_leadership',
                'special_campaigns': ['New Product Launch', 'Customer Appreciation']
            }
        }
        
        # Add more months if period is longer
        if period_days > 30:
            monthly_themes['month_2'] = {
                'primary_theme': 'Community & Connection',
                'sub_themes': ['User Stories', 'Team Highlights', 'Industry Partnerships'],
                'content_focus': 'engagement_building',
                'special_campaigns': ['Community Spotlight', 'Partnership Announcements']
            }
        
        return monthly_themes
    
    def _calculate_content_distribution(self, daily_schedule: Dict) -> Dict[str, Any]:
        """Calculate overall content distribution"""
        
        platform_distribution = {}
        format_distribution = {}
        pillar_distribution = {}
        
        for date, day_data in daily_schedule.items():
            for piece in day_data['content_pieces']:
                # Platform distribution
                platform = piece['platform']
                platform_distribution[platform] = platform_distribution.get(platform, 0) + 1
                
                # Format distribution
                format_type = piece['format']
                format_distribution[format_type] = format_distribution.get(format_type, 0) + 1
                
                # Pillar distribution
                pillar = piece['pillar']
                pillar_distribution[pillar] = pillar_distribution.get(pillar, 0) + 1
        
        return {
            'platform_distribution': platform_distribution,
            'format_distribution': format_distribution,
            'pillar_distribution': pillar_distribution,
            'total_content_pieces': sum(platform_distribution.values())
        }
    
    def _set_performance_targets(self, brand: ContentBrand, period_days: int) -> Dict[str, Any]:
        """Set performance targets for the period"""
        
        # Base targets
        base_engagement_rate = 4.5
        base_reach_multiplier = 1000
        base_conversion_rate = 2.2
        
        # Calculate period targets
        total_content_pieces = period_days * 2  # Assuming 2 pieces per day average
        
        return {
            'engagement_targets': {
                'total_engagements': total_content_pieces * base_engagement_rate * 100,
                'average_engagement_rate': base_engagement_rate,
                'engagement_growth': '15% month-over-month'
            },
            'reach_targets': {
                'total_reach': total_content_pieces * base_reach_multiplier,
                'unique_reach': total_content_pieces * base_reach_multiplier * 0.7,
                'reach_growth': '20% month-over-month'
            },
            'conversion_targets': {
                'total_conversions': total_content_pieces * base_conversion_rate,
                'conversion_rate': base_conversion_rate,
                'conversion_value': total_content_pieces * base_conversion_rate * 50  # $50 avg value
            },
            'brand_awareness_targets': {
                'brand_mention_increase': '25%',
                'hashtag_performance': '30% increase',
                'share_of_voice': '10% industry share'
            }
        }
    
    def _generate_platform_specific_content(self, brand: ContentBrand, calendar: Dict) -> Dict[str, Any]:
        """Generate platform-specific optimized content"""
        
        platform_content = {}
        
        for platform in Platform:
            platform_content[platform.value] = {
                'content_specifications': self._get_platform_content_specs(platform),
                'optimized_content': self._create_platform_optimized_content(platform, brand, calendar),
                'posting_strategy': self._develop_platform_posting_strategy(platform),
                'engagement_tactics': self._define_platform_engagement_tactics(platform)
            }
        
        return platform_content
    
    def _get_platform_content_specs(self, platform: Platform) -> Dict[str, Any]:
        """Get content specifications for each platform"""
        
        specs = {
            Platform.INSTAGRAM: {
                'image_size': '1080x1080',
                'video_length': '15-60 seconds',
                'caption_length': '2200 characters max',
                'hashtags': '20-30 hashtags',
                'optimal_posting_frequency': '1-2 posts daily'
            },
            Platform.LINKEDIN: {
                'image_size': '1200x627',
                'video_length': '30-300 seconds',
                'caption_length': '1300 characters optimal',
                'hashtags': '3-5 hashtags',
                'optimal_posting_frequency': '1 post daily'
            },
            Platform.FACEBOOK: {
                'image_size': '1200x630',
                'video_length': '15-240 seconds',
                'caption_length': '500 characters optimal',
                'hashtags': '1-2 hashtags',
                'optimal_posting_frequency': '1 post daily'
            },
            Platform.TWITTER: {
                'image_size': '1200x675',
                'video_length': '15-140 seconds',
                'caption_length': '280 characters max',
                'hashtags': '1-2 hashtags',
                'optimal_posting_frequency': '3-5 posts daily'
            },
            Platform.YOUTUBE: {
                'thumbnail_size': '1280x720',
                'video_length': '2-15 minutes',
                'description_length': '5000 characters max',
                'hashtags': '10-15 hashtags',
                'optimal_posting_frequency': '1-3 posts weekly'
            }
        }
        
        return specs.get(platform, {})
    
    def _create_platform_optimized_content(self, platform: Platform, brand: ContentBrand, calendar: Dict) -> List[Dict[str, Any]]:
        """Create optimized content for specific platform"""
        
        optimized_content = []
        
        # Extract relevant content from calendar
        daily_schedule = calendar.get('daily_schedule', {})
        
        for date, day_data in daily_schedule.items():
            platform_pieces = [p for p in day_data['content_pieces'] if p['platform'] == platform.value]
            
            for piece in platform_pieces:
                optimized_piece = self._optimize_content_for_platform(piece, platform, brand)
                optimized_piece['scheduled_date'] = date
                optimized_content.append(optimized_piece)
        
        return optimized_content[:10]  # Return first 10 for example
    
    def _optimize_content_for_platform(self, content_piece: Dict, platform: Platform, brand: ContentBrand) -> Dict[str, Any]:
        """Optimize individual content piece for platform"""
        
        # Generate platform-specific content using AI
        content_prompt = f"""
        Create {platform.value} content for a {content_piece['format']} about {content_piece['pillar']}.
        Brand voice: {brand.tone_attributes or 'professional, friendly, authentic'}
        Brand values: {brand.brand_voice or 'innovation, quality, customer-centricity'}
        
        Requirements:
        - Platform: {platform.value}
        - Format: {content_piece['format']}
        - Pillar: {content_piece['pillar']}
        - Estimated engagement: {content_piece['estimated_engagement']}%
        
        Generate engaging, platform-optimized content that drives {content_piece['estimated_engagement']}% engagement.
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are an expert {platform.value} content creator who creates highly engaging, platform-optimized content."},
                {"role": "user", "content": content_prompt}
            ],
            max_tokens=800
        )
        
        generated_content = response.choices[0].message.content
        
        return {
            'platform': platform.value,
            'format': content_piece['format'],
            'pillar': content_piece['pillar'],
            'content': generated_content,
            'hashtags': self._generate_platform_hashtags(platform, content_piece['pillar']),
            'visual_requirements': self._define_visual_requirements_for_platform(platform, content_piece),
            'engagement_optimization': self._add_engagement_optimization(platform, content_piece),
            'call_to_action': self._generate_platform_cta(platform, content_piece['pillar'])
        }
    
    def _generate_platform_hashtags(self, platform: Platform, pillar: str) -> List[str]:
        """Generate platform-specific hashtags"""
        
        base_hashtags = {
            'brand_awareness': ['#Innovation', '#Leadership', '#Excellence'],
            'education': ['#LearningTogether', '#DidYouKnow', '#Education'],
            'engagement': ['#Community', '#EngageWithUs', '#YourVoiceMatters'],
            'conversion': ['#GetStarted', '#TransformToday', '#JoinUs'],
            'retention': ['#CustomerSuccess', '#ThankYou', '#LoyaltyRewards']
        }
        
        platform_specific = {
            Platform.INSTAGRAM: ['#InstaGood', '#PhotoOfTheDay', '#Inspiration'],
            Platform.LINKEDIN: ['#ProfessionalDevelopment', '#B2B', '#Networking'],
            Platform.TWITTER: ['#TweetThis', '#Breaking', '#Trending'],
            Platform.FACEBOOK: ['#CommunityFirst', '#ShareTheJoy', '#TogetherWeCan'],
            Platform.YOUTUBE: ['#VideoContent', '#Subscribe', '#Tutorial']
        }
        
        hashtags = base_hashtags.get(pillar, ['#BrandContent'])
        hashtags.extend(platform_specific.get(platform, []))
        
        return hashtags[:10]  # Limit to 10 hashtags
    
    def _define_visual_requirements_for_platform(self, platform: Platform, content_piece: Dict) -> Dict[str, Any]:
        """Define visual requirements for platform-specific content"""
        
        specs = self._get_platform_content_specs(platform)
        
        return {
            'image_specifications': specs.get('image_size', '1080x1080'),
            'video_specifications': specs.get('video_length', '60 seconds'),
            'design_style': 'brand_consistent',
            'color_palette': 'brand_primary_secondary',
            'typography': 'brand_fonts',
            'visual_hierarchy': 'clear_message_focus',
            'brand_elements': 'logo_watermark_subtle'
        }
    
    def _add_engagement_optimization(self, platform: Platform, content_piece: Dict) -> Dict[str, Any]:
        """Add platform-specific engagement optimization"""
        
        optimization_tactics = {
            Platform.INSTAGRAM: {
                'engagement_hooks': ['Ask questions in caption', 'Use interactive stickers', 'Create shareable moments'],
                'posting_timing': 'Peak audience hours',
                'content_features': ['Stories integration', 'Reels optimization', 'IGTV consideration']
            },
            Platform.LINKEDIN: {
                'engagement_hooks': ['Industry insights', 'Professional questions', 'Career tips'],
                'posting_timing': 'Business hours',
                'content_features': ['Article publication', 'Professional networking', 'Thought leadership']
            },
            Platform.TWITTER: {
                'engagement_hooks': ['Trending hashtags', 'Reply prompts', 'Retweet requests'],
                'posting_timing': 'News cycle alignment',
                'content_features': ['Thread creation', 'Twitter Spaces', 'Live tweeting']
            }
        }
        
        return optimization_tactics.get(platform, {
            'engagement_hooks': ['Ask questions', 'Encourage sharing'],
            'posting_timing': 'Optimal hours',
            'content_features': ['Standard posting']
        })
    
    def _generate_platform_cta(self, platform: Platform, pillar: str) -> str:
        """Generate platform-appropriate call-to-action"""
        
        cta_templates = {
            Platform.INSTAGRAM: {
                'brand_awareness': 'Double tap if you agree! 💝',
                'education': 'Save this post for later! 📚',
                'engagement': 'Tell us in the comments! 👇',
                'conversion': 'Link in bio to get started! 🔗',
                'retention': 'Tag someone who needs to see this! 👥'
            },
            Platform.LINKEDIN: {
                'brand_awareness': 'Connect with us for more insights',
                'education': 'Share your thoughts in the comments',
                'engagement': 'What\'s your experience with this?',
                'conversion': 'Ready to transform your business?',
                'retention': 'Thank you for being part of our community'
            },
            Platform.TWITTER: {
                'brand_awareness': 'RT if you found this valuable',
                'education': 'Thread: More insights below 🧵',
                'engagement': 'What do you think? Reply below',
                'conversion': 'Get started today: [link]',
                'retention': 'Thanks for following our journey'
            }
        }
        
        platform_ctas = cta_templates.get(platform, {})
        return platform_ctas.get(pillar, 'Engage with us!')
    
    def _develop_platform_posting_strategy(self, platform: Platform) -> Dict[str, Any]:
        """Develop comprehensive posting strategy for platform"""
        
        strategies = {
            Platform.INSTAGRAM: {
                'primary_strategy': 'visual_storytelling',
                'content_mix': '70% original, 20% user_generated, 10% curated',
                'engagement_focus': 'community_building',
                'growth_tactics': ['hashtag_optimization', 'story_highlights', 'reels_creation']
            },
            Platform.LINKEDIN: {
                'primary_strategy': 'thought_leadership',
                'content_mix': '80% original, 15% industry_news, 5% curated',
                'engagement_focus': 'professional_networking',
                'growth_tactics': ['article_publishing', 'professional_insights', 'industry_participation']
            },
            Platform.YOUTUBE: {
                'primary_strategy': 'educational_content',
                'content_mix': '90% original, 10% collaborations',
                'engagement_focus': 'subscriber_growth',
                'growth_tactics': ['series_creation', 'tutorial_content', 'community_engagement']
            }
        }
        
        return strategies.get(platform, {
            'primary_strategy': 'brand_awareness',
            'content_mix': '80% original, 20% curated',
            'engagement_focus': 'audience_growth',
            'growth_tactics': ['consistent_posting', 'engagement_optimization']
        })
    
    def _define_platform_engagement_tactics(self, platform: Platform) -> List[Dict[str, Any]]:
        """Define specific engagement tactics for platform"""
        
        tactics = {
            Platform.INSTAGRAM: [
                {'tactic': 'story_polls', 'frequency': 'daily', 'impact': 'high'},
                {'tactic': 'user_generated_content', 'frequency': 'weekly', 'impact': 'high'},
                {'tactic': 'hashtag_challenges', 'frequency': 'monthly', 'impact': 'very_high'},
                {'tactic': 'behind_scenes_content', 'frequency': 'weekly', 'impact': 'medium'}
            ],
            Platform.LINKEDIN: [
                {'tactic': 'industry_discussions', 'frequency': 'daily', 'impact': 'high'},
                {'tactic': 'professional_tips', 'frequency': 'weekly', 'impact': 'medium'},
                {'tactic': 'company_updates', 'frequency': 'weekly', 'impact': 'medium'},
                {'tactic': 'thought_leadership_articles', 'frequency': 'bi_weekly', 'impact': 'very_high'}
            ],
            Platform.TWITTER: [
                {'tactic': 'trending_hashtags', 'frequency': 'daily', 'impact': 'high'},
                {'tactic': 'real_time_engagement', 'frequency': 'continuous', 'impact': 'high'},
                {'tactic': 'twitter_threads', 'frequency': 'weekly', 'impact': 'medium'},
                {'tactic': 'live_tweeting_events', 'frequency': 'monthly', 'impact': 'high'}
            ]
        }
        
        return tactics.get(platform, [
            {'tactic': 'consistent_posting', 'frequency': 'daily', 'impact': 'medium'},
            {'tactic': 'audience_interaction', 'frequency': 'daily', 'impact': 'high'}
        ])
    
    def _optimize_for_engagement(self, brand: ContentBrand, platform_content: Dict) -> Dict[str, Any]:
        """Optimize content strategy for maximum engagement"""
        
        # Analyze engagement patterns
        engagement_analysis = self._analyze_engagement_patterns(platform_content)
        
        # Identify optimization opportunities
        optimization_opportunities = self._identify_engagement_opportunities(engagement_analysis)
        
        # Generate optimization recommendations
        recommendations = self._generate_engagement_recommendations(optimization_opportunities, brand)
        
        return {
            'engagement_analysis': engagement_analysis,
            'optimization_opportunities': optimization_opportunities,
            'recommendations': recommendations,
            'implementation_priority': self._prioritize_optimizations(recommendations)
        }
    
    def _analyze_engagement_patterns(self, platform_content: Dict) -> Dict[str, Any]:
        """Analyze engagement patterns across platforms"""
        
        patterns = {}
        
        for platform, content_data in platform_content.items():
            optimized_content = content_data.get('optimized_content', [])
            
            if optimized_content:
                # Analyze content characteristics
                avg_engagement_potential = np.mean([
                    float(content.get('engagement_optimization', {}).get('estimated_engagement', 4.0))
                    for content in optimized_content
                ])
                
                patterns[platform] = {
                    'content_count': len(optimized_content),
                    'avg_engagement_potential': avg_engagement_potential,
                    'engagement_tactics_count': len(content_data.get('engagement_tactics', [])),
                    'posting_frequency': content_data.get('content_specifications', {}).get('optimal_posting_frequency', 'daily')
                }
        
        return patterns
    
    def _identify_engagement_opportunities(self, analysis: Dict) -> List[Dict[str, Any]]:
        """Identify opportunities to improve engagement"""
        
        opportunities = []
        
        for platform, data in analysis.items():
            # Low engagement potential opportunity
            if data['avg_engagement_potential'] < 4.0:
                opportunities.append({
                    'platform': platform,
                    'opportunity_type': 'engagement_optimization',
                    'current_performance': data['avg_engagement_potential'],
                    'improvement_potential': '25-40%',
                    'priority': 'high'
                })
            
            # Low tactics utilization
            if data['engagement_tactics_count'] < 3:
                opportunities.append({
                    'platform': platform,
                    'opportunity_type': 'tactics_expansion',
                    'current_tactics': data['engagement_tactics_count'],
                    'improvement_potential': '15-30%',
                    'priority': 'medium'
                })
        
        # Cross-platform coordination opportunity
        if len(analysis) > 2:
            opportunities.append({
                'platform': 'cross_platform',
                'opportunity_type': 'coordination_enhancement',
                'improvement_potential': '20-35%',
                'priority': 'medium'
            })
        
        return opportunities
    
    def _generate_engagement_recommendations(self, opportunities: List[Dict], brand: ContentBrand) -> List[Dict[str, Any]]:
        """Generate specific engagement improvement recommendations"""
        
        recommendations = []
        
        for opportunity in opportunities:
            if opportunity['opportunity_type'] == 'engagement_optimization':
                recommendations.append({
                    'category': 'content_optimization',
                    'platform': opportunity['platform'],
                    'recommendation': 'Enhance content with interactive elements and emotional hooks',
                    'expected_impact': opportunity['improvement_potential'],
                    'implementation_effort': 'medium',
                    'timeline': '2-4 weeks'
                })
            
            elif opportunity['opportunity_type'] == 'tactics_expansion':
                recommendations.append({
                    'category': 'engagement_tactics',
                    'platform': opportunity['platform'],
                    'recommendation': 'Implement additional platform-specific engagement tactics',
                    'expected_impact': opportunity['improvement_potential'],
                    'implementation_effort': 'low',
                    'timeline': '1-2 weeks'
                })
            
            elif opportunity['opportunity_type'] == 'coordination_enhancement':
                recommendations.append({
                    'category': 'cross_platform_optimization',
                    'platform': 'all',
                    'recommendation': 'Implement coordinated cross-platform content strategy',
                    'expected_impact': opportunity['improvement_potential'],
                    'implementation_effort': 'high',
                    'timeline': '4-6 weeks'
                })
        
        return recommendations
    
    def _prioritize_optimizations(self, recommendations: List[Dict]) -> List[Dict[str, Any]]:
        """Prioritize optimization recommendations"""
        
        # Score recommendations based on impact and effort
        scored_recommendations = []
        
        for rec in recommendations:
            impact_score = self._calculate_impact_score(rec['expected_impact'])
            effort_score = self._calculate_effort_score(rec['implementation_effort'])
            
            # Priority score: higher impact, lower effort = higher priority
            priority_score = impact_score / max(effort_score, 1)
            
            scored_recommendations.append({
                'recommendation': rec,
                'priority_score': priority_score,
                'priority_level': self._get_priority_level(priority_score)
            })
        
        # Sort by priority score
        scored_recommendations.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return scored_recommendations
    
    def _calculate_impact_score(self, impact: str) -> float:
        """Calculate impact score from impact string"""
        
        if '40%' in impact or '35%' in impact:
            return 4.0
        elif '30%' in impact or '25%' in impact:
            return 3.0
        elif '20%' in impact or '15%' in impact:
            return 2.0
        else:
            return 1.0
    
    def _calculate_effort_score(self, effort: str) -> float:
        """Calculate effort score from effort string"""
        
        effort_scores = {
            'low': 1.0,
            'medium': 2.0,
            'high': 3.0
        }
        
        return effort_scores.get(effort, 2.0)
    
    def _get_priority_level(self, score: float) -> str:
        """Get priority level from score"""
        
        if score >= 3.0:
            return 'high'
        elif score >= 2.0:
            return 'medium'
        else:
            return 'low'
    
    def _project_content_performance(self, platform_content: Dict) -> Dict[str, Any]:
        """Project content performance based on strategy"""
        
        # Calculate aggregate projections
        total_content_pieces = sum(
            len(data.get('optimized_content', []))
            for data in platform_content.values()
        )
        
        # Estimate performance metrics
        projected_performance = {
            'total_content_pieces': total_content_pieces,
            'estimated_total_reach': total_content_pieces * 2500,  # Avg reach per piece
            'estimated_total_engagement': total_content_pieces * 125,  # Avg engagement per piece
            'estimated_conversion_rate': 2.8,
            'estimated_brand_awareness_lift': '18%',
            'confidence_interval': '85%'
        }
        
        # Platform-specific projections
        platform_projections = {}
        for platform, data in platform_content.items():
            content_count = len(data.get('optimized_content', []))
            
            platform_projections[platform] = {
                'content_pieces': content_count,
                'estimated_reach': content_count * self._get_platform_reach_multiplier(platform),
                'estimated_engagement': content_count * self._get_platform_engagement_multiplier(platform),
                'growth_projection': self._calculate_platform_growth_projection(platform)
            }
        
        projected_performance['platform_projections'] = platform_projections
        
        return projected_performance
    
    def _get_platform_reach_multiplier(self, platform: str) -> int:
        """Get reach multiplier for platform"""
        
        multipliers = {
            'instagram': 1800,
            'facebook': 2200,
            'linkedin': 1500,
            'twitter': 3200,
            'youtube': 4500
        }
        
        return multipliers.get(platform, 2000)
    
    def _get_platform_engagement_multiplier(self, platform: str) -> int:
        """Get engagement multiplier for platform"""
        
        multipliers = {
            'instagram': 90,
            'facebook': 70,
            'linkedin': 85,
            'twitter': 150,
            'youtube': 200
        }
        
        return multipliers.get(platform, 100)
    
    def _calculate_platform_growth_projection(self, platform: str) -> str:
        """Calculate growth projection for platform"""
        
        # Simplified growth projections
        projections = {
            'instagram': '15% monthly follower growth',
            'linkedin': '12% monthly connection growth',
            'facebook': '8% monthly page like growth',
            'twitter': '20% monthly follower growth',
            'youtube': '25% monthly subscriber growth'
        }
        
        return projections.get(platform, '10% monthly audience growth')

# Initialize content creator engine
content_engine = OmnichannelContentEngine()

# Routes
@app.route('/omnichannel-content')
def content_dashboard():
    """Omnichannel Content Creator dashboard"""
    
    recent_brands = ContentBrand.query.order_by(ContentBrand.created_at.desc()).limit(10).all()
    
    return render_template('content/dashboard.html',
                         brands=recent_brands)

@app.route('/omnichannel-content/api/comprehensive-strategy', methods=['POST'])
def create_content_strategy():
    """API endpoint for comprehensive content strategy"""
    
    data = request.get_json()
    brand_id = data.get('brand_id')
    period_days = data.get('period_days', 30)
    
    if not brand_id:
        return jsonify({'error': 'Brand ID required'}), 400
    
    strategy = content_engine.generate_comprehensive_content_strategy(brand_id, period_days)
    return jsonify(strategy)

# Initialize database
with app.app_context():
    db.create_all()
    
    # Create sample data
    if ContentBrand.query.count() == 0:
        sample_brand = ContentBrand(
            brand_id='CONTENT_DEMO_001',
            brand_name='Demo Content Brand',
            content_pillars=['innovation', 'education', 'community'],
            content_themes=['technology', 'customer_success', 'industry_insights'],
            publishing_frequency={'daily_posts': 2, 'weekly_videos': 1}
        )
        
        db.session.add(sample_brand)
        db.session.commit()
        logger.info("Sample omnichannel content data created")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5032, debug=True)