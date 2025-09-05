"""
Master Digital Marketing Strategist Agent - Complete AI-Powered Marketing Intelligence
Strategic Marketing Planning, Campaign Optimization & Multi-Channel Performance Management
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
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "marketing-strategist-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///marketing_strategist.db")

db.init_app(app)

# Marketing Strategy Enums
class CampaignType(Enum):
    BRAND_AWARENESS = "brand_awareness"
    LEAD_GENERATION = "lead_generation"
    CONVERSION = "conversion"
    RETENTION = "retention"
    UPSELL = "upsell"

class MarketingChannel(Enum):
    SEO = "seo"
    SEM = "sem"
    SOCIAL_MEDIA = "social_media"
    CONTENT_MARKETING = "content_marketing"
    EMAIL_MARKETING = "email_marketing"
    INFLUENCER = "influencer"
    AFFILIATE = "affiliate"
    DISPLAY = "display"
    VIDEO = "video"

class IndustryVertical(Enum):
    ECOMMERCE = "ecommerce"
    SAAS = "saas"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    ENTERTAINMENT = "entertainment"
    REAL_ESTATE = "real_estate"
    AUTOMOTIVE = "automotive"

# Data Models
class MarketingAccount(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    account_id = db.Column(db.String(100), unique=True, nullable=False)
    company_name = db.Column(db.String(200), nullable=False)
    industry = db.Column(db.Enum(IndustryVertical), nullable=False)
    
    # Business Profile
    monthly_revenue = db.Column(db.Float, default=0.0)
    marketing_budget = db.Column(db.Float, default=10000.0)
    target_market = db.Column(db.JSON)  # Geographic and demographic data
    business_goals = db.Column(db.JSON)  # Primary business objectives
    
    # Current Performance
    current_roas = db.Column(db.Float, default=3.0)
    monthly_leads = db.Column(db.Integer, default=500)
    conversion_rate = db.Column(db.Float, default=2.5)  # percentage
    customer_acquisition_cost = db.Column(db.Float, default=150.0)
    lifetime_value = db.Column(db.Float, default=500.0)
    
    # Competitive Analysis
    main_competitors = db.Column(db.JSON)
    competitive_advantages = db.Column(db.JSON)
    market_share = db.Column(db.Float, default=5.0)  # percentage
    
    # AI Insights
    growth_potential = db.Column(db.Float, default=25.0)  # percentage growth potential
    optimization_opportunities = db.Column(db.JSON)
    predicted_performance = db.Column(db.JSON)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Campaign(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    campaign_id = db.Column(db.String(100), unique=True, nullable=False)
    account_id = db.Column(db.String(100), db.ForeignKey('marketing_account.account_id'), nullable=False)
    
    # Campaign Details
    campaign_name = db.Column(db.String(200), nullable=False)
    campaign_type = db.Column(db.Enum(CampaignType), nullable=False)
    primary_channel = db.Column(db.Enum(MarketingChannel), nullable=False)
    
    # Campaign Strategy
    target_audience = db.Column(db.JSON)  # Detailed audience targeting
    campaign_objectives = db.Column(db.JSON)  # Specific goals and KPIs
    budget_allocation = db.Column(db.JSON)  # Budget distribution across channels
    
    # Timeline
    start_date = db.Column(db.Date, nullable=False)
    end_date = db.Column(db.Date, nullable=False)
    estimated_duration = db.Column(db.Integer, default=30)  # days
    
    # Budget and Performance
    total_budget = db.Column(db.Float, nullable=False)
    spent_budget = db.Column(db.Float, default=0.0)
    target_roas = db.Column(db.Float, default=4.0)
    actual_roas = db.Column(db.Float, default=0.0)
    
    # Performance Metrics
    impressions = db.Column(db.Integer, default=0)
    clicks = db.Column(db.Integer, default=0)
    conversions = db.Column(db.Integer, default=0)
    revenue_generated = db.Column(db.Float, default=0.0)
    
    # Quality Scores
    relevance_score = db.Column(db.Float, default=7.0)  # 1-10 scale
    quality_score = db.Column(db.Float, default=6.5)  # 1-10 scale
    ad_strength = db.Column(db.Float, default=75.0)  # 0-100 percentage
    
    # AI Optimization
    performance_prediction = db.Column(db.JSON)
    optimization_recommendations = db.Column(db.JSON)
    success_probability = db.Column(db.Float, default=75.0)  # percentage
    
    is_active = db.Column(db.Boolean, default=True)

class ChannelPerformance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    performance_id = db.Column(db.String(100), unique=True, nullable=False)
    account_id = db.Column(db.String(100), db.ForeignKey('marketing_account.account_id'), nullable=False)
    channel = db.Column(db.Enum(MarketingChannel), nullable=False)
    
    # Performance Metrics
    measurement_date = db.Column(db.Date, nullable=False)
    spend = db.Column(db.Float, default=0.0)
    revenue = db.Column(db.Float, default=0.0)
    roas = db.Column(db.Float, default=0.0)
    
    # Traffic Metrics
    sessions = db.Column(db.Integer, default=0)
    page_views = db.Column(db.Integer, default=0)
    bounce_rate = db.Column(db.Float, default=50.0)  # percentage
    avg_session_duration = db.Column(db.Float, default=120.0)  # seconds
    
    # Conversion Metrics
    leads = db.Column(db.Integer, default=0)
    conversions = db.Column(db.Integer, default=0)
    conversion_rate = db.Column(db.Float, default=2.0)  # percentage
    cost_per_lead = db.Column(db.Float, default=50.0)
    cost_per_acquisition = db.Column(db.Float, default=200.0)
    
    # Engagement Metrics
    engagement_rate = db.Column(db.Float, default=3.5)  # percentage
    social_shares = db.Column(db.Integer, default=0)
    comments = db.Column(db.Integer, default=0)
    time_on_page = db.Column(db.Float, default=90.0)  # seconds
    
    # AI Analysis
    performance_score = db.Column(db.Float, default=70.0)  # 0-100
    growth_trend = db.Column(db.String(50), default='stable')  # growing, stable, declining
    optimization_potential = db.Column(db.Float, default=20.0)  # percentage improvement

class CompetitorAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.String(100), unique=True, nullable=False)
    account_id = db.Column(db.String(100), db.ForeignKey('marketing_account.account_id'), nullable=False)
    
    # Competitor Information
    competitor_name = db.Column(db.String(200), nullable=False)
    competitor_website = db.Column(db.String(500))
    analysis_date = db.Column(db.Date, nullable=False)
    
    # Marketing Strategy Analysis
    estimated_ad_spend = db.Column(db.Float, default=0.0)
    active_channels = db.Column(db.JSON)  # Channels they're using
    content_strategy = db.Column(db.JSON)  # Content themes and frequency
    
    # Performance Estimation
    estimated_traffic = db.Column(db.Integer, default=0)
    estimated_conversions = db.Column(db.Integer, default=0)
    estimated_revenue = db.Column(db.Float, default=0.0)
    market_share_estimate = db.Column(db.Float, default=0.0)
    
    # Competitive Intelligence
    strengths = db.Column(db.JSON)
    weaknesses = db.Column(db.JSON)
    opportunities = db.Column(db.JSON)  # Opportunities for us
    threats = db.Column(db.JSON)
    
    # AI Insights
    competitive_score = db.Column(db.Float, default=75.0)  # 0-100
    threat_level = db.Column(db.String(50), default='medium')  # low, medium, high
    recommended_actions = db.Column(db.JSON)

class MarketTrend(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    trend_id = db.Column(db.String(100), unique=True, nullable=False)
    industry = db.Column(db.Enum(IndustryVertical), nullable=False)
    
    # Trend Information
    trend_name = db.Column(db.String(200), nullable=False)
    trend_description = db.Column(db.Text)
    trend_category = db.Column(db.String(100))  # technology, behavior, market, etc.
    
    # Trend Metrics
    trend_strength = db.Column(db.Float, default=50.0)  # 0-100
    adoption_rate = db.Column(db.Float, default=25.0)  # percentage
    growth_velocity = db.Column(db.Float, default=5.0)  # percentage per month
    
    # Timeline
    emergence_date = db.Column(db.Date)
    peak_prediction = db.Column(db.Date)
    decline_prediction = db.Column(db.Date)
    
    # Impact Analysis
    market_impact = db.Column(db.Float, default=50.0)  # 0-100
    opportunity_score = db.Column(db.Float, default=60.0)  # 0-100
    implementation_difficulty = db.Column(db.Float, default=40.0)  # 0-100
    
    # AI Predictions
    trend_longevity = db.Column(db.Integer, default=12)  # months
    roi_potential = db.Column(db.Float, default=15.0)  # percentage
    recommended_investment = db.Column(db.Float, default=5000.0)
    
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

# Master Digital Marketing Strategist Engine
class DigitalMarketingStrategistEngine:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    def develop_comprehensive_strategy(self, account_id: str) -> Dict[str, Any]:
        """AI-powered comprehensive marketing strategy development"""
        
        # Get account data
        account = MarketingAccount.query.filter_by(account_id=account_id).first()
        if not account:
            return {'error': 'Account not found'}
        
        # Analyze current performance
        current_performance = self._analyze_current_performance(account_id)
        
        # Competitive analysis
        competitive_landscape = self._analyze_competitive_landscape(account_id)
        
        # Market trend analysis
        market_opportunities = self._analyze_market_trends(account.industry)
        
        # Channel optimization
        channel_strategy = self._optimize_channel_mix(account_id, account)
        
        # Budget allocation optimization
        budget_optimization = self._optimize_budget_allocation(account, channel_strategy)
        
        # Performance predictions
        performance_forecast = self._forecast_performance(account, channel_strategy, budget_optimization)
        
        return {
            'account_id': account_id,
            'strategy_date': datetime.utcnow().isoformat(),
            'current_performance_analysis': current_performance,
            'competitive_landscape': competitive_landscape,
            'market_opportunities': market_opportunities,
            'channel_strategy': channel_strategy,
            'budget_optimization': budget_optimization,
            'performance_forecast': performance_forecast,
            'strategic_recommendations': self._generate_strategic_recommendations(
                current_performance, competitive_landscape, market_opportunities
            )
        }
    
    def _analyze_current_performance(self, account_id: str) -> Dict[str, Any]:
        """Analyze current marketing performance across all channels"""
        
        # Get recent performance data
        recent_performance = ChannelPerformance.query.filter_by(account_id=account_id)\
                                                    .filter(ChannelPerformance.measurement_date >= datetime.now().date() - timedelta(days=90))\
                                                    .all()
        
        if not recent_performance:
            return {'status': 'insufficient_data'}
        
        # Calculate overall metrics
        total_spend = sum(p.spend for p in recent_performance)
        total_revenue = sum(p.revenue for p in recent_performance)
        overall_roas = total_revenue / total_spend if total_spend > 0 else 0
        
        # Channel performance analysis
        channel_performance = {}
        for channel in MarketingChannel:
            channel_data = [p for p in recent_performance if p.channel == channel]
            if channel_data:
                channel_spend = sum(p.spend for p in channel_data)
                channel_revenue = sum(p.revenue for p in channel_data)
                channel_roas = channel_revenue / channel_spend if channel_spend > 0 else 0
                
                channel_performance[channel.value] = {
                    'spend': channel_spend,
                    'revenue': channel_revenue,
                    'roas': channel_roas,
                    'performance_score': np.mean([p.performance_score for p in channel_data]),
                    'growth_trend': self._calculate_growth_trend([p.revenue for p in channel_data])
                }
        
        # Performance benchmarking
        industry_benchmarks = self._get_industry_benchmarks()
        performance_vs_benchmark = self._compare_to_benchmarks(channel_performance, industry_benchmarks)
        
        return {
            'overall_metrics': {
                'total_spend': total_spend,
                'total_revenue': total_revenue,
                'overall_roas': overall_roas,
                'performance_period': '90_days'
            },
            'channel_performance': channel_performance,
            'performance_vs_benchmark': performance_vs_benchmark,
            'key_insights': self._generate_performance_insights(channel_performance, overall_roas)
        }
    
    def _calculate_growth_trend(self, revenue_data: List[float]) -> str:
        """Calculate growth trend from revenue data"""
        
        if len(revenue_data) < 2:
            return 'insufficient_data'
        
        # Simple linear trend calculation
        x = np.arange(len(revenue_data))
        z = np.polyfit(x, revenue_data, 1)
        slope = z[0]
        
        if slope > 0.1:
            return 'growing'
        elif slope < -0.1:
            return 'declining'
        else:
            return 'stable'
    
    def _get_industry_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Get industry benchmark data for performance comparison"""
        
        # Industry benchmarks (simplified - would come from real data sources)
        return {
            'seo': {'roas': 5.2, 'conversion_rate': 3.1, 'cpc': 1.2},
            'sem': {'roas': 4.8, 'conversion_rate': 2.8, 'cpc': 2.5},
            'social_media': {'roas': 3.5, 'conversion_rate': 1.9, 'cpc': 1.8},
            'email_marketing': {'roas': 7.2, 'conversion_rate': 4.5, 'cpc': 0.1},
            'content_marketing': {'roas': 4.1, 'conversion_rate': 2.2, 'cpc': 0.8}
        }
    
    def _compare_to_benchmarks(self, performance: Dict, benchmarks: Dict) -> Dict[str, Any]:
        """Compare performance to industry benchmarks"""
        
        comparison = {}
        for channel, metrics in performance.items():
            if channel in benchmarks:
                benchmark = benchmarks[channel]
                comparison[channel] = {
                    'roas_vs_benchmark': (metrics['roas'] / benchmark['roas']) * 100 if benchmark['roas'] > 0 else 0,
                    'performance_rating': self._rate_performance(metrics['roas'], benchmark['roas']),
                    'improvement_potential': max(0, benchmark['roas'] - metrics['roas'])
                }
        
        return comparison
    
    def _rate_performance(self, actual: float, benchmark: float) -> str:
        """Rate performance compared to benchmark"""
        
        ratio = actual / benchmark if benchmark > 0 else 0
        
        if ratio >= 1.2:
            return 'excellent'
        elif ratio >= 1.0:
            return 'good'
        elif ratio >= 0.8:
            return 'average'
        else:
            return 'below_average'
    
    def _generate_performance_insights(self, channel_performance: Dict, overall_roas: float) -> List[str]:
        """Generate actionable performance insights"""
        
        insights = []
        
        # Overall performance insight
        if overall_roas > 4.0:
            insights.append("Strong overall ROAS indicates effective marketing mix")
        elif overall_roas < 2.0:
            insights.append("Low ROAS suggests need for campaign optimization and budget reallocation")
        
        # Channel-specific insights
        best_channel = max(channel_performance.items(), key=lambda x: x[1]['roas']) if channel_performance else None
        worst_channel = min(channel_performance.items(), key=lambda x: x[1]['roas']) if channel_performance else None
        
        if best_channel:
            insights.append(f"{best_channel[0].title()} is your top performing channel with {best_channel[1]['roas']:.1f} ROAS")
        
        if worst_channel and worst_channel[1]['roas'] < 2.0:
            insights.append(f"{worst_channel[0].title()} underperforming - consider optimization or budget reallocation")
        
        # Growth trends
        growing_channels = [ch for ch, data in channel_performance.items() if data['growth_trend'] == 'growing']
        if growing_channels:
            insights.append(f"Growing channels ({', '.join(growing_channels)}) present scaling opportunities")
        
        return insights
    
    def _analyze_competitive_landscape(self, account_id: str) -> Dict[str, Any]:
        """Analyze competitive landscape and positioning"""
        
        # Get competitor analysis data
        competitor_analyses = CompetitorAnalysis.query.filter_by(account_id=account_id)\
                                                     .filter(CompetitorAnalysis.analysis_date >= datetime.now().date() - timedelta(days=180))\
                                                     .all()
        
        if not competitor_analyses:
            return {'status': 'no_competitor_data'}
        
        # Competitive positioning analysis
        competitive_metrics = {
            'total_competitors_analyzed': len(competitor_analyses),
            'average_competitor_ad_spend': np.mean([c.estimated_ad_spend for c in competitor_analyses]),
            'market_saturation': self._calculate_market_saturation(competitor_analyses),
            'competitive_threats': [c for c in competitor_analyses if c.threat_level == 'high']
        }
        
        # Opportunity identification
        market_gaps = self._identify_market_gaps(competitor_analyses)
        
        # Competitive advantages
        our_advantages = self._identify_competitive_advantages(account_id, competitor_analyses)
        
        return {
            'competitive_metrics': competitive_metrics,
            'market_gaps': market_gaps,
            'competitive_advantages': our_advantages,
            'strategic_positioning': self._recommend_positioning_strategy(competitor_analyses)
        }
    
    def _calculate_market_saturation(self, competitors: List[CompetitorAnalysis]) -> float:
        """Calculate market saturation level"""
        
        total_estimated_spend = sum(c.estimated_ad_spend for c in competitors)
        
        # Simplified saturation calculation
        if total_estimated_spend > 1000000:  # $1M+ monthly spend
            return 85.0  # High saturation
        elif total_estimated_spend > 500000:  # $500K+ monthly spend
            return 65.0  # Medium saturation
        else:
            return 35.0  # Low saturation
    
    def _identify_market_gaps(self, competitors: List[CompetitorAnalysis]) -> List[Dict[str, Any]]:
        """Identify gaps in the competitive landscape"""
        
        gaps = []
        
        # Channel gap analysis
        all_channels = set()
        for competitor in competitors:
            all_channels.update(competitor.active_channels or [])
        
        underutilized_channels = []
        for channel in MarketingChannel:
            channel_usage = sum(1 for c in competitors if channel.value in (c.active_channels or []))
            if channel_usage < len(competitors) * 0.5:  # Less than 50% using this channel
                underutilized_channels.append(channel.value)
        
        if underutilized_channels:
            gaps.append({
                'type': 'channel_opportunity',
                'channels': underutilized_channels,
                'opportunity_score': 75.0
            })
        
        # Content strategy gaps
        content_themes = []
        for competitor in competitors:
            content_strategy = competitor.content_strategy or {}
            content_themes.extend(content_strategy.get('themes', []))
        
        # Identify underrepresented content themes
        theme_counts = {}
        for theme in content_themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        underrepresented_themes = [theme for theme, count in theme_counts.items() if count < 2]
        
        if underrepresented_themes:
            gaps.append({
                'type': 'content_opportunity',
                'themes': underrepresented_themes[:5],  # Top 5 opportunities
                'opportunity_score': 60.0
            })
        
        return gaps
    
    def _identify_competitive_advantages(self, account_id: str, competitors: List[CompetitorAnalysis]) -> List[Dict[str, Any]]:
        """Identify our competitive advantages"""
        
        advantages = []
        
        # Get our account data
        account = MarketingAccount.query.filter_by(account_id=account_id).first()
        if not account:
            return advantages
        
        # ROAS advantage
        avg_competitor_roas = 3.0  # Estimated average
        if account.current_roas > avg_competitor_roas * 1.2:
            advantages.append({
                'type': 'performance_advantage',
                'description': 'Superior ROAS performance vs competitors',
                'strength_score': 85.0
            })
        
        # Market positioning advantages
        if hasattr(account, 'competitive_advantages') and account.competitive_advantages:
            for advantage in account.competitive_advantages:
                advantages.append({
                    'type': 'positioning_advantage',
                    'description': advantage,
                    'strength_score': 75.0
                })
        
        return advantages
    
    def _recommend_positioning_strategy(self, competitors: List[CompetitorAnalysis]) -> Dict[str, Any]:
        """Recommend competitive positioning strategy"""
        
        # Analyze competitor positioning
        high_spend_competitors = [c for c in competitors if c.estimated_ad_spend > 50000]
        
        if len(high_spend_competitors) >= 3:
            strategy = 'differentiation'
            recommendation = 'Focus on unique value propositions and niche targeting'
        elif len(competitors) < 3:
            strategy = 'market_leadership'
            recommendation = 'Aggressive expansion to capture market share'
        else:
            strategy = 'selective_competition'
            recommendation = 'Target specific segments where competitors are weak'
        
        return {
            'recommended_strategy': strategy,
            'strategy_description': recommendation,
            'implementation_priority': 'high',
            'expected_timeframe': '3-6 months'
        }
    
    def _analyze_market_trends(self, industry: IndustryVertical) -> Dict[str, Any]:
        """Analyze market trends and opportunities"""
        
        # Get recent market trends for the industry
        trends = MarketTrend.query.filter_by(industry=industry)\
                                 .filter(MarketTrend.last_updated >= datetime.now() - timedelta(days=30))\
                                 .order_by(MarketTrend.opportunity_score.desc())\
                                 .limit(10).all()
        
        if not trends:
            return {'status': 'no_trend_data'}
        
        # Categorize trends by impact and timeline
        high_impact_trends = [t for t in trends if t.opportunity_score > 70]
        emerging_trends = [t for t in trends if t.emergence_date and t.emergence_date > datetime.now().date() - timedelta(days=90)]
        
        # Investment recommendations
        recommended_investments = []
        for trend in high_impact_trends[:3]:  # Top 3 opportunities
            recommended_investments.append({
                'trend_name': trend.trend_name,
                'investment_amount': trend.recommended_investment,
                'expected_roi': trend.roi_potential,
                'implementation_difficulty': trend.implementation_difficulty,
                'timeline': f"{trend.trend_longevity} months"
            })
        
        return {
            'total_trends_analyzed': len(trends),
            'high_impact_opportunities': len(high_impact_trends),
            'emerging_trends': len(emerging_trends),
            'recommended_investments': recommended_investments,
            'market_outlook': self._generate_market_outlook(trends)
        }
    
    def _generate_market_outlook(self, trends: List[MarketTrend]) -> Dict[str, Any]:
        """Generate market outlook based on trends"""
        
        avg_growth_velocity = np.mean([t.growth_velocity for t in trends])
        avg_opportunity_score = np.mean([t.opportunity_score for t in trends])
        
        if avg_growth_velocity > 8 and avg_opportunity_score > 70:
            outlook = 'very_positive'
        elif avg_growth_velocity > 5 and avg_opportunity_score > 60:
            outlook = 'positive'
        elif avg_growth_velocity > 2:
            outlook = 'stable'
        else:
            outlook = 'cautious'
        
        return {
            'overall_outlook': outlook,
            'growth_potential': avg_growth_velocity,
            'opportunity_level': avg_opportunity_score,
            'recommended_approach': self._get_approach_recommendation(outlook)
        }
    
    def _get_approach_recommendation(self, outlook: str) -> str:
        """Get strategic approach recommendation based on outlook"""
        
        recommendations = {
            'very_positive': 'Aggressive growth strategy with increased investment',
            'positive': 'Moderate expansion with focus on high-ROI opportunities',
            'stable': 'Steady optimization with selective new initiatives',
            'cautious': 'Conservative approach with emphasis on proven channels'
        }
        
        return recommendations.get(outlook, 'Balanced approach with regular performance monitoring')
    
    def _optimize_channel_mix(self, account_id: str, account: MarketingAccount) -> Dict[str, Any]:
        """Optimize marketing channel mix based on performance and potential"""
        
        # Get current channel performance
        current_performance = ChannelPerformance.query.filter_by(account_id=account_id)\
                                                     .filter(ChannelPerformance.measurement_date >= datetime.now().date() - timedelta(days=60))\
                                                     .all()
        
        # Calculate channel efficiency scores
        channel_scores = {}
        for channel in MarketingChannel:
            channel_data = [p for p in current_performance if p.channel == channel]
            if channel_data:
                avg_roas = np.mean([p.roas for p in channel_data])
                avg_performance_score = np.mean([p.performance_score for p in channel_data])
                optimization_potential = np.mean([p.optimization_potential for p in channel_data])
                
                # Calculate composite efficiency score
                efficiency_score = (avg_roas * 0.4 + avg_performance_score * 0.4 + (100 - optimization_potential) * 0.2)
                
                channel_scores[channel.value] = {
                    'efficiency_score': efficiency_score,
                    'current_roas': avg_roas,
                    'optimization_potential': optimization_potential,
                    'recommendation': self._get_channel_recommendation(efficiency_score, optimization_potential)
                }
        
        # Identify optimal channel mix
        optimal_mix = self._calculate_optimal_channel_mix(channel_scores, account.marketing_budget)
        
        return {
            'channel_efficiency_analysis': channel_scores,
            'optimal_channel_mix': optimal_mix,
            'reallocation_recommendations': self._generate_reallocation_recommendations(channel_scores)
        }
    
    def _get_channel_recommendation(self, efficiency_score: float, optimization_potential: float) -> str:
        """Get recommendation for individual channel"""
        
        if efficiency_score > 80:
            return 'scale_up' if optimization_potential < 20 else 'optimize_and_scale'
        elif efficiency_score > 60:
            return 'optimize' if optimization_potential > 30 else 'maintain'
        else:
            return 'reduce_or_pause' if optimization_potential < 40 else 'major_optimization_required'
    
    def _calculate_optimal_channel_mix(self, channel_scores: Dict, total_budget: float) -> Dict[str, Any]:
        """Calculate optimal budget allocation across channels"""
        
        # Sort channels by efficiency score
        sorted_channels = sorted(channel_scores.items(), key=lambda x: x[1]['efficiency_score'], reverse=True)
        
        # Allocate budget based on efficiency scores with constraints
        optimal_allocation = {}
        remaining_budget = total_budget
        
        for i, (channel, data) in enumerate(sorted_channels):
            if data['efficiency_score'] > 70:  # High performing channels
                allocation_percentage = 0.3 - (i * 0.05)  # Decreasing allocation
            elif data['efficiency_score'] > 50:  # Medium performing channels
                allocation_percentage = 0.15 - (i * 0.02)
            else:  # Low performing channels
                allocation_percentage = 0.05
            
            allocation_amount = min(remaining_budget * allocation_percentage, remaining_budget)
            optimal_allocation[channel] = {
                'budget_allocation': allocation_amount,
                'percentage': (allocation_amount / total_budget) * 100,
                'expected_roas': data['current_roas'] * 1.1  # Expected improvement
            }
            remaining_budget -= allocation_amount
        
        return optimal_allocation
    
    def _generate_reallocation_recommendations(self, channel_scores: Dict) -> List[Dict[str, Any]]:
        """Generate specific budget reallocation recommendations"""
        
        recommendations = []
        
        # Identify high and low performers
        high_performers = {k: v for k, v in channel_scores.items() if v['efficiency_score'] > 75}
        low_performers = {k: v for k, v in channel_scores.items() if v['efficiency_score'] < 50}
        
        # Recommendations for scaling up high performers
        for channel, data in high_performers.items():
            recommendations.append({
                'action': 'increase_budget',
                'channel': channel,
                'current_score': data['efficiency_score'],
                'recommended_increase': '25-50%',
                'expected_impact': 'Significant ROAS improvement'
            })
        
        # Recommendations for optimizing or reducing low performers
        for channel, data in low_performers.items():
            if data['optimization_potential'] > 40:
                recommendations.append({
                    'action': 'optimize_campaign',
                    'channel': channel,
                    'current_score': data['efficiency_score'],
                    'optimization_focus': 'Targeting and creative optimization',
                    'expected_impact': 'Improved efficiency'
                })
            else:
                recommendations.append({
                    'action': 'reduce_budget',
                    'channel': channel,
                    'current_score': data['efficiency_score'],
                    'recommended_reduction': '50-75%',
                    'expected_impact': 'Reallocate to higher performing channels'
                })
        
        return recommendations
    
    def _optimize_budget_allocation(self, account: MarketingAccount, channel_strategy: Dict) -> Dict[str, Any]:
        """Optimize budget allocation based on channel strategy and business goals"""
        
        total_budget = account.marketing_budget
        optimal_allocation = channel_strategy.get('optimal_channel_mix', {})
        
        # Calculate expected performance with new allocation
        expected_performance = {
            'total_budget': total_budget,
            'expected_revenue': 0,
            'expected_roas': 0,
            'channel_breakdown': {}
        }
        
        total_expected_revenue = 0
        for channel, allocation_data in optimal_allocation.items():
            budget = allocation_data['budget_allocation']
            expected_roas = allocation_data['expected_roas']
            expected_revenue = budget * expected_roas
            
            expected_performance['channel_breakdown'][channel] = {
                'budget': budget,
                'expected_revenue': expected_revenue,
                'expected_roas': expected_roas
            }
            
            total_expected_revenue += expected_revenue
        
        expected_performance['expected_revenue'] = total_expected_revenue
        expected_performance['expected_roas'] = total_expected_revenue / total_budget if total_budget > 0 else 0
        
        # Budget optimization recommendations
        optimization_recommendations = self._generate_budget_optimization_recommendations(
            account, expected_performance
        )
        
        return {
            'current_budget': total_budget,
            'optimal_allocation': optimal_allocation,
            'expected_performance': expected_performance,
            'optimization_recommendations': optimization_recommendations,
            'roi_improvement': self._calculate_roi_improvement(account.current_roas, expected_performance['expected_roas'])
        }
    
    def _generate_budget_optimization_recommendations(self, account: MarketingAccount, expected_performance: Dict) -> List[Dict[str, Any]]:
        """Generate budget optimization recommendations"""
        
        recommendations = []
        
        # Budget increase recommendation
        current_revenue = account.monthly_revenue
        expected_revenue_increase = expected_performance['expected_revenue'] - (account.marketing_budget * account.current_roas)
        
        if expected_revenue_increase > account.marketing_budget * 0.5:  # If 50%+ budget could be recovered
            recommendations.append({
                'type': 'budget_increase',
                'recommendation': 'Consider increasing marketing budget by 25-50%',
                'reasoning': 'High ROI potential identified across multiple channels',
                'expected_impact': f'Additional ${expected_revenue_increase:,.0f} monthly revenue'
            })
        
        # Seasonal budget allocation
        recommendations.append({
            'type': 'seasonal_optimization',
            'recommendation': 'Implement seasonal budget adjustments',
            'reasoning': 'Optimize spending based on seasonal performance patterns',
            'expected_impact': '15-25% efficiency improvement'
        })
        
        # Testing budget recommendation
        testing_budget = account.marketing_budget * 0.15  # 15% for testing
        recommendations.append({
            'type': 'testing_allocation',
            'recommendation': f'Allocate ${testing_budget:,.0f} for testing new channels/strategies',
            'reasoning': 'Continuous testing drives long-term growth',
            'expected_impact': 'Discovery of new high-ROI opportunities'
        })
        
        return recommendations
    
    def _calculate_roi_improvement(self, current_roas: float, expected_roas: float) -> Dict[str, Any]:
        """Calculate ROI improvement from optimization"""
        
        improvement_percentage = ((expected_roas - current_roas) / current_roas) * 100 if current_roas > 0 else 0
        
        return {
            'current_roas': current_roas,
            'expected_roas': expected_roas,
            'improvement_percentage': improvement_percentage,
            'improvement_category': 'significant' if improvement_percentage > 25 else 'moderate' if improvement_percentage > 10 else 'minor'
        }
    
    def _forecast_performance(self, account: MarketingAccount, channel_strategy: Dict, budget_optimization: Dict) -> Dict[str, Any]:
        """Forecast performance based on strategic recommendations"""
        
        # Get expected performance from budget optimization
        expected_performance = budget_optimization['expected_performance']
        
        # Generate 12-month forecast
        monthly_forecast = []
        base_revenue = expected_performance['expected_revenue']
        
        for month in range(1, 13):
            # Apply growth curve and seasonal factors
            growth_factor = 1 + (month * 0.02)  # 2% monthly growth
            seasonal_factor = self._get_seasonal_factor(month, account.industry)
            
            forecast_revenue = base_revenue * growth_factor * seasonal_factor
            forecast_spend = account.marketing_budget
            forecast_roas = forecast_revenue / forecast_spend if forecast_spend > 0 else 0
            
            monthly_forecast.append({
                'month': month,
                'expected_revenue': forecast_revenue,
                'expected_spend': forecast_spend,
                'expected_roas': forecast_roas,
                'confidence_interval': 0.85  # 85% confidence
            })
        
        # Calculate annual projections
        annual_revenue = sum(f['expected_revenue'] for f in monthly_forecast)
        annual_spend = sum(f['expected_spend'] for f in monthly_forecast)
        annual_roas = annual_revenue / annual_spend if annual_spend > 0 else 0
        
        return {
            'monthly_forecast': monthly_forecast,
            'annual_projections': {
                'total_revenue': annual_revenue,
                'total_spend': annual_spend,
                'average_roas': annual_roas,
                'revenue_growth': ((annual_revenue - (account.monthly_revenue * 12)) / (account.monthly_revenue * 12)) * 100 if account.monthly_revenue > 0 else 0
            },
            'key_milestones': self._identify_key_milestones(monthly_forecast),
            'risk_factors': self._identify_risk_factors(account, channel_strategy)
        }
    
    def _get_seasonal_factor(self, month: int, industry: IndustryVertical) -> float:
        """Get seasonal adjustment factor for specific month and industry"""
        
        # Simplified seasonal factors by industry
        seasonal_patterns = {
            IndustryVertical.ECOMMERCE: {
                11: 1.4, 12: 1.6, 1: 0.8, 2: 0.9, 7: 1.1, 8: 1.1  # Holiday season boost
            },
            IndustryVertical.FINANCE: {
                1: 1.2, 4: 1.1, 7: 0.9, 12: 1.1  # Tax season and year-end
            },
            IndustryVertical.EDUCATION: {
                8: 1.3, 9: 1.2, 1: 1.1, 6: 0.8, 7: 0.7  # Back to school seasons
            }
        }
        
        pattern = seasonal_patterns.get(industry, {})
        return pattern.get(month, 1.0)  # Default to 1.0 (no adjustment)
    
    def _identify_key_milestones(self, monthly_forecast: List[Dict]) -> List[Dict[str, Any]]:
        """Identify key performance milestones"""
        
        milestones = []
        
        # ROAS milestone
        for i, month_data in enumerate(monthly_forecast, 1):
            if month_data['expected_roas'] > 5.0 and i <= 6:  # High ROAS within 6 months
                milestones.append({
                    'month': i,
                    'type': 'roas_achievement',
                    'description': f'Target ROAS of {month_data["expected_roas"]:.1f} achieved',
                    'significance': 'high'
                })
                break
        
        # Revenue milestone
        cumulative_revenue = 0
        for i, month_data in enumerate(monthly_forecast, 1):
            cumulative_revenue += month_data['expected_revenue']
            if cumulative_revenue > 1000000 and i <= 12:  # $1M revenue milestone
                milestones.append({
                    'month': i,
                    'type': 'revenue_milestone',
                    'description': f'$1M cumulative revenue reached',
                    'significance': 'high'
                })
                break
        
        return milestones
    
    def _identify_risk_factors(self, account: MarketingAccount, channel_strategy: Dict) -> List[Dict[str, Any]]:
        """Identify potential risk factors"""
        
        risks = []
        
        # Budget concentration risk
        optimal_mix = channel_strategy.get('optimal_channel_mix', {})
        if optimal_mix:
            max_allocation = max(data['percentage'] for data in optimal_mix.values())
            if max_allocation > 60:  # More than 60% in one channel
                risks.append({
                    'type': 'concentration_risk',
                    'severity': 'medium',
                    'description': 'High budget concentration in single channel',
                    'mitigation': 'Diversify across multiple channels'
                })
        
        # Market saturation risk
        if account.market_share > 30:  # High market share
            risks.append({
                'type': 'market_saturation',
                'severity': 'medium',
                'description': 'Potential market saturation limiting growth',
                'mitigation': 'Explore new markets or product extensions'
            })
        
        # Competitive pressure risk
        risks.append({
            'type': 'competitive_pressure',
            'severity': 'low',
            'description': 'Increased competition may impact performance',
            'mitigation': 'Continuous optimization and differentiation'
        })
        
        return risks
    
    def _generate_strategic_recommendations(self, current_performance: Dict, competitive_landscape: Dict, market_opportunities: Dict) -> List[Dict[str, Any]]:
        """Generate comprehensive strategic recommendations"""
        
        recommendations = []
        
        # Performance-based recommendations
        if current_performance.get('overall_metrics', {}).get('overall_roas', 0) < 3.0:
            recommendations.append({
                'category': 'performance_optimization',
                'priority': 'high',
                'recommendation': 'Immediate campaign optimization required to improve ROAS',
                'expected_impact': '25-50% ROAS improvement',
                'timeline': '30-60 days'
            })
        
        # Competitive recommendations
        market_gaps = competitive_landscape.get('market_gaps', [])
        if market_gaps:
            recommendations.append({
                'category': 'competitive_advantage',
                'priority': 'medium',
                'recommendation': 'Exploit identified market gaps for competitive advantage',
                'expected_impact': '15-30% market share increase',
                'timeline': '3-6 months'
            })
        
        # Market opportunity recommendations
        recommended_investments = market_opportunities.get('recommended_investments', [])
        if recommended_investments:
            top_opportunity = recommended_investments[0]
            recommendations.append({
                'category': 'market_opportunity',
                'priority': 'medium',
                'recommendation': f'Invest in {top_opportunity["trend_name"]} trend',
                'expected_impact': f'{top_opportunity["expected_roi"]}% ROI',
                'timeline': top_opportunity['timeline']
            })
        
        # Strategic growth recommendations
        recommendations.append({
            'category': 'strategic_growth',
            'priority': 'high',
            'recommendation': 'Implement comprehensive attribution modeling',
            'expected_impact': '20-40% improved budget allocation efficiency',
            'timeline': '2-3 months'
        })
        
        return recommendations

# Initialize strategist engine
strategist_engine = DigitalMarketingStrategistEngine()

# Routes
@app.route('/digital-marketing-strategist')
def strategist_dashboard():
    """Digital Marketing Strategist dashboard"""
    
    recent_accounts = MarketingAccount.query.order_by(MarketingAccount.created_at.desc()).limit(10).all()
    
    return render_template('digital_marketing/strategist_dashboard.html',
                         accounts=recent_accounts)

@app.route('/digital-marketing-strategist/api/comprehensive-strategy', methods=['POST'])
def develop_strategy():
    """API endpoint for comprehensive strategy development"""
    
    data = request.get_json()
    account_id = data.get('account_id')
    
    if not account_id:
        return jsonify({'error': 'Account ID required'}), 400
    
    strategy = strategist_engine.develop_comprehensive_strategy(account_id)
    return jsonify(strategy)

# Initialize database
with app.app_context():
    db.create_all()
    
    # Create sample data
    if MarketingAccount.query.count() == 0:
        sample_account = MarketingAccount(
            account_id='MKT_DEMO_001',
            company_name='Demo Marketing Company',
            industry=IndustryVertical.ECOMMERCE,
            monthly_revenue=250000,
            marketing_budget=50000,
            current_roas=3.2,
            monthly_leads=1200,
            conversion_rate=3.1
        )
        
        db.session.add(sample_account)
        db.session.commit()
        logger.info("Sample digital marketing strategist data created")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5030, debug=True)