"""
Advanced Media Buying Agent - AI-Powered Performance Marketing & ROAS Optimization
Real-Time Bid Management, Cross-Platform Campaign Optimization & Advanced Attribution
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
app.secret_key = os.environ.get("SESSION_SECRET", "media-buying-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///media_buying.db")

db.init_app(app)

# Media Buying Enums
class AdPlatform(Enum):
    GOOGLE_ADS = "google_ads"
    FACEBOOK_ADS = "facebook_ads"
    INSTAGRAM_ADS = "instagram_ads"
    LINKEDIN_ADS = "linkedin_ads"
    TWITTER_ADS = "twitter_ads"
    YOUTUBE_ADS = "youtube_ads"
    TIKTOK_ADS = "tiktok_ads"
    SNAPCHAT_ADS = "snapchat_ads"

class CampaignObjective(Enum):
    AWARENESS = "awareness"
    REACH = "reach"
    TRAFFIC = "traffic"
    ENGAGEMENT = "engagement"
    LEADS = "leads"
    CONVERSIONS = "conversions"
    SALES = "sales"
    RETENTION = "retention"

class BiddingStrategy(Enum):
    MAXIMIZE_CLICKS = "maximize_clicks"
    MAXIMIZE_CONVERSIONS = "maximize_conversions"
    TARGET_CPA = "target_cpa"
    TARGET_ROAS = "target_roas"
    MANUAL_CPC = "manual_cpc"
    ENHANCED_CPC = "enhanced_cpc"
    MAXIMIZE_CONVERSION_VALUE = "maximize_conversion_value"

# Data Models
class MediaBuyingAccount(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    account_id = db.Column(db.String(100), unique=True, nullable=False)
    account_name = db.Column(db.String(200), nullable=False)
    
    # Account Configuration
    industry_vertical = db.Column(db.String(100))
    target_markets = db.Column(db.JSON)
    business_model = db.Column(db.String(50))
    
    # Budget Management
    monthly_budget = db.Column(db.Float, nullable=False)
    daily_budget_cap = db.Column(db.Float)
    emergency_budget_stop = db.Column(db.Float)
    
    # Performance Targets
    target_roas = db.Column(db.Float, default=4.0)
    target_cpa = db.Column(db.Float, default=50.0)
    target_ctr = db.Column(db.Float, default=2.0)
    target_conversion_rate = db.Column(db.Float, default=3.0)
    
    # Risk Management
    risk_tolerance = db.Column(db.String(20), default='medium')
    max_daily_loss = db.Column(db.Float)
    auto_pause_threshold = db.Column(db.Float, default=0.5)  # ROAS threshold
    
    # Attribution Settings
    attribution_model = db.Column(db.String(50), default='data_driven')
    conversion_window = db.Column(db.Integer, default=30)  # days
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Campaign(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    campaign_id = db.Column(db.String(100), unique=True, nullable=False)
    account_id = db.Column(db.String(100), db.ForeignKey('media_buying_account.account_id'), nullable=False)
    
    # Campaign Details
    campaign_name = db.Column(db.String(200), nullable=False)
    platform = db.Column(db.Enum(AdPlatform), nullable=False)
    objective = db.Column(db.Enum(CampaignObjective), nullable=False)
    bidding_strategy = db.Column(db.Enum(BiddingStrategy), nullable=False)
    
    # Campaign Settings
    target_audience = db.Column(db.JSON)
    geographic_targets = db.Column(db.JSON)
    demographic_targets = db.Column(db.JSON)
    interest_targets = db.Column(db.JSON)
    
    # Budget and Bidding
    daily_budget = db.Column(db.Float, nullable=False)
    max_bid = db.Column(db.Float)
    target_bid = db.Column(db.Float)
    bid_adjustments = db.Column(db.JSON)
    
    # Creative Assets
    ad_creatives = db.Column(db.JSON)
    landing_pages = db.Column(db.JSON)
    tracking_urls = db.Column(db.JSON)
    
    # Performance Metrics
    impressions = db.Column(db.Integer, default=0)
    clicks = db.Column(db.Integer, default=0)
    conversions = db.Column(db.Integer, default=0)
    cost = db.Column(db.Float, default=0.0)
    revenue = db.Column(db.Float, default=0.0)
    
    # Calculated Metrics
    ctr = db.Column(db.Float, default=0.0)
    cpc = db.Column(db.Float, default=0.0)
    cpa = db.Column(db.Float, default=0.0)
    roas = db.Column(db.Float, default=0.0)
    
    # AI Optimization
    optimization_score = db.Column(db.Float, default=0.0)
    ai_recommendations = db.Column(db.JSON)
    auto_optimization_enabled = db.Column(db.Boolean, default=True)
    
    # Campaign Status
    status = db.Column(db.String(20), default='active')
    start_date = db.Column(db.Date, nullable=False)
    end_date = db.Column(db.Date)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_optimized = db.Column(db.DateTime, default=datetime.utcnow)

class BidOptimization(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    optimization_id = db.Column(db.String(100), unique=True, nullable=False)
    campaign_id = db.Column(db.String(100), db.ForeignKey('campaign.campaign_id'), nullable=False)
    
    # Optimization Details
    optimization_type = db.Column(db.String(50), nullable=False)
    trigger_condition = db.Column(db.String(100))
    
    # Before/After Metrics
    before_metrics = db.Column(db.JSON)
    after_metrics = db.Column(db.JSON)
    
    # Optimization Actions
    bid_changes = db.Column(db.JSON)
    audience_adjustments = db.Column(db.JSON)
    budget_reallocations = db.Column(db.JSON)
    
    # Performance Impact
    performance_impact = db.Column(db.JSON)
    confidence_score = db.Column(db.Float, default=0.0)
    expected_improvement = db.Column(db.Float, default=0.0)
    actual_improvement = db.Column(db.Float, default=0.0)
    
    # AI Decision Making
    ai_reasoning = db.Column(db.Text)
    optimization_algorithm = db.Column(db.String(50))
    data_sources = db.Column(db.JSON)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    implemented_at = db.Column(db.DateTime)

class PerformanceAttribution(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    attribution_id = db.Column(db.String(100), unique=True, nullable=False)
    account_id = db.Column(db.String(100), db.ForeignKey('media_buying_account.account_id'), nullable=False)
    
    # Attribution Period
    attribution_date = db.Column(db.Date, nullable=False)
    attribution_window = db.Column(db.Integer, default=30)
    
    # Channel Attribution
    channel_attribution = db.Column(db.JSON)  # Attribution by channel
    touchpoint_analysis = db.Column(db.JSON)  # Customer journey touchpoints
    conversion_paths = db.Column(db.JSON)     # Full conversion paths
    
    # Cross-Platform Attribution
    cross_platform_impact = db.Column(db.JSON)
    platform_synergies = db.Column(db.JSON)
    incremental_lift = db.Column(db.JSON)
    
    # Advanced Attribution Models
    first_touch_attribution = db.Column(db.JSON)
    last_touch_attribution = db.Column(db.JSON)
    linear_attribution = db.Column(db.JSON)
    time_decay_attribution = db.Column(db.JSON)
    data_driven_attribution = db.Column(db.JSON)
    
    # Attribution Insights
    top_converting_paths = db.Column(db.JSON)
    undervalued_touchpoints = db.Column(db.JSON)
    optimization_opportunities = db.Column(db.JSON)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Advanced Media Buying Engine
class AdvancedMediaBuyingEngine:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    def generate_comprehensive_media_strategy(self, account_id: str) -> Dict[str, Any]:
        """Generate comprehensive AI-powered media buying strategy"""
        
        account = MediaBuyingAccount.query.filter_by(account_id=account_id).first()
        if not account:
            return {'error': 'Account not found'}
        
        # Performance analysis
        performance_analysis = self._analyze_current_performance(account_id)
        
        # Cross-platform optimization
        cross_platform_strategy = self._develop_cross_platform_strategy(account, performance_analysis)
        
        # Real-time bidding optimization
        bidding_optimization = self._optimize_bidding_strategies(account, performance_analysis)
        
        # Attribution modeling
        attribution_insights = self._generate_attribution_insights(account_id)
        
        # Budget allocation optimization
        budget_optimization = self._optimize_budget_allocation(account, cross_platform_strategy)
        
        # AI-powered audience optimization
        audience_optimization = self._optimize_audience_targeting(account, performance_analysis)
        
        return {
            'account_id': account_id,
            'strategy_date': datetime.utcnow().isoformat(),
            'performance_analysis': performance_analysis,
            'cross_platform_strategy': cross_platform_strategy,
            'bidding_optimization': bidding_optimization,
            'attribution_insights': attribution_insights,
            'budget_optimization': budget_optimization,
            'audience_optimization': audience_optimization,
            'automation_framework': self._design_automation_framework(account),
            'performance_projections': self._project_media_performance(account, cross_platform_strategy)
        }
    
    def _analyze_current_performance(self, account_id: str) -> Dict[str, Any]:
        """Analyze current media buying performance across all platforms"""
        
        # Get recent campaigns
        recent_campaigns = Campaign.query.filter_by(account_id=account_id)\
                                        .filter(Campaign.created_at >= datetime.utcnow() - timedelta(days=30))\
                                        .all()
        
        if not recent_campaigns:
            return {'status': 'no_campaign_data'}
        
        # Platform performance analysis
        platform_performance = {}
        for platform in AdPlatform:
            platform_campaigns = [c for c in recent_campaigns if c.platform == platform]
            
            if platform_campaigns:
                # Calculate aggregate metrics
                total_cost = sum(c.cost for c in platform_campaigns)
                total_revenue = sum(c.revenue for c in platform_campaigns)
                total_conversions = sum(c.conversions for c in platform_campaigns)
                total_clicks = sum(c.clicks for c in platform_campaigns)
                total_impressions = sum(c.impressions for c in platform_campaigns)
                
                # Calculate performance metrics
                platform_roas = total_revenue / total_cost if total_cost > 0 else 0
                platform_ctr = (total_clicks / total_impressions) * 100 if total_impressions > 0 else 0
                platform_conversion_rate = (total_conversions / total_clicks) * 100 if total_clicks > 0 else 0
                platform_cpa = total_cost / total_conversions if total_conversions > 0 else 0
                
                platform_performance[platform.value] = {
                    'campaign_count': len(platform_campaigns),
                    'total_spend': total_cost,
                    'total_revenue': total_revenue,
                    'roas': platform_roas,
                    'ctr': platform_ctr,
                    'conversion_rate': platform_conversion_rate,
                    'cpa': platform_cpa,
                    'performance_rating': self._rate_platform_performance(platform_roas, platform_ctr),
                    'optimization_score': np.mean([c.optimization_score for c in platform_campaigns])
                }
        
        # Objective performance analysis
        objective_performance = self._analyze_objective_performance(recent_campaigns)
        
        # Bidding strategy effectiveness
        bidding_effectiveness = self._analyze_bidding_effectiveness(recent_campaigns)
        
        # Creative performance analysis
        creative_performance = self._analyze_creative_performance(recent_campaigns)
        
        # Audience performance analysis
        audience_performance = self._analyze_audience_performance(recent_campaigns)
        
        return {
            'analysis_period': '30 days',
            'total_campaigns': len(recent_campaigns),
            'platform_performance': platform_performance,
            'objective_performance': objective_performance,
            'bidding_effectiveness': bidding_effectiveness,
            'creative_performance': creative_performance,
            'audience_performance': audience_performance,
            'key_insights': self._generate_performance_insights(platform_performance, objective_performance)
        }
    
    def _rate_platform_performance(self, roas: float, ctr: float) -> str:
        """Rate platform performance based on key metrics"""
        
        if roas >= 5.0 and ctr >= 3.0:
            return 'excellent'
        elif roas >= 3.0 and ctr >= 2.0:
            return 'good'
        elif roas >= 2.0 and ctr >= 1.0:
            return 'fair'
        else:
            return 'needs_optimization'
    
    def _analyze_objective_performance(self, campaigns: List[Campaign]) -> Dict[str, Any]:
        """Analyze performance by campaign objective"""
        
        objective_stats = {}
        
        for objective in CampaignObjective:
            objective_campaigns = [c for c in campaigns if c.objective == objective]
            
            if objective_campaigns:
                avg_roas = np.mean([c.roas for c in objective_campaigns if c.roas > 0])
                avg_ctr = np.mean([c.ctr for c in objective_campaigns if c.ctr > 0])
                avg_cpa = np.mean([c.cpa for c in objective_campaigns if c.cpa > 0])
                total_spend = sum(c.cost for c in objective_campaigns)
                
                objective_stats[objective.value] = {
                    'campaign_count': len(objective_campaigns),
                    'avg_roas': avg_roas,
                    'avg_ctr': avg_ctr,
                    'avg_cpa': avg_cpa,
                    'total_spend': total_spend,
                    'spend_percentage': 0,  # Will be calculated after all objectives
                    'efficiency_score': self._calculate_objective_efficiency(avg_roas, avg_ctr, avg_cpa)
                }
        
        # Calculate spend percentages
        total_spend = sum(stats['total_spend'] for stats in objective_stats.values())
        for stats in objective_stats.values():
            stats['spend_percentage'] = (stats['total_spend'] / total_spend * 100) if total_spend > 0 else 0
        
        return objective_stats
    
    def _calculate_objective_efficiency(self, roas: float, ctr: float, cpa: float) -> float:
        """Calculate efficiency score for campaign objective"""
        
        # Normalize metrics and create composite score
        roas_score = min(roas / 5.0, 1.0) * 40  # 40% weight
        ctr_score = min(ctr / 5.0, 1.0) * 30   # 30% weight
        cpa_score = max(0, 1 - (cpa / 100)) * 30  # 30% weight, lower CPA is better
        
        return roas_score + ctr_score + cpa_score
    
    def _analyze_bidding_effectiveness(self, campaigns: List[Campaign]) -> Dict[str, Any]:
        """Analyze effectiveness of different bidding strategies"""
        
        bidding_stats = {}
        
        for strategy in BiddingStrategy:
            strategy_campaigns = [c for c in campaigns if c.bidding_strategy == strategy]
            
            if strategy_campaigns:
                avg_roas = np.mean([c.roas for c in strategy_campaigns if c.roas > 0])
                avg_cpc = np.mean([c.cpc for c in strategy_campaigns if c.cpc > 0])
                avg_conversion_rate = np.mean([c.conversions / c.clicks * 100 for c in strategy_campaigns if c.clicks > 0])
                
                bidding_stats[strategy.value] = {
                    'campaign_count': len(strategy_campaigns),
                    'avg_roas': avg_roas,
                    'avg_cpc': avg_cpc,
                    'avg_conversion_rate': avg_conversion_rate,
                    'effectiveness_score': self._calculate_bidding_effectiveness_score(avg_roas, avg_cpc, avg_conversion_rate),
                    'recommendation': self._get_bidding_strategy_recommendation(avg_roas, avg_cpc)
                }
        
        return bidding_stats
    
    def _calculate_bidding_effectiveness_score(self, roas: float, cpc: float, conversion_rate: float) -> float:
        """Calculate effectiveness score for bidding strategy"""
        
        roas_score = min(roas / 5.0, 1.0) * 50    # 50% weight
        cpc_score = max(0, 1 - (cpc / 10)) * 25   # 25% weight, lower CPC is better
        conv_score = min(conversion_rate / 10, 1.0) * 25  # 25% weight
        
        return roas_score + cpc_score + conv_score
    
    def _get_bidding_strategy_recommendation(self, roas: float, cpc: float) -> str:
        """Get recommendation for bidding strategy"""
        
        if roas >= 4.0 and cpc <= 2.0:
            return 'scale_up'
        elif roas >= 3.0:
            return 'optimize_and_scale'
        elif roas >= 2.0:
            return 'optimize_targeting'
        else:
            return 'revise_strategy'
    
    def _analyze_creative_performance(self, campaigns: List[Campaign]) -> Dict[str, Any]:
        """Analyze creative asset performance"""
        
        # This would integrate with actual creative performance data
        # For now, providing a framework
        
        creative_insights = {
            'top_performing_formats': [
                {'format': 'video_ads', 'avg_ctr': 4.2, 'avg_conversion_rate': 3.8},
                {'format': 'carousel_ads', 'avg_ctr': 3.1, 'avg_conversion_rate': 2.9},
                {'format': 'single_image', 'avg_ctr': 2.4, 'avg_conversion_rate': 2.1}
            ],
            'creative_fatigue_analysis': {
                'frequency_threshold': 3.5,
                'performance_decline_rate': '15% after 5 exposures',
                'refresh_recommendation': 'every_7_days'
            },
            'message_testing_results': {
                'pain_point_focused': {'conversion_lift': '23%'},
                'benefit_focused': {'conversion_lift': '18%'},
                'feature_focused': {'conversion_lift': '12%'}
            },
            'visual_element_insights': {
                'human_faces': {'engagement_lift': '31%'},
                'product_shots': {'conversion_lift': '28%'},
                'lifestyle_imagery': {'brand_recall_lift': '22%'}
            }
        }
        
        return creative_insights
    
    def _analyze_audience_performance(self, campaigns: List[Campaign]) -> Dict[str, Any]:
        """Analyze audience targeting performance"""
        
        # Framework for audience performance analysis
        audience_insights = {
            'demographic_performance': {
                'age_groups': {
                    '25-34': {'roas': 4.2, 'volume': 'high'},
                    '35-44': {'roas': 3.8, 'volume': 'medium'},
                    '45-54': {'roas': 3.1, 'volume': 'low'}
                },
                'gender_performance': {
                    'female': {'roas': 3.9, 'ctr': 2.8},
                    'male': {'roas': 3.4, 'ctr': 2.1}
                }
            },
            'interest_performance': {
                'high_intent_keywords': {'conversion_rate': 4.2, 'cpa': 45},
                'broad_interests': {'conversion_rate': 2.1, 'cpa': 78},
                'lookalike_audiences': {'conversion_rate': 3.5, 'cpa': 52}
            },
            'behavioral_insights': {
                'previous_purchasers': {'roas': 8.5, 'frequency_cap': 2},
                'website_visitors': {'roas': 3.2, 'retargeting_window': '30_days'},
                'engagement_audiences': {'roas': 2.8, 'scale_potential': 'high'}
            },
            'geographic_performance': {
                'tier_1_cities': {'roas': 4.1, 'cpa': 48},
                'tier_2_cities': {'roas': 3.6, 'cpa': 42},
                'rural_areas': {'roas': 2.9, 'cpa': 65}
            }
        }
        
        return audience_insights
    
    def _generate_performance_insights(self, platform_performance: Dict, objective_performance: Dict) -> List[str]:
        """Generate actionable performance insights"""
        
        insights = []
        
        # Platform insights
        if platform_performance:
            best_platform = max(platform_performance.items(), key=lambda x: x[1]['roas'])
            insights.append(f"{best_platform[0].title()} delivers highest ROAS at {best_platform[1]['roas']:.1f}")
            
            underperforming = [p for p, data in platform_performance.items() if data['performance_rating'] == 'needs_optimization']
            if underperforming:
                insights.append(f"Optimization needed for {', '.join(underperforming)}")
        
        # Objective insights
        if objective_performance:
            efficient_objectives = [obj for obj, data in objective_performance.items() if data['efficiency_score'] > 70]
            if efficient_objectives:
                insights.append(f"High-performing objectives: {', '.join(efficient_objectives)}")
        
        return insights
    
    def _develop_cross_platform_strategy(self, account: MediaBuyingAccount, analysis: Dict) -> Dict[str, Any]:
        """Develop comprehensive cross-platform media strategy"""
        
        # Platform allocation strategy
        platform_allocation = self._optimize_platform_allocation(account, analysis)
        
        # Cross-platform synergies
        platform_synergies = self._identify_platform_synergies(analysis)
        
        # Audience coordination
        audience_coordination = self._coordinate_cross_platform_audiences(account, analysis)
        
        # Message coordination
        message_coordination = self._coordinate_cross_platform_messaging(account)
        
        # Budget flow optimization
        budget_flow = self._optimize_cross_platform_budget_flow(account, platform_allocation)
        
        return {
            'platform_allocation_strategy': platform_allocation,
            'platform_synergies': platform_synergies,
            'audience_coordination': audience_coordination,
            'message_coordination': message_coordination,
            'budget_flow_optimization': budget_flow,
            'performance_amplification': self._design_performance_amplification_strategy(platform_synergies)
        }
    
    def _optimize_platform_allocation(self, account: MediaBuyingAccount, analysis: Dict) -> Dict[str, Any]:
        """Optimize budget allocation across platforms"""
        
        platform_performance = analysis.get('platform_performance', {})
        total_budget = account.monthly_budget
        
        # Base allocation strategy
        if not platform_performance:
            # Default allocation for new accounts
            base_allocation = {
                'google_ads': 35,
                'facebook_ads': 25,
                'instagram_ads': 15,
                'linkedin_ads': 10,
                'youtube_ads': 10,
                'twitter_ads': 3,
                'tiktok_ads': 2
            }
        else:
            # Performance-based allocation
            base_allocation = self._calculate_performance_based_allocation(platform_performance)
        
        # Apply optimization factors
        optimized_allocation = self._apply_allocation_optimization_factors(base_allocation, account)
        
        # Calculate budget amounts
        budget_allocation = {}
        for platform, percentage in optimized_allocation.items():
            budget_allocation[platform] = {
                'percentage': percentage,
                'monthly_budget': (percentage / 100) * total_budget,
                'daily_budget': ((percentage / 100) * total_budget) / 30,
                'allocation_rationale': self._get_allocation_rationale(platform, percentage, platform_performance)
            }
        
        return {
            'allocation_strategy': 'performance_weighted',
            'budget_allocation': budget_allocation,
            'rebalancing_frequency': 'weekly',
            'performance_thresholds': self._define_rebalancing_thresholds()
        }
    
    def _calculate_performance_based_allocation(self, platform_performance: Dict) -> Dict[str, float]:
        """Calculate allocation based on platform performance"""
        
        # Weight platforms by performance score
        performance_scores = {}
        for platform, data in platform_performance.items():
            roas = data.get('roas', 2.0)
            ctr = data.get('ctr', 1.0)
            optimization_score = data.get('optimization_score', 50.0)
            
            # Composite performance score
            performance_scores[platform] = (roas * 0.5 + ctr * 0.3 + optimization_score/100 * 0.2) * 100
        
        # Convert to allocation percentages
        total_score = sum(performance_scores.values())
        allocation = {}
        
        for platform, score in performance_scores.items():
            base_percentage = (score / total_score) * 100
            # Apply bounds (minimum 5%, maximum 40%)
            allocation[platform] = max(5, min(40, base_percentage))
        
        # Normalize to 100%
        total_allocation = sum(allocation.values())
        normalized_allocation = {p: (alloc / total_allocation) * 100 for p, alloc in allocation.items()}
        
        return normalized_allocation
    
    def _apply_allocation_optimization_factors(self, base_allocation: Dict, account: MediaBuyingAccount) -> Dict[str, float]:
        """Apply optimization factors to base allocation"""
        
        optimized = base_allocation.copy()
        
        # Industry-specific adjustments
        industry_adjustments = {
            'b2b': {'linkedin_ads': 1.5, 'facebook_ads': 0.8},
            'ecommerce': {'google_ads': 1.2, 'facebook_ads': 1.3, 'instagram_ads': 1.2},
            'saas': {'google_ads': 1.3, 'linkedin_ads': 1.4},
            'retail': {'instagram_ads': 1.3, 'facebook_ads': 1.2}
        }
        
        industry = account.industry_vertical
        if industry in industry_adjustments:
            adjustments = industry_adjustments[industry]
            for platform, multiplier in adjustments.items():
                if platform in optimized:
                    optimized[platform] *= multiplier
        
        # Risk tolerance adjustments
        if account.risk_tolerance == 'low':
            # Favor proven platforms
            optimized['google_ads'] = optimized.get('google_ads', 30) * 1.1
            optimized['facebook_ads'] = optimized.get('facebook_ads', 25) * 1.1
        elif account.risk_tolerance == 'high':
            # Increase allocation to emerging platforms
            optimized['tiktok_ads'] = optimized.get('tiktok_ads', 2) * 2.0
            optimized['snapchat_ads'] = optimized.get('snapchat_ads', 1) * 2.0
        
        # Normalize back to 100%
        total = sum(optimized.values())
        normalized = {p: (alloc / total) * 100 for p, alloc in optimized.items()}
        
        return normalized
    
    def _get_allocation_rationale(self, platform: str, percentage: float, performance_data: Dict) -> str:
        """Get rationale for platform allocation"""
        
        if platform in performance_data:
            data = performance_data[platform]
            roas = data.get('roas', 0)
            if roas > 4.0:
                return f"High allocation due to strong ROAS of {roas:.1f}"
            elif roas > 2.0:
                return f"Balanced allocation with ROAS of {roas:.1f}"
            else:
                return f"Conservative allocation pending optimization (ROAS: {roas:.1f})"
        else:
            return "Strategic allocation for testing and expansion"
    
    def _define_rebalancing_thresholds(self) -> Dict[str, Any]:
        """Define thresholds for automatic budget rebalancing"""
        
        return {
            'roas_threshold': {
                'increase_budget': 5.0,
                'decrease_budget': 2.0,
                'pause_spending': 1.0
            },
            'cpa_threshold': {
                'target_multiplier': 1.5,  # Pause if CPA > 1.5x target
                'optimization_trigger': 1.2
            },
            'volume_threshold': {
                'minimum_conversions': 10,  # Per week for statistical significance
                'minimum_spend': 500  # Minimum weekly spend for consideration
            },
            'rebalancing_frequency': {
                'emergency_rebalancing': 'immediate',  # For severe underperformance
                'routine_rebalancing': 'weekly',
                'strategic_rebalancing': 'monthly'
            }
        }
    
    def _identify_platform_synergies(self, analysis: Dict) -> Dict[str, Any]:
        """Identify synergies between platforms"""
        
        synergies = {
            'awareness_to_conversion_funnel': {
                'awareness_platforms': ['youtube_ads', 'facebook_ads'],
                'consideration_platforms': ['google_ads', 'linkedin_ads'],
                'conversion_platforms': ['google_ads', 'facebook_ads'],
                'synergy_multiplier': 1.3
            },
            'retargeting_synergies': {
                'primary_touchpoint': 'google_ads',
                'retargeting_platforms': ['facebook_ads', 'instagram_ads'],
                'audience_overlap_optimization': 'exclude_converters',
                'frequency_capping': 'cross_platform'
            },
            'lookalike_amplification': {
                'seed_platform': 'facebook_ads',
                'expansion_platforms': ['google_ads', 'linkedin_ads'],
                'audience_modeling': 'cross_platform_lookalikes'
            },
            'creative_testing_synergies': {
                'testing_platform': 'facebook_ads',
                'scale_platforms': ['google_ads', 'youtube_ads'],
                'creative_transfer_efficiency': '85%'
            }
        }
        
        return synergies
    
    def _coordinate_cross_platform_audiences(self, account: MediaBuyingAccount, analysis: Dict) -> Dict[str, Any]:
        """Coordinate audience targeting across platforms"""
        
        return {
            'audience_segmentation_strategy': {
                'cold_audiences': {
                    'platforms': ['facebook_ads', 'google_ads', 'youtube_ads'],
                    'targeting_approach': 'broad_reach_with_interest_overlay',
                    'budget_allocation': '40%'
                },
                'warm_audiences': {
                    'platforms': ['facebook_ads', 'google_ads', 'instagram_ads'],
                    'targeting_approach': 'website_visitors_and_engagers',
                    'budget_allocation': '35%'
                },
                'hot_audiences': {
                    'platforms': ['google_ads', 'facebook_ads'],
                    'targeting_approach': 'cart_abandoners_and_high_intent',
                    'budget_allocation': '25%'
                }
            },
            'audience_exclusion_strategy': {
                'prevent_overlap': 'exclude_converters_from_cold_campaigns',
                'frequency_management': 'cross_platform_frequency_capping',
                'audience_hierarchy': 'prioritize_high_intent_audiences'
            },
            'lookalike_coordination': {
                'seed_audience_sharing': 'use_best_converting_segments_across_platforms',
                'similarity_thresholds': 'optimize_for_conversion_similarity',
                'expansion_strategy': 'gradual_expansion_with_performance_monitoring'
            },
            'dynamic_audience_optimization': {
                'real_time_adjustments': 'based_on_performance_data',
                'ai_powered_audience_discovery': 'continuous_audience_expansion',
                'performance_based_prioritization': 'allocate_budget_to_best_performing_segments'
            }
        }
    
    def _coordinate_cross_platform_messaging(self, account: MediaBuyingAccount) -> Dict[str, Any]:
        """Coordinate messaging strategy across platforms"""
        
        return {
            'message_framework': {
                'awareness_stage': {
                    'primary_message': 'problem_identification_and_solution_introduction',
                    'platforms': ['youtube_ads', 'facebook_ads', 'linkedin_ads'],
                    'content_type': 'educational_and_inspirational'
                },
                'consideration_stage': {
                    'primary_message': 'solution_benefits_and_differentiation',
                    'platforms': ['google_ads', 'linkedin_ads', 'facebook_ads'],
                    'content_type': 'comparison_and_proof_points'
                },
                'decision_stage': {
                    'primary_message': 'urgency_and_call_to_action',
                    'platforms': ['google_ads', 'facebook_ads'],
                    'content_type': 'offers_and_social_proof'
                }
            },
            'platform_specific_adaptations': {
                'google_ads': 'search_intent_focused_messaging',
                'facebook_ads': 'lifestyle_and_emotional_connection',
                'linkedin_ads': 'professional_benefits_and_roi',
                'youtube_ads': 'storytelling_and_demonstration',
                'instagram_ads': 'visual_appeal_and_lifestyle'
            },
            'message_testing_coordination': {
                'testing_framework': 'systematic_message_variation_testing',
                'cross_platform_learnings': 'apply_winning_messages_across_platforms',
                'performance_optimization': 'continuous_message_refinement'
            }
        }

# Initialize media buying engine
media_engine = AdvancedMediaBuyingEngine()

# Routes
@app.route('/advanced-media-buying')
def media_dashboard():
    """Advanced Media Buying dashboard"""
    
    recent_accounts = MediaBuyingAccount.query.order_by(MediaBuyingAccount.created_at.desc()).limit(10).all()
    
    return render_template('media/dashboard.html',
                         accounts=recent_accounts)

@app.route('/advanced-media-buying/api/comprehensive-strategy', methods=['POST'])
def create_media_strategy():
    """API endpoint for comprehensive media strategy"""
    
    data = request.get_json()
    account_id = data.get('account_id')
    
    if not account_id:
        return jsonify({'error': 'Account ID required'}), 400
    
    strategy = media_engine.generate_comprehensive_media_strategy(account_id)
    return jsonify(strategy)

# Initialize database
with app.app_context():
    db.create_all()
    
    # Create sample data
    if MediaBuyingAccount.query.count() == 0:
        sample_account = MediaBuyingAccount(
            account_id='MEDIA_DEMO_001',
            account_name='Demo Media Buying Account',
            monthly_budget=50000.0,
            target_roas=4.0,
            industry_vertical='ecommerce'
        )
        
        db.session.add(sample_account)
        db.session.commit()
        logger.info("Sample media buying data created")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5035, debug=True)