"""
Content Strategy & Marketing Agent
Content optimization and marketing campaign intelligence
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

class ContentStrategyAgent:
    """
    Comprehensive Content Strategy & Marketing System
    - Content performance optimization
    - Marketing campaign intelligence
    - Audience engagement analysis
    - Content ROI measurement
    """
    
    def __init__(self):
        self.name = "Content Strategy & Marketing Agent"
        self.version = "1.0.0"
        self.capabilities = [
            "Content Performance Analysis",
            "Marketing Campaign Optimization",
            "Audience Segmentation",
            "Content ROI Analysis",
            "Trend Identification",
            "Content Planning"
        ]
        
    def analyze_content_strategy(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive content strategy analysis"""
        try:
            company_name = content_data.get('company_name', 'Unknown Company')
            
            # Content performance analysis
            performance_analysis = self._analyze_content_performance(content_data)
            
            # Audience analysis
            audience_analysis = self._analyze_audience_engagement(content_data)
            
            # Campaign optimization
            campaign_optimization = self._optimize_campaigns(content_data)
            
            # Content planning
            content_planning = self._generate_content_plan(performance_analysis, audience_analysis)
            
            # Generate recommendations
            recommendations = self._generate_content_recommendations(
                performance_analysis, audience_analysis, campaign_optimization
            )
            
            return {
                'company': company_name,
                'analysis_date': datetime.now().isoformat(),
                'performance_analysis': performance_analysis,
                'audience_analysis': audience_analysis,
                'campaign_optimization': campaign_optimization,
                'content_planning': content_planning,
                'recommendations': recommendations,
                'next_review_date': (datetime.now() + timedelta(days=30)).isoformat()
            }
            
        except Exception as e:
            logging.error(f"Content strategy analysis failed: {str(e)}")
            return {'error': f'Content strategy analysis failed: {str(e)}'}
            
    def _analyze_content_performance(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content performance metrics"""
        
        content_pieces = content_data.get('content_pieces', [])
        
        # Performance metrics
        total_views = sum(piece.get('views', 0) for piece in content_pieces)
        total_engagements = sum(piece.get('engagements', 0) for piece in content_pieces)
        avg_engagement_rate = (total_engagements / total_views * 100) if total_views > 0 else 0
        
        # Top performing content
        top_content = sorted(content_pieces, key=lambda x: x.get('engagement_rate', 0), reverse=True)[:5]
        
        # Content type performance
        type_performance = {}
        for piece in content_pieces:
            content_type = piece.get('type', 'unknown')
            if content_type not in type_performance:
                type_performance[content_type] = {'views': 0, 'engagements': 0, 'count': 0}
            
            type_performance[content_type]['views'] += piece.get('views', 0)
            type_performance[content_type]['engagements'] += piece.get('engagements', 0)
            type_performance[content_type]['count'] += 1
            
        # Calculate type-level engagement rates
        for content_type in type_performance:
            views = type_performance[content_type]['views']
            engagements = type_performance[content_type]['engagements']
            type_performance[content_type]['engagement_rate'] = (engagements / views * 100) if views > 0 else 0
            
        return {
            'total_content_pieces': len(content_pieces),
            'total_views': total_views,
            'total_engagements': total_engagements,
            'average_engagement_rate': avg_engagement_rate,
            'top_performing_content': top_content,
            'content_type_performance': type_performance,
            'performance_trends': self._identify_performance_trends(content_pieces)
        }
        
    def _identify_performance_trends(self, content_pieces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify performance trends"""
        
        # Group by publish date
        monthly_performance = {}
        for piece in content_pieces:
            publish_date = piece.get('publish_date', '2025-01-01')
            month = publish_date[:7]  # YYYY-MM format
            
            if month not in monthly_performance:
                monthly_performance[month] = {'views': 0, 'engagements': 0}
                
            monthly_performance[month]['views'] += piece.get('views', 0)
            monthly_performance[month]['engagements'] += piece.get('engagements', 0)
            
        # Calculate trend
        months = sorted(monthly_performance.keys())
        if len(months) >= 2:
            recent_month = monthly_performance[months[-1]]
            previous_month = monthly_performance[months[-2]]
            
            view_trend = ((recent_month['views'] - previous_month['views']) / 
                         max(1, previous_month['views'])) * 100
            engagement_trend = ((recent_month['engagements'] - previous_month['engagements']) / 
                              max(1, previous_month['engagements'])) * 100
        else:
            view_trend = 0
            engagement_trend = 0
            
        return {
            'monthly_performance': monthly_performance,
            'view_trend_percentage': view_trend,
            'engagement_trend_percentage': engagement_trend,
            'trend_direction': 'Growing' if view_trend > 5 else 'Stable' if view_trend > -5 else 'Declining'
        }
        
    def _analyze_audience_engagement(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze audience engagement patterns"""
        
        audience_segments = content_data.get('audience_segments', [])
        
        # Segment performance
        segment_performance = {}
        for segment in audience_segments:
            segment_name = segment.get('name', 'unknown')
            segment_performance[segment_name] = {
                'size': segment.get('size', 0),
                'engagement_rate': segment.get('engagement_rate', 0),
                'conversion_rate': segment.get('conversion_rate', 0),
                'preferred_content_types': segment.get('preferred_content_types', [])
            }
            
        # Identify high-value segments
        high_value_segments = []
        for name, metrics in segment_performance.items():
            value_score = (metrics['engagement_rate'] * 0.6 + metrics['conversion_rate'] * 0.4)
            if value_score > 15:  # High value threshold
                high_value_segments.append({
                    'name': name,
                    'value_score': value_score,
                    'size': metrics['size']
                })
                
        return {
            'total_audience_size': sum(seg.get('size', 0) for seg in audience_segments),
            'segment_performance': segment_performance,
            'high_value_segments': high_value_segments,
            'engagement_insights': self._generate_engagement_insights(segment_performance)
        }
        
    def _generate_engagement_insights(self, segment_performance: Dict[str, Any]) -> List[str]:
        """Generate audience engagement insights"""
        
        insights = []
        
        # Find best performing segment
        if segment_performance:
            best_segment = max(segment_performance.items(), 
                             key=lambda x: x[1]['engagement_rate'])
            insights.append(f"'{best_segment[0]}' segment shows highest engagement at {best_segment[1]['engagement_rate']:.1f}%")
            
        # Find largest segment
        if segment_performance:
            largest_segment = max(segment_performance.items(), 
                                key=lambda x: x[1]['size'])
            insights.append(f"'{largest_segment[0]}' is the largest segment with {largest_segment[1]['size']:,} members")
            
        return insights
        
    def _optimize_campaigns(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize marketing campaigns"""
        
        campaigns = content_data.get('campaigns', [])
        
        campaign_analysis = []
        total_spend = 0
        total_revenue = 0
        
        for campaign in campaigns:
            spend = campaign.get('spend', 0)
            revenue = campaign.get('revenue', 0)
            roi = ((revenue - spend) / spend * 100) if spend > 0 else 0
            
            total_spend += spend
            total_revenue += revenue
            
            campaign_analysis.append({
                'campaign_name': campaign.get('name', 'unknown'),
                'spend': spend,
                'revenue': revenue,
                'roi': roi,
                'performance': 'Excellent' if roi > 300 else 'Good' if roi > 200 else 'Fair' if roi > 100 else 'Poor'
            })
            
        # Sort by ROI
        campaign_analysis.sort(key=lambda x: x['roi'], reverse=True)
        
        # Overall campaign metrics
        overall_roi = ((total_revenue - total_spend) / total_spend * 100) if total_spend > 0 else 0
        
        return {
            'total_campaigns': len(campaigns),
            'total_spend': total_spend,
            'total_revenue': total_revenue,
            'overall_roi': overall_roi,
            'campaign_analysis': campaign_analysis,
            'optimization_opportunities': self._identify_campaign_optimizations(campaign_analysis)
        }
        
    def _identify_campaign_optimizations(self, campaign_analysis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify campaign optimization opportunities"""
        
        optimizations = []
        
        # Find underperforming campaigns
        poor_campaigns = [c for c in campaign_analysis if c['performance'] == 'Poor']
        for campaign in poor_campaigns:
            optimizations.append({
                'campaign': campaign['campaign_name'],
                'type': 'Performance Improvement',
                'issue': f'Low ROI of {campaign["roi"]:.1f}%',
                'recommendation': 'Review targeting, messaging, and budget allocation'
            })
            
        # Find high-performing campaigns to scale
        excellent_campaigns = [c for c in campaign_analysis if c['performance'] == 'Excellent']
        for campaign in excellent_campaigns[:3]:  # Top 3
            optimizations.append({
                'campaign': campaign['campaign_name'],
                'type': 'Scale Opportunity',
                'opportunity': f'High ROI of {campaign["roi"]:.1f}%',
                'recommendation': 'Consider increasing budget allocation'
            })
            
        return optimizations
        
    def _generate_content_plan(self, performance: Dict[str, Any], audience: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content planning recommendations"""
        
        # Identify best performing content types
        type_performance = performance.get('content_type_performance', {})
        top_types = sorted(type_performance.items(), 
                          key=lambda x: x[1]['engagement_rate'], reverse=True)[:3]
        
        # Content recommendations based on performance
        content_recommendations = []
        for content_type, metrics in top_types:
            content_recommendations.append({
                'content_type': content_type,
                'frequency': 'Weekly' if metrics['engagement_rate'] > 8 else 'Bi-weekly',
                'rationale': f'High engagement rate of {metrics["engagement_rate"]:.1f}%'
            })
            
        # Audience-specific content
        high_value_segments = audience.get('high_value_segments', [])
        segment_content = []
        for segment in high_value_segments:
            segment_content.append({
                'target_segment': segment['name'],
                'content_focus': 'Personalized content for high-value audience',
                'frequency': 'Weekly'
            })
            
        return {
            'content_type_recommendations': content_recommendations,
            'segment_specific_content': segment_content,
            'publishing_schedule': self._create_publishing_schedule(content_recommendations),
            'content_calendar_suggestions': self._suggest_content_calendar(performance, audience)
        }
        
    def _create_publishing_schedule(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create optimal publishing schedule"""
        
        # Calculate total content pieces per week
        weekly_content = 0
        for rec in recommendations:
            if rec['frequency'] == 'Weekly':
                weekly_content += 1
            elif rec['frequency'] == 'Bi-weekly':
                weekly_content += 0.5
                
        return {
            'recommended_weekly_posts': int(weekly_content),
            'optimal_posting_days': ['Tuesday', 'Wednesday', 'Thursday'],  # Best engagement days
            'optimal_posting_times': ['9:00 AM', '1:00 PM', '3:00 PM'],
            'content_distribution': {
                'educational': 0.4,
                'promotional': 0.2,
                'entertaining': 0.3,
                'news_updates': 0.1
            }
        }
        
    def _suggest_content_calendar(self, performance: Dict[str, Any], audience: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest content calendar themes"""
        
        calendar_suggestions = [
            {
                'theme': 'Industry Insights',
                'frequency': 'Weekly',
                'content_types': ['Blog posts', 'Infographics'],
                'target_audience': 'Professional segment'
            },
            {
                'theme': 'Customer Success Stories',
                'frequency': 'Bi-weekly',
                'content_types': ['Case studies', 'Video testimonials'],
                'target_audience': 'Prospects and customers'
            },
            {
                'theme': 'Product Updates',
                'frequency': 'Monthly',
                'content_types': ['Announcements', 'Demo videos'],
                'target_audience': 'Existing customers'
            },
            {
                'theme': 'Thought Leadership',
                'frequency': 'Bi-weekly',
                'content_types': ['Articles', 'Webinars'],
                'target_audience': 'Industry professionals'
            }
        ]
        
        return calendar_suggestions
        
    def _generate_content_recommendations(self, performance: Dict[str, Any], 
                                        audience: Dict[str, Any], 
                                        campaigns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate content strategy recommendations"""
        
        recommendations = []
        
        # Performance-based recommendations
        if performance['average_engagement_rate'] < 5:
            recommendations.append({
                'category': 'Engagement Optimization',
                'priority': 'High',
                'recommendation': 'Improve content engagement through better targeting and formats',
                'actions': [
                    'Focus on top-performing content types',
                    'Optimize content for audience preferences',
                    'A/B test different content formats',
                    'Improve content timing and frequency'
                ],
                'expected_impact': 'Increase engagement rate by 50-100%'
            })
            
        # Campaign optimization recommendations
        if campaigns['overall_roi'] < 200:
            recommendations.append({
                'category': 'Campaign Optimization',
                'priority': 'Medium',
                'recommendation': 'Optimize marketing campaigns for better ROI',
                'actions': [
                    'Reallocate budget from poor-performing campaigns',
                    'Scale high-performing campaigns',
                    'Improve targeting and messaging',
                    'Implement better tracking and attribution'
                ],
                'expected_impact': f'Improve overall ROI from {campaigns["overall_roi"]:.1f}% to 250%+'
            })
            
        # Audience development recommendations
        high_value_segments = audience.get('high_value_segments', [])
        if len(high_value_segments) < 2:
            recommendations.append({
                'category': 'Audience Development',
                'priority': 'Medium',
                'recommendation': 'Develop more high-value audience segments',
                'actions': [
                    'Analyze audience data for new segments',
                    'Create targeted content for different personas',
                    'Implement personalization strategies',
                    'Expand reach through strategic partnerships'
                ],
                'expected_impact': 'Increase high-value audience segments by 100%'
            })
            
        # Content diversification recommendations
        type_performance = performance.get('content_type_performance', {})
        if len(type_performance) < 4:
            recommendations.append({
                'category': 'Content Diversification',
                'priority': 'Low',
                'recommendation': 'Diversify content types to reach broader audience',
                'actions': [
                    'Experiment with video content',
                    'Create interactive content',
                    'Develop podcast or audio content',
                    'Create visual content like infographics'
                ],
                'expected_impact': 'Reach new audience segments and improve engagement'
            })
            
        return recommendations

def test_content_strategy_agent():
    """Test the Content Strategy Agent"""
    print("🧪 Testing Content Strategy & Marketing Agent")
    print("=" * 50)
    
    try:
        agent = ContentStrategyAgent()
        
        test_data = {
            'company_name': 'ContentMasters Inc',
            'content_pieces': [
                {
                    'type': 'blog_post',
                    'views': 5000,
                    'engagements': 250,
                    'engagement_rate': 5.0,
                    'publish_date': '2025-08-15'
                },
                {
                    'type': 'video',
                    'views': 3000,
                    'engagements': 450,
                    'engagement_rate': 15.0,
                    'publish_date': '2025-08-20'
                }
            ],
            'audience_segments': [
                {
                    'name': 'Professional Users',
                    'size': 10000,
                    'engagement_rate': 8.5,
                    'conversion_rate': 12.0
                },
                {
                    'name': 'Enterprise Customers',
                    'size': 5000,
                    'engagement_rate': 12.0,
                    'conversion_rate': 18.0
                }
            ],
            'campaigns': [
                {
                    'name': 'Q3 Lead Generation',
                    'spend': 10000,
                    'revenue': 35000
                },
                {
                    'name': 'Brand Awareness',
                    'spend': 15000,
                    'revenue': 25000
                }
            ]
        }
        
        analysis = agent.analyze_content_strategy(test_data)
        print(f"✅ Content strategy analysis completed for {test_data['company_name']}")
        print(f"   Content pieces analyzed: {analysis['performance_analysis']['total_content_pieces']}")
        print(f"   Average engagement rate: {analysis['performance_analysis']['average_engagement_rate']:.1f}%")
        print(f"   Campaign ROI: {analysis['campaign_optimization']['overall_roi']:.1f}%")
        
        return {
            'agent_initialized': True,
            'content_analyzed': analysis['performance_analysis']['total_content_pieces'],
            'engagement_rate': analysis['performance_analysis']['average_engagement_rate']
        }
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return {'error': str(e)}

if __name__ == "__main__":
    test_content_strategy_agent()