"""
Market Intelligence & Competitive Agent
Real-time competitive analysis and market positioning
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class CompetitorProfile:
    name: str
    market_share: float
    strengths: List[str]
    weaknesses: List[str]
    threat_level: str

class MarketIntelligenceAgent:
    """
    Comprehensive Market Intelligence System
    - Real-time competitive analysis
    - Market share prediction
    - Competitive response modeling
    - Strategic positioning optimization
    """
    
    def __init__(self):
        self.name = "Market Intelligence & Competitive Agent"
        self.version = "1.0.0"
        self.capabilities = [
            "Competitive Analysis",
            "Market Share Tracking",
            "Trend Analysis",
            "Positioning Strategy",
            "Competitive Response Modeling",
            "Market Opportunity Identification"
        ]
        
    def analyze_market_position(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive market position analysis"""
        try:
            company_name = market_data.get('company_name', 'Unknown Company')
            
            # Competitive landscape analysis
            competitive_analysis = self._analyze_competitive_landscape(market_data)
            
            # Market trends analysis
            trends_analysis = self._analyze_market_trends(market_data)
            
            # Positioning analysis
            positioning_analysis = self._analyze_market_positioning(market_data)
            
            # Generate strategic recommendations
            recommendations = self._generate_market_recommendations(
                competitive_analysis, trends_analysis, positioning_analysis
            )
            
            return {
                'company': company_name,
                'analysis_date': datetime.now().isoformat(),
                'competitive_analysis': competitive_analysis,
                'trends_analysis': trends_analysis,
                'positioning_analysis': positioning_analysis,
                'recommendations': recommendations,
                'next_review_date': (datetime.now() + timedelta(days=30)).isoformat()
            }
            
        except Exception as e:
            logging.error(f"Market intelligence analysis failed: {str(e)}")
            return {'error': f'Market intelligence analysis failed: {str(e)}'}
            
    def _analyze_competitive_landscape(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze competitive landscape"""
        competitors = market_data.get('competitors', [])
        company_metrics = market_data.get('company_metrics', {})
        
        competitor_profiles = []
        for competitor in competitors:
            profile = self._create_competitor_profile(competitor, company_metrics)
            competitor_profiles.append(profile)
            
        # Sort by threat level
        competitor_profiles.sort(key=lambda x: {'High': 3, 'Medium': 2, 'Low': 1}[x.threat_level], reverse=True)
        
        market_concentration = self._calculate_market_concentration(competitors)
        competitive_intensity = self._assess_competitive_intensity(competitors, company_metrics)
        
        return {
            'competitor_profiles': [profile.__dict__ for profile in competitor_profiles],
            'market_concentration': market_concentration,
            'competitive_intensity': competitive_intensity,
            'key_competitive_threats': [p.name for p in competitor_profiles if p.threat_level == 'High'],
            'competitive_advantages': self._identify_competitive_advantages(company_metrics, competitors)
        }
        
    def _create_competitor_profile(self, competitor: Dict[str, Any], company_metrics: Dict[str, Any]) -> CompetitorProfile:
        """Create detailed competitor profile"""
        
        name = competitor.get('name', 'Unknown Competitor')
        market_share = competitor.get('market_share', 10)
        
        # Analyze strengths and weaknesses
        strengths = []
        weaknesses = []
        
        # Financial strength
        if competitor.get('revenue', 0) > company_metrics.get('revenue', 0):
            strengths.append('Strong financial position')
        else:
            weaknesses.append('Weaker financial position')
            
        # Market presence
        if market_share > company_metrics.get('market_share', 15):
            strengths.append('Larger market share')
        else:
            weaknesses.append('Smaller market share')
            
        # Product quality
        product_rating = competitor.get('product_rating', 7)
        if product_rating >= 8:
            strengths.append('High product quality')
        elif product_rating < 6:
            weaknesses.append('Lower product quality')
            
        # Innovation
        if competitor.get('innovation_score', 6) >= 8:
            strengths.append('Strong innovation capability')
        elif competitor.get('innovation_score', 6) < 5:
            weaknesses.append('Limited innovation')
            
        # Calculate threat level
        threat_score = (market_share + competitor.get('growth_rate', 5) + product_rating) / 3
        threat_level = 'High' if threat_score >= 8 else 'Medium' if threat_score >= 6 else 'Low'
        
        return CompetitorProfile(
            name=name,
            market_share=market_share,
            strengths=strengths,
            weaknesses=weaknesses,
            threat_level=threat_level
        )
        
    def _calculate_market_concentration(self, competitors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate market concentration metrics"""
        
        market_shares = [comp.get('market_share', 0) for comp in competitors]
        total_tracked_share = sum(market_shares)
        
        # Calculate HHI (Herfindahl-Hirschman Index)
        hhi = sum(share ** 2 for share in market_shares)
        
        # Top 4 concentration ratio
        top_4_shares = sorted(market_shares, reverse=True)[:4]
        cr4 = sum(top_4_shares)
        
        concentration_level = 'High' if hhi > 2500 else 'Medium' if hhi > 1500 else 'Low'
        
        return {
            'hhi': hhi,
            'cr4': cr4,
            'concentration_level': concentration_level,
            'total_tracked_share': total_tracked_share,
            'market_fragmentation': 100 - total_tracked_share
        }
        
    def _assess_competitive_intensity(self, competitors: List[Dict[str, Any]], company_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess competitive intensity in the market"""
        
        intensity_factors = {
            'number_of_competitors': min(10, len(competitors)),
            'market_growth_rate': company_metrics.get('market_growth_rate', 5),
            'product_differentiation': company_metrics.get('product_differentiation_score', 6),
            'switching_costs': company_metrics.get('customer_switching_costs', 5),
            'barriers_to_entry': company_metrics.get('entry_barriers_score', 5)
        }
        
        # Calculate intensity score
        intensity_score = (
            intensity_factors['number_of_competitors'] * 0.2 +
            (10 - intensity_factors['market_growth_rate']) * 0.2 +  # Lower growth = higher intensity
            (10 - intensity_factors['product_differentiation']) * 0.2 +  # Lower differentiation = higher intensity
            (10 - intensity_factors['switching_costs']) * 0.2 +  # Lower switching costs = higher intensity
            (10 - intensity_factors['barriers_to_entry']) * 0.2  # Lower barriers = higher intensity
        )
        
        intensity_level = 'High' if intensity_score >= 7 else 'Medium' if intensity_score >= 4 else 'Low'
        
        return {
            'intensity_score': intensity_score,
            'intensity_level': intensity_level,
            'intensity_factors': intensity_factors,
            'competitive_pressure': 'Significant' if intensity_score >= 6 else 'Moderate' if intensity_score >= 3 else 'Low'
        }
        
    def _identify_competitive_advantages(self, company_metrics: Dict[str, Any], competitors: List[Dict[str, Any]]) -> List[str]:
        """Identify company's competitive advantages"""
        
        advantages = []
        
        # Cost advantage
        company_cost_structure = company_metrics.get('cost_efficiency', 70)
        avg_competitor_cost = sum(comp.get('cost_efficiency', 70) for comp in competitors) / len(competitors) if competitors else 70
        
        if company_cost_structure > avg_competitor_cost + 10:
            advantages.append('Cost leadership advantage')
            
        # Quality advantage
        company_quality = company_metrics.get('product_rating', 7)
        avg_competitor_quality = sum(comp.get('product_rating', 7) for comp in competitors) / len(competitors) if competitors else 7
        
        if company_quality > avg_competitor_quality + 0.5:
            advantages.append('Product quality differentiation')
            
        # Innovation advantage
        company_innovation = company_metrics.get('innovation_score', 6)
        avg_competitor_innovation = sum(comp.get('innovation_score', 6) for comp in competitors) / len(competitors) if competitors else 6
        
        if company_innovation > avg_competitor_innovation + 1:
            advantages.append('Innovation leadership')
            
        # Customer service advantage
        if company_metrics.get('customer_satisfaction', 75) >= 85:
            advantages.append('Superior customer service')
            
        # Brand strength
        if company_metrics.get('brand_strength', 6) >= 8:
            advantages.append('Strong brand recognition')
            
        return advantages if advantages else ['Focus on developing competitive advantages']
        
    def _analyze_market_trends(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market trends and patterns"""
        
        historical_data = market_data.get('historical_market_data', {})
        current_trends = market_data.get('current_trends', [])
        
        # Market growth analysis
        market_growth = self._calculate_market_growth(historical_data)
        
        # Trend analysis
        trend_impact = self._assess_trend_impact(current_trends)
        
        # Opportunity identification
        opportunities = self._identify_market_opportunities(market_growth, trend_impact)
        
        # Threat assessment
        threats = self._identify_market_threats(market_growth, trend_impact)
        
        return {
            'market_growth': market_growth,
            'trend_impact': trend_impact,
            'market_opportunities': opportunities,
            'market_threats': threats,
            'trend_recommendations': self._generate_trend_recommendations(opportunities, threats)
        }
        
    def _calculate_market_growth(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate market growth metrics"""
        
        market_size_history = historical_data.get('market_size', [1000, 1050, 1100, 1160])
        
        if len(market_size_history) >= 2:
            # Calculate CAGR
            periods = len(market_size_history) - 1
            start_value = market_size_history[0]
            end_value = market_size_history[-1]
            
            cagr = ((end_value / start_value) ** (1/periods) - 1) * 100
            
            # Calculate recent growth
            recent_growth = ((market_size_history[-1] / market_size_history[-2]) - 1) * 100
        else:
            cagr = 5.0  # Default assumption
            recent_growth = 5.0
            
        growth_trend = 'Accelerating' if recent_growth > cagr else 'Stable' if abs(recent_growth - cagr) < 2 else 'Decelerating'
        
        return {
            'cagr': cagr,
            'recent_growth': recent_growth,
            'growth_trend': growth_trend,
            'market_maturity': 'Mature' if cagr < 3 else 'Growing' if cagr < 10 else 'High Growth'
        }
        
    def _assess_trend_impact(self, trends: List[str]) -> Dict[str, Any]:
        """Assess impact of current market trends"""
        
        # Categorize trends by impact
        trend_categories = {
            'technology': [],
            'consumer_behavior': [],
            'regulatory': [],
            'economic': [],
            'social': []
        }
        
        # Simple keyword-based categorization
        for trend in trends:
            trend_lower = trend.lower()
            if any(keyword in trend_lower for keyword in ['ai', 'digital', 'automation', 'tech']):
                trend_categories['technology'].append(trend)
            elif any(keyword in trend_lower for keyword in ['consumer', 'customer', 'user']):
                trend_categories['consumer_behavior'].append(trend)
            elif any(keyword in trend_lower for keyword in ['regulation', 'compliance', 'legal']):
                trend_categories['regulatory'].append(trend)
            elif any(keyword in trend_lower for keyword in ['economic', 'inflation', 'recession']):
                trend_categories['economic'].append(trend)
            else:
                trend_categories['social'].append(trend)
                
        # Assess overall trend impact
        total_trends = len(trends)
        trend_intensity = 'High' if total_trends >= 10 else 'Medium' if total_trends >= 5 else 'Low'
        
        return {
            'trend_categories': trend_categories,
            'total_trends': total_trends,
            'trend_intensity': trend_intensity,
            'dominant_trend_category': max(trend_categories.items(), key=lambda x: len(x[1]))[0] if trends else 'none'
        }
        
    def _identify_market_opportunities(self, growth: Dict[str, Any], trends: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify market opportunities"""
        
        opportunities = []
        
        # Growth-based opportunities
        if growth['cagr'] > 8:
            opportunities.append({
                'type': 'Market Growth',
                'opportunity': 'High market growth rate presents expansion opportunities',
                'priority': 'High',
                'potential_impact': 'Revenue growth of 15-25% annually'
            })
            
        # Technology trend opportunities
        if len(trends['trend_categories']['technology']) >= 2:
            opportunities.append({
                'type': 'Technology Innovation',
                'opportunity': 'Multiple technology trends create innovation opportunities',
                'priority': 'Medium',
                'potential_impact': 'Competitive differentiation and market leadership'
            })
            
        # Consumer behavior opportunities
        if len(trends['trend_categories']['consumer_behavior']) >= 2:
            opportunities.append({
                'type': 'Consumer Evolution',
                'opportunity': 'Changing consumer preferences create new market segments',
                'priority': 'Medium',
                'potential_impact': 'New customer acquisition and retention'
            })
            
        # Default opportunity if none identified
        if not opportunities:
            opportunities.append({
                'type': 'Market Optimization',
                'opportunity': 'Focus on operational efficiency and customer satisfaction',
                'priority': 'Medium',
                'potential_impact': 'Improved profitability and market position'
            })
            
        return opportunities
        
    def _identify_market_threats(self, growth: Dict[str, Any], trends: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify market threats"""
        
        threats = []
        
        # Market maturity threats
        if growth['market_maturity'] == 'Mature' and growth['cagr'] < 2:
            threats.append({
                'type': 'Market Saturation',
                'threat': 'Mature market with low growth limits expansion opportunities',
                'severity': 'Medium',
                'mitigation': 'Focus on market share gains and new market development'
            })
            
        # Regulatory threats
        if len(trends['trend_categories']['regulatory']) >= 2:
            threats.append({
                'type': 'Regulatory Pressure',
                'threat': 'Increasing regulatory requirements may impact operations',
                'severity': 'Medium',
                'mitigation': 'Proactive compliance and regulatory engagement'
            })
            
        # Economic threats
        if len(trends['trend_categories']['economic']) >= 1:
            threats.append({
                'type': 'Economic Uncertainty',
                'threat': 'Economic trends may impact customer demand and spending',
                'severity': 'Medium',
                'mitigation': 'Diversify customer base and improve cost flexibility'
            })
            
        return threats
        
    def _generate_trend_recommendations(self, opportunities: List[Dict[str, Any]], threats: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on trends"""
        
        recommendations = []
        
        # Opportunity-based recommendations
        high_priority_opportunities = [opp for opp in opportunities if opp.get('priority') == 'High']
        if high_priority_opportunities:
            recommendations.append('Prioritize investment in high-growth market opportunities')
            
        # Threat mitigation recommendations
        if threats:
            recommendations.append('Develop contingency plans for identified market threats')
            
        # General recommendations
        recommendations.extend([
            'Monitor market trends continuously for early identification of changes',
            'Maintain agility to quickly respond to market opportunities and threats',
            'Invest in market research and competitive intelligence capabilities'
        ])
        
        return recommendations
        
    def _analyze_market_positioning(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current market positioning"""
        
        company_metrics = market_data.get('company_metrics', {})
        target_segments = market_data.get('target_segments', [])
        
        # Current position analysis
        current_position = self._assess_current_position(company_metrics)
        
        # Segment analysis
        segment_analysis = self._analyze_target_segments(target_segments, company_metrics)
        
        # Positioning gaps
        positioning_gaps = self._identify_positioning_gaps(current_position, segment_analysis)
        
        # Positioning recommendations
        positioning_strategy = self._develop_positioning_strategy(current_position, positioning_gaps)
        
        return {
            'current_position': current_position,
            'segment_analysis': segment_analysis,
            'positioning_gaps': positioning_gaps,
            'positioning_strategy': positioning_strategy
        }
        
    def _assess_current_position(self, company_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess current market position"""
        
        position_dimensions = {
            'price_position': self._categorize_price_position(company_metrics.get('price_level', 50)),
            'quality_position': self._categorize_quality_position(company_metrics.get('product_rating', 7)),
            'innovation_position': self._categorize_innovation_position(company_metrics.get('innovation_score', 6)),
            'service_position': self._categorize_service_position(company_metrics.get('customer_satisfaction', 75))
        }
        
        # Overall positioning
        position_scores = {
            'Premium': sum(1 for pos in position_dimensions.values() if pos == 'Premium'),
            'Mid-market': sum(1 for pos in position_dimensions.values() if pos == 'Mid-market'),
            'Value': sum(1 for pos in position_dimensions.values() if pos == 'Value')
        }
        
        overall_position = max(position_scores.items(), key=lambda x: x[1])[0]
        
        return {
            'position_dimensions': position_dimensions,
            'overall_position': overall_position,
            'position_consistency': max(position_scores.values()) / len(position_dimensions)
        }
        
    def _categorize_price_position(self, price_level: float) -> str:
        """Categorize price position"""
        if price_level >= 80:
            return 'Premium'
        elif price_level >= 40:
            return 'Mid-market'
        else:
            return 'Value'
            
    def _categorize_quality_position(self, quality_rating: float) -> str:
        """Categorize quality position"""
        if quality_rating >= 8.5:
            return 'Premium'
        elif quality_rating >= 6.5:
            return 'Mid-market'
        else:
            return 'Value'
            
    def _categorize_innovation_position(self, innovation_score: float) -> str:
        """Categorize innovation position"""
        if innovation_score >= 8:
            return 'Premium'
        elif innovation_score >= 5:
            return 'Mid-market'
        else:
            return 'Value'
            
    def _categorize_service_position(self, satisfaction_score: float) -> str:
        """Categorize service position"""
        if satisfaction_score >= 85:
            return 'Premium'
        elif satisfaction_score >= 70:
            return 'Mid-market'
        else:
            return 'Value'
            
    def _analyze_target_segments(self, segments: List[Dict[str, Any]], company_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze target market segments"""
        
        segment_performance = []
        
        for segment in segments:
            performance = {
                'segment_name': segment.get('name', 'Unknown'),
                'market_share': segment.get('our_market_share', 10),
                'growth_rate': segment.get('growth_rate', 5),
                'competitive_position': segment.get('competitive_position', 'Average'),
                'segment_attractiveness': self._calculate_segment_attractiveness(segment),
                'strategic_fit': self._assess_strategic_fit(segment, company_metrics)
            }
            segment_performance.append(performance)
            
        # Identify best performing segments
        best_segments = sorted(segment_performance, key=lambda x: x['segment_attractiveness'], reverse=True)[:3]
        
        return {
            'segment_performance': segment_performance,
            'best_performing_segments': [seg['segment_name'] for seg in best_segments],
            'underperforming_segments': [seg['segment_name'] for seg in segment_performance if seg['market_share'] < 5],
            'growth_segments': [seg['segment_name'] for seg in segment_performance if seg['growth_rate'] > 10]
        }
        
    def _calculate_segment_attractiveness(self, segment: Dict[str, Any]) -> float:
        """Calculate segment attractiveness score"""
        
        factors = {
            'size': segment.get('size_score', 5),
            'growth': segment.get('growth_rate', 5),
            'profitability': segment.get('profitability_score', 5),
            'competitive_intensity': 10 - segment.get('competitive_intensity', 5)  # Lower intensity is better
        }
        
        # Weighted average
        weights = {'size': 0.3, 'growth': 0.3, 'profitability': 0.25, 'competitive_intensity': 0.15}
        attractiveness = sum(factors[factor] * weights[factor] for factor in factors)
        
        return attractiveness
        
    def _assess_strategic_fit(self, segment: Dict[str, Any], company_metrics: Dict[str, Any]) -> float:
        """Assess strategic fit with company capabilities"""
        
        fit_factors = {
            'capability_match': segment.get('capability_requirements_match', 70),
            'resource_requirements': 100 - segment.get('resource_intensity', 50),  # Lower requirements = better fit
            'brand_alignment': segment.get('brand_fit_score', 70),
            'channel_compatibility': segment.get('channel_fit_score', 80)
        }
        
        strategic_fit = sum(fit_factors.values()) / len(fit_factors) / 100
        
        return strategic_fit
        
    def _identify_positioning_gaps(self, current_position: Dict[str, Any], segment_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify positioning gaps"""
        
        gaps = []
        
        # Position consistency gap
        if current_position['position_consistency'] < 0.75:
            gaps.append({
                'type': 'Position Consistency',
                'gap': 'Inconsistent positioning across different dimensions',
                'impact': 'Medium',
                'recommendation': 'Align pricing, quality, and service positioning'
            })
            
        # Segment coverage gaps
        underperforming_segments = segment_analysis.get('underperforming_segments', [])
        if underperforming_segments:
            gaps.append({
                'type': 'Segment Coverage',
                'gap': f'Underperformance in {len(underperforming_segments)} market segments',
                'impact': 'High',
                'recommendation': 'Develop targeted strategies for underperforming segments'
            })
            
        # Growth segment gaps
        growth_segments = segment_analysis.get('growth_segments', [])
        if len(growth_segments) > len(segment_analysis.get('best_performing_segments', [])):
            gaps.append({
                'type': 'Growth Opportunity',
                'gap': 'Missing opportunities in high-growth segments',
                'impact': 'High',
                'recommendation': 'Increase focus on high-growth market segments'
            })
            
        return gaps
        
    def _develop_positioning_strategy(self, current_position: Dict[str, Any], gaps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Develop positioning strategy recommendations"""
        
        strategy = {
            'recommended_position': current_position['overall_position'],
            'positioning_objectives': [],
            'strategic_priorities': [],
            'implementation_timeline': '6-18 months'
        }
        
        # Address gaps
        for gap in gaps:
            if gap['type'] == 'Position Consistency':
                strategy['positioning_objectives'].append('Achieve consistent positioning across all dimensions')
                strategy['strategic_priorities'].append('Align product, pricing, and service strategies')
                
            elif gap['type'] == 'Segment Coverage':
                strategy['positioning_objectives'].append('Improve performance in underperforming segments')
                strategy['strategic_priorities'].append('Develop segment-specific value propositions')
                
            elif gap['type'] == 'Growth Opportunity':
                strategy['positioning_objectives'].append('Capture opportunities in high-growth segments')
                strategy['strategic_priorities'].append('Increase investment in growth segments')
                
        # Default objectives if no gaps
        if not strategy['positioning_objectives']:
            strategy['positioning_objectives'] = ['Strengthen current market position', 'Defend against competitive threats']
            strategy['strategic_priorities'] = ['Enhance competitive advantages', 'Improve customer retention']
            
        return strategy
        
    def _generate_market_recommendations(self, competitive: Dict[str, Any], 
                                       trends: Dict[str, Any], 
                                       positioning: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate comprehensive market recommendations"""
        
        recommendations = []
        
        # Competitive response recommendations
        high_threat_competitors = competitive.get('key_competitive_threats', [])
        if high_threat_competitors:
            recommendations.append({
                'category': 'Competitive Response',
                'priority': 'High',
                'recommendation': f'Develop strategies to counter {len(high_threat_competitors)} high-threat competitors',
                'actions': [
                    'Analyze competitor strategies and weaknesses',
                    'Strengthen competitive advantages',
                    'Consider strategic partnerships or acquisitions'
                ],
                'timeline': '3-6 months'
            })
            
        # Market opportunity recommendations
        high_priority_opportunities = [opp for opp in trends.get('market_opportunities', []) if opp.get('priority') == 'High']
        if high_priority_opportunities:
            recommendations.append({
                'category': 'Market Opportunities',
                'priority': 'High',
                'recommendation': 'Capitalize on high-priority market opportunities',
                'actions': [
                    'Develop business cases for opportunity pursuit',
                    'Allocate resources to high-potential areas',
                    'Establish metrics to track opportunity capture'
                ],
                'timeline': '6-12 months'
            })
            
        # Positioning improvement recommendations
        positioning_gaps = positioning.get('positioning_gaps', [])
        if positioning_gaps:
            recommendations.append({
                'category': 'Market Positioning',
                'priority': 'Medium',
                'recommendation': 'Address identified positioning gaps',
                'actions': [gap['recommendation'] for gap in positioning_gaps],
                'timeline': '6-18 months'
            })
            
        # Market intelligence recommendations
        recommendations.append({
            'category': 'Market Intelligence',
            'priority': 'Medium',
            'recommendation': 'Enhance market intelligence capabilities',
            'actions': [
                'Implement competitive monitoring systems',
                'Establish market trend analysis processes',
                'Develop customer insight programs'
            ],
            'timeline': '3-9 months'
        })
        
        return recommendations

def test_market_intelligence_agent():
    """Test the Market Intelligence Agent"""
    print("🧪 Testing Market Intelligence & Competitive Agent")
    print("=" * 55)
    
    try:
        agent = MarketIntelligenceAgent()
        
        test_data = {
            'company_name': 'Market Leaders Inc',
            'company_metrics': {
                'market_share': 15,
                'revenue': 50000000,
                'product_rating': 8.2,
                'innovation_score': 7.5,
                'customer_satisfaction': 82,
                'price_level': 65
            },
            'competitors': [
                {
                    'name': 'Competitor A',
                    'market_share': 25,
                    'revenue': 80000000,
                    'product_rating': 8.5,
                    'innovation_score': 8.0,
                    'growth_rate': 12
                },
                {
                    'name': 'Competitor B', 
                    'market_share': 18,
                    'revenue': 60000000,
                    'product_rating': 7.8,
                    'innovation_score': 6.5,
                    'growth_rate': 8
                }
            ],
            'current_trends': [
                'AI adoption acceleration',
                'Digital transformation focus',
                'Increased regulatory scrutiny',
                'Economic uncertainty'
            ],
            'target_segments': [
                {
                    'name': 'Enterprise Customers',
                    'our_market_share': 20,
                    'growth_rate': 15,
                    'size_score': 8,
                    'profitability_score': 9
                }
            ]
        }
        
        analysis = agent.analyze_market_position(test_data)
        print(f"✅ Market analysis completed for {test_data['company_name']}")
        print(f"   Competitor profiles: {len(analysis['competitive_analysis']['competitor_profiles'])}")
        print(f"   Market opportunities: {len(analysis['trends_analysis']['market_opportunities'])}")
        print(f"   Strategic recommendations: {len(analysis['recommendations'])}")
        
        return {
            'agent_initialized': True,
            'competitors_analyzed': len(analysis['competitive_analysis']['competitor_profiles']),
            'opportunities_identified': len(analysis['trends_analysis']['market_opportunities']),
            'recommendations_generated': len(analysis['recommendations'])
        }
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return {'error': str(e)}

if __name__ == "__main__":
    test_market_intelligence_agent()