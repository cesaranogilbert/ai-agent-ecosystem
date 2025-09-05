"""
Comprehensive test suite for Metaverse Economy Architect Agent
Tests all functionality including virtual real estate, NFT optimization, avatar management, and interoperability
"""

import pytest
import asyncio
import json
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add services directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services'))

from metaverse_economy_architect_agent import (
    MetaverseEconomyArchitectAgent,
    VirtualAsset,
    MetaversePlatform,
    EconomicModel
)

class TestMetaverseEconomyArchitectAgent:
    """Test suite for Metaverse Economy Architect Agent"""
    
    @pytest.fixture
    def agent(self):
        """Create agent instance for testing"""
        return MetaverseEconomyArchitectAgent()
    
    @pytest.fixture
    def sample_virtual_asset_data(self):
        """Sample virtual asset data for testing"""
        return {
            'asset_id': 'test_land_001',
            'platform': 'decentraland',
            'coordinates': {'x': 100, 'y': 150},
            'size': 16,  # 4x4 parcel
            'current_price': 5000,  # MANA
            'asset_type': 'land'
        }
    
    @pytest.fixture
    def sample_nft_marketplace_config(self):
        """Sample NFT marketplace configuration"""
        return {
            'marketplace_id': 'test_marketplace_001',
            'platform': 'ethereum',
            'collection_size': 5000,
            'target_audience': 'gaming',
            'launch_strategy': 'phased_release'
        }
    
    @pytest.fixture
    def sample_identity_config(self):
        """Sample avatar identity configuration"""
        return {
            'user_id': 'test_user_001',
            'platforms': ['decentraland', 'sandbox', 'horizon_worlds'],
            'current_assets': [
                {'type': 'avatar', 'platform': 'decentraland', 'value': 1000},
                {'type': 'wearables', 'platform': 'sandbox', 'value': 500}
            ],
            'goals': ['cross_platform_presence', 'monetization', 'reputation_building']
        }
    
    def test_agent_initialization(self, agent):
        """Test agent initialization and basic properties"""
        assert agent.agent_id == "metaverse_economy_architect"
        assert agent.version == "1.0.0"
        assert len(agent.platforms) > 0
        assert hasattr(agent, 'economic_models')
        assert hasattr(agent, 'portfolio_strategies')
        assert hasattr(agent, 'interoperability_protocols')
    
    def test_platform_initialization(self, agent):
        """Test platform initialization"""
        platforms = agent.platforms
        assert len(platforms) >= 3
        
        # Check that platforms have required attributes
        for platform in platforms:
            assert hasattr(platform, 'platform_id')
            assert hasattr(platform, 'name')
            assert hasattr(platform, 'blockchain')
            assert hasattr(platform, 'native_currency')
            assert platform.user_base > 0
            assert 0 <= platform.interoperability_score <= 1
    
    @pytest.mark.asyncio
    async def test_virtual_real_estate_analysis(self, agent, sample_virtual_asset_data):
        """Test virtual real estate analysis functionality"""
        result = await agent.analyze_virtual_real_estate(sample_virtual_asset_data)
        
        # Verify response structure
        assert 'asset_id' in result
        assert 'platform' in result
        assert 'current_valuation' in result
        assert 'investment_analysis' in result
        assert 'yield_opportunities' in result
        assert 'risk_factors' in result
        assert 'market_trends' in result
        assert 'price_predictions' in result
        
        # Verify data types and ranges
        valuation = result['current_valuation']
        assert isinstance(valuation['location_score'], float)
        assert 0 <= valuation['location_score'] <= 1
        
        investment_analysis = result['investment_analysis']
        assert investment_analysis['recommendation'] in ['strong_buy', 'buy', 'hold', 'sell']
        assert 0 <= investment_analysis['confidence_score'] <= 1
        
        # Verify price predictions have reasonable values
        predictions = result['price_predictions']
        assert predictions['fair_value'] > 0
        assert predictions['confidence_interval'] > 0
    
    @pytest.mark.asyncio
    async def test_nft_marketplace_optimization(self, agent, sample_nft_marketplace_config):
        """Test NFT marketplace strategy optimization"""
        result = await agent.optimize_nft_marketplace_strategy(sample_nft_marketplace_config)
        
        # Verify response structure
        assert 'marketplace_id' in result
        assert 'optimized_tokenomics' in result
        assert 'pricing_strategy' in result
        assert 'launch_strategy' in result
        assert 'marketing_strategy' in result
        assert 'revenue_optimization' in result
        assert 'risk_mitigation' in result
        assert 'projected_metrics' in result
        
        # Verify tokenomics
        tokenomics = result['optimized_tokenomics']
        assert tokenomics['total_supply'] == sample_nft_marketplace_config['collection_size']
        assert tokenomics['mint_price'] > 0
        assert 0 <= tokenomics['royalty_percentage'] <= 0.1  # Reasonable royalty range
        
        # Verify revenue projections
        revenue = result['revenue_optimization']
        assert revenue['total_revenue'] > 0
        assert revenue['creator_roi'] > 1  # Should be profitable
        
        # Verify risk mitigation
        risk_mitigation = result['risk_mitigation']
        assert risk_mitigation['smart_contract_audit'] is True
        assert risk_mitigation['legal_compliance'] == 'full'
    
    @pytest.mark.asyncio
    async def test_avatar_identity_management(self, agent, sample_identity_config):
        """Test avatar identity and portfolio management"""
        result = await agent.manage_avatar_identity_portfolio(sample_identity_config)
        
        # Verify response structure
        assert 'user_id' in result
        assert 'identity_optimization' in result
        assert 'portfolio_strategy' in result
        assert 'reputation_management' in result
        assert 'interoperability_plan' in result
        assert 'monetization_opportunities' in result
        assert 'implementation_roadmap' in result
        
        # Verify identity optimization
        identity_opt = result['identity_optimization']
        assert identity_opt['unified_identity'] is True
        assert len(identity_opt['avatar_standards']) > 0
        
        # Verify portfolio strategy
        portfolio = result['portfolio_strategy']
        assert 0 <= portfolio['diversification_score'] <= 1
        assert portfolio['yield_optimization'] > 0
        
        # Verify monetization opportunities
        monetization = result['monetization_opportunities']
        assert len(monetization['revenue_streams']) > 0
        assert monetization['projected_income'] > 0
        
        # Verify implementation roadmap
        roadmap = result['implementation_roadmap']
        assert len(roadmap['immediate_actions']) > 0
        assert len(roadmap['medium_term_goals']) > 0
    
    @pytest.mark.asyncio
    async def test_interoperability_solution_design(self, agent):
        """Test cross-platform interoperability solution design"""
        interop_requirements = {
            'source_platforms': ['decentraland', 'sandbox'],
            'target_platforms': ['horizon_worlds', 'roblox'],
            'asset_types': ['avatars', 'wearables', 'land'],
            'goals': ['asset_portability', 'identity_continuity']
        }
        
        result = await agent.design_interoperability_solution(interop_requirements)
        
        # Verify response structure
        assert 'solution_id' in result
        assert 'technical_architecture' in result
        assert 'protocol_strategy' in result
        assert 'bridge_implementation' in result
        assert 'security_framework' in result
        assert 'cost_analysis' in result
        assert 'implementation_plan' in result
        assert 'success_metrics' in result
        
        # Verify technical architecture
        tech_arch = result['technical_architecture']
        assert tech_arch['architecture_type'] in ['hub_and_spoke', 'mesh', 'federated']
        assert tech_arch['expected_tps'] > 0
        
        # Verify security framework
        security = result['security_framework']
        assert len(security['security_layers']) > 0
        assert 0.8 <= security['security_score'] <= 1.0
        
        # Verify cost analysis
        cost_analysis = result['cost_analysis']
        assert cost_analysis['transaction_cost'] > 0
        assert 0 <= cost_analysis['savings_percentage'] <= 1
    
    @pytest.mark.asyncio
    async def test_location_score_calculation(self, agent):
        """Test location scoring for virtual real estate"""
        platform = "decentraland"
        coordinates = {"x": 0, "y": 0}  # Prime location
        
        score = await agent._calculate_location_score(platform, coordinates)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
    
    @pytest.mark.asyncio
    async def test_market_trends_analysis(self, agent):
        """Test market trends analysis"""
        platform = "sandbox"
        trends = await agent._analyze_market_trends(platform)
        
        assert 'trend_direction' in trends
        assert 'momentum_score' in trends
        assert 'volume_trend' in trends
        assert 'user_growth_rate' in trends
        assert 'price_volatility' in trends
        
        assert trends['trend_direction'] in ['bullish', 'bearish', 'neutral']
        assert 0 <= trends['momentum_score'] <= 1
        assert trends['user_growth_rate'] >= 0
    
    @pytest.mark.asyncio
    async def test_yield_potential_calculation(self, agent, sample_virtual_asset_data):
        """Test yield potential calculation"""
        yield_analysis = await agent._calculate_yield_potential(sample_virtual_asset_data)
        
        assert 'rental_yield' in yield_analysis
        assert 'staking_opportunities' in yield_analysis
        assert 'development_roi' in yield_analysis
        assert 'passive_income_potential' in yield_analysis
        
        assert yield_analysis['rental_yield'] > 0
        assert yield_analysis['development_roi'] >= 1
        assert yield_analysis['passive_income_potential'] >= 0
    
    @pytest.mark.asyncio
    async def test_investment_risk_assessment(self, agent, sample_virtual_asset_data):
        """Test investment risk assessment"""
        risk_assessment = await agent._assess_investment_risks(sample_virtual_asset_data)
        
        assert 'platform_risk' in risk_assessment
        assert 'liquidity_risk' in risk_assessment
        assert 'regulatory_risk' in risk_assessment
        assert 'technology_risk' in risk_assessment
        assert 'market_risk' in risk_assessment
        assert 'overall_risk_score' in risk_assessment
        
        # Verify risk levels are valid
        risk_levels = ['low', 'medium', 'high']
        assert risk_assessment['platform_risk'] in risk_levels
        assert risk_assessment['liquidity_risk'] in risk_levels
        assert risk_assessment['regulatory_risk'] in risk_levels
        
        assert 0 <= risk_assessment['overall_risk_score'] <= 1
    
    @pytest.mark.asyncio
    async def test_price_prediction(self, agent, sample_virtual_asset_data):
        """Test asset price prediction"""
        market_trends = {
            'momentum_score': 0.8,
            'trend_direction': 'bullish',
            'volume_trend': 'increasing'
        }
        
        prediction = await agent._predict_asset_price(sample_virtual_asset_data, market_trends)
        
        assert 'fair_value' in prediction
        assert '30_day_prediction' in prediction
        assert '90_day_prediction' in prediction
        assert '1_year_prediction' in prediction
        assert 'confidence_interval' in prediction
        
        current_price = sample_virtual_asset_data['current_price']
        assert prediction['fair_value'] > 0
        assert prediction['30_day_prediction'] > 0
        assert prediction['confidence_interval'] > 0
    
    def test_get_agent_capabilities(self, agent):
        """Test agent capabilities reporting"""
        capabilities = agent.get_agent_capabilities()
        
        assert 'agent_id' in capabilities
        assert 'version' in capabilities
        assert 'capabilities' in capabilities
        assert 'supported_platforms' in capabilities
        assert 'market_coverage' in capabilities
        assert 'specializations' in capabilities
        
        assert capabilities['agent_id'] == agent.agent_id
        assert len(capabilities['capabilities']) >= 4
        assert len(capabilities['specializations']) >= 4
        assert '$800B' in capabilities['market_coverage']
    
    @pytest.mark.asyncio
    async def test_error_handling(self, agent):
        """Test error handling for invalid inputs"""
        # Test with missing required data
        invalid_data = {}
        
        result = await agent.analyze_virtual_real_estate(invalid_data)
        assert 'error' not in result or 'asset_id' in result  # Should handle gracefully
        
        # Test with invalid platform
        invalid_marketplace = {
            'marketplace_id': 'test',
            'platform': 'invalid_platform',
            'collection_size': -1  # Invalid size
        }
        
        result = await agent.optimize_nft_marketplace_strategy(invalid_marketplace)
        # Should handle gracefully without crashing
        assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, agent, sample_virtual_asset_data, sample_nft_marketplace_config):
        """Test concurrent operation handling"""
        # Test multiple concurrent operations
        tasks = [
            agent.analyze_virtual_real_estate(sample_virtual_asset_data),
            agent.optimize_nft_marketplace_strategy(sample_nft_marketplace_config),
            agent.manage_avatar_identity_portfolio({
                'user_id': 'test_user',
                'platforms': ['decentraland'],
                'current_assets': [],
                'goals': ['test']
            })
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All operations should complete successfully
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, dict)
    
    def test_platform_scoring_logic(self, agent):
        """Test platform scoring and selection logic"""
        platforms = agent.platforms
        
        # Verify each platform has scoring attributes
        for platform in platforms:
            assert hasattr(platform, 'user_base')
            assert hasattr(platform, 'daily_active_users')
            assert hasattr(platform, 'transaction_volume')
            assert hasattr(platform, 'interoperability_score')
            
            # Verify reasonable values
            assert platform.user_base > 0
            assert platform.daily_active_users <= platform.user_base
            assert platform.transaction_volume >= 0
            assert 0 <= platform.interoperability_score <= 1
    
    @pytest.mark.asyncio
    async def test_economic_modeling(self, agent):
        """Test economic modeling capabilities"""
        # Test economic model creation
        model_data = {
            'platform': 'test_platform',
            'supply_mechanisms': ['minting', 'staking'],
            'demand_drivers': ['utility', 'speculation'],
            'inflation_rate': 0.05
        }
        
        # The agent should handle economic modeling
        # This tests the economic modeling logic indirectly through real estate analysis
        asset_data = {
            'asset_id': 'economic_test',
            'platform': 'decentraland',
            'current_price': 1000,
            'coordinates': {'x': 50, 'y': 50}
        }
        
        result = await agent.analyze_virtual_real_estate(asset_data)
        
        # Verify economic factors are considered
        assert 'market_trends' in result
        assert 'yield_opportunities' in result
        market_trends = result['market_trends']
        assert 'user_growth_rate' in market_trends
        assert 'price_volatility' in market_trends
    
    @pytest.mark.asyncio
    async def test_portfolio_optimization(self, agent):
        """Test portfolio optimization strategies"""
        portfolio_data = {
            'user_id': 'portfolio_test',
            'platforms': ['decentraland', 'sandbox', 'horizon_worlds'],
            'current_assets': [
                {'type': 'land', 'platform': 'decentraland', 'value': 5000, 'yield': 0.12},
                {'type': 'avatar', 'platform': 'sandbox', 'value': 1000, 'yield': 0.08},
                {'type': 'wearables', 'platform': 'horizon_worlds', 'value': 500, 'yield': 0.15}
            ],
            'goals': ['diversification', 'yield_optimization', 'risk_management']
        }
        
        result = await agent.manage_avatar_identity_portfolio(portfolio_data)
        
        # Verify portfolio optimization
        portfolio_strategy = result['portfolio_strategy']
        assert 'diversification_score' in portfolio_strategy
        assert 'yield_optimization' in portfolio_strategy
        assert 'risk_adjusted_return' in portfolio_strategy
        
        # Verify optimization metrics are reasonable
        assert 0 <= portfolio_strategy['diversification_score'] <= 1
        assert portfolio_strategy['yield_optimization'] > 0
        assert portfolio_strategy['risk_adjusted_return'] > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])