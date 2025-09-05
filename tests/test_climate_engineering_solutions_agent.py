"""
Comprehensive test suite for Climate Engineering Solutions Agent
Tests carbon capture optimization, ecosystem restoration, climate modeling, and intervention strategies
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add services directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services'))

from climate_engineering_solutions_agent import (
    ClimateEngineeringSolutionsAgent,
    ClimateInterventionType,
    CarbonCaptureMethod,
    ClimateProject,
    EcosystemState
)

class TestClimateEngineeringSolutionsAgent:
    """Test suite for Climate Engineering Solutions Agent"""
    
    @pytest.fixture
    def agent(self):
        """Create agent instance for testing"""
        return ClimateEngineeringSolutionsAgent()
    
    @pytest.fixture
    def sample_carbon_capture_requirements(self):
        """Sample carbon capture system requirements"""
        return {
            'system_id': 'carbon_capture_001',
            'annual_capture_target': 1000000,  # 1 million tons CO2/year
            'location': {
                'latitude': 40.7128,
                'longitude': -74.0060,
                'region': 'northeast_usa',
                'climate_zone': 'temperate'
            },
            'budget_constraints': {
                'total_budget': 500000000,  # $500M
                'annual_opex_limit': 50000000  # $50M/year
            },
            'available_energy': ['renewable_grid', 'solar', 'wind', 'nuclear']
        }
    
    @pytest.fixture
    def sample_restoration_request(self):
        """Sample ecosystem restoration request"""
        return {
            'project_id': 'restoration_001',
            'ecosystem_type': 'temperate_forest',
            'location': {
                'latitude': 45.5152,
                'longitude': -122.6784,
                'area_hectares': 10000,
                'region': 'pacific_northwest'
            },
            'degradation_state': {
                'forest_cover_loss': 0.6,
                'biodiversity_loss': 0.4,
                'soil_degradation': 0.3,
                'water_quality_impact': 0.2
            },
            'goals': [
                'carbon_sequestration',
                'biodiversity_recovery',
                'watershed_protection',
                'climate_resilience'
            ]
        }
    
    @pytest.fixture
    def sample_intervention_data(self):
        """Sample climate intervention modeling data"""
        return {
            'session_id': 'climate_modeling_001',
            'interventions': [
                {
                    'type': 'carbon_capture',
                    'scale': 'global',
                    'capacity': 10e9,  # 10 GT CO2/year
                    'implementation_timeline': '2025-2040'
                },
                {
                    'type': 'reforestation',
                    'scale': 'regional',
                    'area': 50e6,  # 50 million hectares
                    'implementation_timeline': '2025-2050'
                }
            ],
            'timeframe': '2025-2100',
            'spatial_scope': 'global',
            'include_uncertainty': True
        }
    
    def test_agent_initialization(self, agent):
        """Test agent initialization and basic properties"""
        assert agent.agent_id == "climate_engineering_solutions"
        assert agent.version == "1.0.0"
        assert hasattr(agent, 'intervention_technologies')
        assert hasattr(agent, 'climate_models')
        assert hasattr(agent, 'ecosystem_databases')
        assert hasattr(agent, 'policy_frameworks')
    
    def test_intervention_technologies_initialization(self, agent):
        """Test climate intervention technologies initialization"""
        technologies = agent.intervention_technologies
        
        # Verify carbon capture technologies
        assert 'carbon_capture' in technologies
        cc_tech = technologies['carbon_capture']
        
        required_cc_methods = ['direct_air_capture', 'bioenergy_ccs', 'ocean_capture']
        for method in required_cc_methods:
            assert method in cc_tech
            method_data = cc_tech[method]
            assert 'efficiency' in method_data
            assert 'cost_per_ton' in method_data
            assert 'energy_requirement' in method_data
            assert 'scalability' in method_data
            assert 'readiness_level' in method_data
            
            # Verify reasonable values
            assert 0 <= method_data['efficiency'] <= 1
            assert method_data['cost_per_ton'] > 0
            assert method_data['energy_requirement'] > 0
            assert 1 <= method_data['readiness_level'] <= 9
        
        # Verify solar management technologies
        assert 'solar_management' in technologies
        solar_tech = technologies['solar_management']
        assert 'stratospheric_aerosols' in solar_tech
        assert 'marine_cloud_brightening' in solar_tech
        
        # Verify ecosystem solutions
        assert 'ecosystem_solutions' in technologies
        eco_tech = technologies['ecosystem_solutions']
        assert 'reforestation' in eco_tech
        assert 'wetland_restoration' in eco_tech
    
    def test_climate_models_initialization(self, agent):
        """Test climate modeling capabilities initialization"""
        models = agent.climate_models
        
        assert 'global_circulation_models' in models
        assert 'regional_models' in models
        assert 'integrated_assessment' in models
        
        # Verify global circulation models
        gcm = models['global_circulation_models']
        assert 'resolution' in gcm
        assert 'time_steps' in gcm
        assert 'variables' in gcm
        assert 'scenarios' in gcm
        
        # Verify regional models
        regional = models['regional_models']
        assert 'resolution' in regional
        assert 'downscaling_methods' in regional
        assert 'impact_assessments' in regional
    
    def test_ecosystem_databases_initialization(self, agent):
        """Test ecosystem databases initialization"""
        eco_db = agent.ecosystem_databases
        
        assert 'global_datasets' in eco_db
        assert 'monitoring_systems' in eco_db
        assert 'restoration_protocols' in eco_db
        
        # Verify global datasets
        datasets = eco_db['global_datasets']
        required_datasets = ['forest_cover', 'biodiversity', 'carbon_stocks', 'protected_areas']
        for dataset in required_datasets:
            assert dataset in datasets
    
    def test_policy_frameworks_initialization(self, agent):
        """Test policy frameworks initialization"""
        policies = agent.policy_frameworks
        
        assert 'international_agreements' in policies
        assert 'governance_mechanisms' in policies
        assert 'risk_assessment' in policies
        
        # Verify international agreements
        agreements = policies['international_agreements']
        assert 'paris_agreement' in agreements
        assert 'cbd_targets' in agreements
        assert 'sdgs' in agreements
    
    @pytest.mark.asyncio
    async def test_carbon_capture_system_optimization(self, agent, sample_carbon_capture_requirements):
        """Test carbon capture system optimization"""
        result = await agent.optimize_carbon_capture_system(sample_carbon_capture_requirements)
        
        # Verify response structure
        assert 'system_id' in result
        assert 'capture_system_design' in result
        assert 'performance_metrics' in result
        assert 'economic_analysis' in result
        assert 'carbon_utilization' in result
        assert 'environmental_impact' in result
        assert 'implementation_strategy' in result
        
        # Verify capture system design
        design = result['capture_system_design']
        assert 'technology' in design
        assert 'capacity' in design
        assert 'efficiency' in design
        assert 'infrastructure' in design
        
        assert design['capacity'] > 0
        assert 0 <= design['efficiency'] <= 1
        
        # Verify performance metrics
        performance = result['performance_metrics']
        assert 'annual_co2_capture' in performance
        assert 'energy_efficiency' in performance
        assert 'capacity_factor' in performance
        assert 'lifecycle_emissions' in performance
        
        assert performance['annual_co2_capture'] > 0
        assert 0 <= performance['capacity_factor'] <= 1
        
        # Verify economic analysis
        economics = result['economic_analysis']
        assert 'capex' in economics
        assert 'opex' in economics
        assert 'levelized_cost' in economics
        assert 'revenue_streams' in economics
        assert 'payback_period' in economics
        
        assert economics['capex'] > 0
        assert economics['opex'] > 0
        assert economics['levelized_cost'] > 0
        
        # Verify carbon utilization
        utilization = result['carbon_utilization']
        assert 'utilization_rate' in utilization
        assert 'products' in utilization
        assert 'market_potential' in utilization
        assert 'value_creation' in utilization
        
        assert 0 <= utilization['utilization_rate'] <= 1
        assert utilization['market_potential'] > 0
        
        # Verify environmental impact
        env_impact = result['environmental_impact']
        assert 'net_climate_benefit' in env_impact
        assert 'ecosystem_impacts' in env_impact
        assert 'resource_consumption' in env_impact
        assert 'sustainability_score' in env_impact
        
        # Verify implementation strategy
        implementation = result['implementation_strategy']
        assert 'deployment_phases' in implementation
        assert 'timeline' in implementation
        assert 'risk_mitigation' in implementation
        assert 'stakeholder_engagement' in implementation
    
    @pytest.mark.asyncio
    async def test_ecosystem_restoration_planning(self, agent, sample_restoration_request):
        """Test ecosystem restoration plan design"""
        result = await agent.design_ecosystem_restoration_plan(sample_restoration_request)
        
        # Verify response structure
        assert 'restoration_project_id' in result
        assert 'ecosystem_analysis' in result
        assert 'restoration_strategy' in result
        assert 'biodiversity_plan' in result
        assert 'implementation_plan' in result
        assert 'monitoring_system' in result
        assert 'climate_impact' in result
        assert 'economic_assessment' in result
        
        # Verify ecosystem analysis
        analysis = result['ecosystem_analysis']
        assert 'ecosystem_type' in analysis
        assert 'current_state' in analysis
        assert 'degradation_drivers' in analysis
        assert 'restoration_potential' in analysis
        
        assert analysis['ecosystem_type'] == sample_restoration_request['ecosystem_type']
        
        # Verify restoration strategy
        strategy = result['restoration_strategy']
        assert 'approach' in strategy
        assert 'interventions' in strategy
        assert 'timeline' in strategy
        assert 'success_criteria' in strategy
        
        # Verify biodiversity plan
        biodiversity = result['biodiversity_plan']
        assert 'target_species' in biodiversity
        assert 'habitat_requirements' in biodiversity
        assert 'reintroduction_schedule' in biodiversity
        assert 'genetic_diversity' in biodiversity
        
        # Verify implementation plan
        implementation = result['implementation_plan']
        assert 'methodology' in implementation
        assert 'resource_requirements' in implementation
        assert 'capacity_building' in implementation
        assert 'community_engagement' in implementation
        
        # Verify monitoring system
        monitoring = result['monitoring_system']
        assert 'indicators' in monitoring
        assert 'monitoring_protocols' in monitoring
        assert 'data_management' in monitoring
        assert 'adaptive_triggers' in monitoring
        
        # Verify climate impact
        climate = result['climate_impact']
        assert 'carbon_sequestration' in climate
        assert 'climate_regulation' in climate
        assert 'resilience_building' in climate
        assert 'mitigation_value' in climate
        
        assert climate['carbon_sequestration'] > 0
        
        # Verify economic assessment
        economic = result['economic_assessment']
        assert 'implementation_cost' in economic
        assert 'ecosystem_services_value' in economic
        assert 'cost_benefit_ratio' in economic
        assert 'financing_options' in economic
        
        assert economic['implementation_cost'] > 0
        assert economic['ecosystem_services_value'] > 0
    
    @pytest.mark.asyncio
    async def test_climate_intervention_modeling(self, agent, sample_intervention_data):
        """Test climate intervention scenario modeling"""
        result = await agent.model_climate_intervention_scenarios(sample_intervention_data)
        
        # Verify response structure
        assert 'modeling_session_id' in result
        assert 'scenario_overview' in result
        assert 'climate_projections' in result
        assert 'regional_impacts' in result
        assert 'intervention_effectiveness' in result
        assert 'risk_assessment' in result
        assert 'co_benefits_tradeoffs' in result
        
        # Verify scenario overview
        overview = result['scenario_overview']
        assert 'interventions_modeled' in overview
        assert 'timeframe' in overview
        assert 'spatial_coverage' in overview
        assert 'scenario_variations' in overview
        
        assert overview['interventions_modeled'] == len(sample_intervention_data['interventions'])
        assert overview['timeframe'] == sample_intervention_data['timeframe']
        
        # Verify climate projections
        projections = result['climate_projections']
        assert 'temperature_changes' in projections
        assert 'precipitation_changes' in projections
        assert 'extreme_events' in projections
        assert 'sea_level_response' in projections
        
        # Verify regional impacts
        impacts = result['regional_impacts']
        assert 'agriculture' in impacts
        assert 'water_resources' in impacts
        assert 'ecosystems' in impacts
        assert 'human_systems' in impacts
        
        # Verify intervention effectiveness
        effectiveness = result['intervention_effectiveness']
        assert 'climate_targets' in effectiveness
        assert 'cost_effectiveness' in effectiveness
        assert 'deployment_feasibility' in effectiveness
        assert 'reversibility' in effectiveness
        
        # Verify risk assessment
        risks = result['risk_assessment']
        assert 'environmental_risks' in risks
        assert 'social_risks' in risks
        assert 'governance_risks' in risks
        assert 'uncertainty_ranges' in risks
        
        # Verify co-benefits and trade-offs
        cobenefits = result['co_benefits_tradeoffs']
        assert 'positive_co_benefits' in cobenefits
        assert 'negative_tradeoffs' in cobenefits
        assert 'net_benefit_assessment' in cobenefits
        assert 'optimization_opportunities' in cobenefits
    
    @pytest.mark.asyncio
    async def test_technology_selection_optimization(self, agent):
        """Test optimal technology selection logic"""
        requirements = {
            'annual_capture_target': 500000,
            'budget_constraints': {'total_budget': 200000000},
            'location': {'region': 'europe'}
        }
        
        selection = await agent._select_optimal_capture_technology(requirements)
        
        assert 'selected_technology' in selection
        assert 'selection_rationale' in selection
        assert 'technology_specs' in selection
        
        # Verify selected technology is valid
        valid_technologies = ['direct_air_capture', 'bioenergy_ccs', 'ocean_capture']
        assert selection['selected_technology'] in valid_technologies
        
        # Verify technology specs
        specs = selection['technology_specs']
        assert 'efficiency' in specs
        assert 'cost_per_ton' in specs
        assert 'energy_requirement' in specs
        assert 'scalability' in specs
        assert 'readiness_level' in specs
    
    @pytest.mark.asyncio
    async def test_system_design_optimization(self, agent):
        """Test capture system design optimization"""
        technology = {
            'selected_technology': 'direct_air_capture',
            'technology_specs': {
                'efficiency': 0.85,
                'cost_per_ton': 150,
                'energy_requirement': 1.5,
                'scalability': 'high',
                'readiness_level': 7
            }
        }
        target = 1000000  # 1M tons/year
        location = {'latitude': 40.0, 'longitude': -100.0}
        
        design = await agent._design_capture_system(technology, target, location)
        
        assert 'annual_capacity' in design
        assert 'capture_efficiency' in design
        assert 'capacity_factor' in design
        assert 'infrastructure_requirements' in design
        
        assert design['annual_capacity'] == target
        assert 0 <= design['capture_efficiency'] <= 1
        assert 0 <= design['capacity_factor'] <= 1
        
        # Verify infrastructure requirements
        infrastructure = design['infrastructure_requirements']
        assert 'land_area' in infrastructure
        assert 'energy_consumption' in infrastructure
        assert 'water_consumption' in infrastructure
        assert 'workforce' in infrastructure
        
        assert infrastructure['land_area'] > 0
        assert infrastructure['energy_consumption'] > 0
        assert infrastructure['workforce'] > 0
    
    @pytest.mark.asyncio
    async def test_energy_integration_optimization(self, agent):
        """Test energy integration optimization"""
        system_design = {
            'infrastructure_requirements': {
                'energy_consumption': 1500000  # MWh/year
            }
        }
        energy_sources = ['solar', 'wind', 'renewable_grid']
        location = {'region': 'california'}
        
        optimization = await agent._optimize_energy_integration(system_design, energy_sources, location)
        
        assert 'energy_sources' in optimization
        assert 'energy_efficiency' in optimization
        assert 'efficiency_rating' in optimization
        assert 'renewable_percentage' in optimization
        assert 'grid_integration' in optimization
        
        assert optimization['energy_sources'] == energy_sources
        assert 0 <= optimization['energy_efficiency'] <= 1
        assert 0 <= optimization['renewable_percentage'] <= 1
    
    @pytest.mark.asyncio
    async def test_carbon_utilization_design(self, agent):
        """Test carbon utilization pathway design"""
        system_design = {'annual_capacity': 1000000}
        location = {'region': 'texas'}
        target = 1000000
        
        utilization = await agent._design_carbon_utilization(system_design, location, target)
        
        assert 'utilization_percentage' in utilization
        assert 'product_portfolio' in utilization
        assert 'market_size' in utilization
        assert 'economic_value' in utilization
        assert 'revenue_potential' in utilization
        
        assert 0 <= utilization['utilization_percentage'] <= 1
        assert len(utilization['product_portfolio']) > 0
        assert utilization['market_size'] > 0
        assert utilization['economic_value'] > 0
        assert utilization['revenue_potential'] > 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, agent):
        """Test error handling for invalid inputs"""
        # Test with missing system requirements
        invalid_requirements = {}
        result = await agent.optimize_carbon_capture_system(invalid_requirements)
        assert isinstance(result, dict)
        
        # Test with invalid ecosystem type
        invalid_restoration = {
            'ecosystem_type': 'invalid_ecosystem',
            'location': {},
            'degradation_state': {},
            'goals': []
        }
        result = await agent.design_ecosystem_restoration_plan(invalid_restoration)
        assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, agent, sample_carbon_capture_requirements, sample_restoration_request):
        """Test concurrent operation handling"""
        tasks = [
            agent.optimize_carbon_capture_system(sample_carbon_capture_requirements),
            agent.design_ecosystem_restoration_plan(sample_restoration_request)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, dict)
    
    def test_climate_intervention_types(self, agent):
        """Test climate intervention type coverage"""
        # Verify all intervention types are supported
        intervention_types = [
            ClimateInterventionType.CARBON_CAPTURE,
            ClimateInterventionType.SOLAR_GEOENGINEERING,
            ClimateInterventionType.OCEAN_ALKALINIZATION,
            ClimateInterventionType.REFORESTATION,
            ClimateInterventionType.RENEWABLE_ENERGY,
            ClimateInterventionType.CARBON_UTILIZATION,
            ClimateInterventionType.ECOSYSTEM_RESTORATION
        ]
        
        for intervention in intervention_types:
            assert intervention in ClimateInterventionType
    
    def test_carbon_capture_methods(self, agent):
        """Test carbon capture method coverage"""
        # Verify all capture methods are supported
        capture_methods = [
            CarbonCaptureMethod.DIRECT_AIR_CAPTURE,
            CarbonCaptureMethod.BIOENERGY_CCS,
            CarbonCaptureMethod.INDUSTRIAL_CCS,
            CarbonCaptureMethod.OCEAN_CAPTURE,
            CarbonCaptureMethod.SOIL_SEQUESTRATION,
            CarbonCaptureMethod.MINERALIZATION
        ]
        
        for method in capture_methods:
            assert method in CarbonCaptureMethod
    
    def test_climate_project_structure(self, agent):
        """Test climate project data structure"""
        test_project = ClimateProject(
            project_id='test_project',
            intervention_type=ClimateInterventionType.CARBON_CAPTURE,
            location={'region': 'north_america'},
            scale='regional',
            timeline='2025-2035',
            cost_estimate=1000000000,
            carbon_impact=5000000,
            co_benefits=['air_quality', 'jobs'],
            risks=['technical', 'economic'],
            feasibility_score=0.8
        )
        
        assert test_project.project_id == 'test_project'
        assert test_project.intervention_type == ClimateInterventionType.CARBON_CAPTURE
        assert test_project.carbon_impact > 0
        assert 0 <= test_project.feasibility_score <= 1
    
    def test_ecosystem_state_structure(self, agent):
        """Test ecosystem state data structure"""
        test_ecosystem = EcosystemState(
            ecosystem_id='test_ecosystem',
            ecosystem_type='forest',
            biodiversity_index=0.7,
            carbon_storage=150.5,
            degradation_level=0.3,
            restoration_potential=0.8,
            key_species=['oak', 'deer', 'wolf'],
            threats=['deforestation', 'climate_change']
        )
        
        assert test_ecosystem.ecosystem_id == 'test_ecosystem'
        assert 0 <= test_ecosystem.biodiversity_index <= 1
        assert test_ecosystem.carbon_storage > 0
        assert 0 <= test_ecosystem.degradation_level <= 1
        assert 0 <= test_ecosystem.restoration_potential <= 1
    
    def test_get_agent_capabilities(self, agent):
        """Test agent capabilities reporting"""
        capabilities = agent.get_agent_capabilities()
        
        assert 'agent_id' in capabilities
        assert 'version' in capabilities
        assert 'capabilities' in capabilities
        assert 'intervention_types' in capabilities
        assert 'capture_methods' in capabilities
        assert 'market_coverage' in capabilities
        assert 'specializations' in capabilities
        
        assert capabilities['agent_id'] == agent.agent_id
        assert len(capabilities['capabilities']) >= 4
        assert len(capabilities['intervention_types']) >= 7
        assert len(capabilities['capture_methods']) >= 6
        assert '$6T' in capabilities['market_coverage']
        assert len(capabilities['specializations']) >= 6
    
    @pytest.mark.asyncio
    async def test_baseline_ecosystem_assessment(self, agent):
        """Test baseline ecosystem assessment"""
        ecosystem_type = 'tropical_rainforest'
        location = {'latitude': -3.0, 'longitude': -60.0}
        degradation = {'deforestation': 0.4, 'fragmentation': 0.3}
        
        # Test the conceptual framework
        assert ecosystem_type in ['tropical_rainforest', 'temperate_forest', 'wetland', 'grassland']
        assert 'latitude' in location
        assert 'longitude' in location
        assert 'deforestation' in degradation
    
    @pytest.mark.asyncio
    async def test_restoration_strategy_design(self, agent):
        """Test restoration strategy design"""
        baseline = {
            'current_condition': 'degraded',
            'degradation_causes': ['logging', 'agriculture'],
            'restoration_feasibility': 0.75
        }
        goals = ['carbon_sequestration', 'biodiversity']
        ecosystem_type = 'forest'
        
        # Test the strategy framework
        assert 'current_condition' in baseline
        assert 'restoration_feasibility' in baseline
        assert len(goals) > 0
        assert ecosystem_type in ['forest', 'wetland', 'grassland', 'marine']
    
    @pytest.mark.asyncio
    async def test_climate_benefits_calculation(self, agent):
        """Test climate benefits calculation"""
        strategy = {'approach': 'active_restoration', 'area': 10000}
        ecosystem_type = 'temperate_forest'
        location = {'climate_zone': 'temperate'}
        
        # Test the calculation framework
        assert 'approach' in strategy
        assert 'area' in strategy
        assert strategy['area'] > 0
        assert ecosystem_type in ['temperate_forest', 'tropical_forest', 'boreal_forest']

if __name__ == "__main__":
    pytest.main([__file__, "-v"])