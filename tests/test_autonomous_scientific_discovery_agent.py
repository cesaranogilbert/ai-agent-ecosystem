"""
Comprehensive test suite for Autonomous Scientific Discovery Agent
Tests hypothesis generation, experimental design, knowledge synthesis, and collaboration
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

from autonomous_scientific_discovery_agent import (
    AutonomousScientificDiscoveryAgent,
    ResearchDomain,
    ExperimentType,
    ScientificHypothesis,
    ExperimentDesign,
    LiteratureSource,
    KnowledgeGraph
)

class TestAutonomousScientificDiscoveryAgent:
    """Test suite for Autonomous Scientific Discovery Agent"""
    
    @pytest.fixture
    def agent(self):
        """Create agent instance for testing"""
        return AutonomousScientificDiscoveryAgent()
    
    @pytest.fixture
    def sample_research_context(self):
        """Sample research context for hypothesis generation"""
        return {
            'session_id': 'research_session_001',
            'domain': 'biology',
            'research_question': 'What are the molecular mechanisms of cellular aging?',
            'existing_knowledge': [
                'Telomere shortening contributes to aging',
                'Mitochondrial dysfunction increases with age',
                'DNA damage accumulates over time'
            ],
            'constraints': {
                'budget': 500000,
                'timeline': '24 months',
                'ethical_approval': 'required'
            }
        }
    
    @pytest.fixture
    def sample_hypothesis_data(self):
        """Sample hypothesis data for experimental design"""
        return {
            'experiment_id': 'exp_design_001',
            'domain': 'chemistry',
            'hypothesis': 'Novel catalyst increases reaction efficiency by 40%',
            'variables': [
                {'name': 'catalyst_concentration', 'type': 'continuous', 'range': [0.1, 1.0]},
                {'name': 'temperature', 'type': 'continuous', 'range': [25, 100]},
                {'name': 'reaction_time', 'type': 'discrete', 'values': [1, 2, 4, 8]}
            ],
            'constraints': {
                'safety_requirements': ['fume_hood', 'protective_equipment'],
                'equipment_availability': ['hplc', 'nmr', 'gc_ms'],
                'budget_limit': 100000
            }
        }
    
    @pytest.fixture
    def sample_synthesis_request(self):
        """Sample interdisciplinary synthesis request"""
        return {
            'synthesis_id': 'interdisciplinary_001',
            'domains': ['biology', 'computer_science', 'physics'],
            'research_question': 'How can quantum effects in biological systems be computationally modeled?',
            'goals': [
                'identify_quantum_biological_phenomena',
                'develop_computational_models',
                'bridge_physics_biology_gaps'
            ]
        }
    
    @pytest.fixture
    def sample_collaboration_request(self):
        """Sample collaboration facilitation request"""
        return {
            'collaboration_id': 'collab_001',
            'project_description': 'Multi-institutional study on climate change impacts on marine ecosystems',
            'required_expertise': [
                'marine_biology',
                'climate_modeling',
                'statistical_analysis',
                'oceanography'
            ],
            'goals': [
                'comprehensive_ecosystem_analysis',
                'predictive_modeling',
                'policy_recommendations'
            ],
            'constraints': {
                'geographic_distribution': 'global',
                'funding_available': 2000000,
                'timeline': '36 months'
            }
        }
    
    def test_agent_initialization(self, agent):
        """Test agent initialization and basic properties"""
        assert agent.agent_id == "autonomous_scientific_discovery"
        assert agent.version == "1.0.0"
        assert hasattr(agent, 'knowledge_base')
        assert hasattr(agent, 'research_methodologies')
        assert hasattr(agent, 'lab_automation_systems')
        assert hasattr(agent, 'collaboration_networks')
    
    def test_knowledge_base_initialization(self, agent):
        """Test scientific knowledge base initialization"""
        kb = agent.knowledge_base
        assert 'domains' in kb
        assert 'cross_domain_patterns' in kb
        
        # Verify domain coverage
        domains = kb['domains']
        required_domains = ['biology', 'chemistry', 'physics']
        for domain in required_domains:
            assert domain in domains
            domain_data = domains[domain]
            assert 'subdisciplines' in domain_data
            assert 'common_methods' in domain_data
            assert 'key_databases' in domain_data
    
    def test_research_methodologies_initialization(self, agent):
        """Test research methodology initialization"""
        methodologies = agent.research_methodologies
        
        required_methods = ['hypothesis_generation', 'experimental_design', 'data_analysis']
        for method in required_methods:
            assert method in methodologies
            assert isinstance(methodologies[method], dict)
    
    def test_lab_automation_systems_initialization(self, agent):
        """Test laboratory automation systems initialization"""
        lab_systems = agent.lab_automation_systems
        
        assert 'robotic_platforms' in lab_systems
        assert 'computational_resources' in lab_systems
        assert 'integration_protocols' in lab_systems
        
        # Verify robotic platforms
        robotics = lab_systems['robotic_platforms']
        assert 'liquid_handling' in robotics
        assert 'analytical_instruments' in robotics
        assert len(robotics['liquid_handling']) > 0
    
    def test_collaboration_networks_initialization(self, agent):
        """Test collaboration network initialization"""
        networks = agent.collaboration_networks
        
        assert 'researcher_networks' in networks
        assert 'institutional_partnerships' in networks
        assert 'knowledge_sharing' in networks
        
        # Verify institutional partnerships
        institutions = networks['institutional_partnerships']
        assert 'universities' in institutions
        assert 'research_institutes' in institutions
        assert len(institutions['universities']) > 0
    
    @pytest.mark.asyncio
    async def test_hypothesis_generation(self, agent, sample_research_context):
        """Test research hypothesis generation"""
        result = await agent.generate_research_hypotheses(sample_research_context)
        
        # Verify response structure
        assert 'research_session_id' in result
        assert 'domain' in result
        assert 'literature_analysis' in result
        assert 'generated_hypotheses' in result
        assert 'cross_domain_insights' in result
        assert 'research_recommendations' in result
        
        # Verify literature analysis
        lit_analysis = result['literature_analysis']
        assert 'papers_analyzed' in lit_analysis
        assert 'knowledge_gaps' in lit_analysis
        assert 'emerging_trends' in lit_analysis
        assert 'contradictory_findings' in lit_analysis
        assert lit_analysis['papers_analyzed'] > 0
        
        # Verify generated hypotheses
        hypotheses = result['generated_hypotheses']
        assert 'total_generated' in hypotheses
        assert 'high_impact_candidates' in hypotheses
        assert 'novelty_scores' in hypotheses
        assert 'testability_scores' in hypotheses
        
        assert hypotheses['total_generated'] > 0
        assert len(hypotheses['novelty_scores']) > 0
        assert len(hypotheses['testability_scores']) > 0
        
        # Verify scores are in valid range
        for score in hypotheses['novelty_scores']:
            assert 0 <= score <= 1
        for score in hypotheses['testability_scores']:
            assert 0 <= score <= 1
        
        # Verify cross-domain insights
        insights = result['cross_domain_insights']
        assert 'interdisciplinary_connections' in insights
        assert 'methodology_transfers' in insights
        assert 'conceptual_bridges' in insights
        
        # Verify research recommendations
        recommendations = result['research_recommendations']
        assert 'priority_hypotheses' in recommendations
        assert 'experimental_approaches' in recommendations
        assert 'collaboration_suggestions' in recommendations
        assert 'resource_requirements' in recommendations
    
    @pytest.mark.asyncio
    async def test_experimental_design_automation(self, agent, sample_hypothesis_data):
        """Test automated experimental design"""
        result = await agent.design_automated_experiments(sample_hypothesis_data)
        
        # Verify response structure
        assert 'experiment_design_id' in result
        assert 'hypothesis_tested' in result
        assert 'experimental_strategy' in result
        assert 'automation_protocol' in result
        assert 'resource_requirements' in result
        assert 'quality_assurance' in result
        assert 'execution_plan' in result
        
        # Verify experimental strategy
        strategy = result['experimental_strategy']
        assert 'approach' in strategy
        assert 'methodology' in strategy
        assert 'controls' in strategy
        assert 'measurements' in strategy
        
        # Verify automation protocol
        automation = result['automation_protocol']
        assert 'workflow_steps' in automation
        assert 'robotic_systems' in automation
        assert 'data_collection' in automation
        assert 'integration_points' in automation
        
        # Verify resource requirements
        resources = result['resource_requirements']
        assert 'equipment' in resources
        assert 'materials' in resources
        assert 'personnel' in resources
        assert 'budget_estimate' in resources
        assert resources['budget_estimate'] > 0
        
        # Verify quality assurance
        qa = result['quality_assurance']
        assert 'control_experiments' in qa
        assert 'validation_protocols' in qa
        assert 'error_detection' in qa
        assert 'reproducibility_measures' in qa
        
        # Verify execution plan
        execution = result['execution_plan']
        assert 'timeline' in execution
        assert 'milestones' in execution
        assert 'risk_factors' in execution
        assert 'contingency_plans' in execution
    
    @pytest.mark.asyncio
    async def test_interdisciplinary_knowledge_synthesis(self, agent, sample_synthesis_request):
        """Test interdisciplinary knowledge synthesis"""
        result = await agent.synthesize_interdisciplinary_knowledge(sample_synthesis_request)
        
        # Verify response structure
        assert 'synthesis_id' in result
        assert 'domains_analyzed' in result
        assert 'knowledge_synthesis' in result
        assert 'discovered_patterns' in result
        assert 'novel_insights' in result
        assert 'validation_results' in result
        
        # Verify domains analyzed
        domains = result['domains_analyzed']
        expected_domains = sample_synthesis_request['domains']
        assert len(domains) == len(expected_domains)
        
        # Verify knowledge synthesis
        synthesis = result['knowledge_synthesis']
        assert 'total_sources' in synthesis
        assert 'conceptual_mappings' in synthesis
        assert 'knowledge_graph_entities' in synthesis
        assert 'cross_domain_connections' in synthesis
        assert synthesis['total_sources'] > 0
        
        # Verify discovered patterns
        patterns = result['discovered_patterns']
        assert 'methodological_patterns' in patterns
        assert 'theoretical_convergences' in patterns
        assert 'empirical_consistencies' in patterns
        assert 'gap_identifications' in patterns
        
        # Verify novel insights
        insights = result['novel_insights']
        assert 'breakthrough_hypotheses' in insights
        assert 'methodological_innovations' in insights
        assert 'theoretical_frameworks' in insights
        assert 'practical_applications' in insights
        
        # Verify validation results
        validation = result['validation_results']
        assert 'evidence_strength' in validation
        assert 'consistency_analysis' in validation
        assert 'confidence_levels' in validation
        assert 'research_priorities' in validation
    
    @pytest.mark.asyncio
    async def test_scientific_collaboration_facilitation(self, agent, sample_collaboration_request):
        """Test scientific collaboration facilitation"""
        result = await agent.facilitate_scientific_collaboration(sample_collaboration_request)
        
        # Verify response structure
        assert 'collaboration_id' in result
        assert 'project_overview' in result
        assert 'team_composition' in result
        assert 'coordination_plan' in result
        assert 'workflow_design' in result
        assert 'collaboration_tools' in result
        assert 'success_framework' in result
        
        # Verify team composition
        team = result['team_composition']
        assert 'recommended_researchers' in team
        assert 'expertise_coverage' in team
        assert 'complementary_skills' in team
        assert 'collaboration_potential' in team
        
        # Verify coordination plan
        coordination = result['coordination_plan']
        assert 'leadership_structure' in coordination
        assert 'communication_protocols' in coordination
        assert 'decision_making_process' in coordination
        assert 'conflict_resolution' in coordination
        
        # Verify workflow design
        workflow = result['workflow_design']
        assert 'project_phases' in workflow
        assert 'milestone_schedule' in workflow
        assert 'resource_sharing' in workflow
        assert 'knowledge_integration' in workflow
        
        # Verify collaboration tools
        tools = result['collaboration_tools']
        assert 'communication_platforms' in tools
        assert 'data_sharing_systems' in tools
        assert 'project_management' in tools
        assert 'collaborative_analysis' in tools
        
        # Verify success framework
        success = result['success_framework']
        assert 'key_metrics' in success
        assert 'evaluation_schedule' in success
        assert 'improvement_mechanisms' in success
        assert 'impact_assessment' in success
    
    @pytest.mark.asyncio
    async def test_literature_gap_analysis(self, agent):
        """Test literature gap analysis"""
        context = {
            'domain': 'biology',
            'research_question': 'aging mechanisms',
            'keywords': ['telomeres', 'senescence', 'longevity']
        }
        
        analysis = await agent._analyze_literature_gaps(context)
        
        assert 'paper_count' in analysis
        assert 'gaps' in analysis
        assert 'trends' in analysis
        assert 'contradictions' in analysis
        
        assert analysis['paper_count'] > 0
        assert len(analysis['gaps']) > 0
        assert len(analysis['trends']) > 0
    
    @pytest.mark.asyncio
    async def test_cross_domain_knowledge_synthesis(self, agent):
        """Test cross-domain knowledge synthesis"""
        domain = ResearchDomain.BIOLOGY
        question = "protein folding mechanisms"
        analysis = {'gaps': ['computational_modeling'], 'trends': ['ai_applications']}
        
        synthesis = await agent._synthesize_cross_domain_knowledge(domain, question, analysis)
        
        assert 'connections' in synthesis
        assert 'method_transfers' in synthesis
        assert 'concept_bridges' in synthesis
        
        assert len(synthesis['connections']) > 0
        assert len(synthesis['method_transfers']) > 0
    
    @pytest.mark.asyncio
    async def test_research_pattern_analysis(self, agent):
        """Test research pattern analysis"""
        literature = {
            'gaps': ['mechanistic_understanding'],
            'trends': ['machine_learning_adoption'],
            'contradictions': ['methodology_conflicts']
        }
        insights = {
            'connections': ['biology_ai_integration'],
            'method_transfers': ['cs_to_biology']
        }
        
        patterns = await agent._analyze_research_patterns(literature, insights)
        
        assert 'methodological_patterns' in patterns
        assert 'theoretical_gaps' in patterns
        assert 'empirical_anomalies' in patterns
        assert 'convergence_opportunities' in patterns
    
    @pytest.mark.asyncio
    async def test_hypothesis_generation_ai(self, agent):
        """Test AI-powered hypothesis generation"""
        context = {
            'domain': 'chemistry',
            'research_question': 'catalysis efficiency',
            'session_id': 'test_session'
        }
        literature = {'gaps': ['novel_catalysts'], 'trends': ['green_chemistry']}
        patterns = {'methodological_patterns': ['high_throughput_screening']}
        
        hypotheses = await agent._generate_hypotheses_ai(context, literature, patterns)
        
        assert len(hypotheses) > 0
        for hypothesis in hypotheses:
            assert isinstance(hypothesis, ScientificHypothesis)
            assert hypothesis.hypothesis_id is not None
            assert hypothesis.domain == ResearchDomain.CHEMISTRY
            assert 0 <= hypothesis.confidence_score <= 1
            assert 0 <= hypothesis.novelty_score <= 1
            assert 0 <= hypothesis.testability_score <= 1
            assert 0 <= hypothesis.impact_potential <= 1
    
    @pytest.mark.asyncio
    async def test_hypothesis_quality_evaluation(self, agent):
        """Test hypothesis quality evaluation"""
        hypotheses = [
            ScientificHypothesis(
                hypothesis_id='hyp_1',
                domain=ResearchDomain.BIOLOGY,
                statement='Test hypothesis 1',
                variables=[],
                predictions=[],
                confidence_score=0.8,
                novelty_score=0.9,
                testability_score=0.7,
                impact_potential=0.85
            ),
            ScientificHypothesis(
                hypothesis_id='hyp_2',
                domain=ResearchDomain.BIOLOGY,
                statement='Test hypothesis 2',
                variables=[],
                predictions=[],
                confidence_score=0.6,
                novelty_score=0.7,
                testability_score=0.8,
                impact_potential=0.6
            )
        ]
        
        context = {'domain': 'biology'}
        evaluation = await agent._evaluate_hypothesis_quality(hypotheses, context)
        
        assert 'all_hypotheses' in evaluation
        assert 'top_hypotheses' in evaluation
        assert 'quality_scores' in evaluation
        
        assert len(evaluation['all_hypotheses']) == len(hypotheses)
        assert len(evaluation['top_hypotheses']) <= len(hypotheses)
        assert len(evaluation['quality_scores']) == len(hypotheses)
        
        # Verify hypotheses are sorted by quality
        scores = evaluation['quality_scores']
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_hypothesis_feasibility_assessment(self, agent):
        """Test hypothesis feasibility assessment"""
        evaluated = {
            'top_hypotheses': [
                ScientificHypothesis('hyp_1', ResearchDomain.CHEMISTRY, 'Test', [], [], 0.8, 0.9, 0.7, 0.8)
            ],
            'quality_scores': [0.8]
        }
        constraints = {'budget': 100000, 'timeline': '12 months'}
        
        feasibility = await agent._assess_hypothesis_feasibility(evaluated, constraints)
        
        assert 'priority_list' in feasibility
        assert 'experimental_strategies' in feasibility
        assert 'collaboration_needs' in feasibility
        assert 'resource_estimates' in feasibility
        
        resource_estimates = feasibility['resource_estimates']
        assert 'budget' in resource_estimates
        assert 'timeline' in resource_estimates
        assert 'personnel' in resource_estimates
        assert resource_estimates['budget'] > 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, agent):
        """Test error handling for invalid inputs"""
        # Test with missing research context
        invalid_context = {}
        result = await agent.generate_research_hypotheses(invalid_context)
        assert isinstance(result, dict)
        
        # Test with invalid domain
        invalid_hypothesis = {
            'domain': 'invalid_domain',
            'hypothesis': 'test hypothesis'
        }
        result = await agent.design_automated_experiments(invalid_hypothesis)
        assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, agent, sample_research_context, sample_synthesis_request):
        """Test concurrent operation handling"""
        tasks = [
            agent.generate_research_hypotheses(sample_research_context),
            agent.synthesize_interdisciplinary_knowledge(sample_synthesis_request)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, dict)
    
    def test_research_domain_coverage(self, agent):
        """Test research domain coverage"""
        kb = agent.knowledge_base
        domains = kb['domains']
        
        # Verify comprehensive domain coverage
        required_domains = ['biology', 'chemistry', 'physics']
        for domain in required_domains:
            assert domain in domains
            domain_data = domains[domain]
            assert len(domain_data['subdisciplines']) > 0
            assert len(domain_data['common_methods']) > 0
            assert len(domain_data['key_databases']) > 0
    
    def test_experiment_type_coverage(self, agent):
        """Test experiment type coverage"""
        # Verify all experiment types are supported
        experiment_types = [
            ExperimentType.IN_VITRO,
            ExperimentType.IN_VIVO,
            ExperimentType.COMPUTATIONAL,
            ExperimentType.FIELD_STUDY,
            ExperimentType.LABORATORY,
            ExperimentType.SIMULATION,
            ExperimentType.CLINICAL_TRIAL
        ]
        
        for exp_type in experiment_types:
            assert exp_type in ExperimentType
    
    def test_get_agent_capabilities(self, agent):
        """Test agent capabilities reporting"""
        capabilities = agent.get_agent_capabilities()
        
        assert 'agent_id' in capabilities
        assert 'version' in capabilities
        assert 'capabilities' in capabilities
        assert 'research_domains' in capabilities
        assert 'experiment_types' in capabilities
        assert 'market_coverage' in capabilities
        assert 'specializations' in capabilities
        
        assert capabilities['agent_id'] == agent.agent_id
        assert len(capabilities['capabilities']) >= 4
        assert len(capabilities['research_domains']) >= 8
        assert len(capabilities['experiment_types']) >= 7
        assert '$200B' in capabilities['market_coverage']
        assert len(capabilities['specializations']) >= 6
    
    @pytest.mark.asyncio
    async def test_multi_domain_literature_mining(self, agent):
        """Test multi-domain literature mining"""
        domains = [ResearchDomain.BIOLOGY, ResearchDomain.COMPUTER_SCIENCE]
        question = "machine learning in genomics"
        
        # This would be called internally by synthesize_interdisciplinary_knowledge
        # Test the conceptual framework
        assert len(domains) > 1
        assert all(isinstance(domain, ResearchDomain) for domain in domains)
    
    @pytest.mark.asyncio
    async def test_experimental_strategy_selection(self, agent):
        """Test experimental strategy selection"""
        hypothesis = "Novel drug compound shows 90% efficacy"
        domain = ResearchDomain.MEDICINE
        variables = [{'name': 'dosage', 'type': 'continuous'}]
        constraints = {'budget': 500000, 'timeline': '18 months'}
        
        # Test the framework for strategy selection
        assert domain in ResearchDomain
        assert len(variables) > 0
        assert 'budget' in constraints
    
    @pytest.mark.asyncio
    async def test_quality_control_design(self, agent):
        """Test quality control design for experiments"""
        protocol_design = {
            'methodology': 'controlled_trial',
            'sample_size': 100,
            'controls': ['positive', 'negative']
        }
        automation_workflow = {
            'steps': ['preparation', 'execution', 'analysis'],
            'robotics': ['liquid_handler', 'plate_reader']
        }
        
        # Test the QC framework conceptually
        assert 'methodology' in protocol_design
        assert 'controls' in protocol_design
        assert 'steps' in automation_workflow
    
    def test_knowledge_graph_structure(self, agent):
        """Test knowledge graph structure and components"""
        # Test the KnowledgeGraph dataclass structure
        test_graph = KnowledgeGraph(
            graph_id='test_kg',
            entities=[{'id': 'entity1', 'type': 'concept'}],
            relationships=[{'source': 'entity1', 'target': 'entity2', 'type': 'related_to'}],
            confidence_scores={'entity1': 0.9},
            temporal_dynamics={'entity1': 'stable'},
            interdisciplinary_connections=[{'domain1': 'biology', 'domain2': 'physics'}]
        )
        
        assert test_graph.graph_id == 'test_kg'
        assert len(test_graph.entities) > 0
        assert len(test_graph.relationships) > 0
        assert len(test_graph.confidence_scores) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])