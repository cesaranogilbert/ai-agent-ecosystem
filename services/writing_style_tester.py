"""
Comprehensive Testing Framework for Gilbert's Sympathetic Writing Style AI Agent

This module provides extensive testing capabilities to validate the AI agent's
ability to transform content into Gilbert's authentic sympathetic writing style.
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add the services directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sympathetic_writing_agent import GilbertSympatheticWritingAgent, create_sympathetic_writing_agent


@dataclass
class TestResult:
    """Results from a single test case"""
    test_name: str
    original_content: str
    transformed_content: str
    validation_scores: Dict[str, any]
    passed: bool
    execution_time: float
    notes: str


@dataclass
class TestSuite:
    """Complete test suite results"""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    average_score: float
    test_results: List[TestResult]
    execution_time: float


class WritingStyleTester:
    """Comprehensive testing framework for Gilbert's writing style agent"""
    
    def __init__(self):
        self.agent = create_sympathetic_writing_agent()
        self.test_cases = self._create_test_cases()
        
    def _create_test_cases(self) -> Dict[str, List[Dict]]:
        """Create comprehensive test cases for different content types"""
        
        return {
            "business_content": [
                {
                    "name": "Corporate Leadership",
                    "content": "Effective leadership requires clear communication, strategic vision, and the ability to inspire teams. Leaders must make tough decisions while maintaining team morale and driving results.",
                    "expected_elements": ["vulnerability", "personal_failure", "team_experience"],
                    "minimum_score": 7.0
                },
                {
                    "name": "Digital Transformation",
                    "content": "Organizations must embrace digital transformation to remain competitive. This involves adopting new technologies, changing processes, and developing digital capabilities across the enterprise.",
                    "expected_elements": ["personal_struggle", "learning_journey", "cultural_adaptation"],
                    "minimum_score": 7.0
                },
                {
                    "name": "Data-Driven Decision Making",
                    "content": "Data analytics enables better decision making by providing insights into customer behavior, market trends, and operational efficiency. Companies that leverage data effectively gain competitive advantages.",
                    "expected_elements": ["analysis_failures", "learning_process", "emotional_honesty"],
                    "minimum_score": 7.0
                }
            ],
            
            "personal_development": [
                {
                    "name": "Goal Setting",
                    "content": "Setting clear goals is essential for personal and professional success. SMART goals provide direction and help measure progress toward desired outcomes.",
                    "expected_elements": ["goal_failures", "personal_journey", "family_impact"],
                    "minimum_score": 7.5
                },
                {
                    "name": "Time Management",
                    "content": "Effective time management involves prioritizing tasks, eliminating distractions, and focusing on high-impact activities that drive results.",
                    "expected_elements": ["time_management_struggles", "family_balance", "cultural_perspectives"],
                    "minimum_score": 7.5
                },
                {
                    "name": "Mindfulness and Focus",
                    "content": "Mindfulness practices improve focus, reduce stress, and enhance decision-making capabilities. Regular meditation and mindful awareness create better outcomes.",
                    "expected_elements": ["meditation_struggles", "spiritual_journey", "training_grounds_concept"],
                    "minimum_score": 8.0
                }
            ],
            
            "cultural_intelligence": [
                {
                    "name": "Cross-Cultural Communication",
                    "content": "Working effectively across cultures requires understanding different communication styles, building trust, and adapting approaches to cultural contexts.",
                    "expected_elements": ["cultural_misunderstandings", "learning_from_mistakes", "multicultural_wisdom"],
                    "minimum_score": 8.0
                },
                {
                    "name": "Global Team Leadership",
                    "content": "Leading global teams involves navigating time zones, cultural differences, and varying work styles while maintaining team cohesion and productivity.",
                    "expected_elements": ["leadership_failures", "cultural_learning", "personal_growth"],
                    "minimum_score": 8.0
                }
            ],
            
            "technology_content": [
                {
                    "name": "AI Implementation",
                    "content": "Implementing AI systems requires careful planning, data preparation, and change management to ensure successful adoption and business value creation.",
                    "expected_elements": ["implementation_failures", "learning_curve", "team_challenges"],
                    "minimum_score": 7.0
                },
                {
                    "name": "Cloud Architecture",
                    "content": "Cloud architecture design involves balancing performance, cost, security, and scalability requirements to create robust and efficient systems.",
                    "expected_elements": ["technical_mistakes", "learning_journey", "client_stories"],
                    "minimum_score": 7.0
                }
            ]
        }
    
    def run_comprehensive_tests(self) -> TestSuite:
        """Run all test cases and return comprehensive results"""
        
        print("🚀 Starting Comprehensive Writing Style Agent Tests...")
        print("=" * 70)
        
        start_time = time.time()
        all_results = []
        total_tests = 0
        passed_tests = 0
        total_scores = []
        
        for category, test_cases in self.test_cases.items():
            print(f"\n📁 Testing Category: {category.upper()}")
            print("-" * 50)
            
            for test_case in test_cases:
                total_tests += 1
                result = self._run_single_test(test_case, category)
                all_results.append(result)
                
                if result.passed:
                    passed_tests += 1
                    status = "✅ PASSED"
                else:
                    status = "❌ FAILED"
                
                overall_score = result.validation_scores.get('overall_score', 0)
                total_scores.append(overall_score)
                
                print(f"{status} | {result.test_name} | Score: {overall_score}/10 | Time: {result.execution_time:.2f}s")
        
        execution_time = time.time() - start_time
        average_score = sum(total_scores) / len(total_scores) if total_scores else 0
        
        test_suite = TestSuite(
            suite_name="Gilbert's Sympathetic Writing Style Agent",
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=total_tests - passed_tests,
            average_score=average_score,
            test_results=all_results,
            execution_time=execution_time
        )
        
        self._print_test_summary(test_suite)
        return test_suite
    
    def _run_single_test(self, test_case: Dict, category: str) -> TestResult:
        """Run a single test case"""
        
        start_time = time.time()
        
        try:
            # Transform content using the agent
            transformed = self.agent.transform_to_sympathetic_style(
                test_case["content"], 
                target_enhancement="comprehensive"
            )
            
            # Validate the transformation
            validation = self.agent.validate_style_authenticity(transformed)
            
            # Check if test passed based on minimum score requirement
            overall_score = validation.get('overall_score', 0)
            passed = overall_score >= test_case.get('minimum_score', 7.0)
            
            # Check for expected elements
            expected_elements = test_case.get('expected_elements', [])
            elements_found = self._check_expected_elements(transformed, expected_elements)
            
            execution_time = time.time() - start_time
            
            return TestResult(
                test_name=test_case['name'],
                original_content=test_case['content'],
                transformed_content=transformed,
                validation_scores=validation,
                passed=passed and len(elements_found) >= len(expected_elements) * 0.6,  # 60% of expected elements
                execution_time=execution_time,
                notes=f"Expected elements found: {elements_found}"
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                test_name=test_case['name'],
                original_content=test_case['content'],
                transformed_content=f"ERROR: {str(e)}",
                validation_scores={'error': str(e)},
                passed=False,
                execution_time=execution_time,
                notes=f"Test failed with error: {str(e)}"
            )
    
    def _check_expected_elements(self, content: str, expected_elements: List[str]) -> List[str]:
        """Check if expected sympathetic elements are present in content"""
        
        content_lower = content.lower()
        found_elements = []
        
        element_indicators = {
            "vulnerability": ["failed", "mistake", "embarrassing", "struggled", "wrong", "learned the hard way"],
            "personal_failure": ["failed", "disaster", "embarrassing", "humiliating", "completely wrong"],
            "team_experience": ["team", "together", "collaboration", "we learned", "our mistake"],
            "personal_struggle": ["struggle", "difficult", "challenging", "hard time", "difficulty"],
            "learning_journey": ["learned", "journey", "process", "gradually", "over time"],
            "cultural_adaptation": ["culture", "german", "italian", "swiss", "cultural", "different approach"],
            "analysis_failures": ["analysis failed", "data wrong", "misunderstood", "interpreted incorrectly"],
            "learning_process": ["learning", "process", "step by step", "gradually understood"],
            "emotional_honesty": ["felt", "emotion", "heart", "afraid", "anxious", "worried"],
            "goal_failures": ["goal failed", "didn't achieve", "missed target", "fell short"],
            "family_impact": ["family", "wife", "children", "home", "personal cost"],
            "time_management_struggles": ["time management", "overwhelmed", "too busy", "couldn't balance"],
            "family_balance": ["balance", "family time", "missing", "sacrifice"],
            "cultural_perspectives": ["perspective", "culture taught", "learned from"],
            "meditation_struggles": ["meditation", "difficult", "mind wandered", "struggled to focus"],
            "spiritual_journey": ["spiritual", "prayer", "faith", "god", "sanctuary"],
            "training_grounds_concept": ["training ground", "classroom", "teaches", "learn from"],
            "cultural_misunderstandings": ["misunderstood", "cultural mistake", "offended", "confused"],
            "learning_from_mistakes": ["mistake taught", "learned from", "error showed"],
            "multicultural_wisdom": ["german", "italian", "swiss", "culture", "different approach"],
            "leadership_failures": ["leadership failed", "team didn't follow", "made wrong decision"],
            "personal_growth": ["grew", "developed", "became better", "improved"],
            "implementation_failures": ["implementation failed", "didn't work", "had to restart"],
            "learning_curve": ["learning curve", "took time", "gradually understood"],
            "team_challenges": ["team struggled", "team challenges", "team conflict"],
            "technical_mistakes": ["technical mistake", "wrong approach", "failed implementation"],
            "client_stories": ["client", "customer", "worked with", "helped"]
        }
        
        for element in expected_elements:
            if element in element_indicators:
                indicators = element_indicators[element]
                if any(indicator in content_lower for indicator in indicators):
                    found_elements.append(element)
        
        return found_elements
    
    def _print_test_summary(self, test_suite: TestSuite):
        """Print comprehensive test summary"""
        
        print("\n" + "=" * 70)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 70)
        print(f"Suite: {test_suite.suite_name}")
        print(f"Total Tests: {test_suite.total_tests}")
        print(f"Passed: {test_suite.passed_tests} ✅")
        print(f"Failed: {test_suite.failed_tests} ❌")
        print(f"Success Rate: {(test_suite.passed_tests/test_suite.total_tests)*100:.1f}%")
        print(f"Average Score: {test_suite.average_score:.1f}/10")
        print(f"Total Execution Time: {test_suite.execution_time:.2f}s")
        
        if test_suite.failed_tests > 0:
            print("\n❌ FAILED TESTS:")
            for result in test_suite.test_results:
                if not result.passed:
                    score = result.validation_scores.get('overall_score', 0)
                    print(f"   • {result.test_name}: Score {score}/10")
        
        print("\n🎯 TOP PERFORMING TESTS:")
        top_tests = sorted(test_suite.test_results, 
                          key=lambda x: x.validation_scores.get('overall_score', 0), 
                          reverse=True)[:3]
        for i, result in enumerate(top_tests, 1):
            score = result.validation_scores.get('overall_score', 0)
            print(f"   {i}. {result.test_name}: {score}/10")
        
        print("\n" + "=" * 70)
    
    def test_specific_transformation(self, content: str, enhancement_type: str = "comprehensive") -> Dict:
        """Test a specific content transformation and return detailed results"""
        
        print(f"🧪 Testing {enhancement_type} transformation...")
        print("-" * 50)
        
        start_time = time.time()
        
        # Transform content
        transformed = self.agent.transform_to_sympathetic_style(content, enhancement_type)
        
        # Validate transformation
        validation = self.agent.validate_style_authenticity(transformed)
        
        execution_time = time.time() - start_time
        
        results = {
            "original_content": content,
            "transformed_content": transformed,
            "validation_scores": validation,
            "execution_time": execution_time,
            "enhancement_type": enhancement_type
        }
        
        # Print results
        print(f"Transformation Time: {execution_time:.2f}s")
        print(f"Overall Score: {validation.get('overall_score', 0)}/10")
        print(f"Strengths: {', '.join(validation.get('strengths', []))}")
        if validation.get('improvements'):
            print(f"Areas for Improvement: {', '.join(validation.get('improvements', []))}")
        
        print("\n📝 TRANSFORMED CONTENT:")
        print("-" * 30)
        print(transformed)
        print("-" * 30)
        
        return results
    
    def benchmark_agent_performance(self) -> Dict:
        """Benchmark the agent's performance across multiple metrics"""
        
        print("⚡ Benchmarking Agent Performance...")
        print("=" * 50)
        
        benchmark_content = [
            "Leadership involves making difficult decisions and inspiring teams to achieve common goals.",
            "Data analytics helps businesses understand customer behavior and optimize operations.",
            "Cultural intelligence is essential for global business success and team collaboration.",
            "Personal development requires continuous learning and self-reflection to grow.",
            "Technology implementation must balance innovation with practical business needs."
        ]
        
        benchmark_results = {
            "transformation_times": [],
            "validation_scores": [],
            "authenticity_scores": [],
            "vulnerability_scores": [],
            "relatability_scores": []
        }
        
        for i, content in enumerate(benchmark_content, 1):
            print(f"Benchmarking sample {i}/5...")
            
            start_time = time.time()
            transformed = self.agent.transform_to_sympathetic_style(content)
            transform_time = time.time() - start_time
            
            validation = self.agent.validate_style_authenticity(transformed)
            
            benchmark_results["transformation_times"].append(transform_time)
            benchmark_results["validation_scores"].append(validation.get('overall_score', 0))
            benchmark_results["authenticity_scores"].append(validation.get('authenticity', 0))
            benchmark_results["vulnerability_scores"].append(validation.get('vulnerability', 0))
            benchmark_results["relatability_scores"].append(validation.get('relatability', 0))
        
        # Calculate averages
        avg_transform_time = sum(benchmark_results["transformation_times"]) / len(benchmark_results["transformation_times"])
        avg_validation_score = sum(benchmark_results["validation_scores"]) / len(benchmark_results["validation_scores"])
        avg_authenticity = sum(benchmark_results["authenticity_scores"]) / len(benchmark_results["authenticity_scores"])
        avg_vulnerability = sum(benchmark_results["vulnerability_scores"]) / len(benchmark_results["vulnerability_scores"])
        avg_relatability = sum(benchmark_results["relatability_scores"]) / len(benchmark_results["relatability_scores"])
        
        print(f"\n📈 BENCHMARK RESULTS:")
        print(f"Average Transformation Time: {avg_transform_time:.2f}s")
        print(f"Average Validation Score: {avg_validation_score:.1f}/10")
        print(f"Average Authenticity Score: {avg_authenticity:.1f}/10")
        print(f"Average Vulnerability Score: {avg_vulnerability:.1f}/10")
        print(f"Average Relatability Score: {avg_relatability:.1f}/10")
        
        return {
            "avg_transformation_time": avg_transform_time,
            "avg_validation_score": avg_validation_score,
            "avg_authenticity_score": avg_authenticity,
            "avg_vulnerability_score": avg_vulnerability,
            "avg_relatability_score": avg_relatability,
            "benchmark_data": benchmark_results
        }


def main():
    """Run comprehensive testing of Gilbert's Sympathetic Writing Style AI Agent"""
    
    print("🎯 Gilbert's Sympathetic Writing Style AI Agent - Comprehensive Testing")
    print("=" * 80)
    
    # Initialize tester
    tester = WritingStyleTester()
    
    # Run comprehensive test suite
    test_results = tester.run_comprehensive_tests()
    
    # Run performance benchmark
    benchmark_results = tester.benchmark_agent_performance()
    
    # Test content generation
    print("\n🎨 Testing Content Generation...")
    print("-" * 50)
    
    agent = create_sympathetic_writing_agent()
    generated_content = agent.generate_sympathetic_content(
        topic="The importance of failure in personal growth",
        content_type="blog_post",
        target_length="medium"
    )
    
    print("Generated Content Sample:")
    print("-" * 30)
    print(generated_content[:500] + "..." if len(generated_content) > 500 else generated_content)
    
    # Final assessment
    print("\n" + "=" * 80)
    print("🏆 FINAL ASSESSMENT")
    print("=" * 80)
    
    if test_results.passed_tests >= test_results.total_tests * 0.8:  # 80% pass rate
        print("✅ AGENT READY FOR DEPLOYMENT")
        print("The Sympathetic Writing Style AI Agent meets quality standards.")
    else:
        print("⚠️  AGENT NEEDS IMPROVEMENT")
        print("Additional training or refinement may be needed.")
    
    print(f"Overall Performance: {(test_results.passed_tests/test_results.total_tests)*100:.1f}%")
    print(f"Average Quality Score: {test_results.average_score:.1f}/10")
    print(f"Average Response Time: {benchmark_results['avg_transformation_time']:.2f}s")


if __name__ == "__main__":
    main()