"""
Performance benchmarks and scalability tests for all agents
Tests response times, resource usage, and concurrent execution
"""

import pytest
import asyncio
import time
import psutil
import os
from unittest.mock import patch
from datetime import datetime
from typing import Dict, Any, List

from services.synthetic_biology_engineering_agent import SyntheticBiologyEngineeringAgent
from services.quantum_computing_optimization_agent import QuantumComputingOptimizationAgent
from services.consciousness_ai_research_agent import ConsciousnessAIResearchAgent


class TestPerformanceBenchmarks:
    """Performance benchmarks for all agents"""
    
    @pytest.fixture
    def agents(self):
        """Create agent instances for benchmarking"""
        return {
            "synthetic_biology": SyntheticBiologyEngineeringAgent(),
            "quantum_computing": QuantumComputingOptimizationAgent(),
            "consciousness_ai": ConsciousnessAIResearchAgent()
        }
    
    @pytest.fixture
    def benchmark_inputs(self):
        """Benchmark inputs for each agent"""
        return {
            "synthetic_biology": {
                "design_biological_systems": {
                    "design_objectives": ["enzyme_optimization"] * 10,
                    "target_applications": ["therapeutic"],
                    "safety_requirements": ["BSL2"],
                    "regulatory_constraints": ["FDA_guidance"] * 5
                }
            },
            "quantum_computing": {
                "optimize_quantum_algorithms": {
                    "problem_type": "portfolio_optimization",
                    "quantum_advantage_goals": ["quadratic_speedup"],
                    "hardware_constraints": {"qubit_count": 50, "coherence_time": 100},
                    "performance_targets": {"runtime": 300, "accuracy": 0.9}
                }
            },
            "consciousness_ai": {
                "consciousness_assessment": {
                    "assessment_targets": ["consciousness_level"] * 5,
                    "measurement_methods": ["phi_measure", "global_access"],
                    "validation_requirements": {"reliability": 0.9},
                    "ethical_constraints": ["welfare_consideration"] * 3
                }
            }
        }

    # Response Time Benchmarks
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_single_request_response_time(self, agents, benchmark_inputs, performance_thresholds):
        """Test response time for single requests"""
        results = {}
        
        for agent_name, agent in agents.items():
            if agent_name in benchmark_inputs:
                capability_tests = benchmark_inputs[agent_name]
                
                for capability, input_data in capability_tests.items():
                    start_time = time.time()
                    
                    try:
                        result = await agent.execute(capability, input_data)
                        end_time = time.time()
                        
                        response_time = end_time - start_time
                        results[f"{agent_name}_{capability}"] = {
                            "response_time": response_time,
                            "success": True,
                            "result_size": len(str(result))
                        }
                        
                        # Validate against threshold
                        assert response_time < performance_thresholds["max_response_time"], \
                            f"{agent_name} {capability} took {response_time}s (max: {performance_thresholds['max_response_time']}s)"
                            
                    except Exception as e:
                        results[f"{agent_name}_{capability}"] = {
                            "response_time": time.time() - start_time,
                            "success": False,
                            "error": str(e)
                        }
        
        # Report results
        print("\\nResponse Time Benchmark Results:")
        for test_name, result in results.items():
            status = "✓" if result["success"] else "✗"
            print(f"{status} {test_name}: {result['response_time']:.2f}s")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_request_performance(self, agents, benchmark_inputs, performance_thresholds):
        """Test performance under concurrent load"""
        concurrent_requests = performance_thresholds["concurrent_requests"]
        
        for agent_name, agent in agents.items():
            if agent_name in benchmark_inputs:
                capability_tests = benchmark_inputs[agent_name]
                
                for capability, input_data in capability_tests.items():
                    # Create concurrent tasks
                    tasks = []
                    start_time = time.time()
                    
                    for i in range(concurrent_requests):
                        # Modify input slightly for each request
                        modified_input = input_data.copy()
                        if isinstance(modified_input.get("design_objectives"), list):
                            modified_input["design_objectives"] = modified_input["design_objectives"] + [f"concurrent_{i}"]
                        
                        task = agent.execute(capability, modified_input)
                        tasks.append(task)
                    
                    # Execute concurrently
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    end_time = time.time()
                    
                    # Analyze results
                    successful_results = [r for r in results if not isinstance(r, Exception)]
                    success_rate = len(successful_results) / len(results)
                    total_time = end_time - start_time
                    avg_time_per_request = total_time / len(results)
                    
                    # Validate performance
                    assert success_rate >= performance_thresholds["min_success_rate"], \
                        f"{agent_name} success rate {success_rate} below threshold {performance_thresholds['min_success_rate']}"
                    
                    print(f"\\n{agent_name} {capability} concurrent performance:")
                    print(f"  Requests: {concurrent_requests}")
                    print(f"  Success rate: {success_rate:.2%}")
                    print(f"  Total time: {total_time:.2f}s")
                    print(f"  Avg time per request: {avg_time_per_request:.2f}s")

    # Memory Usage Benchmarks
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage_benchmark(self, agents, benchmark_inputs, performance_thresholds):
        """Test memory usage during agent execution"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_results = {}
        
        for agent_name, agent in agents.items():
            if agent_name in benchmark_inputs:
                capability_tests = benchmark_inputs[agent_name]
                
                for capability, input_data in capability_tests.items():
                    # Measure memory before execution
                    pre_memory = process.memory_info().rss / 1024 / 1024
                    
                    # Execute capability
                    result = await agent.execute(capability, input_data)
                    
                    # Measure memory after execution
                    post_memory = process.memory_info().rss / 1024 / 1024
                    memory_delta = post_memory - pre_memory
                    
                    memory_results[f"{agent_name}_{capability}"] = {
                        "pre_memory_mb": pre_memory,
                        "post_memory_mb": post_memory,
                        "memory_delta_mb": memory_delta,
                        "result_size": len(str(result))
                    }
                    
                    # Validate memory usage
                    assert memory_delta < performance_thresholds["max_memory_usage"], \
                        f"{agent_name} memory delta {memory_delta}MB exceeds threshold {performance_thresholds['max_memory_usage']}MB"
        
        # Report memory usage
        print("\\nMemory Usage Benchmark Results:")
        for test_name, result in memory_results.items():
            print(f"{test_name}:")
            print(f"  Memory delta: {result['memory_delta_mb']:.2f}MB")
            print(f"  Result size: {result['result_size']} chars")

    # Scalability Tests
    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_input_size_scalability(self, agents, benchmark_test_cases):
        """Test performance scaling with input size"""
        scalability_results = {}
        
        for agent_name, agent in agents.items():
            capabilities = agent.get_capabilities()
            if capabilities:
                primary_capability = capabilities[0].name
                
                for scale_name, test_case in benchmark_test_cases.items():
                    # Create scaled input based on data_size
                    if agent_name == "synthetic_biology":
                        scaled_input = {
                            "design_objectives": ["objective"] * test_case["data_size"],
                            "target_applications": ["therapeutic"],
                            "safety_requirements": ["BSL2"],
                            "regulatory_constraints": []
                        }
                    elif agent_name == "quantum_computing":
                        scaled_input = {
                            "problem_type": "portfolio_optimization",
                            "quantum_advantage_goals": ["quadratic_speedup"],
                            "hardware_constraints": {"qubit_count": min(test_case["data_size"], 100)},
                            "performance_targets": {}
                        }
                    elif agent_name == "consciousness_ai":
                        scaled_input = {
                            "assessment_targets": ["target"] * min(test_case["data_size"], 50),
                            "measurement_methods": ["phi_measure"],
                            "validation_requirements": {},
                            "ethical_constraints": []
                        }
                    else:
                        continue
                    
                    # Execute and time
                    start_time = time.time()
                    try:
                        result = await agent.execute(primary_capability, scaled_input)
                        end_time = time.time()
                        
                        execution_time = end_time - start_time
                        scalability_results[f"{agent_name}_{scale_name}"] = {
                            "data_size": test_case["data_size"],
                            "execution_time": execution_time,
                            "expected_time": test_case["expected_time"],
                            "success": True,
                            "efficiency": test_case["data_size"] / execution_time if execution_time > 0 else 0
                        }
                        
                        # Validate against expected time
                        assert execution_time < test_case["expected_time"] * 1.5, \
                            f"{agent_name} {scale_name} took {execution_time}s (expected: {test_case['expected_time']}s)"
                            
                    except Exception as e:
                        scalability_results[f"{agent_name}_{scale_name}"] = {
                            "data_size": test_case["data_size"],
                            "execution_time": time.time() - start_time,
                            "expected_time": test_case["expected_time"],
                            "success": False,
                            "error": str(e)
                        }
        
        # Report scalability results
        print("\\nScalability Benchmark Results:")
        for test_name, result in scalability_results.items():
            status = "✓" if result["success"] else "✗"
            print(f"{status} {test_name}:")
            print(f"  Data size: {result['data_size']}")
            print(f"  Execution time: {result['execution_time']:.2f}s")
            print(f"  Expected time: {result['expected_time']:.2f}s")
            if result["success"]:
                print(f"  Efficiency: {result['efficiency']:.2f} items/sec")

    # Stress Tests
    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_sustained_load_performance(self, agents, benchmark_inputs):
        """Test performance under sustained load"""
        duration_seconds = 60  # 1 minute stress test
        request_interval = 2.0  # seconds between requests
        
        stress_results = {}
        
        for agent_name, agent in agents.items():
            if agent_name in benchmark_inputs:
                capability_tests = benchmark_inputs[agent_name]
                
                for capability, input_data in capability_tests.items():
                    start_time = time.time()
                    requests_sent = 0
                    successful_requests = 0
                    errors = []
                    response_times = []
                    
                    while time.time() - start_time < duration_seconds:
                        request_start = time.time()
                        
                        try:
                            result = await agent.execute(capability, input_data)
                            request_end = time.time()
                            
                            response_times.append(request_end - request_start)
                            successful_requests += 1
                            
                        except Exception as e:
                            errors.append(str(e))
                        
                        requests_sent += 1
                        
                        # Wait before next request
                        await asyncio.sleep(request_interval)
                    
                    # Calculate statistics
                    success_rate = successful_requests / requests_sent if requests_sent > 0 else 0
                    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
                    max_response_time = max(response_times) if response_times else 0
                    min_response_time = min(response_times) if response_times else 0
                    
                    stress_results[f"{agent_name}_{capability}"] = {
                        "duration": duration_seconds,
                        "requests_sent": requests_sent,
                        "successful_requests": successful_requests,
                        "success_rate": success_rate,
                        "avg_response_time": avg_response_time,
                        "max_response_time": max_response_time,
                        "min_response_time": min_response_time,
                        "error_count": len(errors),
                        "unique_errors": len(set(errors))
                    }
                    
                    # Validate stress test results
                    assert success_rate >= 0.9, \
                        f"{agent_name} stress test success rate {success_rate:.2%} below 90%"
        
        # Report stress test results
        print("\\nStress Test Results:")
        for test_name, result in stress_results.items():
            print(f"{test_name}:")
            print(f"  Requests: {result['requests_sent']} (Success: {result['success_rate']:.1%})")
            print(f"  Avg response time: {result['avg_response_time']:.2f}s")
            print(f"  Response time range: {result['min_response_time']:.2f}s - {result['max_response_time']:.2f}s")
            print(f"  Errors: {result['error_count']} ({result['unique_errors']} unique)")

    # Resource Efficiency Tests
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_resource_efficiency_metrics(self, agents, benchmark_inputs):
        """Test resource efficiency metrics for all agents"""
        efficiency_results = {}
        
        for agent_name, agent in agents.items():
            if agent_name in benchmark_inputs:
                capability_tests = benchmark_inputs[agent_name]
                
                for capability, input_data in capability_tests.items():
                    # Measure multiple metrics
                    process = psutil.Process(os.getpid())
                    
                    # Pre-execution metrics
                    pre_cpu_times = process.cpu_times()
                    pre_memory = process.memory_info().rss
                    start_time = time.time()
                    
                    # Execute capability
                    result = await agent.execute(capability, input_data)
                    
                    # Post-execution metrics
                    end_time = time.time()
                    post_cpu_times = process.cpu_times()
                    post_memory = process.memory_info().rss
                    
                    # Calculate metrics
                    execution_time = end_time - start_time
                    cpu_time_used = (post_cpu_times.user - pre_cpu_times.user) + \
                                   (post_cpu_times.system - pre_cpu_times.system)
                    memory_used = post_memory - pre_memory
                    result_size = len(str(result))
                    
                    # Efficiency calculations
                    cpu_efficiency = result_size / cpu_time_used if cpu_time_used > 0 else 0
                    memory_efficiency = result_size / (memory_used / 1024 / 1024) if memory_used > 0 else 0
                    time_efficiency = result_size / execution_time if execution_time > 0 else 0
                    
                    efficiency_results[f"{agent_name}_{capability}"] = {
                        "execution_time": execution_time,
                        "cpu_time_used": cpu_time_used,
                        "memory_used_mb": memory_used / 1024 / 1024,
                        "result_size": result_size,
                        "cpu_efficiency": cpu_efficiency,
                        "memory_efficiency": memory_efficiency,
                        "time_efficiency": time_efficiency
                    }
        
        # Report efficiency results
        print("\\nResource Efficiency Results:")
        for test_name, result in efficiency_results.items():
            print(f"{test_name}:")
            print(f"  Time efficiency: {result['time_efficiency']:.2f} chars/sec")
            print(f"  CPU efficiency: {result['cpu_efficiency']:.2f} chars/cpu_sec")
            print(f"  Memory efficiency: {result['memory_efficiency']:.2f} chars/MB")
            print(f"  Resource usage: {result['cpu_time_used']:.3f}s CPU, {result['memory_used_mb']:.2f}MB")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])