"""
Cloud Infrastructure Optimization Agent
Cloud cost optimization and performance management
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

class CloudOptimizationAgent:
    """
    Comprehensive Cloud Infrastructure Optimization System
    - Cost optimization and monitoring
    - Performance optimization
    - Resource allocation efficiency
    - Multi-cloud management
    """
    
    def __init__(self):
        self.name = "Cloud Infrastructure Optimization Agent"
        self.version = "1.0.0"
        self.capabilities = [
            "Cost Optimization",
            "Performance Monitoring",
            "Resource Right-sizing",
            "Multi-cloud Management",
            "Security Optimization",
            "Compliance Management"
        ]
        
    def optimize_cloud_infrastructure(self, cloud_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive cloud infrastructure optimization"""
        try:
            company_name = cloud_data.get('company_name', 'Unknown Company')
            
            # Cost analysis and optimization
            cost_optimization = self._analyze_cost_optimization(cloud_data)
            
            # Performance analysis
            performance_analysis = self._analyze_performance(cloud_data)
            
            # Resource optimization
            resource_optimization = self._optimize_resources(cloud_data)
            
            # Generate recommendations
            recommendations = self._generate_cloud_recommendations(
                cost_optimization, performance_analysis, resource_optimization
            )
            
            return {
                'company': company_name,
                'analysis_date': datetime.now().isoformat(),
                'cost_optimization': cost_optimization,
                'performance_analysis': performance_analysis,
                'resource_optimization': resource_optimization,
                'recommendations': recommendations,
                'next_review_date': (datetime.now() + timedelta(days=14)).isoformat()
            }
            
        except Exception as e:
            logging.error(f"Cloud optimization failed: {str(e)}")
            return {'error': f'Cloud optimization failed: {str(e)}'}
            
    def _analyze_cost_optimization(self, cloud_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cloud cost optimization opportunities"""
        
        resources = cloud_data.get('resources', [])
        monthly_spend = cloud_data.get('monthly_spend', 10000)
        
        # Identify cost optimization opportunities
        opportunities = []
        potential_savings = 0
        
        for resource in resources:
            utilization = resource.get('utilization', 50)
            cost = resource.get('monthly_cost', 100)
            
            if utilization < 30:  # Underutilized
                savings = cost * 0.6  # 60% savings potential
                opportunities.append({
                    'resource_id': resource.get('id', 'unknown'),
                    'resource_type': resource.get('type', 'unknown'),
                    'optimization_type': 'Right-sizing',
                    'current_cost': cost,
                    'potential_savings': savings,
                    'recommendation': 'Downsize or terminate underutilized resource'
                })
                potential_savings += savings
                
        return {
            'current_monthly_spend': monthly_spend,
            'optimization_opportunities': opportunities,
            'potential_monthly_savings': potential_savings,
            'savings_percentage': (potential_savings / monthly_spend * 100) if monthly_spend > 0 else 0,
            'cost_efficiency_score': self._calculate_cost_efficiency(resources)
        }
        
    def _calculate_cost_efficiency(self, resources: List[Dict[str, Any]]) -> float:
        """Calculate cost efficiency score"""
        if not resources:
            return 0
            
        efficiency_scores = []
        for resource in resources:
            utilization = resource.get('utilization', 50)
            # Higher utilization = better efficiency
            efficiency_scores.append(min(100, utilization * 1.5))
            
        return sum(efficiency_scores) / len(efficiency_scores)
        
    def _analyze_performance(self, cloud_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cloud performance metrics"""
        
        resources = cloud_data.get('resources', [])
        
        performance_issues = []
        overall_performance = 0
        
        for resource in resources:
            cpu_usage = resource.get('cpu_usage', 50)
            memory_usage = resource.get('memory_usage', 50)
            response_time = resource.get('response_time_ms', 200)
            
            # Performance score calculation
            perf_score = (100 - max(0, cpu_usage - 80)) * 0.4 + \
                        (100 - max(0, memory_usage - 80)) * 0.4 + \
                        (100 - max(0, (response_time - 100) / 10)) * 0.2
                        
            overall_performance += perf_score
            
            # Identify performance issues
            if cpu_usage > 80:
                performance_issues.append({
                    'resource_id': resource.get('id', 'unknown'),
                    'issue': 'High CPU Usage',
                    'severity': 'High',
                    'recommendation': 'Scale up or optimize workload'
                })
                
        avg_performance = overall_performance / len(resources) if resources else 0
        
        return {
            'average_performance_score': avg_performance,
            'performance_issues': performance_issues,
            'performance_level': 'Good' if avg_performance >= 80 else 'Fair' if avg_performance >= 60 else 'Poor'
        }
        
    def _optimize_resources(self, cloud_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize cloud resource allocation"""
        
        resources = cloud_data.get('resources', [])
        
        optimizations = []
        
        for resource in resources:
            optimization = self._analyze_resource_optimization(resource)
            if optimization:
                optimizations.append(optimization)
                
        return {
            'optimization_recommendations': optimizations,
            'total_optimizations': len(optimizations),
            'resource_efficiency': self._calculate_resource_efficiency(resources)
        }
        
    def _analyze_resource_optimization(self, resource: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze optimization for individual resource"""
        
        utilization = resource.get('utilization', 50)
        resource_type = resource.get('type', 'unknown')
        
        if utilization < 20:
            return {
                'resource_id': resource.get('id', 'unknown'),
                'resource_type': resource_type,
                'action': 'Terminate',
                'reason': 'Extremely low utilization',
                'potential_savings': resource.get('monthly_cost', 100)
            }
        elif utilization < 40:
            return {
                'resource_id': resource.get('id', 'unknown'),
                'resource_type': resource_type,
                'action': 'Downsize',
                'reason': 'Low utilization',
                'potential_savings': resource.get('monthly_cost', 100) * 0.5
            }
        elif utilization > 90:
            return {
                'resource_id': resource.get('id', 'unknown'),
                'resource_type': resource_type,
                'action': 'Scale Up',
                'reason': 'High utilization may cause performance issues',
                'additional_cost': resource.get('monthly_cost', 100) * 0.5
            }
            
        return None
        
    def _calculate_resource_efficiency(self, resources: List[Dict[str, Any]]) -> float:
        """Calculate overall resource efficiency"""
        if not resources:
            return 0
            
        efficiency_scores = []
        for resource in resources:
            utilization = resource.get('utilization', 50)
            # Optimal utilization is around 70-80%
            if 60 <= utilization <= 80:
                efficiency = 100
            else:
                efficiency = max(0, 100 - abs(utilization - 70))
            efficiency_scores.append(efficiency)
            
        return sum(efficiency_scores) / len(efficiency_scores)
        
    def _generate_cloud_recommendations(self, cost_opt: Dict[str, Any], 
                                      performance: Dict[str, Any], 
                                      resource_opt: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate cloud optimization recommendations"""
        
        recommendations = []
        
        # Cost optimization recommendations
        if cost_opt['potential_monthly_savings'] > 1000:
            recommendations.append({
                'category': 'Cost Optimization',
                'priority': 'High',
                'recommendation': f'Implement cost optimization strategies to save ${cost_opt["potential_monthly_savings"]:,.0f}/month',
                'actions': [
                    'Right-size underutilized resources',
                    'Implement automated scaling',
                    'Use reserved instances for stable workloads'
                ],
                'expected_impact': f'{cost_opt["savings_percentage"]:.1f}% cost reduction'
            })
            
        # Performance optimization recommendations
        if performance['performance_level'] == 'Poor':
            recommendations.append({
                'category': 'Performance Optimization',
                'priority': 'High',
                'recommendation': 'Address performance issues to improve system reliability',
                'actions': [
                    'Scale resources experiencing high utilization',
                    'Optimize application performance',
                    'Implement monitoring and alerting'
                ],
                'expected_impact': 'Improved system performance and user experience'
            })
            
        # Resource optimization recommendations
        if resource_opt['total_optimizations'] > 0:
            recommendations.append({
                'category': 'Resource Management',
                'priority': 'Medium',
                'recommendation': f'Optimize {resource_opt["total_optimizations"]} resources for better efficiency',
                'actions': [
                    'Implement resource right-sizing',
                    'Set up automated scaling policies',
                    'Regular resource utilization reviews'
                ],
                'expected_impact': 'Improved resource efficiency and cost optimization'
            })
            
        return recommendations

def test_cloud_optimization_agent():
    """Test the Cloud Optimization Agent"""
    print("🧪 Testing Cloud Infrastructure Optimization Agent")
    print("=" * 55)
    
    try:
        agent = CloudOptimizationAgent()
        
        test_data = {
            'company_name': 'CloudTech Solutions',
            'monthly_spend': 15000,
            'resources': [
                {
                    'id': 'RES001',
                    'type': 'EC2 Instance',
                    'utilization': 25,
                    'monthly_cost': 800,
                    'cpu_usage': 30,
                    'memory_usage': 40,
                    'response_time_ms': 150
                },
                {
                    'id': 'RES002',
                    'type': 'RDS Database',
                    'utilization': 85,
                    'monthly_cost': 1200,
                    'cpu_usage': 75,
                    'memory_usage': 80,
                    'response_time_ms': 80
                }
            ]
        }
        
        analysis = agent.optimize_cloud_infrastructure(test_data)
        print(f"✅ Cloud optimization completed for {test_data['company_name']}")
        print(f"   Monthly spend: ${analysis['cost_optimization']['current_monthly_spend']:,}")
        print(f"   Potential savings: ${analysis['cost_optimization']['potential_monthly_savings']:,.0f}")
        print(f"   Recommendations: {len(analysis['recommendations'])}")
        
        return {
            'agent_initialized': True,
            'resources_analyzed': len(test_data['resources']),
            'potential_savings': analysis['cost_optimization']['potential_monthly_savings']
        }
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return {'error': str(e)}

if __name__ == "__main__":
    test_cloud_optimization_agent()