"""
Manufacturing Optimization Agent
Production efficiency and quality optimization
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

class ManufacturingOptimizationAgent:
    """
    Comprehensive Manufacturing Optimization System
    - Production efficiency analysis
    - Quality control optimization
    - Equipment utilization
    - Supply chain integration
    """
    
    def __init__(self):
        self.name = "Manufacturing Optimization Agent"
        self.version = "1.0.0"
        self.capabilities = [
            "Production Efficiency",
            "Quality Control",
            "Equipment Optimization",
            "Supply Chain Integration",
            "Waste Reduction",
            "Performance Analytics"
        ]
        
    def optimize_manufacturing_operations(self, manufacturing_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive manufacturing optimization"""
        try:
            company_name = manufacturing_data.get('company_name', 'Unknown Company')
            
            # Production efficiency analysis
            efficiency_analysis = self._analyze_production_efficiency(manufacturing_data)
            
            # Quality control analysis
            quality_analysis = self._analyze_quality_control(manufacturing_data)
            
            # Equipment optimization
            equipment_optimization = self._optimize_equipment_utilization(manufacturing_data)
            
            # Waste reduction analysis
            waste_analysis = self._analyze_waste_reduction(manufacturing_data)
            
            # Generate recommendations
            recommendations = self._generate_manufacturing_recommendations(
                efficiency_analysis, quality_analysis, equipment_optimization, waste_analysis
            )
            
            return {
                'company': company_name,
                'analysis_date': datetime.now().isoformat(),
                'efficiency_analysis': efficiency_analysis,
                'quality_analysis': quality_analysis,
                'equipment_optimization': equipment_optimization,
                'waste_analysis': waste_analysis,
                'recommendations': recommendations,
                'next_review_date': (datetime.now() + timedelta(days=7)).isoformat()
            }
            
        except Exception as e:
            logging.error(f"Manufacturing optimization failed: {str(e)}")
            return {'error': f'Manufacturing optimization failed: {str(e)}'}
            
    def _analyze_production_efficiency(self, manufacturing_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze production efficiency metrics"""
        
        production_lines = manufacturing_data.get('production_lines', [])
        
        if not production_lines:
            return {'overall_efficiency': 0}
            
        # Calculate efficiency metrics
        line_efficiencies = []
        total_output = 0
        total_capacity = 0
        
        for line in production_lines:
            actual_output = line.get('actual_output', 0)
            planned_output = line.get('planned_output', 0)
            capacity = line.get('capacity', 0)
            
            # Line efficiency calculation
            if planned_output > 0:
                efficiency = (actual_output / planned_output) * 100
            else:
                efficiency = 0
                
            line_efficiencies.append({
                'line_id': line.get('id', 'unknown'),
                'efficiency': efficiency,
                'actual_output': actual_output,
                'planned_output': planned_output,
                'capacity_utilization': (actual_output / capacity * 100) if capacity > 0 else 0,
                'performance_status': self._categorize_performance(efficiency)
            })
            
            total_output += actual_output
            total_capacity += capacity
            
        # Overall metrics
        avg_efficiency = sum(line['efficiency'] for line in line_efficiencies) / len(line_efficiencies)
        overall_utilization = (total_output / total_capacity * 100) if total_capacity > 0 else 0
        
        # Identify bottlenecks
        bottlenecks = [line for line in line_efficiencies if line['efficiency'] < 70]
        
        return {
            'overall_efficiency': avg_efficiency,
            'overall_utilization': overall_utilization,
            'line_efficiencies': line_efficiencies,
            'bottlenecks': bottlenecks,
            'efficiency_level': self._categorize_efficiency_level(avg_efficiency),
            'improvement_opportunities': self._identify_efficiency_improvements(line_efficiencies)
        }
        
    def _categorize_performance(self, efficiency: float) -> str:
        """Categorize performance level"""
        if efficiency >= 90:
            return 'Excellent'
        elif efficiency >= 80:
            return 'Good'
        elif efficiency >= 70:
            return 'Fair'
        else:
            return 'Poor'
            
    def _categorize_efficiency_level(self, efficiency: float) -> str:
        """Categorize overall efficiency level"""
        if efficiency >= 85:
            return 'High Efficiency'
        elif efficiency >= 75:
            return 'Medium Efficiency'
        else:
            return 'Low Efficiency'
            
    def _identify_efficiency_improvements(self, line_efficiencies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify efficiency improvement opportunities"""
        
        improvements = []
        
        for line in line_efficiencies:
            if line['efficiency'] < 80:
                improvement = {
                    'line_id': line['line_id'],
                    'current_efficiency': line['efficiency'],
                    'target_efficiency': 85,
                    'improvement_potential': 85 - line['efficiency'],
                    'priority': 'High' if line['efficiency'] < 60 else 'Medium',
                    'suggested_actions': self._suggest_line_improvements(line)
                }
                improvements.append(improvement)
                
        return improvements
        
    def _suggest_line_improvements(self, line: Dict[str, Any]) -> List[str]:
        """Suggest improvements for production line"""
        
        suggestions = []
        
        if line['efficiency'] < 60:
            suggestions.extend([
                'Conduct detailed process analysis',
                'Implement preventive maintenance program',
                'Review and optimize workflow'
            ])
        elif line['efficiency'] < 80:
            suggestions.extend([
                'Optimize changeover times',
                'Improve operator training',
                'Implement continuous improvement initiatives'
            ])
            
        if line['capacity_utilization'] < 70:
            suggestions.append('Increase throughput or reduce bottlenecks')
            
        return suggestions
        
    def _analyze_quality_control(self, manufacturing_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze quality control metrics"""
        
        quality_data = manufacturing_data.get('quality_metrics', {})
        
        # Quality metrics
        defect_rate = quality_data.get('defect_rate', 2.0)
        rework_rate = quality_data.get('rework_rate', 1.5)
        first_pass_yield = quality_data.get('first_pass_yield', 95.0)
        customer_complaints = quality_data.get('customer_complaints', 10)
        
        # Quality score calculation
        quality_factors = {
            'defect_rate': max(0, 100 - (defect_rate * 20)),  # Lower is better
            'rework_rate': max(0, 100 - (rework_rate * 25)),  # Lower is better
            'first_pass_yield': first_pass_yield,              # Higher is better
            'customer_satisfaction': max(0, 100 - (customer_complaints * 2))  # Lower complaints is better
        }
        
        overall_quality_score = sum(quality_factors.values()) / len(quality_factors)
        
        # Quality issues identification
        quality_issues = []
        
        if defect_rate > 3.0:
            quality_issues.append({
                'issue': 'High Defect Rate',
                'severity': 'High',
                'metric': defect_rate,
                'target': 2.0,
                'action': 'Implement quality control improvements'
            })
            
        if rework_rate > 2.0:
            quality_issues.append({
                'issue': 'Excessive Rework',
                'severity': 'Medium',
                'metric': rework_rate,
                'target': 1.0,
                'action': 'Improve process standardization'
            })
            
        return {
            'overall_quality_score': overall_quality_score,
            'quality_metrics': {
                'defect_rate': defect_rate,
                'rework_rate': rework_rate,
                'first_pass_yield': first_pass_yield,
                'customer_complaints': customer_complaints
            },
            'quality_level': self._categorize_quality_level(overall_quality_score),
            'quality_issues': quality_issues,
            'quality_improvement_plan': self._create_quality_improvement_plan(quality_issues)
        }
        
    def _categorize_quality_level(self, quality_score: float) -> str:
        """Categorize quality level"""
        if quality_score >= 90:
            return 'Excellent Quality'
        elif quality_score >= 80:
            return 'Good Quality'
        elif quality_score >= 70:
            return 'Average Quality'
        else:
            return 'Poor Quality'
            
    def _create_quality_improvement_plan(self, quality_issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create quality improvement plan"""
        
        improvement_plan = []
        
        for issue in quality_issues:
            plan_item = {
                'issue': issue['issue'],
                'current_state': issue['metric'],
                'target_state': issue['target'],
                'recommended_actions': self._get_quality_actions(issue['issue']),
                'timeline': '3-6 months',
                'priority': issue['severity']
            }
            improvement_plan.append(plan_item)
            
        return improvement_plan
        
    def _get_quality_actions(self, issue: str) -> List[str]:
        """Get specific actions for quality issues"""
        
        action_map = {
            'High Defect Rate': [
                'Implement statistical process control',
                'Enhance quality inspection procedures',
                'Improve supplier quality requirements',
                'Provide additional operator training'
            ],
            'Excessive Rework': [
                'Standardize work procedures',
                'Implement error-proofing techniques',
                'Improve first-time quality measures',
                'Enhance quality checkpoints'
            ]
        }
        
        return action_map.get(issue, ['Implement general quality improvements'])
        
    def _optimize_equipment_utilization(self, manufacturing_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize equipment utilization"""
        
        equipment = manufacturing_data.get('equipment', [])
        
        if not equipment:
            return {'overall_utilization': 0}
            
        equipment_analysis = []
        total_utilization = 0
        
        for item in equipment:
            utilization = item.get('utilization_percentage', 0)
            efficiency = item.get('efficiency_percentage', 0)
            downtime_hours = item.get('downtime_hours', 0)
            
            # Calculate OEE (Overall Equipment Effectiveness)
            availability = max(0, 100 - (downtime_hours / 168 * 100))  # Assuming 168 hours per week
            oee = (availability * utilization * efficiency) / 10000  # Convert to percentage
            
            equipment_analysis.append({
                'equipment_id': item.get('id', 'unknown'),
                'equipment_name': item.get('name', 'Unknown'),
                'utilization': utilization,
                'efficiency': efficiency,
                'availability': availability,
                'oee': oee,
                'downtime_hours': downtime_hours,
                'optimization_opportunity': self._assess_equipment_optimization(utilization, efficiency, availability)
            })
            
            total_utilization += utilization
            
        avg_utilization = total_utilization / len(equipment)
        
        # Identify optimization opportunities
        optimization_opportunities = []
        for item in equipment_analysis:
            if item['oee'] < 60:  # Industry benchmark
                optimization_opportunities.append({
                    'equipment': item['equipment_name'],
                    'current_oee': item['oee'],
                    'improvement_potential': 85 - item['oee'],  # Target 85% OEE
                    'focus_areas': self._identify_oee_focus_areas(item)
                })
                
        return {
            'overall_utilization': avg_utilization,
            'equipment_analysis': equipment_analysis,
            'optimization_opportunities': optimization_opportunities,
            'utilization_level': self._categorize_utilization_level(avg_utilization)
        }
        
    def _assess_equipment_optimization(self, utilization: float, efficiency: float, availability: float) -> str:
        """Assess equipment optimization potential"""
        
        if utilization < 70 or efficiency < 80 or availability < 90:
            return 'High optimization potential'
        elif utilization < 80 or efficiency < 85 or availability < 95:
            return 'Medium optimization potential'
        else:
            return 'Low optimization potential'
            
    def _identify_oee_focus_areas(self, equipment: Dict[str, Any]) -> List[str]:
        """Identify focus areas for OEE improvement"""
        
        focus_areas = []
        
        if equipment['availability'] < 90:
            focus_areas.append('Reduce downtime and improve maintenance')
        if equipment['utilization'] < 70:
            focus_areas.append('Increase equipment utilization')
        if equipment['efficiency'] < 80:
            focus_areas.append('Improve operational efficiency')
            
        return focus_areas
        
    def _categorize_utilization_level(self, utilization: float) -> str:
        """Categorize utilization level"""
        if utilization >= 85:
            return 'High Utilization'
        elif utilization >= 70:
            return 'Medium Utilization'
        else:
            return 'Low Utilization'
            
    def _analyze_waste_reduction(self, manufacturing_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze waste reduction opportunities"""
        
        waste_data = manufacturing_data.get('waste_metrics', {})
        
        # Waste categories (Lean Manufacturing 8 Wastes)
        waste_categories = {
            'overproduction': waste_data.get('overproduction_percentage', 5),
            'waiting': waste_data.get('waiting_time_percentage', 8),
            'transportation': waste_data.get('transportation_waste', 3),
            'over_processing': waste_data.get('over_processing', 4),
            'inventory': waste_data.get('excess_inventory', 6),
            'motion': waste_data.get('unnecessary_motion', 3),
            'defects': waste_data.get('defect_waste', 2),
            'unused_talent': waste_data.get('unused_talent', 5)
        }
        
        # Calculate total waste
        total_waste_percentage = sum(waste_categories.values())
        
        # Identify major waste areas
        major_waste_areas = []
        for category, percentage in waste_categories.items():
            if percentage > 5:  # Threshold for significant waste
                major_waste_areas.append({
                    'category': category,
                    'waste_percentage': percentage,
                    'priority': 'High' if percentage > 8 else 'Medium',
                    'reduction_actions': self._get_waste_reduction_actions(category)
                })
                
        # Calculate potential savings
        annual_revenue = manufacturing_data.get('annual_revenue', 10000000)
        potential_savings = annual_revenue * (total_waste_percentage / 100) * 0.5  # 50% reduction target
        
        return {
            'total_waste_percentage': total_waste_percentage,
            'waste_categories': waste_categories,
            'major_waste_areas': major_waste_areas,
            'potential_annual_savings': potential_savings,
            'waste_level': self._categorize_waste_level(total_waste_percentage),
            'lean_opportunities': self._identify_lean_opportunities(waste_categories)
        }
        
    def _get_waste_reduction_actions(self, category: str) -> List[str]:
        """Get waste reduction actions for category"""
        
        action_map = {
            'overproduction': [
                'Implement pull-based production',
                'Improve demand forecasting',
                'Reduce batch sizes'
            ],
            'waiting': [
                'Balance production lines',
                'Improve scheduling',
                'Implement preventive maintenance'
            ],
            'transportation': [
                'Optimize facility layout',
                'Reduce material handling',
                'Implement cellular manufacturing'
            ],
            'inventory': [
                'Implement just-in-time delivery',
                'Improve inventory turnover',
                'Reduce safety stock levels'
            ]
        }
        
        return action_map.get(category, ['Implement lean manufacturing principles'])
        
    def _categorize_waste_level(self, waste_percentage: float) -> str:
        """Categorize waste level"""
        if waste_percentage <= 15:
            return 'Low Waste'
        elif waste_percentage <= 25:
            return 'Medium Waste'
        else:
            return 'High Waste'
            
    def _identify_lean_opportunities(self, waste_categories: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify lean manufacturing opportunities"""
        
        opportunities = []
        
        # Sort waste categories by impact
        sorted_waste = sorted(waste_categories.items(), key=lambda x: x[1], reverse=True)
        
        for category, percentage in sorted_waste[:3]:  # Top 3 waste areas
            opportunities.append({
                'opportunity': f'Reduce {category} waste',
                'current_impact': f'{percentage}% of operations',
                'improvement_potential': f'{percentage * 0.5:.1f}% reduction possible',
                'implementation_difficulty': self._assess_implementation_difficulty(category),
                'expected_timeline': '3-6 months'
            })
            
        return opportunities
        
    def _assess_implementation_difficulty(self, category: str) -> str:
        """Assess implementation difficulty for waste reduction"""
        
        difficulty_map = {
            'overproduction': 'Medium',
            'waiting': 'Low',
            'transportation': 'High',
            'inventory': 'Medium',
            'motion': 'Low',
            'defects': 'Medium'
        }
        
        return difficulty_map.get(category, 'Medium')
        
    def _generate_manufacturing_recommendations(self, efficiency: Dict[str, Any], 
                                              quality: Dict[str, Any], 
                                              equipment: Dict[str, Any], 
                                              waste: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate manufacturing optimization recommendations"""
        
        recommendations = []
        
        # Efficiency recommendations
        if efficiency['efficiency_level'] == 'Low Efficiency':
            recommendations.append({
                'category': 'Production Efficiency',
                'priority': 'High',
                'recommendation': 'Implement comprehensive efficiency improvement program',
                'actions': [
                    'Address production line bottlenecks',
                    'Optimize workflow and processes',
                    'Improve operator training and skills',
                    'Implement lean manufacturing principles'
                ],
                'expected_impact': f'Increase efficiency from {efficiency["overall_efficiency"]:.1f}% to 85%+',
                'timeline': '6-12 months'
            })
            
        # Quality recommendations
        if quality['quality_level'] in ['Poor Quality', 'Average Quality']:
            recommendations.append({
                'category': 'Quality Improvement',
                'priority': 'High',
                'recommendation': 'Enhance quality control and management systems',
                'actions': [
                    'Implement statistical process control',
                    'Improve quality inspection procedures',
                    'Enhance supplier quality management',
                    'Provide quality training for operators'
                ],
                'expected_impact': f'Improve quality score from {quality["overall_quality_score"]:.1f} to 90+',
                'timeline': '4-8 months'
            })
            
        # Equipment optimization recommendations
        if equipment['utilization_level'] == 'Low Utilization':
            recommendations.append({
                'category': 'Equipment Optimization',
                'priority': 'Medium',
                'recommendation': 'Optimize equipment utilization and effectiveness',
                'actions': [
                    'Implement Total Productive Maintenance (TPM)',
                    'Improve equipment scheduling',
                    'Reduce setup and changeover times',
                    'Enhance preventive maintenance programs'
                ],
                'expected_impact': f'Increase utilization from {equipment["overall_utilization"]:.1f}% to 85%+',
                'timeline': '3-9 months'
            })
            
        # Waste reduction recommendations
        if waste['waste_level'] in ['High Waste', 'Medium Waste']:
            recommendations.append({
                'category': 'Waste Reduction',
                'priority': 'Medium',
                'recommendation': 'Implement lean manufacturing and waste reduction initiatives',
                'actions': [
                    'Conduct value stream mapping',
                    'Implement 5S workplace organization',
                    'Reduce inventory and work-in-process',
                    'Optimize material flow and layout'
                ],
                'expected_impact': f'Reduce waste from {waste["total_waste_percentage"]:.1f}% to <15%',
                'timeline': '6-18 months'
            })
            
        return recommendations

def test_manufacturing_optimization_agent():
    """Test the Manufacturing Optimization Agent"""
    print("🧪 Testing Manufacturing Optimization Agent")
    print("=" * 45)
    
    try:
        agent = ManufacturingOptimizationAgent()
        
        test_data = {
            'company_name': 'Advanced Manufacturing Corp',
            'production_lines': [
                {
                    'id': 'LINE001',
                    'actual_output': 850,
                    'planned_output': 1000,
                    'capacity': 1200
                },
                {
                    'id': 'LINE002',
                    'actual_output': 950,
                    'planned_output': 1000,
                    'capacity': 1100
                }
            ],
            'quality_metrics': {
                'defect_rate': 2.5,
                'rework_rate': 1.8,
                'first_pass_yield': 96.5,
                'customer_complaints': 8
            },
            'equipment': [
                {
                    'id': 'EQ001',
                    'name': 'CNC Machine 1',
                    'utilization_percentage': 75,
                    'efficiency_percentage': 85,
                    'downtime_hours': 12
                }
            ],
            'waste_metrics': {
                'overproduction_percentage': 6,
                'waiting_time_percentage': 10,
                'excess_inventory': 8
            },
            'annual_revenue': 15000000
        }
        
        optimization = agent.optimize_manufacturing_operations(test_data)
        print(f"✅ Manufacturing optimization completed for {test_data['company_name']}")
        print(f"   Production efficiency: {optimization['efficiency_analysis']['efficiency_level']}")
        print(f"   Quality level: {optimization['quality_analysis']['quality_level']}")
        print(f"   Equipment utilization: {optimization['equipment_optimization']['utilization_level']}")
        print(f"   Recommendations: {len(optimization['recommendations'])}")
        
        return {
            'agent_initialized': True,
            'efficiency_score': optimization['efficiency_analysis']['overall_efficiency'],
            'quality_score': optimization['quality_analysis']['overall_quality_score']
        }
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return {'error': str(e)}

if __name__ == "__main__":
    test_manufacturing_optimization_agent()