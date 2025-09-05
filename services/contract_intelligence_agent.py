"""
Contract Intelligence & Negotiation Agent
Contract risk analysis and negotiation optimization
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class ContractRisk:
    risk_type: str
    severity: str
    description: str
    mitigation: str

class ContractIntelligenceAgent:
    """
    Comprehensive Contract Intelligence System
    - Contract risk analysis
    - Negotiation strategy optimization
    - Compliance clause verification
    - Vendor performance tracking
    """
    
    def __init__(self):
        self.name = "Contract Intelligence & Negotiation Agent"
        self.version = "1.0.0"
        self.capabilities = [
            "Contract Risk Analysis",
            "Negotiation Strategy Development",
            "Compliance Verification",
            "Performance Tracking",
            "Cost Optimization",
            "Risk Mitigation"
        ]
        
    def analyze_contract_portfolio(self, contract_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze contract portfolio for risks and opportunities"""
        try:
            company_name = contract_data.get('company_name', 'Unknown Company')
            
            # Analyze individual contracts
            contract_analyses = self._analyze_contracts(contract_data.get('contracts', []))
            
            # Portfolio-level analysis
            portfolio_analysis = self._analyze_portfolio_level_risks(contract_analyses)
            
            # Vendor performance analysis
            vendor_analysis = self._analyze_vendor_performance(contract_data)
            
            # Generate optimization recommendations
            recommendations = self._generate_contract_recommendations(
                contract_analyses, portfolio_analysis, vendor_analysis
            )
            
            return {
                'company': company_name,
                'analysis_date': datetime.now().isoformat(),
                'contract_analyses': contract_analyses,
                'portfolio_analysis': portfolio_analysis,
                'vendor_analysis': vendor_analysis,
                'recommendations': recommendations,
                'next_review_date': (datetime.now() + timedelta(days=90)).isoformat()
            }
            
        except Exception as e:
            logging.error(f"Contract analysis failed: {str(e)}")
            return {'error': f'Contract analysis failed: {str(e)}'}
            
    def _analyze_contracts(self, contracts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze individual contracts"""
        analyses = []
        
        for contract in contracts:
            analysis = {
                'contract_id': contract.get('id', 'unknown'),
                'contract_type': contract.get('type', 'general'),
                'vendor': contract.get('vendor', 'Unknown'),
                'value': contract.get('value', 0),
                'risk_assessment': self._assess_contract_risks(contract),
                'compliance_status': self._check_compliance_status(contract),
                'performance_metrics': self._calculate_performance_metrics(contract),
                'optimization_opportunities': self._identify_optimization_opportunities(contract)
            }
            analyses.append(analysis)
            
        return analyses
        
    def _assess_contract_risks(self, contract: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risks in individual contract"""
        risks = []
        
        # Financial risks
        if contract.get('value', 0) > 1000000:  # High-value contract
            risks.append(ContractRisk(
                risk_type='Financial',
                severity='High',
                description='High-value contract exposure',
                mitigation='Implement milestone-based payments'
            ))
            
        # Term risks
        contract_term = contract.get('term_months', 12)
        if contract_term > 36:  # Long-term contract
            risks.append(ContractRisk(
                risk_type='Term',
                severity='Medium',
                description='Long-term commitment risk',
                mitigation='Include periodic review clauses'
            ))
            
        # Performance risks
        if contract.get('performance_history', 80) < 70:
            risks.append(ContractRisk(
                risk_type='Performance',
                severity='High',
                description='Poor vendor performance history',
                mitigation='Implement enhanced monitoring and penalties'
            ))
            
        # Compliance risks
        if contract.get('regulatory_requirements', []):
            risks.append(ContractRisk(
                risk_type='Compliance',
                severity='Medium',
                description='Regulatory compliance requirements',
                mitigation='Regular compliance audits and reporting'
            ))
            
        # Calculate overall risk score
        risk_scores = {'High': 3, 'Medium': 2, 'Low': 1}
        total_risk_score = sum(risk_scores.get(risk.severity, 1) for risk in risks)
        avg_risk_score = total_risk_score / len(risks) if risks else 0
        
        overall_risk_level = 'High' if avg_risk_score >= 2.5 else 'Medium' if avg_risk_score >= 1.5 else 'Low'
        
        return {
            'identified_risks': [risk.__dict__ for risk in risks],
            'risk_count': len(risks),
            'overall_risk_level': overall_risk_level,
            'risk_score': avg_risk_score
        }
        
    def _check_compliance_status(self, contract: Dict[str, Any]) -> Dict[str, Any]:
        """Check contract compliance status"""
        
        compliance_checks = {
            'terms_compliance': contract.get('terms_met', True),
            'payment_compliance': contract.get('payments_current', True),
            'performance_compliance': contract.get('performance_standards_met', True),
            'regulatory_compliance': contract.get('regulatory_compliant', True)
        }
        
        compliance_score = sum(compliance_checks.values()) / len(compliance_checks) * 100
        
        non_compliant_areas = [area for area, status in compliance_checks.items() if not status]
        
        return {
            'compliance_score': compliance_score,
            'compliance_status': 'Compliant' if compliance_score >= 90 else 'Partially Compliant' if compliance_score >= 70 else 'Non-Compliant',
            'non_compliant_areas': non_compliant_areas,
            'corrective_actions_needed': len(non_compliant_areas) > 0
        }
        
    def _calculate_performance_metrics(self, contract: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate contract performance metrics"""
        
        metrics = {
            'delivery_performance': contract.get('on_time_delivery', 85),
            'quality_performance': contract.get('quality_score', 80),
            'cost_performance': contract.get('cost_efficiency', 75),
            'service_level_achievement': contract.get('sla_achievement', 90)
        }
        
        overall_performance = sum(metrics.values()) / len(metrics)
        
        return {
            'individual_metrics': metrics,
            'overall_performance': overall_performance,
            'performance_trend': 'Improving' if overall_performance > 80 else 'Stable' if overall_performance > 70 else 'Declining',
            'performance_gaps': [metric for metric, score in metrics.items() if score < 75]
        }
        
    def _identify_optimization_opportunities(self, contract: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify contract optimization opportunities"""
        opportunities = []
        
        # Cost optimization
        if contract.get('cost_efficiency', 75) < 80:
            opportunities.append({
                'type': 'Cost Optimization',
                'opportunity': 'Renegotiate pricing terms',
                'potential_savings': f"${contract.get('value', 0) * 0.1:,.0f}",
                'implementation': 'Medium'
            })
            
        # Performance improvement
        if contract.get('quality_score', 80) < 85:
            opportunities.append({
                'type': 'Performance Enhancement',
                'opportunity': 'Implement performance incentives',
                'potential_benefit': 'Improve service quality by 10-15%',
                'implementation': 'Easy'
            })
            
        # Term optimization
        contract_term = contract.get('term_months', 12)
        if contract_term < 12:
            opportunities.append({
                'type': 'Term Optimization',
                'opportunity': 'Negotiate longer-term contract for better rates',
                'potential_benefit': 'Reduce costs by 5-10%',
                'implementation': 'Medium'
            })
            
        return opportunities
        
    def _analyze_portfolio_level_risks(self, contract_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze portfolio-level risks"""
        
        if not contract_analyses:
            return {'total_contracts': 0, 'portfolio_risk': 'No contracts'}
            
        # Aggregate risk analysis
        high_risk_contracts = [c for c in contract_analyses if c['risk_assessment']['overall_risk_level'] == 'High']
        total_portfolio_value = sum(c.get('value', 0) for c in contract_analyses)
        
        # Vendor concentration risk
        vendor_exposure = {}
        for contract in contract_analyses:
            vendor = contract.get('vendor', 'Unknown')
            vendor_exposure[vendor] = vendor_exposure.get(vendor, 0) + contract.get('value', 0)
            
        max_vendor_exposure = max(vendor_exposure.values()) if vendor_exposure else 0
        concentration_risk = (max_vendor_exposure / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
        
        # Contract type distribution
        contract_types = {}
        for contract in contract_analyses:
            contract_type = contract.get('contract_type', 'general')
            contract_types[contract_type] = contract_types.get(contract_type, 0) + 1
            
        return {
            'total_contracts': len(contract_analyses),
            'total_portfolio_value': total_portfolio_value,
            'high_risk_contracts': len(high_risk_contracts),
            'high_risk_percentage': len(high_risk_contracts) / len(contract_analyses) * 100,
            'vendor_concentration_risk': concentration_risk,
            'contract_type_distribution': contract_types,
            'portfolio_risk_level': self._assess_portfolio_risk_level(high_risk_contracts, concentration_risk),
            'diversification_recommendations': self._generate_diversification_recommendations(vendor_exposure, contract_types)
        }
        
    def _assess_portfolio_risk_level(self, high_risk_contracts: List[Dict[str, Any]], concentration_risk: float) -> str:
        """Assess overall portfolio risk level"""
        
        high_risk_percentage = len(high_risk_contracts) / max(1, len(high_risk_contracts)) * 100
        
        if high_risk_percentage > 30 or concentration_risk > 40:
            return 'High'
        elif high_risk_percentage > 15 or concentration_risk > 25:
            return 'Medium'
        else:
            return 'Low'
            
    def _generate_diversification_recommendations(self, vendor_exposure: Dict[str, float], 
                                                contract_types: Dict[str, int]) -> List[str]:
        """Generate portfolio diversification recommendations"""
        recommendations = []
        
        # Vendor diversification
        total_value = sum(vendor_exposure.values())
        max_exposure = max(vendor_exposure.values()) if vendor_exposure else 0
        
        if max_exposure / total_value > 0.4 if total_value > 0 else False:
            recommendations.append('Reduce vendor concentration risk by diversifying supplier base')
            
        # Contract type diversification
        if len(contract_types) < 3:
            recommendations.append('Consider diversifying contract types to reduce portfolio risk')
            
        if not recommendations:
            recommendations.append('Portfolio diversification is adequate')
            
        return recommendations
        
    def _analyze_vendor_performance(self, contract_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze vendor performance across contracts"""
        
        contracts = contract_data.get('contracts', [])
        vendor_performance = {}
        
        for contract in contracts:
            vendor = contract.get('vendor', 'Unknown')
            
            if vendor not in vendor_performance:
                vendor_performance[vendor] = {
                    'contracts': [],
                    'total_value': 0,
                    'average_performance': 0,
                    'risk_level': 'Low'
                }
                
            vendor_performance[vendor]['contracts'].append(contract.get('id', 'unknown'))
            vendor_performance[vendor]['total_value'] += contract.get('value', 0)
            
        # Calculate vendor metrics
        for vendor, data in vendor_performance.items():
            vendor_contracts = [c for c in contracts if c.get('vendor') == vendor]
            
            if vendor_contracts:
                avg_performance = sum(c.get('performance_history', 80) for c in vendor_contracts) / len(vendor_contracts)
                data['average_performance'] = avg_performance
                data['contract_count'] = len(vendor_contracts)
                data['risk_level'] = 'High' if avg_performance < 70 else 'Medium' if avg_performance < 85 else 'Low'
                
        # Identify top and bottom performers
        sorted_vendors = sorted(vendor_performance.items(), key=lambda x: x[1]['average_performance'], reverse=True)
        
        return {
            'vendor_performance': vendor_performance,
            'top_performers': [v[0] for v in sorted_vendors[:3]],
            'underperformers': [v[0] for v in sorted_vendors if v[1]['average_performance'] < 75],
            'vendor_count': len(vendor_performance),
            'performance_summary': self._summarize_vendor_performance(vendor_performance)
        }
        
    def _summarize_vendor_performance(self, vendor_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize overall vendor performance"""
        
        if not vendor_performance:
            return {'average_performance': 0, 'performance_distribution': {}}
            
        all_performances = [data['average_performance'] for data in vendor_performance.values()]
        avg_performance = sum(all_performances) / len(all_performances)
        
        performance_distribution = {
            'high_performers': len([p for p in all_performances if p >= 85]),
            'average_performers': len([p for p in all_performances if 70 <= p < 85]),
            'underperformers': len([p for p in all_performances if p < 70])
        }
        
        return {
            'average_performance': avg_performance,
            'performance_distribution': performance_distribution,
            'improvement_needed': performance_distribution['underperformers'] > 0
        }
        
    def _generate_contract_recommendations(self, contract_analyses: List[Dict[str, Any]], 
                                         portfolio_analysis: Dict[str, Any], 
                                         vendor_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate contract management recommendations"""
        
        recommendations = []
        
        # High-risk contract recommendations
        high_risk_contracts = [c for c in contract_analyses if c['risk_assessment']['overall_risk_level'] == 'High']
        if high_risk_contracts:
            recommendations.append({
                'category': 'Risk Management',
                'priority': 'High',
                'recommendation': f'Address {len(high_risk_contracts)} high-risk contracts immediately',
                'actions': [
                    'Conduct detailed risk assessments',
                    'Implement additional risk mitigation measures',
                    'Consider contract renegotiation or termination'
                ],
                'timeline': '1-3 months',
                'expected_impact': 'Reduce portfolio risk by 40-60%'
            })
            
        # Vendor performance recommendations
        if vendor_analysis.get('performance_summary', {}).get('improvement_needed', False):
            recommendations.append({
                'category': 'Vendor Management',
                'priority': 'Medium',
                'recommendation': 'Improve underperforming vendor relationships',
                'actions': [
                    'Implement vendor improvement plans',
                    'Establish performance monitoring systems',
                    'Consider vendor replacement for persistent issues'
                ],
                'timeline': '3-6 months',
                'expected_impact': 'Improve overall vendor performance by 15-25%'
            })
            
        # Portfolio optimization recommendations
        if portfolio_analysis.get('portfolio_risk_level') == 'High':
            recommendations.append({
                'category': 'Portfolio Optimization',
                'priority': 'Medium',
                'recommendation': 'Optimize contract portfolio structure',
                'actions': portfolio_analysis.get('diversification_recommendations', []),
                'timeline': '6-12 months',
                'expected_impact': 'Reduce portfolio concentration risk'
            })
            
        # Cost optimization recommendations
        total_optimization_value = 0
        for contract in contract_analyses:
            for opp in contract.get('optimization_opportunities', []):
                if 'potential_savings' in opp:
                    savings_str = str(opp['potential_savings']).replace('$', '').replace(',', '')
                    try:
                        total_optimization_value += float(savings_str)
                    except ValueError:
                        continue
        
        if total_optimization_value > 0:
            recommendations.append({
                'category': 'Cost Optimization',
                'priority': 'Medium',
                'recommendation': 'Implement contract cost optimization initiatives',
                'actions': [
                    'Renegotiate underperforming contracts',
                    'Implement performance-based pricing',
                    'Consolidate similar contracts for better terms'
                ],
                'timeline': '3-9 months',
                'expected_impact': f'Potential savings of ${total_optimization_value:,.0f}'
            })
            
        return recommendations

def test_contract_intelligence_agent():
    """Test the Contract Intelligence & Negotiation Agent"""
    print("🧪 Testing Contract Intelligence & Negotiation Agent")
    print("=" * 60)
    
    try:
        agent = ContractIntelligenceAgent()
        
        test_data = {
            'company_name': 'Contract Management Corp',
            'contracts': [
                {
                    'id': 'CONTRACT001',
                    'type': 'IT Services',
                    'vendor': 'TechCorp Solutions',
                    'value': 2500000,
                    'term_months': 24,
                    'performance_history': 85,
                    'on_time_delivery': 90,
                    'quality_score': 88,
                    'cost_efficiency': 78
                },
                {
                    'id': 'CONTRACT002',
                    'type': 'Consulting',
                    'vendor': 'Business Advisors Inc',
                    'value': 800000,
                    'term_months': 12,
                    'performance_history': 65,
                    'on_time_delivery': 70,
                    'quality_score': 68,
                    'cost_efficiency': 72
                }
            ]
        }
        
        analysis = agent.analyze_contract_portfolio(test_data)
        print(f"✅ Contract analysis completed for {test_data['company_name']}")
        print(f"   Contracts analyzed: {analysis['portfolio_analysis']['total_contracts']}")
        print(f"   Portfolio value: ${analysis['portfolio_analysis']['total_portfolio_value']:,}")
        print(f"   High-risk contracts: {analysis['portfolio_analysis']['high_risk_contracts']}")
        print(f"   Recommendations: {len(analysis['recommendations'])}")
        
        return {
            'agent_initialized': True,
            'contracts_analyzed': analysis['portfolio_analysis']['total_contracts'],
            'portfolio_value': analysis['portfolio_analysis']['total_portfolio_value'],
            'recommendations_generated': len(analysis['recommendations'])
        }
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return {'error': str(e)}

if __name__ == "__main__":
    test_contract_intelligence_agent()