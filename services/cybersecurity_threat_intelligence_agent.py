"""
Cybersecurity Threat Intelligence Agent
Advanced threat detection and security posture optimization
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class ThreatIndicator:
    indicator_type: str
    value: str
    severity: str
    confidence: float
    first_seen: datetime

class CybersecurityThreatIntelligenceAgent:
    """
    Comprehensive Cybersecurity Threat Intelligence System
    - Advanced threat detection and analysis
    - Security posture assessment
    - Vulnerability management
    - Incident response optimization
    """
    
    def __init__(self):
        self.name = "Cybersecurity Threat Intelligence Agent"
        self.version = "1.0.0"
        self.capabilities = [
            "Threat Detection",
            "Security Assessment",
            "Vulnerability Analysis",
            "Incident Response",
            "Risk Evaluation",
            "Security Monitoring"
        ]
        
    def assess_security_posture(self, security_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive security posture assessment"""
        try:
            company_name = security_data.get('company_name', 'Unknown Company')
            
            # Threat landscape analysis
            threat_analysis = self._analyze_threat_landscape(security_data)
            
            # Vulnerability assessment
            vulnerability_assessment = self._assess_vulnerabilities(security_data)
            
            # Security controls evaluation
            controls_evaluation = self._evaluate_security_controls(security_data)
            
            # Risk assessment
            risk_assessment = self._calculate_security_risks(threat_analysis, vulnerability_assessment, controls_evaluation)
            
            # Generate recommendations
            recommendations = self._generate_security_recommendations(
                threat_analysis, vulnerability_assessment, controls_evaluation, risk_assessment
            )
            
            return {
                'company': company_name,
                'assessment_date': datetime.now().isoformat(),
                'threat_analysis': threat_analysis,
                'vulnerability_assessment': vulnerability_assessment,
                'controls_evaluation': controls_evaluation,
                'risk_assessment': risk_assessment,
                'recommendations': recommendations,
                'next_review_date': (datetime.now() + timedelta(days=7)).isoformat()
            }
            
        except Exception as e:
            logging.error(f"Security assessment failed: {str(e)}")
            return {'error': f'Security assessment failed: {str(e)}'}
            
    def _analyze_threat_landscape(self, security_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current threat landscape"""
        
        # Industry-specific threat patterns
        industry = security_data.get('industry', 'general').lower()
        threat_vectors = self._get_industry_threat_vectors(industry)
        
        # Current threat indicators
        current_threats = security_data.get('detected_threats', [])
        
        # Threat intelligence analysis
        threat_intelligence = self._process_threat_intelligence(current_threats)
        
        # Threat actor profiling
        actor_analysis = self._analyze_threat_actors(security_data)
        
        return {
            'industry_threat_vectors': threat_vectors,
            'current_threat_count': len(current_threats),
            'threat_intelligence': threat_intelligence,
            'threat_actor_analysis': actor_analysis,
            'threat_level': self._calculate_overall_threat_level(threat_intelligence, actor_analysis)
        }
        
    def _get_industry_threat_vectors(self, industry: str) -> List[Dict[str, Any]]:
        """Get industry-specific threat vectors"""
        
        threat_vectors_map = {
            'financial': [
                {'vector': 'Banking Trojans', 'likelihood': 'High', 'impact': 'Critical'},
                {'vector': 'Credential Stuffing', 'likelihood': 'High', 'impact': 'High'},
                {'vector': 'Business Email Compromise', 'likelihood': 'Medium', 'impact': 'High'},
                {'vector': 'Ransomware', 'likelihood': 'Medium', 'impact': 'Critical'}
            ],
            'healthcare': [
                {'vector': 'Ransomware', 'likelihood': 'High', 'impact': 'Critical'},
                {'vector': 'Data Breaches', 'likelihood': 'High', 'impact': 'Critical'},
                {'vector': 'IoT Device Attacks', 'likelihood': 'Medium', 'impact': 'Medium'},
                {'vector': 'Insider Threats', 'likelihood': 'Medium', 'impact': 'High'}
            ],
            'technology': [
                {'vector': 'Supply Chain Attacks', 'likelihood': 'Medium', 'impact': 'Critical'},
                {'vector': 'Zero-day Exploits', 'likelihood': 'Medium', 'impact': 'High'},
                {'vector': 'DDoS Attacks', 'likelihood': 'High', 'impact': 'Medium'},
                {'vector': 'Intellectual Property Theft', 'likelihood': 'Medium', 'impact': 'High'}
            ],
            'manufacturing': [
                {'vector': 'Industrial Control System Attacks', 'likelihood': 'Medium', 'impact': 'Critical'},
                {'vector': 'Ransomware', 'likelihood': 'High', 'impact': 'High'},
                {'vector': 'Supply Chain Compromise', 'likelihood': 'Medium', 'impact': 'High'},
                {'vector': 'Insider Threats', 'likelihood': 'Medium', 'impact': 'Medium'}
            ]
        }
        
        return threat_vectors_map.get(industry, [
            {'vector': 'Phishing', 'likelihood': 'High', 'impact': 'Medium'},
            {'vector': 'Malware', 'likelihood': 'Medium', 'impact': 'Medium'},
            {'vector': 'Ransomware', 'likelihood': 'Medium', 'impact': 'High'},
            {'vector': 'Data Breaches', 'likelihood': 'Medium', 'impact': 'High'}
        ])
        
    def _process_threat_intelligence(self, threats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process threat intelligence data"""
        
        threat_categories = {}
        severity_distribution = {'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0}
        
        for threat in threats:
            category = threat.get('category', 'Unknown')
            severity = threat.get('severity', 'Medium')
            
            threat_categories[category] = threat_categories.get(category, 0) + 1
            if severity in severity_distribution:
                severity_distribution[severity] += 1
                
        # Calculate threat velocity (threats per day)
        threat_velocity = len(threats) / 7 if threats else 0  # Assuming 7-day window
        
        return {
            'threat_categories': threat_categories,
            'severity_distribution': severity_distribution,
            'threat_velocity': threat_velocity,
            'active_campaigns': self._identify_active_campaigns(threats),
            'trending_threats': self._identify_trending_threats(threat_categories)
        }
        
    def _identify_active_campaigns(self, threats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify active threat campaigns"""
        
        # Group threats by similar characteristics
        campaigns = []
        
        # Simplified campaign detection based on threat clustering
        threat_groups = {}
        for threat in threats:
            campaign_id = threat.get('campaign_id', 'Unknown')
            if campaign_id not in threat_groups:
                threat_groups[campaign_id] = []
            threat_groups[campaign_id].append(threat)
            
        for campaign_id, campaign_threats in threat_groups.items():
            if len(campaign_threats) >= 3:  # At least 3 related threats
                campaigns.append({
                    'campaign_id': campaign_id,
                    'threat_count': len(campaign_threats),
                    'severity': max(t.get('severity', 'Low') for t in campaign_threats),
                    'first_detected': min(t.get('detected_at', datetime.now().isoformat()) for t in campaign_threats),
                    'status': 'Active'
                })
                
        return campaigns
        
    def _identify_trending_threats(self, threat_categories: Dict[str, int]) -> List[Dict[str, Any]]:
        """Identify trending threat types"""
        
        # Sort categories by frequency
        sorted_threats = sorted(threat_categories.items(), key=lambda x: x[1], reverse=True)
        
        trending = []
        for threat_type, count in sorted_threats[:5]:  # Top 5
            trending.append({
                'threat_type': threat_type,
                'occurrence_count': count,
                'trend_status': 'Rising' if count > 5 else 'Stable'
            })
            
        return trending
        
    def _analyze_threat_actors(self, security_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze threat actor profiles"""
        
        # Actor attribution based on TTPs
        known_actors = security_data.get('attributed_actors', [])
        
        actor_profiles = []
        for actor in known_actors:
            profile = {
                'actor_name': actor.get('name', 'Unknown'),
                'sophistication': actor.get('sophistication', 'Medium'),
                'motivation': actor.get('motivation', 'Financial'),
                'target_industries': actor.get('target_industries', []),
                'threat_level': self._assess_actor_threat_level(actor)
            }
            actor_profiles.append(profile)
            
        return {
            'identified_actors': len(actor_profiles),
            'actor_profiles': actor_profiles,
            'primary_motivations': self._categorize_actor_motivations(actor_profiles),
            'sophistication_levels': self._categorize_sophistication_levels(actor_profiles)
        }
        
    def _assess_actor_threat_level(self, actor: Dict[str, Any]) -> str:
        """Assess threat level for specific actor"""
        
        sophistication = actor.get('sophistication', 'Medium')
        targeting_score = len(actor.get('target_industries', [])) * 10
        capability_score = {'Low': 20, 'Medium': 50, 'High': 80, 'Advanced': 100}.get(sophistication, 50)
        
        total_score = capability_score + targeting_score
        
        if total_score >= 90:
            return 'Critical'
        elif total_score >= 70:
            return 'High'
        elif total_score >= 50:
            return 'Medium'
        else:
            return 'Low'
            
    def _categorize_actor_motivations(self, actor_profiles: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorize actor motivations"""
        
        motivations = {}
        for actor in actor_profiles:
            motivation = actor['motivation']
            motivations[motivation] = motivations.get(motivation, 0) + 1
            
        return motivations
        
    def _categorize_sophistication_levels(self, actor_profiles: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorize sophistication levels"""
        
        sophistication = {}
        for actor in actor_profiles:
            level = actor['sophistication']
            sophistication[level] = sophistication.get(level, 0) + 1
            
        return sophistication
        
    def _calculate_overall_threat_level(self, threat_intel: Dict[str, Any], actor_analysis: Dict[str, Any]) -> str:
        """Calculate overall threat level"""
        
        # Threat velocity factor
        velocity_score = min(100, threat_intel['threat_velocity'] * 10)
        
        # Severity factor
        severity_dist = threat_intel['severity_distribution']
        severity_score = (severity_dist['Critical'] * 4 + severity_dist['High'] * 3 + 
                         severity_dist['Medium'] * 2 + severity_dist['Low'] * 1)
        
        # Actor sophistication factor
        actor_count = actor_analysis['identified_actors']
        actor_score = min(100, actor_count * 20)
        
        # Combined threat level
        overall_score = (velocity_score + severity_score + actor_score) / 3
        
        if overall_score >= 75:
            return 'Critical'
        elif overall_score >= 50:
            return 'High'
        elif overall_score >= 25:
            return 'Medium'
        else:
            return 'Low'
            
    def _assess_vulnerabilities(self, security_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess system vulnerabilities"""
        
        vulnerabilities = security_data.get('vulnerabilities', [])
        
        # Categorize vulnerabilities by severity
        severity_counts = {'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0}
        exploitable_vulns = 0
        
        vuln_categories = {}
        
        for vuln in vulnerabilities:
            severity = vuln.get('severity', 'Medium')
            category = vuln.get('category', 'Other')
            
            if severity in severity_counts:
                severity_counts[severity] += 1
                
            if vuln.get('exploitable', False):
                exploitable_vulns += 1
                
            vuln_categories[category] = vuln_categories.get(category, 0) + 1
            
        # Calculate vulnerability score
        vuln_score = self._calculate_vulnerability_score(severity_counts, exploitable_vulns, len(vulnerabilities))
        
        # Identify critical systems
        critical_systems = self._identify_critical_vulnerable_systems(vulnerabilities)
        
        return {
            'total_vulnerabilities': len(vulnerabilities),
            'severity_distribution': severity_counts,
            'exploitable_vulnerabilities': exploitable_vulns,
            'vulnerability_categories': vuln_categories,
            'vulnerability_score': vuln_score,
            'critical_vulnerable_systems': critical_systems,
            'remediation_priority': self._prioritize_vulnerability_remediation(vulnerabilities)
        }
        
    def _calculate_vulnerability_score(self, severity_counts: Dict[str, int], exploitable: int, total: int) -> float:
        """Calculate overall vulnerability score"""
        
        if total == 0:
            return 0
            
        # Weighted severity score
        weighted_score = (severity_counts['Critical'] * 10 + severity_counts['High'] * 7 + 
                         severity_counts['Medium'] * 4 + severity_counts['Low'] * 1)
        
        # Exploitability factor
        exploitability_factor = (exploitable / total) * 100 if total > 0 else 0
        
        # Combined score (0-100, higher is worse)
        vulnerability_score = min(100, (weighted_score / total * 10) + exploitability_factor)
        
        return vulnerability_score
        
    def _identify_critical_vulnerable_systems(self, vulnerabilities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify systems with critical vulnerabilities"""
        
        critical_systems = []
        system_vuln_counts = {}
        
        for vuln in vulnerabilities:
            system = vuln.get('affected_system', 'Unknown')
            severity = vuln.get('severity', 'Medium')
            
            if system not in system_vuln_counts:
                system_vuln_counts[system] = {'Critical': 0, 'High': 0, 'Medium': 0, 'Low': 0}
                
            if severity in system_vuln_counts[system]:
                system_vuln_counts[system][severity] += 1
                
        for system, counts in system_vuln_counts.items():
            if counts['Critical'] > 0 or counts['High'] > 2:
                critical_systems.append({
                    'system_name': system,
                    'critical_vulns': counts['Critical'],
                    'high_vulns': counts['High'],
                    'total_vulns': sum(counts.values()),
                    'risk_level': 'Critical' if counts['Critical'] > 0 else 'High'
                })
                
        return critical_systems
        
    def _prioritize_vulnerability_remediation(self, vulnerabilities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize vulnerability remediation"""
        
        prioritized = []
        
        for vuln in vulnerabilities:
            priority_score = self._calculate_remediation_priority(vuln)
            
            prioritized.append({
                'vulnerability_id': vuln.get('id', 'unknown'),
                'severity': vuln.get('severity', 'Medium'),
                'affected_system': vuln.get('affected_system', 'Unknown'),
                'exploitable': vuln.get('exploitable', False),
                'priority_score': priority_score,
                'remediation_effort': vuln.get('remediation_effort', 'Medium'),
                'business_impact': vuln.get('business_impact', 'Medium')
            })
            
        # Sort by priority score (highest first)
        prioritized.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return prioritized[:10]  # Top 10 priorities
        
    def _calculate_remediation_priority(self, vuln: Dict[str, Any]) -> float:
        """Calculate remediation priority score"""
        
        severity_weights = {'Critical': 10, 'High': 7, 'Medium': 4, 'Low': 1}
        
        base_score = severity_weights.get(vuln.get('severity', 'Medium'), 4)
        
        # Exploitability bonus
        if vuln.get('exploitable', False):
            base_score *= 1.5
            
        # Business impact factor
        impact_weights = {'Critical': 1.5, 'High': 1.3, 'Medium': 1.0, 'Low': 0.7}
        impact_factor = impact_weights.get(vuln.get('business_impact', 'Medium'), 1.0)
        
        # Effort penalty (higher effort = lower priority)
        effort_penalties = {'Low': 1.0, 'Medium': 0.8, 'High': 0.6}
        effort_factor = effort_penalties.get(vuln.get('remediation_effort', 'Medium'), 0.8)
        
        priority_score = base_score * impact_factor * effort_factor
        
        return priority_score
        
    def _evaluate_security_controls(self, security_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate security controls effectiveness"""
        
        controls = security_data.get('security_controls', {})
        
        # Evaluate different control categories
        control_categories = {
            'preventive': ['firewall', 'antivirus', 'access_control', 'encryption'],
            'detective': ['ids_ips', 'siem', 'monitoring', 'log_analysis'],
            'corrective': ['incident_response', 'backup_recovery', 'patch_management'],
            'administrative': ['security_policies', 'training', 'risk_management']
        }
        
        control_effectiveness = {}
        
        for category, control_list in control_categories.items():
            category_score = 0
            implemented_controls = 0
            
            for control in control_list:
                if control in controls:
                    effectiveness = controls[control].get('effectiveness', 50)
                    category_score += effectiveness
                    implemented_controls += 1
                    
            if implemented_controls > 0:
                avg_effectiveness = category_score / implemented_controls
            else:
                avg_effectiveness = 0
                
            control_effectiveness[category] = {
                'average_effectiveness': avg_effectiveness,
                'implemented_controls': implemented_controls,
                'total_controls': len(control_list),
                'coverage_percentage': (implemented_controls / len(control_list)) * 100
            }
            
        # Calculate overall control maturity
        overall_maturity = self._calculate_control_maturity(control_effectiveness)
        
        return {
            'control_effectiveness': control_effectiveness,
            'overall_maturity': overall_maturity,
            'control_gaps': self._identify_control_gaps(control_effectiveness),
            'maturity_level': self._categorize_maturity_level(overall_maturity)
        }
        
    def _calculate_control_maturity(self, control_effectiveness: Dict[str, Any]) -> float:
        """Calculate overall control maturity"""
        
        total_score = 0
        total_weight = 0
        
        # Weight different categories
        category_weights = {'preventive': 0.3, 'detective': 0.3, 'corrective': 0.25, 'administrative': 0.15}
        
        for category, weight in category_weights.items():
            if category in control_effectiveness:
                effectiveness = control_effectiveness[category]['average_effectiveness']
                coverage = control_effectiveness[category]['coverage_percentage']
                
                # Combined score considers both effectiveness and coverage
                category_score = (effectiveness * 0.7 + coverage * 0.3)
                total_score += category_score * weight
                total_weight += weight
                
        return total_score / total_weight if total_weight > 0 else 0
        
    def _identify_control_gaps(self, control_effectiveness: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify security control gaps"""
        
        gaps = []
        
        for category, metrics in control_effectiveness.items():
            if metrics['coverage_percentage'] < 75:
                gaps.append({
                    'category': category,
                    'gap_type': 'Coverage Gap',
                    'current_coverage': metrics['coverage_percentage'],
                    'target_coverage': 85,
                    'priority': 'High' if metrics['coverage_percentage'] < 50 else 'Medium'
                })
                
            if metrics['average_effectiveness'] < 70:
                gaps.append({
                    'category': category,
                    'gap_type': 'Effectiveness Gap',
                    'current_effectiveness': metrics['average_effectiveness'],
                    'target_effectiveness': 80,
                    'priority': 'High' if metrics['average_effectiveness'] < 50 else 'Medium'
                })
                
        return gaps
        
    def _categorize_maturity_level(self, maturity_score: float) -> str:
        """Categorize security maturity level"""
        
        if maturity_score >= 80:
            return 'Optimized'
        elif maturity_score >= 70:
            return 'Managed'
        elif maturity_score >= 60:
            return 'Defined'
        elif maturity_score >= 50:
            return 'Repeatable'
        else:
            return 'Initial'
            
    def _calculate_security_risks(self, threat_analysis: Dict[str, Any], 
                                vuln_assessment: Dict[str, Any], 
                                controls_evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive security risks"""
        
        # Risk factors
        threat_level_score = {'Critical': 100, 'High': 75, 'Medium': 50, 'Low': 25}.get(
            threat_analysis['threat_level'], 50)
        
        vulnerability_score = vuln_assessment['vulnerability_score']
        
        control_maturity = controls_evaluation['overall_maturity']
        control_factor = 100 - control_maturity  # Lower maturity = higher risk
        
        # Calculate overall risk score
        risk_score = (threat_level_score * 0.4 + vulnerability_score * 0.4 + control_factor * 0.2)
        
        risk_level = self._categorize_risk_level(risk_score)
        
        # Identify specific risk scenarios
        risk_scenarios = self._identify_risk_scenarios(threat_analysis, vuln_assessment)
        
        return {
            'overall_risk_score': risk_score,
            'risk_level': risk_level,
            'threat_factor': threat_level_score,
            'vulnerability_factor': vulnerability_score,
            'control_factor': control_factor,
            'risk_scenarios': risk_scenarios,
            'business_impact_assessment': self._assess_business_impact(risk_score, risk_scenarios)
        }
        
    def _categorize_risk_level(self, risk_score: float) -> str:
        """Categorize overall risk level"""
        
        if risk_score >= 80:
            return 'Critical'
        elif risk_score >= 60:
            return 'High'
        elif risk_score >= 40:
            return 'Medium'
        else:
            return 'Low'
            
    def _identify_risk_scenarios(self, threat_analysis: Dict[str, Any], vuln_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific risk scenarios"""
        
        scenarios = []
        
        # Data breach scenario
        if vuln_assessment['exploitable_vulnerabilities'] > 0:
            scenarios.append({
                'scenario': 'Data Breach',
                'probability': 'High' if vuln_assessment['exploitable_vulnerabilities'] > 5 else 'Medium',
                'impact': 'Critical',
                'description': 'Exploitation of vulnerabilities leading to data exfiltration'
            })
            
        # Ransomware scenario
        if threat_analysis['threat_level'] in ['Critical', 'High']:
            scenarios.append({
                'scenario': 'Ransomware Attack',
                'probability': 'Medium',
                'impact': 'Critical',
                'description': 'Ransomware deployment causing business disruption'
            })
            
        # Insider threat scenario
        scenarios.append({
            'scenario': 'Insider Threat',
            'probability': 'Medium',
            'impact': 'High',
            'description': 'Malicious or negligent insider activities'
        })
        
        return scenarios
        
    def _assess_business_impact(self, risk_score: float, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess potential business impact"""
        
        # Calculate potential financial impact
        base_revenue = 50000000  # Assumed annual revenue
        
        if risk_score >= 80:
            potential_loss_percentage = 0.15  # 15% of revenue
        elif risk_score >= 60:
            potential_loss_percentage = 0.08  # 8% of revenue
        elif risk_score >= 40:
            potential_loss_percentage = 0.03  # 3% of revenue
        else:
            potential_loss_percentage = 0.01  # 1% of revenue
            
        potential_financial_impact = base_revenue * potential_loss_percentage
        
        return {
            'potential_financial_impact': potential_financial_impact,
            'business_disruption_risk': 'High' if risk_score >= 70 else 'Medium' if risk_score >= 50 else 'Low',
            'reputation_impact_risk': 'High' if any(s['impact'] == 'Critical' for s in scenarios) else 'Medium',
            'regulatory_compliance_risk': 'High' if risk_score >= 75 else 'Medium' if risk_score >= 50 else 'Low'
        }
        
    def _generate_security_recommendations(self, threat_analysis: Dict[str, Any], 
                                         vuln_assessment: Dict[str, Any], 
                                         controls_evaluation: Dict[str, Any], 
                                         risk_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate security improvement recommendations"""
        
        recommendations = []
        
        # Critical vulnerability remediation
        if vuln_assessment['severity_distribution']['Critical'] > 0:
            recommendations.append({
                'category': 'Vulnerability Management',
                'priority': 'Critical',
                'recommendation': f'Immediately remediate {vuln_assessment["severity_distribution"]["Critical"]} critical vulnerabilities',
                'actions': [
                    'Deploy emergency patches for critical vulnerabilities',
                    'Implement temporary mitigating controls',
                    'Monitor affected systems continuously'
                ],
                'timeline': '72 hours',
                'expected_impact': 'Significantly reduce attack surface'
            })
            
        # Threat detection enhancement
        if threat_analysis['threat_level'] in ['Critical', 'High']:
            recommendations.append({
                'category': 'Threat Detection',
                'priority': 'High',
                'recommendation': 'Enhance threat detection and response capabilities',
                'actions': [
                    'Implement advanced threat detection tools',
                    'Enhance security monitoring coverage',
                    'Improve incident response procedures'
                ],
                'timeline': '2-4 weeks',
                'expected_impact': 'Improved threat visibility and faster response'
            })
            
        # Security control improvements
        control_gaps = controls_evaluation.get('control_gaps', [])
        high_priority_gaps = [gap for gap in control_gaps if gap['priority'] == 'High']
        
        if high_priority_gaps:
            recommendations.append({
                'category': 'Security Controls',
                'priority': 'Medium',
                'recommendation': f'Address {len(high_priority_gaps)} high-priority control gaps',
                'actions': [
                    'Implement missing security controls',
                    'Improve effectiveness of existing controls',
                    'Establish regular control testing procedures'
                ],
                'timeline': '1-3 months',
                'expected_impact': 'Strengthen overall security posture'
            })
            
        # Risk management
        if risk_assessment['risk_level'] in ['Critical', 'High']:
            recommendations.append({
                'category': 'Risk Management',
                'priority': 'High',
                'recommendation': 'Implement comprehensive risk management program',
                'actions': [
                    'Conduct detailed risk assessment',
                    'Develop risk mitigation strategies',
                    'Establish risk monitoring and reporting'
                ],
                'timeline': '4-8 weeks',
                'expected_impact': 'Better risk visibility and management'
            })
            
        return recommendations

def test_cybersecurity_threat_intelligence_agent():
    """Test the Cybersecurity Threat Intelligence Agent"""
    print("🧪 Testing Cybersecurity Threat Intelligence Agent")
    print("=" * 55)
    
    try:
        agent = CybersecurityThreatIntelligenceAgent()
        
        test_data = {
            'company_name': 'SecureEnterprises Corp',
            'industry': 'financial',
            'detected_threats': [
                {
                    'category': 'Malware',
                    'severity': 'High',
                    'campaign_id': 'Campaign_A',
                    'detected_at': '2025-09-01T10:00:00Z'
                },
                {
                    'category': 'Phishing',
                    'severity': 'Medium',
                    'campaign_id': 'Campaign_A',
                    'detected_at': '2025-09-01T11:00:00Z'
                }
            ],
            'vulnerabilities': [
                {
                    'id': 'CVE-2024-001',
                    'severity': 'Critical',
                    'category': 'SQL Injection',
                    'affected_system': 'Web Application',
                    'exploitable': True,
                    'business_impact': 'High'
                },
                {
                    'id': 'CVE-2024-002',
                    'severity': 'High',
                    'category': 'Buffer Overflow',
                    'affected_system': 'Core Service',
                    'exploitable': False,
                    'business_impact': 'Medium'
                }
            ],
            'security_controls': {
                'firewall': {'effectiveness': 85},
                'antivirus': {'effectiveness': 80},
                'ids_ips': {'effectiveness': 75},
                'siem': {'effectiveness': 70}
            }
        }
        
        assessment = agent.assess_security_posture(test_data)
        print(f"✅ Security assessment completed for {test_data['company_name']}")
        print(f"   Threat level: {assessment['threat_analysis']['threat_level']}")
        print(f"   Vulnerabilities: {assessment['vulnerability_assessment']['total_vulnerabilities']}")
        print(f"   Risk level: {assessment['risk_assessment']['risk_level']}")
        print(f"   Recommendations: {len(assessment['recommendations'])}")
        
        return {
            'agent_initialized': True,
            'threats_analyzed': len(test_data['detected_threats']),
            'vulnerabilities_assessed': len(test_data['vulnerabilities']),
            'risk_level': assessment['risk_assessment']['risk_level']
        }
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return {'error': str(e)}

if __name__ == "__main__":
    test_cybersecurity_threat_intelligence_agent()