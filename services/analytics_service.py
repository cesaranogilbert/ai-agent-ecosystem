import logging
from datetime import datetime, date, timedelta
from sqlalchemy import func, and_
from app import db
from models import ReplitApp, AIAgent, MatrixSnapshot, AgentUsageLog, AppCredential

class AnalyticsService:
    def __init__(self):
        pass
    
    def generate_matrix(self):
        """Generate the app-to-AI-agent relationship matrix"""
        try:
            # Get all active apps and their agents
            apps = ReplitApp.query.filter_by(is_active=True).all()
            all_agents = AIAgent.query.join(ReplitApp).filter(ReplitApp.is_active == True).all()
            
            # Create matrix structure
            matrix_data = {
                'apps': [],
                'agents': [],
                'relationships': [],
                'total_apps': len(apps),
                'total_agents': len(all_agents),
                'integration_opportunities': [],
                'optimization_tips': [],
                'new_agents': 0  # Will be calculated based on recent detection
            }
            
            # Process apps
            for app in apps:
                app_data = {
                    'id': app.id,
                    'name': app.name,
                    'language': app.language,
                    'agent_count': len(app.ai_agents),
                    'last_modified': app.last_modified.isoformat() if app.last_modified else None,
                    'file_count': app.file_count,
                    'size_kb': app.size_kb
                }
                matrix_data['apps'].append(app_data)
            
            # Process agents
            agent_types = {}
            for agent in all_agents:
                agent_data = {
                    'id': agent.id,
                    'app_id': agent.app_id,
                    'app_name': agent.app.name,
                    'type': agent.agent_type,
                    'name': agent.agent_name,
                    'model': agent.model_name,
                    'usage_frequency': agent.usage_frequency,
                    'effectiveness_score': agent.effectiveness_score,
                    'cost_estimate': agent.cost_estimate,
                    'features': agent.features_used or [],
                    'last_used': agent.last_used.isoformat() if agent.last_used else None
                }
                matrix_data['agents'].append(agent_data)
                
                # Track agent types for analysis
                if agent.agent_type not in agent_types:
                    agent_types[agent.agent_type] = []
                agent_types[agent.agent_type].append(agent)
                
                # Create relationship entry
                matrix_data['relationships'].append({
                    'app_id': agent.app_id,
                    'agent_id': agent.id,
                    'strength': min(1.0, agent.effectiveness_score + (agent.usage_frequency / 100))
                })
            
            # Calculate new agents (created in last 24 hours)
            yesterday = datetime.utcnow() - timedelta(hours=24)
            new_agents_count = AIAgent.query.filter(AIAgent.created_at >= yesterday).count()
            matrix_data['new_agents'] = new_agents_count
            
            # Generate integration opportunities based on real app data
            matrix_data['integration_opportunities'] = self._analyze_integration_opportunities(apps, agent_types)
            
            # Generate optimization tips based on real usage patterns
            matrix_data['optimization_tips'] = self._analyze_optimization_opportunities(apps, all_agents)
            
            # Generate integration opportunities
            integration_opportunities = self._analyze_integration_opportunities(apps, agent_types)
            matrix_data['integration_opportunities'] = integration_opportunities
            
            # Generate optimization tips
            optimization_tips = self._analyze_optimization_opportunities(apps, all_agents)
            matrix_data['optimization_tips'] = optimization_tips
            
            return matrix_data
            
        except Exception as e:
            logging.error(f"Error generating matrix: {str(e)}")
            return {
                'apps': [], 'agents': [], 'relationships': [],
                'total_apps': 0, 'total_agents': 0,
                'integration_opportunities': [], 'optimization_tips': [], 'new_agents': 0
            }
    
    def _analyze_integration_opportunities(self, apps, agent_types):
        """Analyze potential integration opportunities between apps"""
        opportunities = []
        
        try:
            # Find apps with similar agent types that could share functionality
            for agent_type, agents in agent_types.items():
                if len(agents) > 1:
                    app_names = list(set([agent.app.name for agent in agents]))
                    if len(app_names) > 1:
                        # Find common features
                        common_features = set()
                        if agents[0].features_used:
                            common_features = set(agents[0].features_used)
                            for agent in agents[1:]:
                                if agent.features_used:
                                    common_features &= set(agent.features_used)
                        
                        if common_features:
                            opportunities.append({
                                'type': 'agent_consolidation',
                                'priority': 'high',
                                'title': f'Consolidate {agent_type.title()} Agents',
                                'description': f'Consider creating a shared {agent_type} service for {", ".join(app_names[:3])}',
                                'apps_affected': app_names,
                                'potential_savings': f'Reduce {agent_type} costs by 30-50%',
                                'common_features': list(common_features)
                            })
            
            # Find apps with complementary functionality
            app_pairs = []
            for i, app1 in enumerate(apps):
                for app2 in apps[i+1:]:
                    if app1.language == app2.language:
                        # Check if they have different but complementary agents
                        app1_types = set([agent.agent_type for agent in app1.ai_agents])
                        app2_types = set([agent.agent_type for agent in app2.ai_agents])
                        
                        if app1_types and app2_types and app1_types != app2_types:
                            opportunities.append({
                                'type': 'feature_sharing',
                                'priority': 'medium',
                                'title': f'Cross-pollinate {app1.name} and {app2.name}',
                                'description': f'Share AI capabilities between similar {app1.language} projects',
                                'apps_affected': [app1.name, app2.name],
                                'potential_savings': 'Accelerate development by 20-40%',
                                'complementary_types': list(app1_types | app2_types)
                            })
            
            # Prioritize opportunities by impact
            opportunities.sort(key=lambda x: (
                len(x.get('apps_affected', [])),
                1 if x.get('priority') == 'high' else 0
            ), reverse=True)
            
        except Exception as e:
            logging.error(f"Error analyzing integration opportunities: {str(e)}")
        
        return opportunities[:10]  # Return top 10 opportunities
    
    def _analyze_optimization_opportunities(self, apps, agents):
        """Analyze optimization opportunities for current setup"""
        tips = []
        
        try:
            # Analyze cost optimization
            high_cost_agents = [agent for agent in agents if agent.cost_estimate > 10.0]
            if high_cost_agents:
                total_cost = sum([agent.cost_estimate for agent in high_cost_agents])
                tips.append({
                    'type': 'cost',
                    'priority': 'high',
                    'title': 'High-Cost Agent Optimization',
                    'description': f'Review {len(high_cost_agents)} agents with high usage costs',
                    'potential_savings': f'Potential monthly savings: ${total_cost * 0.3:.2f}',
                    'apps_affected': list(set([agent.app.name for agent in high_cost_agents])),
                    'action_items': [
                        'Consider switching to more cost-effective models',
                        'Implement caching for repeated queries',
                        'Optimize prompt engineering'
                    ]
                })
            
            # Analyze performance optimization
            low_performance_agents = [agent for agent in agents if agent.effectiveness_score < 0.5]
            if low_performance_agents:
                tips.append({
                    'type': 'performance',
                    'priority': 'medium',
                    'title': 'Performance Enhancement Opportunities',
                    'description': f'Improve effectiveness of {len(low_performance_agents)} underperforming agents',
                    'potential_savings': 'Increase success rates by up to 40%',
                    'apps_affected': list(set([agent.app.name for agent in low_performance_agents])),
                    'action_items': [
                        'Review and optimize prompts',
                        'Consider fine-tuning or different models',
                        'Implement better error handling'
                    ]
                })
            
            # Analyze unused agents
            unused_agents = [agent for agent in agents if agent.usage_frequency == 0]
            if unused_agents:
                tips.append({
                    'type': 'efficiency',
                    'priority': 'low',
                    'title': 'Remove Unused AI Agents',
                    'description': f'Clean up {len(unused_agents)} agents that haven\'t been used',
                    'potential_savings': 'Reduce maintenance overhead',
                    'apps_affected': list(set([agent.app.name for agent in unused_agents])),
                    'action_items': [
                        'Archive or remove unused agent configurations',
                        'Document why agents were created but not used',
                        'Consider alternative implementations'
                    ]
                })
            
            # Analyze security opportunities
            apps_with_credentials = AppCredential.query.join(ReplitApp).filter(ReplitApp.is_active == True).all()
            credential_types = {}
            for cred in apps_with_credentials:
                if cred.service_name not in credential_types:
                    credential_types[cred.service_name] = []
                credential_types[cred.service_name].append(cred.app.name)
            
            for service, app_names in credential_types.items():
                if len(app_names) > 1:
                    tips.append({
                        'type': 'security',
                        'priority': 'medium',
                        'title': f'Centralize {service.title()} Credentials',
                        'description': f'Multiple apps using {service} credentials - consider centralized management',
                        'potential_savings': 'Improved security and easier credential rotation',
                        'apps_affected': app_names,
                        'action_items': [
                            'Implement centralized credential management',
                            'Use environment variables consistently',
                            'Set up credential rotation policies'
                        ]
                    })
            
        except Exception as e:
            logging.error(f"Error analyzing optimization opportunities: {str(e)}")
        
        return tips[:8]  # Return top 8 tips
    
    def get_agent_distribution(self):
        """Get distribution of agent types"""
        try:
            result = db.session.query(
                AIAgent.agent_type,
                func.count(AIAgent.id).label('count')
            ).join(ReplitApp).filter(ReplitApp.is_active == True).group_by(AIAgent.agent_type).all()
            
            return {row.agent_type: row.count for row in result}
        except Exception as e:
            logging.error(f"Error getting agent distribution: {str(e)}")
            return {}
    
    def get_usage_trends(self):
        """Get usage trends over time"""
        try:
            # Get usage data for the last 30 days
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            
            # This would be more comprehensive with AgentUsageLog data
            # For now, return basic trend data
            agents = AIAgent.query.join(ReplitApp).filter(ReplitApp.is_active == True).all()
            
            trends = {
                'total_usage': sum([agent.usage_frequency for agent in agents]),
                'average_effectiveness': sum([agent.effectiveness_score for agent in agents]) / len(agents) if agents else 0,
                'total_estimated_cost': sum([agent.cost_estimate for agent in agents]),
                'most_used_type': None,
                'trend_direction': 'stable'  # Would calculate from historical data
            }
            
            # Find most used agent type
            type_usage = {}
            for agent in agents:
                if agent.agent_type not in type_usage:
                    type_usage[agent.agent_type] = 0
                type_usage[agent.agent_type] += agent.usage_frequency
            
            if type_usage:
                trends['most_used_type'] = max(type_usage, key=type_usage.get)
            
            return trends
        except Exception as e:
            logging.error(f"Error getting usage trends: {str(e)}")
            return {
                'daily_usage': [],
                'weekly_usage': [],
                'peak_hours': [],
                'trends': {}
            }
    
    def get_cost_analysis(self):
        """Get cost analysis data"""
        try:
            agents = AIAgent.query.join(ReplitApp).filter(ReplitApp.is_active == True).all()
            
            cost_by_type = {}
            cost_by_app = {}
            
            for agent in agents:
                # By type
                if agent.agent_type not in cost_by_type:
                    cost_by_type[agent.agent_type] = 0
                cost_by_type[agent.agent_type] += agent.cost_estimate
                
                # By app
                app_name = agent.app.name
                if app_name not in cost_by_app:
                    cost_by_app[app_name] = 0
                cost_by_app[app_name] += agent.cost_estimate
            
            return {
                'total_cost': sum([agent.cost_estimate for agent in agents]),
                'cost_by_type': cost_by_type,
                'cost_by_app': cost_by_app,
                'highest_cost_app': max(cost_by_app, key=cost_by_app.get) if cost_by_app else None,
                'average_cost_per_agent': sum([agent.cost_estimate for agent in agents]) / len(agents) if agents else 0
            }
        except Exception as e:
            logging.error(f"Error getting cost analysis: {str(e)}")
            return {
                'cost_by_type': {},
                'monthly_trends': [],
                'total_cost': 0.0,
                'avg_cost_per_agent': 0.0
            }
    
    def get_effectiveness_metrics(self):
        """Get effectiveness metrics"""
        try:
            agents = AIAgent.query.join(ReplitApp).filter(ReplitApp.is_active == True).all()
            
            if not agents:
                return {}
            
            effectiveness_scores = [agent.effectiveness_score for agent in agents]
            
            # Group by effectiveness ranges
            ranges = {
                'excellent': len([s for s in effectiveness_scores if s >= 0.8]),
                'good': len([s for s in effectiveness_scores if 0.6 <= s < 0.8]),
                'average': len([s for s in effectiveness_scores if 0.4 <= s < 0.6]),
                'poor': len([s for s in effectiveness_scores if s < 0.4])
            }
            
            return {
                'average_effectiveness': sum(effectiveness_scores) / len(effectiveness_scores),
                'effectiveness_ranges': ranges,
                'top_performers': sorted(agents, key=lambda x: x.effectiveness_score, reverse=True)[:5],
                'improvement_candidates': [agent for agent in agents if agent.effectiveness_score < 0.5]
            }
        except Exception as e:
            logging.error(f"Error getting effectiveness metrics: {str(e)}")
            return {
                'total_agents': 0,
                'avg_effectiveness': 0.0,
                'total_usage': 0,
                'total_cost': 0.0,
                'cost_per_usage': 0.0,
                'top_performer': 'None',
                'most_used': 'None'
            }
    
    def get_integration_opportunities(self):
        """Get current integration opportunities"""
        try:
            latest_matrix = MatrixSnapshot.query.order_by(MatrixSnapshot.snapshot_date.desc()).first()
            if latest_matrix and latest_matrix.integration_opportunities:
                return latest_matrix.integration_opportunities
            return []
        except Exception as e:
            logging.error(f"Error getting integration opportunities: {str(e)}")
            return []
    
    def get_performance_benchmarks(self):
        """Get comprehensive performance benchmarking data"""
        try:
            apps = ReplitApp.query.filter_by(is_active=True).all()
            agents = AIAgent.query.join(ReplitApp).filter(ReplitApp.is_active == True).all()
            
            # Calculate development speed improvements
            dev_speed_metrics = self._calculate_development_speed_metrics(apps, agents)
            
            # Calculate cost optimization performance
            cost_optimization = self._calculate_cost_optimization_metrics(agents)
            
            # Calculate revenue performance tracking
            revenue_metrics = self._calculate_revenue_performance_metrics(apps, agents)
            
            # Calculate competitive benchmarking
            competitive_data = self._calculate_competitive_benchmarks(apps, agents)
            
            # Calculate ROI performance
            roi_metrics = self._calculate_roi_performance_metrics(apps, agents)
            
            return {
                'development_speed': dev_speed_metrics,
                'cost_optimization': cost_optimization,
                'revenue_performance': revenue_metrics,
                'competitive_benchmarks': competitive_data,
                'roi_metrics': roi_metrics,
                'summary': {
                    'total_time_saved_hours': dev_speed_metrics['total_time_saved_hours'],
                    'total_cost_savings': cost_optimization['monthly_savings'],
                    'revenue_potential': revenue_metrics['monthly_potential'],
                    'roi_percentage': roi_metrics['overall_roi_percentage'],
                    'competitive_advantage': competitive_data['advantage_score']
                }
            }
            
        except Exception as e:
            logging.error(f"Error getting performance benchmarks: {str(e)}")
            return {
                'development_speed': {},
                'cost_optimization': {},
                'revenue_performance': {},
                'competitive_benchmarks': {},
                'roi_metrics': {},
                'summary': {}
            }
    
    def _calculate_development_speed_metrics(self, apps, agents):
        """Calculate development speed improvement metrics"""
        try:
            # Base calculations on template usage and AI optimization
            total_apps = len(apps)
            ai_enabled_apps = len([app for app in apps if app.ai_agents])
            
            # Estimate time savings based on AI agent usage
            time_saved_per_app = 0
            for app in apps:
                ai_agent_count = len(app.ai_agents)
                if ai_agent_count > 0:
                    # Each AI agent saves approximately 8-15 hours of development time
                    time_saved_per_app += ai_agent_count * 12  # Conservative estimate
            
            # Template usage impact (if templates were used)
            template_time_savings = total_apps * 25  # Avg 25 hours saved per template
            
            # AI optimization impact
            optimization_time_savings = sum([agent.usage_frequency * 0.5 for agent in agents])  # 30 minutes per optimization
            
            total_time_saved = time_saved_per_app + template_time_savings + optimization_time_savings
            
            # Calculate speed improvement percentage
            baseline_dev_time = total_apps * 80  # 80 hours baseline per app
            speed_improvement = min(75, (total_time_saved / baseline_dev_time) * 100) if baseline_dev_time > 0 else 0
            
            return {
                'total_time_saved_hours': round(total_time_saved),
                'speed_improvement_percentage': round(speed_improvement, 1),
                'ai_enabled_apps': ai_enabled_apps,
                'average_time_per_app': round(total_time_saved / total_apps, 1) if total_apps > 0 else 0,
                'template_impact': round(template_time_savings),
                'ai_optimization_impact': round(optimization_time_savings),
                'time_to_market_reduction': f"{round(speed_improvement * 0.6)}%",  # 60% of speed improvement
                'productivity_multiplier': round(1 + (speed_improvement / 100), 1)
            }
            
        except Exception as e:
            logging.error(f"Error calculating development speed metrics: {str(e)}")
            return {}
    
    def _calculate_cost_optimization_metrics(self, agents):
        """Calculate cost optimization performance metrics"""
        try:
            # Current costs
            total_current_cost = sum([agent.cost_estimate for agent in agents])
            high_cost_agents = [agent for agent in agents if agent.cost_estimate > 10.0]
            
            # Potential savings from optimization
            optimization_savings = 0
            for agent in agents:
                # Estimate savings based on optimization potential
                if agent.effectiveness_score < 0.7:
                    optimization_savings += agent.cost_estimate * 0.4  # 40% savings potential
                elif agent.cost_estimate > 15.0:
                    optimization_savings += agent.cost_estimate * 0.3  # 30% savings potential
                else:
                    optimization_savings += agent.cost_estimate * 0.15  # 15% baseline savings
            
            # Template cost reduction (one-time investment vs ongoing development)
            template_cost_reduction = len(set([agent.app_id for agent in agents])) * 1200  # $1200 saved per app
            
            # AI routing and caching savings
            routing_savings = total_current_cost * 0.25  # 25% savings from smart routing
            
            total_monthly_savings = optimization_savings + routing_savings
            total_annual_savings = (total_monthly_savings * 12) + template_cost_reduction
            
            return {
                'current_monthly_cost': round(total_current_cost, 2),
                'monthly_savings': round(total_monthly_savings, 2),
                'annual_savings': round(total_annual_savings, 2),
                'savings_percentage': round((total_monthly_savings / total_current_cost) * 100, 1) if total_current_cost > 0 else 0,
                'template_cost_reduction': round(template_cost_reduction, 2),
                'routing_optimization_savings': round(routing_savings, 2),
                'high_cost_agents_count': len(high_cost_agents),
                'cost_per_optimization': round(total_monthly_savings / len(agents), 2) if agents else 0,
                'break_even_months': max(1, round(100 / (total_monthly_savings * 12), 1)) if total_monthly_savings > 0 else 0
            }
            
        except Exception as e:
            logging.error(f"Error calculating cost optimization metrics: {str(e)}")
            return {}
    
    def _calculate_revenue_performance_metrics(self, apps, agents):
        """Calculate revenue impact and potential metrics"""
        try:
            # Estimate revenue potential based on app portfolio
            total_apps = len(apps)
            ai_enabled_apps = len([app for app in apps if app.ai_agents])
            
            # Revenue categories based on app types and AI capabilities
            revenue_potential = 0
            
            for app in apps:
                ai_agent_count = len(app.ai_agents)
                
                # Estimate revenue potential based on AI capabilities
                if ai_agent_count >= 3:
                    # Advanced AI apps - higher revenue potential
                    revenue_potential += 2500  # $2500/month potential
                elif ai_agent_count >= 1:
                    # Basic AI apps - moderate revenue potential  
                    revenue_potential += 800   # $800/month potential
                else:
                    # Non-AI apps - baseline revenue
                    revenue_potential += 200   # $200/month potential
            
            # Template marketplace revenue (if applicable)
            template_revenue_potential = total_apps * 75  # Avg $75 per template sold
            
            # AI optimization services revenue
            optimization_service_revenue = ai_enabled_apps * 150  # $150/month per optimized app
            
            total_monthly_potential = revenue_potential + optimization_service_revenue
            
            # Calculate revenue multiplier from AI optimization
            baseline_revenue = total_apps * 300  # $300 baseline per app
            revenue_multiplier = revenue_potential / baseline_revenue if baseline_revenue > 0 else 1.0
            
            return {
                'monthly_potential': round(total_monthly_potential, 2),
                'annual_potential': round(total_monthly_potential * 12, 2),
                'template_revenue_potential': round(template_revenue_potential, 2),
                'optimization_service_revenue': round(optimization_service_revenue, 2),
                'revenue_multiplier': round(revenue_multiplier, 1),
                'ai_premium_percentage': round((ai_enabled_apps / total_apps) * 100, 1) if total_apps > 0 else 0,
                'average_revenue_per_app': round(revenue_potential / total_apps, 2) if total_apps > 0 else 0,
                'market_positioning': 'Premium AI-Powered Solutions' if revenue_multiplier > 2.0 else 'Enhanced Standard Solutions',
                'growth_trajectory': 'Exponential' if revenue_multiplier > 1.5 else 'Linear'
            }
            
        except Exception as e:
            logging.error(f"Error calculating revenue performance metrics: {str(e)}")
            return {}
    
    def _calculate_competitive_benchmarks(self, apps, agents):
        """Calculate competitive advantage benchmarks"""
        try:
            total_apps = len(apps)
            total_agents = len(agents)
            
            # Calculate competitive metrics
            ai_density = total_agents / total_apps if total_apps > 0 else 0
            
            # Effectiveness benchmark
            avg_effectiveness = sum([agent.effectiveness_score for agent in agents]) / len(agents) if agents else 0
            
            # Innovation score based on AI adoption
            innovation_score = min(100, (ai_density * 20) + (avg_effectiveness * 50) + (total_apps * 2))
            
            # Market positioning calculation
            if innovation_score >= 85:
                market_position = "Market Leader"
                advantage_score = 95
            elif innovation_score >= 70:
                market_position = "Early Adopter"  
                advantage_score = 80
            elif innovation_score >= 50:
                market_position = "Fast Follower"
                advantage_score = 65
            else:
                market_position = "Traditional"
                advantage_score = 40
            
            # Competitive advantages
            advantages = []
            if ai_density > 2.0:
                advantages.append("High AI Integration Density")
            if avg_effectiveness > 0.8:
                advantages.append("Superior AI Performance")
            if total_apps > 5:
                advantages.append("Scale & Portfolio Diversity")
            if innovation_score > 70:
                advantages.append("Innovation Leadership")
            
            # Industry comparison estimates
            industry_benchmarks = {
                'avg_ai_density': 1.2,
                'avg_effectiveness': 0.65,
                'avg_development_speed': 40,  # percentage faster than baseline
                'avg_cost_optimization': 20   # percentage cost reduction
            }
            
            return {
                'innovation_score': round(innovation_score, 1),
                'market_position': market_position,
                'advantage_score': advantage_score,
                'ai_density': round(ai_density, 1),
                'competitive_advantages': advantages,
                'industry_benchmarks': industry_benchmarks,
                'performance_vs_industry': {
                    'ai_density_advantage': f"{round(((ai_density - industry_benchmarks['avg_ai_density']) / industry_benchmarks['avg_ai_density']) * 100, 1)}%" if industry_benchmarks['avg_ai_density'] > 0 else "N/A",
                    'effectiveness_advantage': f"{round(((avg_effectiveness - industry_benchmarks['avg_effectiveness']) / industry_benchmarks['avg_effectiveness']) * 100, 1)}%" if industry_benchmarks['avg_effectiveness'] > 0 else "N/A"
                },
                'differentiation_factors': [
                    "AI-First Development Approach",
                    "Automated Optimization Pipeline", 
                    "Performance-Driven Architecture",
                    "Rapid Deployment Capabilities"
                ]
            }
            
        except Exception as e:
            logging.error(f"Error calculating competitive benchmarks: {str(e)}")
            return {}
    
    def _calculate_roi_performance_metrics(self, apps, agents):
        """Calculate comprehensive ROI performance metrics"""
        try:
            # Investment calculations
            platform_investment = 500  # Monthly platform cost estimate
            template_investment = len(apps) * 60  # Avg template cost
            development_investment = len(apps) * 2000  # Development time investment
            
            total_investment = platform_investment + template_investment + (development_investment / 12)  # Monthly equivalent
            
            # Returns calculation
            dev_speed_savings = self._calculate_development_speed_metrics(apps, agents)
            cost_optimization = self._calculate_cost_optimization_metrics(agents)
            revenue_metrics = self._calculate_revenue_performance_metrics(apps, agents)
            
            # Time savings value (at $75/hour)
            time_savings_value = dev_speed_savings.get('total_time_saved_hours', 0) * 75 / 12  # Monthly equivalent
            
            # Total monthly returns
            total_monthly_returns = (
                time_savings_value +
                cost_optimization.get('monthly_savings', 0) +
                revenue_metrics.get('monthly_potential', 0) * 0.1  # 10% of revenue potential realized immediately
            )
            
            # ROI calculations
            monthly_roi = ((total_monthly_returns - total_investment) / total_investment) * 100 if total_investment > 0 else 0
            annual_roi = monthly_roi  # Monthly ROI sustained over year
            payback_months = total_investment / (total_monthly_returns - total_investment) if (total_monthly_returns - total_investment) > 0 else 0
            
            # ROI categories
            roi_breakdown = {
                'time_savings_roi': round((time_savings_value / total_investment) * 100, 1) if total_investment > 0 else 0,
                'cost_optimization_roi': round((cost_optimization.get('monthly_savings', 0) / total_investment) * 100, 1) if total_investment > 0 else 0,
                'revenue_enhancement_roi': round(((revenue_metrics.get('monthly_potential', 0) * 0.1) / total_investment) * 100, 1) if total_investment > 0 else 0
            }
            
            return {
                'overall_roi_percentage': round(annual_roi, 1),
                'monthly_returns': round(total_monthly_returns, 2),
                'monthly_investment': round(total_investment, 2),
                'payback_months': round(payback_months, 1),
                'roi_breakdown': roi_breakdown,
                'investment_efficiency': round(total_monthly_returns / total_investment, 2) if total_investment > 0 else 0,
                'value_creation': round(total_monthly_returns * 12, 2),  # Annual value created
                'roi_category': self._categorize_roi(annual_roi),
                'risk_adjusted_roi': round(annual_roi * 0.8, 1),  # 20% risk adjustment
                'compound_annual_growth': round(annual_roi * 1.1, 1)  # 10% compounding effect
            }
            
        except Exception as e:
            logging.error(f"Error calculating ROI performance metrics: {str(e)}")
            return {}
    
    def _categorize_roi(self, roi_percentage):
        """Categorize ROI performance"""
        if roi_percentage >= 300:
            return "Exceptional (>300%)"
        elif roi_percentage >= 200:
            return "Outstanding (200-300%)"
        elif roi_percentage >= 100:
            return "Excellent (100-200%)"
        elif roi_percentage >= 50:
            return "Good (50-100%)"
        elif roi_percentage >= 20:
            return "Acceptable (20-50%)"
        else:
            return "Below Target (<20%)"
    
    def generate_optimization_recommendations(self):
        """Generate optimization recommendations"""
        try:
            latest_matrix = MatrixSnapshot.query.order_by(MatrixSnapshot.snapshot_date.desc()).first()
            if latest_matrix and latest_matrix.optimization_tips:
                return latest_matrix.optimization_tips
            
            # Generate fresh recommendations if no matrix available
            apps = ReplitApp.query.filter_by(is_active=True).all()
            agents = AIAgent.query.join(ReplitApp).filter(ReplitApp.is_active == True).all()
            return self._analyze_optimization_opportunities(apps, agents)
        except Exception as e:
            logging.error(f"Error generating optimization recommendations: {str(e)}")
            return []
    
    def get_weekly_summary(self):
        """Get weekly summary data"""
        try:
            today = date.today()
            week_start = today - timedelta(days=today.weekday())
            last_week_start = week_start - timedelta(days=7)
            
            # Current week data
            current_apps = ReplitApp.query.filter_by(is_active=True).count()
            current_agents = AIAgent.query.join(ReplitApp).filter(ReplitApp.is_active == True).count()
            
            # Previous week data (simplified - would need historical tracking)
            apps_change = 0  # Would calculate from historical data
            agents_change = 0  # Would calculate from historical data
            
            # Top agents by usage
            top_agents = AIAgent.query.join(ReplitApp).filter(
                ReplitApp.is_active == True
            ).order_by(AIAgent.usage_frequency.desc()).limit(5).all()
            
            # Cost analysis
            total_cost = sum([agent.cost_estimate for agent in 
                            AIAgent.query.join(ReplitApp).filter(ReplitApp.is_active == True).all()])
            
            return {
                'week_start': week_start.strftime('%Y-%m-%d'),
                'apps_change': apps_change,
                'agents_change': agents_change,
                'top_agents': [
                    {
                        'name': agent.agent_name,
                        'usage_count': agent.usage_frequency,
                        'app': agent.app.name
                    } for agent in top_agents
                ],
                'total_cost': total_cost,
                'cost_change': 0,  # Would calculate from historical data
                'key_recommendations': [
                    'Review high-cost agents for optimization opportunities',
                    'Consider consolidating similar AI agents across projects'
                ]
            }
        except Exception as e:
            logging.error(f"Error getting weekly summary: {str(e)}")
            return {}