import logging
from datetime import datetime, time
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from services.replit_service import ReplitService
from services.ai_agent_service import AIAgentService
from services.telegram_service import TelegramService
from services.analytics_service import AnalyticsService
from models import ReplitApp, AIAgent, MatrixSnapshot
from app import db
import atexit

scheduler = None

def init_scheduler():
    """Initialize the background scheduler for daily tasks"""
    global scheduler
    
    try:
        scheduler = BackgroundScheduler()
        
        # Daily matrix update and discovery at 8 AM
        scheduler.add_job(
            func=daily_discovery_and_matrix_update,
            trigger=CronTrigger(hour=8, minute=0),
            id='daily_matrix_update',
            name='Daily Discovery and Matrix Update',
            replace_existing=True
        )
        
        # Agent analysis every 6 hours
        scheduler.add_job(
            func=periodic_agent_analysis,
            trigger=CronTrigger(hour='*/6'),
            id='periodic_agent_analysis',
            name='Periodic Agent Analysis',
            replace_existing=True
        )
        
        # Optimization tips every 2 days at 10 AM
        scheduler.add_job(
            func=send_optimization_tips,
            trigger=CronTrigger(day='*/2', hour=10, minute=0),
            id='optimization_tips',
            name='Send Optimization Tips',
            replace_existing=True
        )
        
        # Weekly summary on Sundays at 9 AM
        scheduler.add_job(
            func=send_weekly_summary,
            trigger=CronTrigger(day_of_week='sun', hour=9, minute=0),
            id='weekly_summary',
            name='Send Weekly Summary',
            replace_existing=True
        )
        
        scheduler.start()
        logging.info("Scheduler initialized with daily tasks")
        
        # Shutdown scheduler when app exits
        atexit.register(lambda: scheduler.shutdown())
        
    except Exception as e:
        logging.error(f"Error initializing scheduler: {str(e)}")

def daily_discovery_and_matrix_update():
    """Daily task to discover apps and update matrix"""
    try:
        logging.info("Starting daily discovery and matrix update")
        
        # Discover new/updated apps
        replit_service = ReplitService()
        discovered_count = replit_service.discover_apps()
        
        # Analyze all apps for AI agents
        ai_service = AIAgentService()
        analyzed_count = ai_service.analyze_all_apps()
        
        # Generate new matrix
        analytics_service = AnalyticsService()
        matrix_data = analytics_service.generate_matrix()
        
        # Save matrix snapshot
        from datetime import date
        today = date.today()
        
        # Remove old snapshot if exists
        old_snapshot = MatrixSnapshot.query.filter_by(snapshot_date=today).first()
        if old_snapshot:
            db.session.delete(old_snapshot)
        
        # Create new snapshot
        new_snapshot = MatrixSnapshot(
            snapshot_date=today,
            matrix_data=matrix_data,
            total_apps=matrix_data.get('total_apps', 0),
            total_agents=matrix_data.get('total_agents', 0),
            integration_opportunities=matrix_data.get('integration_opportunities', []),
            optimization_tips=matrix_data.get('optimization_tips', [])
        )
        db.session.add(new_snapshot)
        db.session.commit()
        
        # Send daily summary via Telegram
        telegram_service = TelegramService()
        summary_data = {
            'total_apps': matrix_data.get('total_apps', 0),
            'total_agents': matrix_data.get('total_agents', 0),
            'new_apps': discovered_count,
            'new_agents': matrix_data.get('new_agents', 0),
            'integration_opportunities': matrix_data.get('integration_opportunities', []),
            'optimization_tips': matrix_data.get('optimization_tips', [])
        }
        
        telegram_service.send_daily_summary(summary_data)
        
        logging.info(f"Daily update completed: {discovered_count} apps discovered, {analyzed_count} apps analyzed")
        
    except Exception as e:
        logging.error(f"Error in daily discovery and matrix update: {str(e)}")

def periodic_agent_analysis():
    """Periodic analysis of AI agents for changes"""
    try:
        logging.info("Starting periodic agent analysis")
        
        ai_service = AIAgentService()
        
        # Get apps that have been modified recently
        from datetime import datetime, timedelta
        recent_threshold = datetime.utcnow() - timedelta(hours=6)
        
        recent_apps = ReplitApp.query.filter(
            ReplitApp.is_active == True,
            ReplitApp.updated_at >= recent_threshold
        ).all()
        
        analyzed_count = 0
        new_agents_detected = []
        
        for app in recent_apps:
            try:
                # Store existing agents for comparison
                existing_agents = {
                    (agent.agent_type, agent.agent_name): agent 
                    for agent in app.ai_agents
                }
                
                # Re-analyze the app
                ai_service.analyze_app_for_agents(app)
                
                # Check for new agents
                current_agents = {
                    (agent.agent_type, agent.agent_name): agent 
                    for agent in app.ai_agents
                }
                
                for agent_key, agent in current_agents.items():
                    if agent_key not in existing_agents:
                        new_agents_detected.append({
                            'agent_name': agent.agent_name,
                            'app_name': app.name,
                            'agent_type': agent.agent_type,
                            'action': 'detected'
                        })
                
                analyzed_count += 1
                
            except Exception as e:
                logging.error(f"Error analyzing app {app.name}: {str(e)}")
                continue
        
        db.session.commit()
        
        # Send alerts for new agents
        if new_agents_detected:
            telegram_service = TelegramService()
            for agent_data in new_agents_detected[:5]:  # Limit to 5 alerts
                telegram_service.send_agent_alert(agent_data)
        
        logging.info(f"Periodic analysis completed: {analyzed_count} apps analyzed, {len(new_agents_detected)} new agents detected")
        
    except Exception as e:
        logging.error(f"Error in periodic agent analysis: {str(e)}")

def send_optimization_tips():
    """Send optimization tips based on current analysis"""
    try:
        logging.info("Generating and sending optimization tips")
        
        analytics_service = AnalyticsService()
        optimization_tips = analytics_service.generate_optimization_recommendations()
        
        if optimization_tips:
            telegram_service = TelegramService()
            
            # Send up to 3 most important tips
            for tip in optimization_tips[:3]:
                telegram_service.send_optimization_tip(tip)
                
        logging.info(f"Sent {min(len(optimization_tips), 3)} optimization tips")
        
    except Exception as e:
        logging.error(f"Error sending optimization tips: {str(e)}")

def send_weekly_summary():
    """Send comprehensive weekly summary"""
    try:
        logging.info("Generating weekly summary")
        
        analytics_service = AnalyticsService()
        weekly_data = analytics_service.get_weekly_summary()
        
        telegram_service = TelegramService()
        
        # Format weekly summary message
        message = f"""📊 *Weekly Replit Summary*

🗓️ *Week of {weekly_data.get('week_start', 'N/A')}*

📈 *Growth:*
• Apps: {weekly_data.get('apps_change', 0):+d}
• AI Agents: {weekly_data.get('agents_change', 0):+d}

🔥 *Most Active Agents:*
"""
        
        for agent in weekly_data.get('top_agents', [])[:3]:
            message += f"• {agent['name']} ({agent['usage_count']} uses)\n"
        
        message += f"""
💰 *Cost Analysis:*
• Total Estimated: ${weekly_data.get('total_cost', 0):.2f}
• Change: ${weekly_data.get('cost_change', 0):+.2f}

🎯 *Key Recommendations:*
"""
        
        for rec in weekly_data.get('key_recommendations', [])[:2]:
            message += f"• {rec}\n"
        
        telegram_service.send_notification(message, 'weekly_summary')
        
        logging.info("Weekly summary sent successfully")
        
    except Exception as e:
        logging.error(f"Error sending weekly summary: {str(e)}")

def get_scheduler_status():
    """Get current scheduler status and job information"""
    global scheduler
    
    if not scheduler:
        return {'status': 'not_initialized', 'jobs': []}
    
    try:
        jobs = []
        for job in scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None,
                'trigger': str(job.trigger)
            })
        
        return {
            'status': 'running' if scheduler.running else 'stopped',
            'jobs': jobs
        }
        
    except Exception as e:
        logging.error(f"Error getting scheduler status: {str(e)}")
        return {'status': 'error', 'jobs': [], 'error': str(e)}
