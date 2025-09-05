import os
import requests
import logging
from datetime import datetime
from app import db
from models import TelegramNotification, SystemSettings

class TelegramService:
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN_REPLARCHITECT', '') or os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.chat_id = self._get_chat_id()
        self.base_url = f'https://api.telegram.org/bot{self.bot_token}'
        
    def _get_chat_id(self):
        """Get chat ID from settings or environment"""
        # Try to get from database settings first
        setting = SystemSettings.query.filter_by(setting_key='telegram_chat_id').first()
        if setting and setting.setting_value:
            return setting.setting_value
        
        # Fallback to environment variable
        return os.getenv('TELEGRAM_CHAT_ID', '')
    
    def send_notification(self, message, notification_type='general'):
        """Send a notification via Telegram"""
        try:
            if not self.bot_token or len(self.bot_token) < 10:
                logging.warning("TELEGRAM_BOT_TOKEN invalid or not found - notification logged only")
                self._log_notification_only(message, notification_type)
                return True  # Return True to not break the integration flow
                
            if not self.chat_id:
                logging.warning("TELEGRAM_CHAT_ID not found - notification logged only")
                self._log_notification_only(message, notification_type)
                return True
            
            # Save notification to database first
            notification = TelegramNotification()
            notification.notification_type = notification_type
            notification.message = message
            notification.chat_id = self.chat_id
            db.session.add(notification)
            db.session.commit()
            
            # Send via Telegram API
            url = f'{self.base_url}/sendMessage'
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            }
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                notification.is_sent = True
                db.session.commit()
                logging.info(f"Telegram notification sent successfully: {notification_type}")
                return True
            else:
                logging.warning(f"Telegram API error: {response.status_code} - {response.text}")
                self._log_notification_only(message, notification_type)
                return True  # Continue execution even if Telegram fails
                
        except requests.exceptions.RequestException as e:
            logging.warning(f"Network error sending Telegram notification: {str(e)}")
            self._log_notification_only(message, notification_type)
            return True
        except Exception as e:
            logging.warning(f"Error sending Telegram notification: {str(e)}")
            self._log_notification_only(message, notification_type)
            return True
    
    def _log_notification_only(self, message, notification_type):
        """Log notification to database only when Telegram API is unavailable"""
        try:
            notification = TelegramNotification()
            notification.notification_type = notification_type
            notification.message = message
            notification.chat_id = self.chat_id or 'unknown'
            notification.is_sent = False  # Mark as not sent since Telegram failed
            db.session.add(notification)
            db.session.commit()
            logging.info(f"Notification logged to database: {notification_type}")
        except Exception as e:
            logging.error(f"Failed to log notification: {str(e)}")
    
    def send_daily_summary(self, summary_data):
        """Send daily summary with matrix updates and optimization tips"""
        try:
            # Format the daily summary message
            message = self._format_daily_summary(summary_data)
            return self.send_notification(message, 'daily_summary')
            
        except Exception as e:
            logging.error(f"Error sending daily summary: {str(e)}")
            return False
    
    def send_agent_alert(self, alert_data):
        """Send alert about new AI agents or changes"""
        try:
            message = self._format_agent_alert(alert_data)
            return self.send_notification(message, 'agent_alert')
            
        except Exception as e:
            logging.error(f"Error sending agent alert: {str(e)}")
            return False
    
    def send_optimization_tip(self, tip_data):
        """Send optimization recommendation"""
        try:
            message = self._format_optimization_tip(tip_data)
            return self.send_notification(message, 'optimization_tip')
            
        except Exception as e:
            logging.error(f"Error sending optimization tip: {str(e)}")
            return False
    
    def _format_daily_summary(self, data):
        """Format daily summary message"""
        total_apps = data.get('total_apps', 0)
        total_agents = data.get('total_agents', 0)
        new_apps = data.get('new_apps', 0)
        new_agents = data.get('new_agents', 0)
        integrations = data.get('integration_opportunities', [])
        optimizations = data.get('optimization_tips', [])
        
        message = f"""🚀 *Daily Replit Summary* - {datetime.now().strftime('%Y-%m-%d')}

📊 *Overview:*
• Total Apps: {total_apps} ({f'+{new_apps}' if new_apps > 0 else 'no change'})
• AI Agents: {total_agents} ({f'+{new_agents}' if new_agents > 0 else 'no change'})

"""
        
        if integrations:
            message += "🔗 *Integration Opportunities:*\n"
            for i, integration in enumerate(integrations[:3], 1):
                message += f"• {integration}\n"
            if len(integrations) > 3:
                message += f"• ...and {len(integrations) - 3} more\n"
            message += "\n"
        
        if optimizations:
            message += "💡 *Optimization Tips:*\n"
            for i, tip in enumerate(optimizations[:3], 1):
                message += f"• {tip}\n"
            if len(optimizations) > 3:
                message += f"• ...and {len(optimizations) - 3} more\n"
            message += "\n"
        
        message += "📱 View full details in your [Replit Manager Dashboard](your-app-url)"
        
        return message
    
    def _format_agent_alert(self, data):
        """Format AI agent alert message"""
        agent_name = data.get('agent_name', 'Unknown Agent')
        app_name = data.get('app_name', 'Unknown App')
        agent_type = data.get('agent_type', 'unknown')
        action = data.get('action', 'detected')  # detected, updated, removed
        
        emoji_map = {
            'openai': '🤖',
            'anthropic': '🧠',
            'huggingface': '🤗',
            'local': '🏠',
            'custom': '⚙️'
        }
        
        emoji = emoji_map.get(agent_type, '🔍')
        
        if action == 'detected':
            message = f"""{emoji} *New AI Agent Detected*

🎯 *Agent:* {agent_name}
📱 *App:* {app_name}
🏷️ *Type:* {agent_type.title()}

This agent was automatically discovered during today's analysis. Check the dashboard for integration opportunities!"""
        
        elif action == 'updated':
            message = f"""{emoji} *AI Agent Updated*

🎯 *Agent:* {agent_name}
📱 *App:* {app_name}
🔄 *Changes detected in agent configuration or usage patterns.*"""
        
        else:  # removed
            message = f"""❌ *AI Agent Removed*

🎯 *Agent:* {agent_name}
📱 *App:* {app_name}
🗑️ *Agent is no longer active in this app.*"""
        
        return message
    
    def _format_optimization_tip(self, data):
        """Format optimization tip message"""
        tip_type = data.get('type', 'general')
        title = data.get('title', 'Optimization Opportunity')
        description = data.get('description', '')
        potential_savings = data.get('potential_savings', '')
        apps_affected = data.get('apps_affected', [])
        
        emoji_map = {
            'cost': '💰',
            'performance': '⚡',
            'integration': '🔗',
            'security': '🔒',
            'efficiency': '📈'
        }
        
        emoji = emoji_map.get(tip_type, '💡')
        
        message = f"""{emoji} *{title}*

📝 *Description:*
{description}

"""
        
        if potential_savings:
            message += f"💰 *Potential Savings:* {potential_savings}\n\n"
        
        if apps_affected:
            message += f"📱 *Apps Affected:* {', '.join(apps_affected[:3])}"
            if len(apps_affected) > 3:
                message += f" (+{len(apps_affected) - 3} more)"
            message += "\n\n"
        
        message += "🎯 *Action Required:* Review these recommendations in your dashboard"
        
        return message
    
    def test_connection(self):
        """Test Telegram bot connection"""
        try:
            if not self.bot_token:
                return False, "Bot token not configured"
                
            url = f'{self.base_url}/getMe'
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                bot_info = response.json()
                if bot_info.get('ok'):
                    return True, f"Connected to bot: {bot_info['result']['first_name']}"
                else:
                    return False, "Invalid bot token"
            else:
                return False, f"API error: {response.status_code}"
                
        except Exception as e:
            return False, f"Connection error: {str(e)}"
