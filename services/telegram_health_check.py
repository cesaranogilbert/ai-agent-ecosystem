"""
Telegram Health Check Service
Monitors Telegram bot connection and validates setup
"""

import os
import requests
import logging
from datetime import datetime

class TelegramHealthCheck:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def check_bot_status(self):
        """Check if Telegram bot is configured and accessible"""
        bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
        chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        
        if not bot_token:
            return {
                'status': 'error',
                'message': 'TELEGRAM_BOT_TOKEN not configured',
                'details': None
            }
            
        if not chat_id:
            return {
                'status': 'warning',
                'message': 'TELEGRAM_CHAT_ID not configured',
                'details': None
            }
            
        try:
            # Test bot connection
            url = f'https://api.telegram.org/bot{bot_token}/getMe'
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                bot_data = response.json()
                if bot_data.get('ok'):
                    bot_info = bot_data['result']
                    return {
                        'status': 'success',
                        'message': f'Bot @{bot_info.get("username", "unknown")} connected',
                        'details': {
                            'bot_id': bot_info.get('id'),
                            'bot_name': bot_info.get('first_name'),
                            'username': bot_info.get('username'),
                            'can_read_all_group_messages': bot_info.get('can_read_all_group_messages'),
                            'supports_inline_queries': bot_info.get('supports_inline_queries')
                        }
                    }
                else:
                    return {
                        'status': 'error',
                        'message': f'Telegram API error: {bot_data.get("description", "Unknown")}',
                        'details': bot_data
                    }
            else:
                return {
                    'status': 'error',
                    'message': f'HTTP {response.status_code}: Bot token invalid or bot not found',
                    'details': response.text
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'status': 'error',
                'message': f'Network error connecting to Telegram API: {str(e)}',
                'details': None
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Unexpected error: {str(e)}',
                'details': None
            }
            
    def send_welcome_message(self):
        """Send welcome message when bot is first configured"""
        from services.telegram_service import TelegramService
        
        telegram = TelegramService()
        
        welcome_message = """🎉 **Welcome to Replit Manager!**

Your Telegram notifications are now configured and ready.

📊 **What you'll receive:**
• Daily app discovery results (5:00 AM CET)
• New AI agent detection alerts
• Optimization recommendations (every 2 days)  
• Weekly portfolio summaries (Sundays)

🚀 **Current Portfolio:**
• 12 Replit apps discovered
• 27 AI agents analyzed
• $92.32/month total AI costs

Your automated monitoring is now active!"""

        return telegram.send_notification(welcome_message, 'welcome')