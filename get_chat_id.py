#!/usr/bin/env python3
"""
Helper script to get user's Telegram chat ID after they message the bot
"""

import os
import requests
import json

def get_chat_id():
    """Get the latest message and extract chat ID"""
    
    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
    if not bot_token:
        print("❌ TELEGRAM_BOT_TOKEN not set")
        return None
        
    try:
        # Get updates from Telegram
        url = f'https://api.telegram.org/bot{bot_token}/getUpdates'
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('ok') and data.get('result'):
                updates = data['result']
                if updates:
                    # Get the most recent message
                    latest_update = updates[-1]
                    if 'message' in latest_update:
                        chat = latest_update['message']['chat']
                        chat_id = chat['id']
                        chat_type = chat['type']
                        
                        print("✅ Found your message!")
                        print(f"   Chat ID: {chat_id}")
                        print(f"   Chat Type: {chat_type}")
                        
                        if chat_type == 'private':
                            user = latest_update['message']['from']
                            print(f"   From: {user.get('first_name', '')} {user.get('last_name', '')}")
                            print(f"   Username: @{user.get('username', 'unknown')}")
                        
                        return str(chat_id)
                    else:
                        print("⚠️ No message found in latest update")
                else:
                    print("📫 No messages yet - please send a message to @replitArchitectBot first")
            else:
                print(f"❌ API Error: {data}")
        else:
            print(f"❌ HTTP Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        
    return None

if __name__ == "__main__":
    print("=== GETTING YOUR TELEGRAM CHAT ID ===")
    chat_id = get_chat_id()
    
    if chat_id:
        print(f"\n🎉 SUCCESS! Your chat ID is: {chat_id}")
        print("\nI'll now configure this and send you a welcome message...")
    else:
        print("\n📱 Please send a message to @replitArchitectBot first, then run this again.")