import os
import requests
import logging
from datetime import datetime
from app import db
from models import ReplitApp

class ReplitService:
    def __init__(self):
        self.api_token = os.getenv('REPLIT_TOKEN', '')
        self.base_url = 'https://replit.com/graphql'
        self.headers = {
            'Authorization': f'Bearer {self.api_token}',
            'Content-Type': 'application/json'
        }
        
    def discover_apps(self):
        """Discover all Replit apps for the authenticated user"""
        try:
            if not self.api_token:
                logging.error("REPLIT_TOKEN not found in environment variables")
                return 0
                
            # GraphQL query to get user's repls
            query = """
            query GetUserRepls {
                currentUser {
                    repls(count: 100) {
                        items {
                            id
                            title
                            slug
                            url
                            language
                            description
                            timeCreated
                            timeUpdated
                            isPrivate
                            size
                            files(count: 100) {
                                items {
                                    path
                                    size
                                }
                            }
                        }
                    }
                }
            }
            """
            
            response = requests.post(
                self.base_url,
                json={'query': query},
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code != 200:
                logging.error(f"Replit API error: {response.status_code} - {response.text}")
                return 0
                
            data = response.json()
            
            if 'errors' in data:
                logging.error(f"GraphQL errors: {data['errors']}")
                return 0
                
            repls = data.get('data', {}).get('currentUser', {}).get('repls', {}).get('items', [])
            discovered_count = 0
            
            for repl in repls:
                try:
                    # Check if app already exists
                    existing_app = ReplitApp.query.filter_by(repl_id=repl['id']).first()
                    
                    if existing_app:
                        # Update existing app
                        existing_app.name = repl['title']
                        existing_app.url = repl.get('url', '')
                        existing_app.language = repl.get('language', '')
                        existing_app.description = repl.get('description', '')
                        existing_app.file_count = len(repl.get('files', {}).get('items', []))
                        existing_app.size_kb = repl.get('size', 0)
                        existing_app.last_modified = datetime.fromisoformat(repl['timeUpdated'].replace('Z', '+00:00')) if repl.get('timeUpdated') else None
                        existing_app.updated_at = datetime.utcnow()
                    else:
                        # Create new app
                        new_app = ReplitApp(
                            repl_id=repl['id'],
                            name=repl['title'],
                            url=repl.get('url', ''),
                            language=repl.get('language', ''),
                            description=repl.get('description', ''),
                            file_count=len(repl.get('files', {}).get('items', [])),
                            size_kb=repl.get('size', 0),
                            last_modified=datetime.fromisoformat(repl['timeUpdated'].replace('Z', '+00:00')) if repl.get('timeUpdated') else None
                        )
                        db.session.add(new_app)
                    
                    discovered_count += 1
                    
                except Exception as e:
                    logging.error(f"Error processing repl {repl.get('id', 'unknown')}: {str(e)}")
                    continue
            
            db.session.commit()
            logging.info(f"Successfully discovered {discovered_count} Replit apps")
            return discovered_count
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error while discovering apps: {str(e)}")
            return 0
        except Exception as e:
            logging.error(f"Unexpected error in discover_apps: {str(e)}")
            return 0
    
    def get_app_files(self, repl_id):
        """Get file contents for a specific repl"""
        try:
            if not self.api_token:
                return []
                
            query = """
            query GetReplFiles($replId: String!) {
                repl(id: $replId) {
                    files(count: 100) {
                        items {
                            path
                            content
                            size
                        }
                    }
                }
            }
            """
            
            response = requests.post(
                self.base_url,
                json={
                    'query': query,
                    'variables': {'replId': repl_id}
                },
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code != 200:
                logging.error(f"Error fetching files for repl {repl_id}: {response.status_code}")
                return []
                
            data = response.json()
            
            if 'errors' in data:
                logging.error(f"GraphQL errors fetching files: {data['errors']}")
                return []
                
            files = data.get('data', {}).get('repl', {}).get('files', {}).get('items', [])
            return files
            
        except Exception as e:
            logging.error(f"Error getting app files for {repl_id}: {str(e)}")
            return []
    
    def mark_inactive_apps(self, active_repl_ids):
        """Mark apps as inactive if they're no longer in the user's account"""
        try:
            # Get all current apps
            all_apps = ReplitApp.query.all()
            
            for app in all_apps:
                if app.repl_id not in active_repl_ids:
                    app.is_active = False
                    
            db.session.commit()
            logging.info("Updated inactive app statuses")
            
        except Exception as e:
            logging.error(f"Error marking inactive apps: {str(e)}")
