#!/usr/bin/env python3
"""
GitHub Repository Upload Script
Uploads the complete AI Agent Ecosystem to GitHub using the GitHub API
"""
import os
import json
import base64
import requests
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GitHubUploader:
    def __init__(self):
        self.token = os.environ.get('GITHUB_TOKEN')
        self.username = os.environ.get('GITHUB_USERNAME')
        self.headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.api_base = 'https://api.github.com'
        
    def create_repository(self, repo_name, description, private=False):
        """Create a new GitHub repository"""
        url = f'{self.api_base}/user/repos'
        data = {
            'name': repo_name,
            'description': description,
            'private': private,
            'auto_init': False
        }
        
        response = requests.post(url, headers=self.headers, json=data)
        if response.status_code == 201:
            logger.info(f"Repository '{repo_name}' created successfully")
            return response.json()
        elif response.status_code == 422:
            # Repository already exists
            logger.info(f"Repository '{repo_name}' already exists")
            return self.get_repository(repo_name)
        else:
            logger.error(f"Failed to create repository: {response.status_code} - {response.text}")
            return None
    
    def get_repository(self, repo_name):
        """Get repository information"""
        url = f'{self.api_base}/repos/{self.username}/{repo_name}'
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        return None
    
    def upload_file(self, repo_name, file_path, content, commit_message):
        """Upload a single file to the repository"""
        url = f'{self.api_base}/repos/{self.username}/{repo_name}/contents/{file_path}'
        
        # Encode content as base64
        if isinstance(content, str):
            content_bytes = content.encode('utf-8')
        else:
            content_bytes = content
        content_b64 = base64.b64encode(content_bytes).decode('utf-8')
        
        data = {
            'message': commit_message,
            'content': content_b64
        }
        
        # Check if file already exists
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            # File exists, get SHA for update
            data['sha'] = response.json()['sha']
        
        response = requests.put(url, headers=self.headers, json=data)
        if response.status_code in [200, 201]:
            logger.info(f"Uploaded: {file_path}")
            return True
        else:
            logger.error(f"Failed to upload {file_path}: {response.status_code} - {response.text}")
            return False
    
    def upload_directory(self, repo_name, local_dir, remote_dir="", exclude_patterns=None):
        """Upload entire directory structure to repository"""
        if exclude_patterns is None:
            exclude_patterns = ['.git', '__pycache__', '*.pyc', '.env', 'node_modules']
        
        success_count = 0
        total_count = 0
        
        for root, dirs, files in os.walk(local_dir):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]
            
            for file in files:
                # Skip excluded files
                if any(pattern in file for pattern in exclude_patterns):
                    continue
                
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_dir)
                remote_file_path = os.path.join(remote_dir, relative_path).replace('\\', '/')
                
                try:
                    with open(local_file_path, 'rb') as f:
                        content = f.read()
                    
                    total_count += 1
                    if self.upload_file(repo_name, remote_file_path, content, f"Add {relative_path}"):
                        success_count += 1
                        
                except Exception as e:
                    logger.error(f"Error uploading {local_file_path}: {str(e)}")
        
        return success_count, total_count

def main():
    """Main function to upload the AI Agent Ecosystem to GitHub"""
    uploader = GitHubUploader()
    
    # Repository details
    repo_name = "ai-agent-ecosystem"
    description = "Comprehensive AI Agent Ecosystem - 103 specialized agents targeting $40.94T market opportunity across enterprise, emerging markets, and future research domains"
    
    # Create repository
    logger.info("Creating GitHub repository...")
    repo = uploader.create_repository(repo_name, description, private=False)
    if not repo:
        logger.error("Failed to create repository")
        return False
    
    logger.info(f"Repository URL: {repo['html_url']}")
    
    # Upload all files
    logger.info("Uploading project files...")
    success_count, total_count = uploader.upload_directory(
        repo_name, 
        ".", 
        exclude_patterns=['.git', '__pycache__', '*.pyc', '.env', 'node_modules', '*.log']
    )
    
    logger.info(f"Upload completed: {success_count}/{total_count} files uploaded successfully")
    
    if success_count == total_count:
        logger.info("✅ All files uploaded successfully!")
        return True
    else:
        logger.warning(f"⚠️ {total_count - success_count} files failed to upload")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)