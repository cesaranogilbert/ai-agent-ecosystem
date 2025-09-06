#!/usr/bin/env python3
import os
import requests
import json

def analyze_github_repositories():
    print('🔍 COMPREHENSIVE GITHUB REPOSITORY ANALYSIS')
    print('=' * 60)

    # Get GitHub credentials
    github_token = os.environ.get('GITHUB_TOKEN')
    github_username = os.environ.get('GITHUB_USERNAME')

    if not github_token or not github_username:
        print('❌ GitHub credentials not found')
        return

    print(f'📋 Analyzing repositories for user: {github_username}')
    print('=' * 50)

    headers = {
        'Authorization': f'token {github_token}',
        'Accept': 'application/vnd.github.v3+json'
    }

    try:
        # Get all repositories
        url = f'https://api.github.com/users/{github_username}/repos'
        response = requests.get(url, headers=headers, params={'per_page': 100, 'sort': 'updated'})
        
        if response.status_code == 200:
            repos = response.json()
            print(f'✅ Found {len(repos)} repositories')
            
            # Analyze each repository
            ai_repos = []
            business_repos = []
            development_repos = []
            total_estimated_value = 0
            
            for repo in repos:
                repo_name = repo['name']
                repo_description = repo.get('description', 'No description')
                repo_url = repo['html_url']
                updated_at = repo['updated_at']
                language = repo.get('language', 'Unknown')
                size = repo['size']  # in KB
                
                print(f'\n📁 Repository: {repo_name}')
                print(f'   Description: {repo_description}')
                print(f'   Language: {language} | Size: {size} KB')
                print(f'   Updated: {updated_at}')
                
                # Estimate repository value based on name and description
                estimated_value = 0
                repo_category = 'development'
                
                # Check for AI agent keywords
                ai_keywords = ['agent', 'ai', 'intelligence', 'gpt', 'openai', 'llm']
                business_keywords = ['business', 'consulting', 'enterprise', 'client', 'revenue']
                
                repo_text = (repo_name + ' ' + (repo_description or '')).lower()
                
                # AI Agent repositories (highest value)
                if 'agent' in repo_text and any(kw in repo_text for kw in ai_keywords):
                    estimated_value = 5000000  # $5M for AI agent repos
                    repo_category = 'ai_agents'
                    print('   🤖 IDENTIFIED AS AI AGENT REPOSITORY')
                
                # Business/consulting repositories
                elif any(kw in repo_text for kw in business_keywords):
                    estimated_value = 500000  # $500K for business systems
                    repo_category = 'business'
                    print('   💼 IDENTIFIED AS BUSINESS SYSTEM')
                
                # AI-related but not agents
                elif any(kw in repo_text for kw in ai_keywords):
                    estimated_value = 200000  # $200K for AI utilities
                    repo_category = 'ai_utility'
                    print('   🔧 IDENTIFIED AS AI UTILITY')
                
                # Development/utility repositories
                else:
                    if size > 1000:
                        estimated_value = 50000  # $50K for large repos
                    elif size > 100:
                        estimated_value = 25000  # $25K for medium repos
                    else:
                        estimated_value = 10000  # $10K for small repos
                    print('   🛠️  IDENTIFIED AS DEVELOPMENT TOOL')
                
                # Store repository info
                repo_info = {
                    'name': repo_name,
                    'description': repo_description,
                    'category': repo_category,
                    'language': language,
                    'size': size,
                    'estimated_value': estimated_value,
                    'url': repo_url,
                    'updated_at': updated_at
                }
                
                if repo_category == 'ai_agents':
                    ai_repos.append(repo_info)
                elif repo_category == 'business':
                    business_repos.append(repo_info)
                else:
                    development_repos.append(repo_info)
                
                total_estimated_value += estimated_value
                
                print(f'   💰 Estimated Value: ${estimated_value:,}')
                print(f'   📂 Category: {repo_category.upper()}')
            
            # Summary report
            print(f'\n\n🏆 COMPREHENSIVE REPOSITORY ANALYSIS SUMMARY')
            print('=' * 60)
            
            if ai_repos:
                print(f'\n🤖 AI AGENT REPOSITORIES ({len(ai_repos)}):')
                print('-' * 40)
                ai_total = 0
                for repo in ai_repos:
                    ai_total += repo['estimated_value']
                    print(f'✅ {repo["name"]} - ${repo["estimated_value"]:,}')
                    print(f'   {repo["description"]}')
                print(f'\nAI Agents Total: ${ai_total:,}')
            
            if business_repos:
                print(f'\n💼 BUSINESS REPOSITORIES ({len(business_repos)}):')
                print('-' * 40)
                business_total = 0
                for repo in business_repos:
                    business_total += repo['estimated_value']
                    print(f'✅ {repo["name"]} - ${repo["estimated_value"]:,}')
                print(f'\nBusiness Total: ${business_total:,}')
            
            if development_repos:
                print(f'\n🔧 DEVELOPMENT REPOSITORIES ({len(development_repos)}):')
                print('-' * 45)
                dev_total = 0
                for repo in development_repos:
                    dev_total += repo['estimated_value']
                    print(f'✅ {repo["name"]} - ${repo["estimated_value"]:,}')
                print(f'\nDevelopment Total: ${dev_total:,}')
            
            print(f'\n💰 TOTAL PORTFOLIO VALUE: ${total_estimated_value:,}')
            
            # Save analysis data
            analysis_data = {
                'ai_repos': ai_repos,
                'business_repos': business_repos, 
                'development_repos': development_repos,
                'total_value': total_estimated_value,
                'summary': {
                    'total_repos': len(repos),
                    'ai_agent_repos': len(ai_repos),
                    'business_repos': len(business_repos),
                    'development_repos': len(development_repos)
                }
            }
            
            with open('github_analysis.json', 'w') as f:
                json.dump(analysis_data, f, indent=2)
            
            print('\n✅ Analysis complete! Ready for consolidation.')
            return analysis_data
            
        else:
            print(f'❌ Failed to fetch repositories: {response.status_code}')
            return None

    except Exception as e:
        print(f'❌ Error: {str(e)}')
        return None

if __name__ == "__main__":
    analyze_github_repositories()