"""
Automatic PDF Generation Service for Future Content Requests

This service automatically applies Gilbert's AI agents and generates PDF files 
for all future text generation requests including eBooks, lead magnets, 
articles, and blog posts.
"""

import os
from typing import Dict, Optional, Tuple
from services.sympathetic_writing_agent import create_sympathetic_writing_agent
from services.professional_thought_leader_agent import create_professional_thought_leader_agent
from services.pdf_generation_service import create_pdf_service


class AutomaticPDFGenerator:
    """Automatic PDF generation with AI agent enhancement for all content requests"""
    
    def __init__(self):
        self.sympathetic_agent = create_sympathetic_writing_agent()
        self.professional_agent = create_professional_thought_leader_agent()
        self.pdf_service = create_pdf_service()
        
        # Content type mappings for agent selection
        self.content_type_mapping = {
            # Personal/Emotional Content -> Sympathetic Agent
            "personal_development": "sympathetic",
            "leadership_journey": "sympathetic", 
            "emotional_intelligence": "sympathetic",
            "relationship_building": "sympathetic",
            "vulnerability_in_leadership": "sympathetic",
            "personal_growth": "sympathetic",
            "life_lessons": "sympathetic",
            "cultural_adaptation": "sympathetic",
            "spiritual_business": "sympathetic",
            
            # Business/Executive Content -> Professional Agent
            "strategic_planning": "professional",
            "business_strategy": "professional",
            "enterprise_architecture": "professional",
            "thought_leadership": "professional",
            "executive_communication": "professional",
            "market_analysis": "professional",
            "competitive_intelligence": "professional",
            "financial_planning": "professional",
            "mergers_acquisitions": "professional",
            "tax_optimization": "professional",
            "technology_strategy": "professional",
            "digital_transformation": "professional",
            "analytics_implementation": "professional",
            "cloud_architecture": "professional",
            "ai_strategy": "professional",
            "innovation_management": "professional",
            "change_management": "professional",
            "organizational_development": "professional",
            "performance_optimization": "professional"
        }
    
    def determine_agent_type(self, content: str, content_type: str = None, 
                           target_audience: str = None) -> str:
        """Determine which AI agent to use based on content analysis"""
        
        # Check explicit content type mapping first
        if content_type and content_type.lower() in self.content_type_mapping:
            return self.content_type_mapping[content_type.lower()]
        
        # Analyze target audience
        if target_audience:
            audience_lower = target_audience.lower()
            if any(term in audience_lower for term in ["c-level", "executive", "ceo", "cfo", "board", "enterprise", "fortune 500"]):
                return "professional"
            elif any(term in audience_lower for term in ["personal", "individual", "coaching", "development", "growth"]):
                return "sympathetic"
        
        # Analyze content for keywords
        content_lower = content.lower()
        
        # Professional indicators
        professional_keywords = [
            "strategy", "framework", "methodology", "roi", "revenue", "profit",
            "enterprise", "fortune", "competitive", "market", "analytics", 
            "implementation", "execution", "performance", "optimization",
            "technology", "digital", "cloud", "ai", "data", "insights",
            "transformation", "innovation", "scaling", "growth", "acquisition"
        ]
        
        # Sympathetic indicators  
        sympathetic_keywords = [
            "journey", "story", "experience", "learn", "struggle", "failure",
            "vulnerable", "authentic", "personal", "emotional", "relationship",
            "trust", "connection", "human", "heart", "soul", "spiritual",
            "cultural", "family", "challenge", "overcome", "growth", "wisdom"
        ]
        
        professional_score = sum(1 for keyword in professional_keywords if keyword in content_lower)
        sympathetic_score = sum(1 for keyword in sympathetic_keywords if keyword in content_lower)
        
        # Default to professional for business content, sympathetic for personal
        if professional_score > sympathetic_score:
            return "professional"
        elif sympathetic_score > professional_score:
            return "sympathetic"
        else:
            # Default based on content length and structure
            if len(content) > 5000 and ("chapter" in content_lower or "framework" in content_lower):
                return "professional"
            else:
                return "sympathetic"
    
    def generate_enhanced_content_pdf(self, content: str, title: str, 
                                    content_type: str = None, 
                                    target_audience: str = None,
                                    custom_filename: str = None) -> Tuple[bool, str, str]:
        """
        Generate enhanced content PDF with automatic agent selection
        
        Returns:
            (success, agent_used, github_url)
        """
        
        try:
            # Determine which agent to use
            agent_type = self.determine_agent_type(content, content_type, target_audience)
            
            print(f"🤖 Using {agent_type.title()} AI Agent for content enhancement")
            
            # Apply appropriate AI agent
            if agent_type == "sympathetic":
                enhanced_content = self.sympathetic_agent.transform_to_sympathetic_style(content)
                agent_label = "sympathetic-enhanced"
            else:
                enhanced_content = self.professional_agent.transform_to_professional_style(content, "comprehensive")
                agent_label = "professional-enhanced"
            
            # Generate filename
            if custom_filename:
                base_filename = custom_filename.replace('.pdf', '').replace(' ', '-').lower()
            else:
                base_filename = title.replace(' ', '-').replace(':', '').replace(',', '').lower()
            
            pdf_filename = f"enhanced_content/{base_filename}-{agent_label}.pdf"
            
            # Generate PDF
            pdf_content = self.pdf_service.generate_pdf(enhanced_content, pdf_filename)
            
            if not pdf_content:
                return False, agent_type, ""
            
            # Upload to GitHub
            commit_message = f"Add {agent_type} enhanced content: {title}"
            success = self.pdf_service.upload_pdf_to_github(pdf_content, pdf_filename, commit_message)
            
            if success:
                github_url = f"https://github.com/{self.pdf_service.github_username}/{self.pdf_service.repo_name}/blob/main/{pdf_filename}"
                return True, agent_type, github_url
            else:
                return False, agent_type, ""
                
        except Exception as e:
            print(f"❌ Error generating enhanced PDF: {e}")
            return False, "unknown", ""
    
    def process_content_request(self, content: str, request_type: str, 
                              title: str = None, **kwargs) -> Dict:
        """
        Process any content request and automatically generate enhanced PDF
        
        Args:
            content: The content to enhance and convert
            request_type: Type of request (ebook, lead_magnet, article, blog_post, etc.)
            title: Title for the content
            **kwargs: Additional parameters (target_audience, content_type, etc.)
        """
        
        if not title:
            # Extract title from content
            lines = content.split('\n')
            for line in lines:
                if line.startswith('# '):
                    title = line[2:].strip()
                    break
            if not title:
                title = f"Enhanced {request_type.replace('_', ' ').title()}"
        
        print(f"📝 Processing {request_type.replace('_', ' ').title()}: {title}")
        print("=" * 60)
        
        # Generate enhanced PDF
        success, agent_used, github_url = self.generate_enhanced_content_pdf(
            content=content,
            title=title,
            content_type=kwargs.get('content_type'),
            target_audience=kwargs.get('target_audience'),
            custom_filename=kwargs.get('filename')
        )
        
        results = {
            "success": success,
            "title": title,
            "request_type": request_type,
            "agent_used": agent_used,
            "github_url": github_url,
            "enhancement_applied": True if success else False
        }
        
        if success:
            print(f"✅ {request_type.replace('_', ' ').title()} enhanced with {agent_used.title()} AI Agent")
            print(f"✅ PDF generated and uploaded to GitHub")
            print(f"📁 GitHub URL: {github_url}")
        else:
            print(f"❌ Failed to process {request_type}")
        
        return results


# Factory function for easy instantiation
def create_automatic_pdf_generator() -> AutomaticPDFGenerator:
    """Create and return automatic PDF generator instance"""
    return AutomaticPDFGenerator()


# Convenience functions for different content types
def generate_ebook_pdf(content: str, title: str = None, **kwargs) -> Dict:
    """Generate enhanced eBook PDF"""
    generator = create_automatic_pdf_generator()
    return generator.process_content_request(content, "ebook", title, **kwargs)


def generate_lead_magnet_pdf(content: str, title: str = None, **kwargs) -> Dict:
    """Generate enhanced lead magnet PDF"""
    generator = create_automatic_pdf_generator()
    return generator.process_content_request(content, "lead_magnet", title, **kwargs)


def generate_article_pdf(content: str, title: str = None, **kwargs) -> Dict:
    """Generate enhanced article PDF"""
    generator = create_automatic_pdf_generator()
    return generator.process_content_request(content, "article", title, **kwargs)


def generate_blog_post_pdf(content: str, title: str = None, **kwargs) -> Dict:
    """Generate enhanced blog post PDF"""
    generator = create_automatic_pdf_generator()
    return generator.process_content_request(content, "blog_post", title, **kwargs)


if __name__ == "__main__":
    # Test automatic PDF generation
    generator = create_automatic_pdf_generator()
    
    test_content = """
# The Future of Enterprise AI Strategy
## A Comprehensive Guide for C-Level Executives

In today's rapidly evolving business landscape, artificial intelligence represents both the greatest opportunity and the most significant challenge facing enterprise leaders. Organizations that successfully implement AI strategies will gain sustainable competitive advantages, while those that fail to adapt risk obsolescence.

This guide provides a strategic framework for enterprise AI implementation, drawing from extensive Fortune 500 experience and proven methodologies.

## Strategic AI Implementation Framework

Success in AI implementation requires a systematic approach that addresses technology, people, processes, and organizational culture. The framework consists of four key phases:

1. **Strategic Assessment and Planning**
2. **Technology Infrastructure Development** 
3. **Pilot Implementation and Testing**
4. **Scaling and Optimization**

Each phase requires careful planning, stakeholder alignment, and measurable outcomes to ensure successful transformation.
"""
    
    print("🧪 Testing Automatic PDF Generation")
    print("=" * 50)
    
    result = generator.process_content_request(
        content=test_content,
        request_type="thought_leadership_article",
        title="The Future of Enterprise AI Strategy",
        content_type="ai_strategy",
        target_audience="C-level executives"
    )
    
    print("\n📊 Test Results:")
    print(f"Success: {result['success']}")
    print(f"Agent Used: {result['agent_used']}")
    print(f"Enhancement Applied: {result['enhancement_applied']}")
    
    if result['success']:
        print("✅ Automatic PDF generation system working correctly!")
    else:
        print("❌ System needs adjustment")