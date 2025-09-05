"""
Cognitive Document Intelligence Agent
Intelligent document processing and knowledge extraction
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class DocumentInsight:
    document_id: str
    content_type: str
    key_entities: List[str]
    sentiment_score: float
    confidence_level: float

class CognitiveDocumentIntelligenceAgent:
    """
    Comprehensive Cognitive Document Intelligence System
    - Intelligent document processing
    - Knowledge extraction and synthesis
    - Document classification and routing
    - Automated insights generation
    """
    
    def __init__(self):
        self.name = "Cognitive Document Intelligence Agent"
        self.version = "1.0.0"
        self.capabilities = [
            "Document Classification",
            "Entity Extraction",
            "Sentiment Analysis",
            "Knowledge Synthesis",
            "Automated Summarization",
            "Content Intelligence"
        ]
        
    def process_document_collection(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and analyze document collection"""
        try:
            company_name = document_data.get('company_name', 'Unknown Company')
            
            # Document classification and processing
            processing_results = self._process_documents(document_data.get('documents', []))
            
            # Knowledge extraction
            knowledge_extraction = self._extract_knowledge(processing_results)
            
            # Content analytics
            content_analytics = self._analyze_content_patterns(processing_results)
            
            # Generate insights
            insights = self._generate_document_insights(processing_results, knowledge_extraction)
            
            return {
                'company': company_name,
                'processing_date': datetime.now().isoformat(),
                'processing_results': processing_results,
                'knowledge_extraction': knowledge_extraction,
                'content_analytics': content_analytics,
                'insights': insights,
                'next_review_date': (datetime.now() + timedelta(days=30)).isoformat()
            }
            
        except Exception as e:
            logging.error(f"Document intelligence processing failed: {str(e)}")
            return {'error': f'Document intelligence processing failed: {str(e)}'}
            
    def _process_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process individual documents"""
        
        processed_documents = []
        classification_summary = {}
        
        for doc in documents:
            # Classify document
            doc_type = self._classify_document(doc)
            
            # Extract entities
            entities = self._extract_entities(doc)
            
            # Analyze sentiment
            sentiment = self._analyze_document_sentiment(doc)
            
            # Generate summary
            summary = self._generate_document_summary(doc)
            
            processed_doc = {
                'document_id': doc.get('id', 'unknown'),
                'title': doc.get('title', 'Untitled'),
                'document_type': doc_type,
                'entities': entities,
                'sentiment_score': sentiment,
                'summary': summary,
                'processing_confidence': self._calculate_processing_confidence(doc, entities),
                'word_count': len(doc.get('content', '').split()),
                'processing_timestamp': datetime.now().isoformat()
            }
            
            processed_documents.append(processed_doc)
            
            # Update classification summary
            classification_summary[doc_type] = classification_summary.get(doc_type, 0) + 1
            
        return {
            'processed_documents': processed_documents,
            'total_documents': len(documents),
            'classification_summary': classification_summary,
            'processing_statistics': self._calculate_processing_statistics(processed_documents)
        }
        
    def _classify_document(self, document: Dict[str, Any]) -> str:
        """Classify document type"""
        content = document.get('content', '').lower()
        title = document.get('title', '').lower()
        
        # Simple keyword-based classification
        if any(keyword in content + title for keyword in ['contract', 'agreement', 'terms']):
            return 'Legal Document'
        elif any(keyword in content + title for keyword in ['financial', 'budget', 'revenue', 'profit']):
            return 'Financial Document'
        elif any(keyword in content + title for keyword in ['technical', 'specification', 'architecture']):
            return 'Technical Document'
        elif any(keyword in content + title for keyword in ['policy', 'procedure', 'guideline']):
            return 'Policy Document'
        elif any(keyword in content + title for keyword in ['report', 'analysis', 'summary']):
            return 'Report'
        elif any(keyword in content + title for keyword in ['email', 'message', 'communication']):
            return 'Communication'
        else:
            return 'General Document'
            
    def _extract_entities(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract entities from document"""
        content = document.get('content', '')
        
        # Simplified entity extraction (in production, would use NLP libraries)
        entities = []
        
        # Extract organizations (simplified)
        org_keywords = ['corp', 'inc', 'llc', 'ltd', 'company', 'corporation']
        words = content.split()
        for i, word in enumerate(words):
            if any(keyword in word.lower() for keyword in org_keywords):
                if i > 0:
                    entities.append({
                        'type': 'Organization',
                        'text': f"{words[i-1]} {word}",
                        'confidence': 0.8
                    })
                    
        # Extract dates (simplified pattern)
        import re
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        dates = re.findall(date_pattern, content)
        for date in dates:
            entities.append({
                'type': 'Date',
                'text': date,
                'confidence': 0.9
            })
            
        # Extract monetary amounts
        money_pattern = r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?'
        amounts = re.findall(money_pattern, content)
        for amount in amounts:
            entities.append({
                'type': 'Money',
                'text': amount,
                'confidence': 0.95
            })
            
        return entities
        
    def _analyze_document_sentiment(self, document: Dict[str, Any]) -> float:
        """Analyze document sentiment"""
        content = document.get('content', '').lower()
        
        # Simple sentiment analysis using word lists
        positive_words = ['good', 'excellent', 'great', 'positive', 'success', 'achieve', 'benefit', 'improve', 'outstanding']
        negative_words = ['bad', 'poor', 'negative', 'fail', 'problem', 'issue', 'concern', 'decline', 'loss', 'risk']
        
        positive_count = sum(1 for word in positive_words if word in content)
        negative_count = sum(1 for word in negative_words if word in content)
        
        total_words = len(content.split())
        
        if total_words == 0:
            return 0.5  # Neutral
            
        # Calculate sentiment score (0 = very negative, 1 = very positive, 0.5 = neutral)
        sentiment_score = 0.5 + (positive_count - negative_count) / (total_words * 0.1)
        
        return max(0, min(1, sentiment_score))
        
    def _generate_document_summary(self, document: Dict[str, Any]) -> str:
        """Generate document summary"""
        content = document.get('content', '')
        
        # Simple extractive summarization (first few sentences)
        sentences = content.split('. ')
        
        if len(sentences) <= 3:
            return content
        else:
            # Return first 2 sentences as summary
            return '. '.join(sentences[:2]) + '.'
            
    def _calculate_processing_confidence(self, document: Dict[str, Any], entities: List[Dict[str, Any]]) -> float:
        """Calculate processing confidence"""
        content = document.get('content', '')
        
        confidence_factors = {
            'content_length': min(1.0, len(content) / 1000),  # Longer content = higher confidence
            'entity_count': min(1.0, len(entities) / 10),     # More entities = higher confidence
            'title_availability': 1.0 if document.get('title') else 0.5
        }
        
        return sum(confidence_factors.values()) / len(confidence_factors)
        
    def _calculate_processing_statistics(self, processed_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate processing statistics"""
        
        if not processed_docs:
            return {}
            
        total_entities = sum(len(doc['entities']) for doc in processed_docs)
        avg_sentiment = sum(doc['sentiment_score'] for doc in processed_docs) / len(processed_docs)
        avg_confidence = sum(doc['processing_confidence'] for doc in processed_docs) / len(processed_docs)
        total_words = sum(doc['word_count'] for doc in processed_docs)
        
        return {
            'total_entities_extracted': total_entities,
            'average_sentiment_score': avg_sentiment,
            'average_processing_confidence': avg_confidence,
            'total_word_count': total_words,
            'sentiment_distribution': self._categorize_sentiment_distribution(processed_docs)
        }
        
    def _categorize_sentiment_distribution(self, processed_docs: List[Dict[str, Any]]) -> Dict[str, int]:
        """Categorize sentiment distribution"""
        distribution = {'positive': 0, 'neutral': 0, 'negative': 0}
        
        for doc in processed_docs:
            sentiment = doc['sentiment_score']
            if sentiment > 0.6:
                distribution['positive'] += 1
            elif sentiment < 0.4:
                distribution['negative'] += 1
            else:
                distribution['neutral'] += 1
                
        return distribution
        
    def _extract_knowledge(self, processing_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract knowledge from processed documents"""
        
        processed_docs = processing_results.get('processed_documents', [])
        
        # Aggregate entities across documents
        all_entities = []
        for doc in processed_docs:
            all_entities.extend(doc['entities'])
            
        # Group entities by type
        entities_by_type = {}
        for entity in all_entities:
            entity_type = entity['type']
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity['text'])
            
        # Find key themes
        key_themes = self._identify_key_themes(processed_docs)
        
        # Extract relationships
        relationships = self._extract_relationships(processed_docs)
        
        return {
            'entities_by_type': entities_by_type,
            'key_themes': key_themes,
            'relationships': relationships,
            'knowledge_confidence': self._calculate_knowledge_confidence(all_entities, key_themes),
            'knowledge_coverage': self._assess_knowledge_coverage(processing_results)
        }
        
    def _identify_key_themes(self, processed_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify key themes across documents"""
        
        # Simple theme identification based on document types and content
        theme_keywords = {
            'Financial Performance': ['revenue', 'profit', 'financial', 'budget', 'cost'],
            'Risk Management': ['risk', 'compliance', 'security', 'audit', 'control'],
            'Operations': ['process', 'efficiency', 'operations', 'production', 'quality'],
            'Technology': ['technology', 'system', 'software', 'digital', 'automation'],
            'Human Resources': ['employee', 'talent', 'training', 'performance', 'culture'],
            'Strategy': ['strategy', 'growth', 'market', 'competitive', 'vision']
        }
        
        themes = []
        for theme_name, keywords in theme_keywords.items():
            relevance_score = 0
            document_count = 0
            
            for doc in processed_docs:
                content = doc.get('summary', '').lower()
                keyword_matches = sum(1 for keyword in keywords if keyword in content)
                
                if keyword_matches > 0:
                    relevance_score += keyword_matches / len(keywords)
                    document_count += 1
                    
            if document_count > 0:
                avg_relevance = relevance_score / document_count
                themes.append({
                    'theme': theme_name,
                    'relevance_score': avg_relevance,
                    'document_count': document_count,
                    'importance': 'High' if avg_relevance > 0.3 and document_count > 2 else 'Medium' if avg_relevance > 0.2 else 'Low'
                })
                
        # Sort by relevance
        themes.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return themes
        
    def _extract_relationships(self, processed_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships between entities and concepts"""
        
        relationships = []
        
        # Simple relationship extraction based on co-occurrence
        for doc in processed_docs:
            entities = doc['entities']
            
            # Find relationships between entities in the same document
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    if entity1['type'] != entity2['type']:  # Different types suggest relationship
                        relationships.append({
                            'entity1': entity1['text'],
                            'entity1_type': entity1['type'],
                            'entity2': entity2['text'],
                            'entity2_type': entity2['type'],
                            'relationship_type': 'co-occurrence',
                            'document_id': doc['document_id'],
                            'confidence': min(entity1['confidence'], entity2['confidence'])
                        })
                        
        return relationships[:20]  # Return top 20 relationships
        
    def _calculate_knowledge_confidence(self, entities: List[Dict[str, Any]], themes: List[Dict[str, Any]]) -> float:
        """Calculate confidence in extracted knowledge"""
        
        if not entities and not themes:
            return 0.0
            
        entity_confidence = sum(entity['confidence'] for entity in entities) / len(entities) if entities else 0
        theme_confidence = sum(theme['relevance_score'] for theme in themes) / len(themes) if themes else 0
        
        return (entity_confidence + theme_confidence) / 2
        
    def _assess_knowledge_coverage(self, processing_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess knowledge coverage across document types"""
        
        classification_summary = processing_results.get('classification_summary', {})
        total_docs = processing_results.get('total_documents', 0)
        
        coverage_assessment = {}
        for doc_type, count in classification_summary.items():
            coverage_percentage = (count / total_docs * 100) if total_docs > 0 else 0
            coverage_assessment[doc_type] = {
                'document_count': count,
                'coverage_percentage': coverage_percentage,
                'adequacy': 'Good' if coverage_percentage > 20 else 'Fair' if coverage_percentage > 10 else 'Poor'
            }
            
        return coverage_assessment
        
    def _analyze_content_patterns(self, processing_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze content patterns across documents"""
        
        processed_docs = processing_results.get('processed_documents', [])
        
        # Analyze document length patterns
        word_counts = [doc['word_count'] for doc in processed_docs]
        avg_length = sum(word_counts) / len(word_counts) if word_counts else 0
        
        # Analyze processing quality
        confidence_scores = [doc['processing_confidence'] for doc in processed_docs]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Analyze content complexity
        complexity_scores = [self._assess_content_complexity(doc) for doc in processed_docs]
        avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0
        
        return {
            'average_document_length': avg_length,
            'length_distribution': self._categorize_length_distribution(word_counts),
            'average_processing_confidence': avg_confidence,
            'average_content_complexity': avg_complexity,
            'quality_assessment': 'High' if avg_confidence > 0.8 else 'Medium' if avg_confidence > 0.6 else 'Low',
            'complexity_assessment': 'High' if avg_complexity > 0.7 else 'Medium' if avg_complexity > 0.4 else 'Low'
        }
        
    def _assess_content_complexity(self, document: Dict[str, Any]) -> float:
        """Assess content complexity"""
        
        factors = {
            'entity_density': min(1.0, len(document['entities']) / max(1, document['word_count'] / 100)),
            'document_type_complexity': self._get_type_complexity(document['document_type']),
            'sentiment_neutrality': 1.0 - abs(document['sentiment_score'] - 0.5) * 2  # More neutral = more complex
        }
        
        return sum(factors.values()) / len(factors)
        
    def _get_type_complexity(self, doc_type: str) -> float:
        """Get complexity score for document type"""
        complexity_map = {
            'Legal Document': 0.9,
            'Technical Document': 0.8,
            'Financial Document': 0.7,
            'Policy Document': 0.6,
            'Report': 0.5,
            'Communication': 0.3,
            'General Document': 0.4
        }
        
        return complexity_map.get(doc_type, 0.5)
        
    def _categorize_length_distribution(self, word_counts: List[int]) -> Dict[str, int]:
        """Categorize document length distribution"""
        distribution = {'short': 0, 'medium': 0, 'long': 0}
        
        for count in word_counts:
            if count < 500:
                distribution['short'] += 1
            elif count < 2000:
                distribution['medium'] += 1
            else:
                distribution['long'] += 1
                
        return distribution
        
    def _generate_document_insights(self, processing_results: Dict[str, Any], 
                                  knowledge_extraction: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable insights from document analysis"""
        
        insights = []
        
        # Content quality insights
        avg_confidence = processing_results.get('processing_statistics', {}).get('average_processing_confidence', 0)
        if avg_confidence < 0.7:
            insights.append({
                'category': 'Content Quality',
                'insight': 'Document processing confidence is below optimal levels',
                'recommendation': 'Consider improving document quality and standardizing formats',
                'priority': 'Medium',
                'impact': 'Improved accuracy in document analysis and knowledge extraction'
            })
            
        # Knowledge coverage insights
        themes = knowledge_extraction.get('key_themes', [])
        high_importance_themes = [theme for theme in themes if theme.get('importance') == 'High']
        
        if len(high_importance_themes) >= 3:
            insights.append({
                'category': 'Strategic Insights',
                'insight': f'Identified {len(high_importance_themes)} high-importance business themes',
                'recommendation': 'Focus strategic initiatives on identified key themes',
                'priority': 'High',
                'impact': 'Aligned strategic focus based on document intelligence'
            })
            
        # Sentiment insights
        sentiment_dist = processing_results.get('processing_statistics', {}).get('sentiment_distribution', {})
        negative_docs = sentiment_dist.get('negative', 0)
        total_docs = processing_results.get('total_documents', 1)
        
        if negative_docs / total_docs > 0.3:  # More than 30% negative sentiment
            insights.append({
                'category': 'Sentiment Analysis',
                'insight': 'High proportion of documents show negative sentiment',
                'recommendation': 'Investigate potential issues or concerns reflected in documents',
                'priority': 'Medium',
                'impact': 'Proactive issue identification and resolution'
            })
            
        # Entity relationship insights
        relationships = knowledge_extraction.get('relationships', [])
        if len(relationships) > 10:
            insights.append({
                'category': 'Knowledge Connections',
                'insight': 'Rich network of entity relationships identified',
                'recommendation': 'Leverage relationship insights for business intelligence',
                'priority': 'Low',
                'impact': 'Enhanced understanding of business connections and dependencies'
            })
            
        return insights

def test_cognitive_document_intelligence_agent():
    """Test the Cognitive Document Intelligence Agent"""
    print("🧪 Testing Cognitive Document Intelligence Agent")
    print("=" * 55)
    
    try:
        agent = CognitiveDocumentIntelligenceAgent()
        
        test_data = {
            'company_name': 'Knowledge Systems Corp',
            'documents': [
                {
                    'id': 'DOC001',
                    'title': 'Q3 Financial Report',
                    'content': 'Our company achieved excellent revenue growth of $2.5 million this quarter. The financial performance exceeded expectations with a 15% increase compared to last year.'
                },
                {
                    'id': 'DOC002', 
                    'title': 'IT Security Policy',
                    'content': 'This policy document outlines security procedures and compliance requirements. All employees must follow these guidelines to ensure data protection and risk management.'
                },
                {
                    'id': 'DOC003',
                    'title': 'Customer Feedback Analysis',
                    'content': 'Customer satisfaction has declined due to service issues. Several complaints were received regarding poor response times and quality problems.'
                }
            ]
        }
        
        analysis = agent.process_document_collection(test_data)
        print(f"✅ Document processing completed for {test_data['company_name']}")
        print(f"   Documents processed: {analysis['processing_results']['total_documents']}")
        print(f"   Entities extracted: {analysis['processing_results']['processing_statistics']['total_entities_extracted']}")
        print(f"   Key themes identified: {len(analysis['knowledge_extraction']['key_themes'])}")
        print(f"   Insights generated: {len(analysis['insights'])}")
        
        return {
            'agent_initialized': True,
            'documents_processed': analysis['processing_results']['total_documents'],
            'entities_extracted': analysis['processing_results']['processing_statistics']['total_entities_extracted'],
            'insights_generated': len(analysis['insights'])
        }
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return {'error': str(e)}

if __name__ == "__main__":
    test_cognitive_document_intelligence_agent()