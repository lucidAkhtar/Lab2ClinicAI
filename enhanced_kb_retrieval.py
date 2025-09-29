"""
Enhanced Knowledge Base Retrieval with Smart Content Extraction
Provides complete, meaningful answers instead of fragmented chunks
"""

import re
import os
import logging
from typing import List, Dict, Optional, Tuple
# Import improved KB system instead of basic one
from improved_kb_retrieval import ImprovedKBRetrieval

logger = logging.getLogger("enhanced_kb")
logger.setLevel(logging.INFO)

class EnhancedKBRetrieval:
    def __init__(self, kb_dir: str = "kb"):
        # Use the improved KB system as the backend
        self.kb_index = ImprovedKBRetrieval(kb_dir)
        self.kb_built = False
        
        # Enhanced configuration for better quality
        self.kb_index.config.update({
            "max_features": 4096,        # Large vocabulary for technical terms
            "chunk_size": 2000,          # Larger chunks for more context
            "chunk_overlap": 400,        # Substantial overlap for continuity
            "top_k_base": 5,            # Return more results by default
            "min_similarity": 0.02,      # Lower threshold for more results
            "max_docs": 15,             # Process ALL documents
            "max_chunks": 500,          # Allow many more chunks for all files
            "ngram_range": (1, 2),      # Include bigrams for better matching
        })
        
        # Enhanced content patterns for better extraction
        self.qa_patterns = [
            r'\*\*Q\d+:\s*([^*]+)\*\*\s*\n\s*A:\s*([^*\n]+(?:\n[^*\n]+)*)',  # FAQ Q&A format
            r'#{1,3}\s*([^#\n]+)\n([^#]+?)(?=\n#{1,3}|\n\*\*Q|\Z)',  # Markdown headers
            r'- ([^-\n]+) — ([^-\n]+(?:\n[^-\n]+)*)',  # Glossary format
        ]
        
        # Disabled curated answers to prioritize comprehensive_qa.md content
        self.curated_answers = {}
    
    def build(self) -> bool:
        """Build the knowledge base index with improved configuration"""
        logger.info("Building enhanced KB with improved TF-IDF configuration...")
        self.kb_built = self.kb_index.build()
        if self.kb_built:
            logger.info(f"Enhanced KB built successfully with {len(self.kb_index.documents)} chunks")
            logger.info(f"TF-IDF matrix shape: {self.kb_index.tfidf_matrix.shape}")
        return self.kb_built
    
    def _extract_relevant_content(self, text: str, query: str) -> List[str]:
        """Extract relevant content pieces from text based on query"""
        relevant_pieces = []
        
        # Check for Q&A patterns
        for pattern in self.qa_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                if pattern.startswith(r'\*\*Q'):  # FAQ format
                    question, answer = match.groups()
                    if any(term.lower() in question.lower() or term.lower() in answer.lower() 
                          for term in query.split()):
                        relevant_pieces.append(f"**Q: {question.strip()}**\nA: {answer.strip()}")
                elif pattern.startswith(r'#{1,3}'):  # Header format
                    header, content = match.groups()
                    if any(term.lower() in header.lower() or term.lower() in content.lower() 
                          for term in query.split()):
                        relevant_pieces.append(f"**{header.strip()}**\n{content.strip()}")
                else:  # Glossary format
                    term, definition = match.groups()
                    if any(query_term.lower() in term.lower() or query_term.lower() in definition.lower() 
                          for query_term in query.split()):
                        relevant_pieces.append(f"**{term.strip()}**: {definition.strip()}")
        
        return relevant_pieces
    
    def _get_curated_answer(self, query: str) -> Optional[Dict]:
        """Get curated answer for common queries"""
        query_lower = query.lower()
        
        # Check for exact matches or key terms
        for key, answer_data in self.curated_answers.items():
            if (key in query_lower or 
                any(term in query_lower for term in [key, f"what is {key}", f"define {key}"])):
                return {
                    "text": answer_data["definition"],
                    "source": answer_data["source"],
                    "score": 0.95,  # High confidence for curated answers
                    "details": answer_data.get("details", ""),
                    "type": "curated"
                }
        
        return None
    
    def query(self, query: str, topk: int = 3) -> List[Dict]:
        """
        Enhanced query with smart content extraction and comprehensive_qa.md prioritization
        Returns complete, meaningful answers with comprehensive content first
        """
        if not self.kb_built:
            logger.warning("KB not built, building now...")
            if not self.build():
                return []
        
        results = []
        
        # First, check for curated answers
        curated = self._get_curated_answer(query)
        if curated:
            results.append(curated)
            logger.info("Found curated answer for query: %s", query)
        
        # Get improved KB results (more comprehensive)
        original_results = self.kb_index.query(query, topk=topk*3)  # Get more to filter better
        
        # Prioritize comprehensive_qa.md results
        comprehensive_results = [r for r in original_results if 'comprehensive_qa.md' in r.get('source', '')]
        other_results = [r for r in original_results if 'comprehensive_qa.md' not in r.get('source', '')]
        
        # Boost scores for comprehensive_qa results and reorder
        for result in comprehensive_results:
            original_score = result['score']
            
            # Check if this result contains the exact terms from the query
            text_lower = result.get('text', '').lower()
            query_lower = query.lower()
            
            # Special boost for LogP and TPSA queries
            if 'logp' in query_lower and 'tpsa' in query_lower:
                if 'logp' in text_lower and 'tpsa' in text_lower:
                    result['score'] = original_score * 5.0  # 400% boost for exact match
                else:
                    result['score'] = original_score * 2.0  # Higher standard boost
            else:
                result['score'] = original_score * 2.0  # Higher standard boost
            
            result['priority_source'] = True
        
        # Combine with comprehensive results first
        prioritized_results = comprehensive_results + other_results
        logger.info(f"Prioritized {len(comprehensive_results)} comprehensive_qa results, {len(other_results)} other results")
        
        # Process and enhance results
        processed_results = []
        seen_content = set()
        
        for result in prioritized_results:
            # Extract relevant content from the full text
            relevant_content = self._extract_relevant_content(result["text"], query)
            
            if relevant_content:
                for content in relevant_content:
                    # Avoid duplicates
                    content_hash = hash(content[:100])
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        processed_results.append({
                            "text": content,
                            "source": result["source"],
                            "score": result["score"],
                            "type": "extracted"
                        })
            else:
                # If no specific extraction, use original but clean it up
                text = result["text"].strip()
                # Try to find complete sentences
                sentences = re.split(r'[.!?]+', text)
                relevant_sentences = [s.strip() for s in sentences 
                                    if any(term.lower() in s.lower() for term in query.split())]
                
                if relevant_sentences:
                    clean_text = '. '.join(relevant_sentences[:3]) + '.'  # Take more sentences
                else:
                    clean_text = text[:500] + "..." if len(text) > 500 else text  # Longer snippets
                
                processed_results.append({
                    "text": clean_text,
                    "source": result["source"],
                    "score": result["score"] * 0.9,  # Slight penalty for non-extracted content
                    "type": "original"
                })
        
        # Combine curated and processed results
        if curated:
            all_results = [curated] + processed_results
        else:
            all_results = processed_results
        
        # Sort by score and return top results
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return more results for better coverage
        final_results = all_results[:topk]
        logger.info(f"Enhanced query '{query}' returned {len(final_results)} results")
        
        return final_results

# Test function
def test_enhanced_kb():
    """Test the enhanced KB with problematic queries"""
    enhanced_kb = EnhancedKBRetrieval()
    enhanced_kb.build()
    
    test_queries = [
        "what is TPSA",
        "explain LogP toxicity", 
        "what is blood brain barrier",
        "define toxicophore",
        "solubility prediction"
    ]
    
    print("=== Enhanced KB Test Results ===")
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 50)
        
        results = enhanced_kb.query(query, topk=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n📝 Result {i} ({result.get('type', 'unknown')}):")
            print(f"Source: {result['source']}")
            print(f"Score: {result['score']:.3f}")
            print(f"Content: {result['text'][:400]}...")
            if 'details' in result:
                print(f"Details: {result['details']}")

if __name__ == "__main__":
    test_enhanced_kb()
