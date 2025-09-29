"""
Enhanced Knowledge Base Configuration for Better Answer Quality
Addresses limitations in current TF-IDF setup to provide more comprehensive answers
"""

import os
import pickle
import re
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

logger = logging.getLogger("enhanced_kb")

class ImprovedKBRetrieval:
    """Enhanced KB retrieval with better TF-IDF configuration"""
    
    def __init__(self, kb_dir: str = "kb"):
        self.kb_dir = kb_dir
        self.vectorizer = None
        self.tfidf_matrix = None
        self.documents = []
        self.metadata = []
        self.built = False
        
        # Enhanced configuration for better retrieval
        self.config = {
            "chunk_size": 1500,          # Larger chunks for more context
            "chunk_overlap": 300,        # More overlap for continuity
            "max_features": 2048,        # Much larger vocabulary
            "ngram_range": (1, 2),       # Include bigrams for better matching
            "max_df": 0.85,             # Ignore very common terms
            "min_df": 1,                # Keep rare terms (small corpus)
            "max_docs": 15,             # Process ALL documents
            "max_chunks": 500,          # Allow many more chunks for all files
            "top_k_base": 5,            # Return more results by default
            "min_similarity": 0.05,     # Lower threshold for more results
            "sublinear_tf": True,       # Better handling of term frequencies
        }
    
    def _chunk_document(self, text: str, filename: str) -> List[Dict]:
        """Create chunks from document text with special handling for comprehensive_qa.md"""
        chunks = []
        chunk_size = self.config["chunk_size"]
        overlap = self.config["chunk_overlap"]
        
        # Clean text
        text = text.strip()
        if not text:
            return []
        
        # Special handling for comprehensive_qa.md to preserve complete Q&A sections
        if filename == 'comprehensive_qa.md':
            return self._chunk_qa_document(text, filename)
        
        if len(text) <= chunk_size:
            return [{
                "text": text,
                "source": filename,
                "chunk_id": 0,
                "start": 0,
                "end": len(text)
            }]
        
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings within last 200 chars
                search_start = max(start + chunk_size - 200, start)
                sentence_end = -1
                for i in range(end - 1, search_start - 1, -1):
                    if text[i] in '.!?':
                        # Make sure it's not an abbreviation
                        if i + 1 < len(text) and text[i + 1] in ' \n':
                            sentence_end = i + 1
                            break
                
                if sentence_end > search_start:
                    end = sentence_end
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "source": filename,
                    "chunk_id": chunk_id,
                    "start": start,
                    "end": end
                })
                chunk_id += 1
            
            # Move start position
            start = max(start + 1, end - overlap)
            if start >= len(text):
                break
        
        return chunks
    
    def _chunk_qa_document(self, text: str, filename: str) -> List[Dict]:
        """Special chunking for comprehensive Q&A to preserve complete sections"""
        chunks = []
        
        # Split by major sections (## headers)
        sections = re.split(r'\n## ', text)
        
        for i, section in enumerate(sections):
            if not section.strip():
                continue
                
            # Add back the ## if it was split off
            if i > 0:
                section = "## " + section
            
            # Keep complete sections together (even if long)
            chunks.append({
                "text": section.strip(),
                "source": filename,
                "chunk_id": len(chunks),
                "start": 0,
                "end": len(section)
            })
        
        return chunks
    
    def _load_documents(self) -> bool:
        """Load all documents from KB directory"""
        if not os.path.exists(self.kb_dir):
            logger.error(f"KB directory not found: {self.kb_dir}")
            return False
        
        files = [f for f in os.listdir(self.kb_dir) 
                if f.lower().endswith(('.md', '.txt'))]
        
        # Prioritize comprehensive_qa.md and core files
        priority_files = ['comprehensive_qa.md', 'faq_master.md', 'glossary_pharma.md', 'admet_interpretation.md']
        priority_found = [f for f in files if f in priority_files]
        other_files = [f for f in files if f not in priority_files]
        
        # Process priority files first with more chunks allocated
        files = priority_found + other_files
        files = files[:self.config["max_docs"]]
        logger.info(f"Processing {len(files)} documents from {self.kb_dir} (priority: {len(priority_found)})")
        
        all_chunks = []
        total_chunks = 0
        
        for filename in files:
            filepath = os.path.join(self.kb_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                chunks = self._chunk_document(content, filename)
                
                # Give comprehensive_qa.md priority for chunk allocation
                if filename == 'comprehensive_qa.md':
                    # Reserve more chunks for comprehensive content
                    file_chunk_limit = min(len(chunks), 250)  # Up to 250 chunks for comprehensive_qa
                else:
                    # Limit other files more aggressively
                    remaining_capacity = self.config["max_chunks"] - total_chunks
                    file_chunk_limit = min(len(chunks), remaining_capacity, 50)  # Max 50 chunks per other file
                
                if file_chunk_limit > 0:
                    chunks = chunks[:file_chunk_limit]
                    all_chunks.extend(chunks)
                    total_chunks += len(chunks)
                    logger.info(f"Loaded {len(chunks)} chunks from {filename} (total: {total_chunks})")
                
                if total_chunks >= self.config["max_chunks"]:
                    logger.info(f"Reached max chunks limit ({self.config['max_chunks']})")
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to load {filename}: {e}")
        
        self.documents = [chunk["text"] for chunk in all_chunks]
        self.metadata = all_chunks
        
        logger.info(f"Total loaded: {len(self.documents)} chunks from {len(files)} files")
        return len(self.documents) > 0
    
    def _build_index(self) -> bool:
        """Build TF-IDF index with enhanced configuration"""
        if not self.documents:
            logger.error("No documents to index")
            return False
        
        try:
            logger.info(f"Building TF-IDF index with {len(self.documents)} documents")
            
            # Enhanced TF-IDF configuration
            self.vectorizer = TfidfVectorizer(
                max_features=self.config["max_features"],
                ngram_range=self.config["ngram_range"],
                max_df=self.config["max_df"],
                min_df=self.config["min_df"],
                stop_words='english',
                sublinear_tf=self.config["sublinear_tf"],
                token_pattern=r'\b[a-zA-Z][a-zA-Z0-9_]*\b',  # Include technical terms
                lowercase=True,
                strip_accents='unicode'
            )
            
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
            
            logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
            logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to build TF-IDF index: {e}")
            return False
    
    def build(self) -> bool:
        """Build the complete knowledge base"""
        success = self._load_documents() and self._build_index()
        self.built = success
        return success
    
    def query(self, query: str, topk: int = None) -> List[Dict]:
        """Query the knowledge base with enhanced retrieval"""
        if not self.built:
            logger.warning("KB not built")
            return []
        
        if topk is None:
            topk = self.config["top_k_base"]
        
        try:
            # Transform query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top results above threshold
            min_sim = self.config["min_similarity"]
            valid_indices = np.where(similarities >= min_sim)[0]
            
            if len(valid_indices) == 0:
                logger.info(f"No results above similarity threshold {min_sim}")
                return []
            
            # Sort by similarity
            sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]]
            
            # Take top k results
            top_indices = sorted_indices[:topk]
            
            results = []
            for idx in top_indices:
                results.append({
                    "text": self.documents[idx],
                    "source": self.metadata[idx]["source"],
                    "score": float(similarities[idx]),
                    "chunk_id": self.metadata[idx]["chunk_id"]
                })
            
            logger.info(f"Query '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
    
    def save_index(self, filepath: str):
        """Save the built index to disk"""
        if not self.built:
            raise ValueError("Index not built")
        
        data = {
            "vectorizer": self.vectorizer,
            "tfidf_matrix": self.tfidf_matrix,
            "documents": self.documents,
            "metadata": self.metadata,
            "config": self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str) -> bool:
        """Load a pre-built index from disk"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.vectorizer = data["vectorizer"]
            self.tfidf_matrix = data["tfidf_matrix"]
            self.documents = data["documents"]
            self.metadata = data["metadata"]
            self.config.update(data.get("config", {}))
            self.built = True
            
            logger.info(f"Index loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False


# Replacement for EnhancedKBRetrieval with better configuration
class EnhancedKBRetrieval(ImprovedKBRetrieval):
    """Drop-in replacement with improved KB retrieval"""
    
    def __init__(self, kb_dir: str = "kb"):
        super().__init__(kb_dir)
        
        # Override for even better results
        self.config.update({
            "max_features": 4096,        # Even larger vocabulary
            "top_k_base": 3,            # Return top 3 by default  
            "min_similarity": 0.03,     # Lower threshold
            "chunk_size": 2000,         # Larger chunks
            "chunk_overlap": 400,       # More overlap
        })


def test_improved_kb():
    """Test the improved KB system"""
    print("🧪 Testing Improved KB Retrieval")
    print("=" * 50)
    
    kb = EnhancedKBRetrieval("kb")
    success = kb.build()
    
    if not success:
        print("❌ Failed to build KB")
        return
    
    print(f"✅ Built KB with {len(kb.documents)} chunks")
    print(f"📊 TF-IDF matrix shape: {kb.tfidf_matrix.shape}")
    
    test_queries = [
        "what is SMILES notation?",
        "explain TPSA",
        "LogP toxicity relationship",
        "blood brain barrier penetration",
        "what are toxicophores?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 30)
        
        results = kb.query(query, topk=3)
        
        for i, result in enumerate(results, 1):
            print(f"📝 Result {i}: {result['source']} (score: {result['score']:.3f})")
            print(f"Content: {result['text'][:200]}...")
            print()


if __name__ == "__main__":
    test_improved_kb()
