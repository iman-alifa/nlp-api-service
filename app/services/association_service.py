import numpy as np
from typing import List, Dict, Any
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class AssociationService:
    """Service to calculate Aspect-Topic Association"""
    
    @staticmethod
    def _aspect_name(aspect: Any) -> str:
        """
        Ambil nama aspek dari dua bentuk yang mungkin dikirim AspectService:
        string ('pelayanan') maupun dict ({'aspect': 'pelayanan', ...}).

        _build_document_aspects() mengembalikan List[List[str]], sehingga
        pemanggilan .get() langsung membuat seluruh analisis asosiasi gagal.
        """
        if isinstance(aspect, str):
            return aspect.strip()
        if isinstance(aspect, dict):
            return (aspect.get('aspect') or '').strip()
        return ''

    def analyze(
        self, 
        document_aspects: List[List[Any]], 
        document_topics: List[int], 
        num_topics: int, 
        min_mentions: int = 3
    ) -> Dict[str, Any]:
        """
        Calculate PMI (Pointwise Mutual Information) between aspects and topics.
        
        Args:
            document_aspects: List of aspect extractions for each document.
                Each element is a list of aspect names (str) or dicts
                ({'aspect': str, ...}); keduanya diterima.
            document_topics: List of topic IDs for each document (-1 for outlier)
            num_topics: Total number of valid topics
            min_mentions: Minimum number of aspect mentions to be included
            
        Returns:
            Dict containing PMI scores and co-occurrence matrix
        """
        try:
            if len(document_aspects) != len(document_topics):
                logger.warning(f"Length mismatch: aspects={len(document_aspects)}, topics={len(document_topics)}")
                return {"error": "Length mismatch"}
                
            N = len(document_topics)
            if N == 0:
                return {}
                
            # Filter valid docs (not outlier topic)
            valid_docs = [(i, t) for i, t in enumerate(document_topics) if t != -1]
            N_valid = len(valid_docs)
            if N_valid == 0:
                return {}
                
            # Count topic occurrences
            topic_counts = {t: 0 for t in range(num_topics)}
            for _, t in valid_docs:
                if t in topic_counts:
                    topic_counts[t] += 1
                else:
                    topic_counts[t] = 1
                    
            # Extract unique aspects and their doc occurrences
            # aspect_docs[aspect] = set of doc indices where aspect appears
            aspect_docs = {}
            for i, t in valid_docs:
                aspects = document_aspects[i]
                for asp in aspects:
                    a_name = self._aspect_name(asp)
                    if not a_name:
                        continue
                    if a_name not in aspect_docs:
                        aspect_docs[a_name] = set()
                    aspect_docs[a_name].add(i)
                    
            # Filter aspects by min_mentions
            valid_aspects = {a: docs for a, docs in aspect_docs.items() if len(docs) >= min_mentions}
            
            if not valid_aspects:
                return {}
                
            pmi_results = []
            heatmap_data = []
            
            for aspect, docs in valid_aspects.items():
                P_a = len(docs) / N_valid
                
                aspect_row = {"aspect": aspect}
                
                for t in range(num_topics):
                    if topic_counts.get(t, 0) == 0:
                        aspect_row[f"topic_{t}"] = 0
                        continue
                        
                    # docs that have both aspect and topic t
                    co_occur_docs = sum(1 for d in docs if document_topics[d] == t)
                    P_t = topic_counts[t] / N_valid
                    P_a_t = co_occur_docs / N_valid
                    
                    # Calculate PMI
                    if P_a_t > 0:
                        pmi = np.log2(P_a_t / (P_a * P_t))
                    else:
                        pmi = 0.0
                        
                    aspect_row[f"topic_{t}"] = round(pmi, 4)
                    
                    if pmi > 0:
                        pmi_results.append({
                            "aspect": aspect,
                            "topic_id": t,
                            "pmi": round(pmi, 4),
                            "co_occurrences": co_occur_docs
                        })
                        
                heatmap_data.append(aspect_row)
                
            # Sort PMI results descending
            pmi_results.sort(key=lambda x: x["pmi"], reverse=True)
            
            return {
                "pmi_top_associations": pmi_results[:20], # top 20 strongest
                "heatmap_matrix": heatmap_data
            }
            
        except Exception as e:
            logger.error(f"Association analysis error: {str(e)}", exc_info=True)
            return {"error": str(e)}
