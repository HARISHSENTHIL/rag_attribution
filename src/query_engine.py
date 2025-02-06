from typing import Tuple, List, Dict
from openai import OpenAI

class QueryProcessor:
    def __init__(self, index_manager, config):
        self.index_manager = index_manager
        self.config = config
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

    @staticmethod
    def apply_confidence_decay(citations: List[Dict]) -> List[Dict]:
        """Apply confidence decay to citation weights"""
        total_score = 0
        decayed_citations = []

        for i, citation in enumerate(citations):
            decayed_score = citation['weight'] * (1 / (i + 1))
            boost_factor = 1.5 if "web" in citation['text'].lower() else 1.0
            adjusted_score = decayed_score * boost_factor
            total_score += adjusted_score

            decayed_citations.append({
                'text': citation['text'],
                'channel': citation.get('channel'),
                'date': citation.get('date'),
                'weight': adjusted_score
            })

        for citation in decayed_citations:
            citation['weight'] = round((citation['weight'] / total_score) * 100, 2)

        return decayed_citations

    def process_query(self, query: str) -> Tuple[str, List[Dict]]:
        """Process query and return response with citations"""
        citations = self.index_manager.search(query, limit=3)
        
        citations = sorted(
            # citations, 
            # key=lambda hit: hit.get("metadata", {}).get("date", "") or 
            #             hit.get("payload", {}).get("date", "") or 
            #             "", 
            citations,
            key=lambda x: x.get('date', ''),
            reverse=True
        )
        
        processed_citations = self.apply_confidence_decay(citations)
        # context = "\n\n".join([f"Source {i+1}:\n{citation['text']}"
        #                     for i, citation in enumerate(processed_citations)])
        context = "\n\n".join([
            f"Source {i+1}:\nChannel: {citation['channel']}\n"
            f"Date: {citation['date']}\n{citation['text']}"
            for i, citation in enumerate(processed_citations)
        ])

        response = self.openai_client.chat.completions.create(
            model=self.config.MODEL_NAME,
            messages=[
                {"role": "system",
                "content":"You are an AI trading assistant. Based on the following messages, "
                "analyze sentiment and provide actionable trading insights for the specified token. "
                "Ensure to include the source of each insight, including the channel, message ID, and date."},
                
                {"role": "user","content": f"Based on these sources, analyze: {query}\n\nSources:\n{context}"}
            ]
        )

        return response.choices[0].message.content, processed_citations

