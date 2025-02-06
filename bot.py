from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
import logging
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import PointStruct
import os
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

class SignalsAgent:
    def __init__(self):
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
        self.openai_client = OpenAI(api_key=self.OPENAI_API_KEY)
        self.qdrant_client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333"))
        )
        self.collection_name = os.getenv("COLLECTION_NAME", "signals_collection")
        self.setup_qdrant()

    def setup_qdrant(self):
        collections = self.qdrant_client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
            )

    @staticmethod
    def apply_confidence_decay(citations: list) -> list:
        """Apply confidence decay to citation weights"""
        total_score = 0
        decayed_citations = []

        for i, citation in enumerate(citations):
            decayed_score = citation['score'] * (1 / (i + 1))
            boost_factor = 1.5 if "entry" in citation['text'].lower() else 1.0
            adjusted_score = decayed_score * boost_factor
            total_score += adjusted_score

            decayed_citations.append({
                'text': citation['text'],
                'weight': adjusted_score
            })

        for citation in decayed_citations:
            citation['weight'] = round((citation['weight'] / total_score) * 100, 2)

        return decayed_citations

    def analyze_signal(self, message: str) -> dict:
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a trading signal validator. A valid trading signal must include any of three components: Entry price, Take Profit (TP) level, and Stop Loss (SL) level. Respond with only the word 'Valid' if the message contains any of three components. Respond with 'Invalid' if none of these components are missing or unclear."
                },
                {"role": "user", "content": message}
            ]
        )
        result = response.choices[0].message.content.strip()
        return {"valid": result == "Valid", "details": result}

    def get_embedding(self, text: str) -> list:
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    def store_signal(self, content: str, user: str, channel: str):
        """Store signal in Qdrant using project's format with line breaks"""
        try:
            # Format content string with line breaks
            formatted_content = f'Content: "{content}"\nUser: {user}\nChannel: {channel}'

            # Generate embedding using the formatted content
            embedding = self.get_embedding(formatted_content)

            # Create unique point ID
            point_id = abs(hash(f"{content}{user}{channel}"))

            # Store in Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={"content": formatted_content}
                )]
            )
            logging.info(f"Stored signal in Qdrant: {point_id}")
        except Exception as e:
            logging.error(f"Error storing signal: {e}")

    def search_signals(self, query: str, limit: int = 3) -> list:
        """Search for similar signals"""
        try:
            query_embedding = self.get_embedding(query)
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )
            # Format results with proper line breaks
            citations = [{"text": r.payload["content"], "score": float(r.score)} for r in search_results]
            return self.apply_confidence_decay(citations)
        except Exception as e:
            logging.error(f"Error searching signals: {e}")
            return []

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Ready to analyze trading signals.")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message.text
        user = update.effective_user.username or "UnknownUser"
        channel = update.effective_chat.title or "DirectMessage"
        chat_id = update.effective_chat.id
        message_id = update.message.message_id

        try:
            analysis = self.analyze_signal(message)
            if analysis["valid"]:
                self.store_signal(message, user, channel)
                await context.bot.send_message(chat_id=chat_id, text="👍", reply_to_message_id=message_id)
        except Exception as e:
            logging.error(f"Error: {e}")
            await update.message.reply_text("Processing error occurred.")

    def run(self):
        application = Application.builder().token(self.TELEGRAM_BOT_TOKEN).build()
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        application.run_polling()

if __name__ == "__main__":
    agent = SignalsAgent()
    agent.run()
