import os
import json
from pathlib import Path

class DataManager:
    def __init__(self, json_dir: str, output_dir: str):
        self.json_path = Path(json_dir)  # Keep json_path for compatibility
        self.output_dir = Path(output_dir)

    def process_signals(self) -> str:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        signal_count = 0

        try:
            # Process all JSON files in directory
            for json_file in self.json_path.glob('*.json'):
                with open(json_file, "r") as file:
                    trading_signals = json.load(file)
                    
                documents = [
                    {
                        "content": signal["content"],
                        "metadata": {
                            "user": str(signal["user"]),
                            "channel": signal["channel"],
                            "date": signal["date"]
                        }
                    }
                    for signal in trading_signals
                ]

                for doc in documents:
                    signal_file = self.output_dir / f"signal_{signal_count}.txt"
                    with open(signal_file, "w") as file:
                        file.write(f"Content: {doc['content']}\n")
                        file.write(f"User: {doc['metadata']['user']}\n")
                        file.write(f"Channel: {doc['metadata']['channel']}\n")
                        file.write(f"Date: {doc['metadata']['date']}\n")
                    signal_count += 1

            return str(self.output_dir)

        except Exception as e:
            print(f"Error processing signals: {str(e)}")
            return None

    def validate_dataset(self) -> bool:
        if not self.json_path.exists():
            print(f"Error: Directory '{self.json_path}' not found")
            return False
        return True

    def get_document_stats(self) -> dict:
        stats = {
            'total_documents': 0,
            'users': set(),
            'channels': set(),
            'total_size': 0
        }

        try:
            for json_file in self.json_path.glob('*.json'):
                with open(json_file, "r") as file:
                    signals = json.load(file)
                    stats['total_documents'] += len(signals)
                    stats['users'].update(signal['user'] for signal in signals)
                    stats['channels'].update(signal['channel'] for signal in signals)
                    stats['total_size'] += os.path.getsize(json_file)
        except Exception as e:
            print(f"Error getting stats: {str(e)}")

        return stats