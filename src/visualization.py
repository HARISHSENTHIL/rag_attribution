import matplotlib.pyplot as plt
from typing import List, Dict
import re
class Visualizer:
    @staticmethod
    def plot_contributions(citations: List[Dict[str, float]]) -> str:
        """Create a bar plot of contribution weights"""
        contributor_names = []
        for citation in citations:
            text = citation["text"]
            user_match = re.search(r"User: (\d+)", text)
            channel_match = re.search(r"Channel: ([^\n]+)", text)

            user = user_match.group(1) if user_match else "Unknown"
            channel = channel_match.group(1) if channel_match else "Unknown"

            contributor_names.append(f"channel : {channel}\nUser : {user}")

        weights = [citation["weight"] for citation in citations]

        plt.figure(figsize=(8, 5))
        plt.bar(contributor_names, weights, color='Orange')
        plt.title("Top Contributors", fontsize=16,pad=15)
        plt.xlabel("Contributors", fontsize=12, labelpad=20)
        plt.ylabel("Weight (%)", fontsize=12)
        plt.ylim(0, 100)

        for i, weight in enumerate(weights):
            plt.text(i, weight + 1, f"{weight:.2f}%", ha='center', fontsize=10)

        # plt.xticks(rotation=45, ha='center')
        plt.tight_layout(pad=1.5)
        plt.savefig("contributions_plot.png", bbox_inches='tight')
        plt.close()

        return "contributions_plot.png"
