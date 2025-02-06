import gradio as gr
from typing import Tuple
import json
from src.query_engine import QueryProcessor
from src.visualization import Visualizer

class Interface:
    def __init__(self, query_processor: QueryProcessor, data_path: str):
        self.query_processor = query_processor
        self.visualizer = Visualizer()
        self.data_path = data_path

    def handle_query(self, query: str) -> Tuple[str, str, str]:
        """Handle user query and return formatted results"""
        response, citations = self.query_processor.process_query(query)
        citation_texts = "\n\n".join(
            [f"Weight: {citation['weight']}%\nSource: {citation['text']}"
             for citation in citations]
        )
        plot_path = self.visualizer.plot_contributions(citations)

        return response, citation_texts, plot_path

    def display_dataset(self) -> str:
        """ Load and return the dataset as a string """
        try:
            with open(self.data_path, "r") as file:
                data = json.load(file)
                return json.dumps(data[:10], indent=2)  # Display the first 5 items for brevity
        except Exception as e:
            return f"Error loading dataset: {str(e)}"

    def create_interface(self) -> gr.Blocks:
        with gr.Blocks() as ui:
            ui.css = """
            .centered-image {
                margin-left: auto;
                margin-right: auto;
                background: transparent;
                border-color: transparent;
                }
            """
            with gr.Row():
                gr.Image("Rag Attribution 1.png", elem_classes="centered-image", scale=0.5, show_label=False, show_download_button=False, show_fullscreen_button=False)
                # gr.Markdown("<h1 style='text-align:center; width: 90%;'>RAG--ATTRIBUTION</h1>")
            with gr.Row():
                input_box = gr.Textbox(label="Enter your question",placeholder="E.g., Enter your Prompt")
                
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                data = gr.Button("Clear", variant="secondary")

            with gr.Row():
                response_box = gr.Textbox(label="Response", placeholder="Model's Response", interactive=False)

            with gr.Row():
                citation_box = gr.Textbox(
                    label="Top 3 Citations with Weights",
                    interactive=False
                )

            with gr.Row():
                plot_box = gr.Image(label="Contribution Bar Graph", show_download_button=False, show_fullscreen_button=False)

            # with gr.Row():
            #     submit_btn = gr.Button("Submit", variant="primary")
            #     dataset_btn = gr.Button("Show Dataset", variant="secondary")

            submit_btn.click(
                self.handle_query,
                inputs=[input_box],
                outputs=[response_box, citation_box, plot_box]
            )

            # with gr.Row():
            #     dataset_display = gr.Textbox(label="Dataset", interactive=False, max_lines=20)
            #     # dataset_btn = gr.Button("Show Dataset", variant="secondary")
            #     dataset_btn.click(
            #         self.display_dataset,
            #         inputs=[],
            #         outputs=[dataset_display]
            #     )

        return ui
