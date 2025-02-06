import os
import gradio as gr
import matplotlib.pyplot as plt
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

# Set up the OpenAI API key and models
os.environ["OPENAI_API_KEY"] = ""  # Replace with your actual API key
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

def create_exhaustive_dataset():
    """Create a larger synthetic dataset with detailed contributions from multiple individuals."""
    dataset_dir = "data/programming_languages_exhaustive"
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Individual contributions
    data = {
        "alice.txt": """Alice's Contribution:
Programming languages have evolved significantly since the 1950s. 
Fortran, developed in the 1950s, was one of the first high-level programming languages 
and laid the foundation for modern computational methods. It was primarily used for numerical
and scientific computation, making it popular in academic and research institutions.
Fortran's ability to handle complex mathematical operations made it a staple for scientists
and engineers for decades, influencing the design of many subsequent languages.""",
        
        "bob.txt": """Bob's Contribution:
The 1970s saw the rise of structured programming with languages like C. 
C introduced features like functions, pointers, and memory management, 
which are critical to system programming. Its versatility and portability 
made it a foundational language for operating systems like UNIX.
Pascal, another language from this era, became a popular choice for teaching programming principles. 
Pascal emphasized clarity and structure, which helped beginners understand core programming concepts.""",
        
        "carol.txt": """Carol's Contribution:
Object-oriented programming (OOP) became popular in the 1980s with languages like Smalltalk and C++. 
This paradigm focused on encapsulating data and behavior into objects, making programs more modular 
and reusable. Smalltalk introduced the world to true OOP, while C++ combined OOP with the power 
and efficiency of C. The 1980s also saw the rise of functional programming languages like Lisp 
and ML, which emphasized immutability and declarative logic, influencing the development of modern 
languages like Haskell and Scala.""",
        
        "dave.txt": """Dave's Contribution:
The 1990s introduced web-focused languages like Java and JavaScript, 
which revolutionized web development by enabling dynamic, interactive web applications. 
Java became known for its "write once, run anywhere" philosophy, making it ideal for enterprise
applications. JavaScript, on the other hand, transformed static web pages into dynamic ones, 
introducing interactivity and user engagement. PHP and Perl became widely used for server-side scripting, 
further expanding the possibilities of web-based applications.""",
        
        "eve.txt": """Eve's Contribution:
The 2000s marked the rise of modern scripting languages like Python and Ruby. 
Python emphasized simplicity and readability, making it popular for data science, automation, 
and education. Its extensive libraries and frameworks like NumPy and Pandas helped it gain 
traction in machine learning and AI. Ruby gained traction for web development with the Ruby on Rails 
framework, which streamlined the process of building full-stack web applications and encouraged convention
over configuration.""",
        
        "frank.txt": """Frank's Contribution:
In the 2010s, the focus shifted to scalability, concurrency, and performance with languages like Go and Rust. 
Go simplified building cloud-native applications with its lightweight goroutines and robust standard library. 
Rust, on the other hand, prioritized memory safety without sacrificing speed, making it a favorite 
for systems programming. These languages were designed to address the challenges of modern computing,
such as parallelism, distributed systems, and secure memory management.""",
        
        "grace.txt": """Grace's Contribution:
AI and data processing drove the rise of domain-specific languages in the 2020s, such as TensorFlow and PyTorch. 
These languages enable efficient development of machine learning models and data pipelines, catering 
to the increasing demand for AI-driven solutions. Additionally, innovations like ONNX (Open Neural Network 
Exchange) standardized model deployment across different frameworks, making AI more accessible and interoperable. 
This era also saw the rise of no-code and low-code platforms, allowing non-programmers to contribute 
to AI development.""",
    }
    
    # Save contributions to files
    for filename, content in data.items():
        with open(os.path.join(dataset_dir, filename), "w") as f:
            f.write(content)
    
    print(f"Larger dataset created at {dataset_dir}")
    return dataset_dir


def create_or_load_index(dataset_dir, persist_dir="./citation_exhaustive"):
    """Create or load a vector store index."""
    if not os.path.exists(persist_dir):
        print("Creating a new index...")
        documents = SimpleDirectoryReader(dataset_dir).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=persist_dir)
    else:
        print("Loading existing index...")
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=persist_dir))
    return index

def apply_confidence_decay(scores):
    """Apply confidence decay to scores to reduce influence of lower-ranked sources."""
    return [score * (1 / (i + 1)) for i, score in enumerate(scores)]

def query_with_advanced_scoring(query, query_engine):
    """Handle a query and return response with advanced scoring."""
    response = query_engine.query(query)
    
    # Apply confidence decay
    scores = [node.score for node in response.source_nodes]
    decayed_scores = apply_confidence_decay(scores)
    total_score = sum(decayed_scores)
    
    citations = []
    for node, decayed_score in zip(response.source_nodes, decayed_scores):
        # Boost weight for nodes with relevant keywords (e.g., "web")
        boost_factor = 1.5 if "web" in node.node.get_text().lower() else 1.0
        adjusted_score = decayed_score * boost_factor
        weight = adjusted_score / total_score if total_score > 0 else 0
        citations.append({
            "text": node.node.get_text(),
            "weight": round(weight * 100, 2)  # Convert to percentage
        })
    
    # Sort by weight and keep the top 3 results
    sorted_citations = sorted(citations, key=lambda x: x["weight"], reverse=True)[:3]
    return response.response, sorted_citations
def plot_contributions(citations):
    """Plot the contributions of the top individuals as a bar graph."""
    names = [f"Source {i+1}" for i in range(len(citations))]
    contributor_names = [citation["text"].split("'s Contribution:")[0].strip() for citation in citations]
    weights = [citation["weight"] for citation in citations]

    plt.figure(figsize=(8, 5))
    plt.bar(contributor_names, weights, color='skyblue')
    plt.title("Top Contributions by Weight", fontsize=16)
    plt.xlabel("Contributors", fontsize=12)
    plt.ylabel("Weight (%)", fontsize=12)
    plt.ylim(0, 100)
    for i, weight in enumerate(weights):
        plt.text(i, weight + 1, f"{weight}%", ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig("contributions_plot.png")
    plt.close()
    return "contributions_plot.png"

def gradio_interface():
    # Create the dataset and load the index
    dataset_dir = create_exhaustive_dataset()
    persist_dir = "./citation_exhaustive"
    index = create_or_load_index(dataset_dir, persist_dir)
    
    # Create the CitationQueryEngine
    query_engine = CitationQueryEngine.from_args(
        index,
        citation_chunk_size=256,  # Smaller chunks for granularity
        similarity_top_k=5,      # Retrieve top 5 results
    )
    
    # Define Gradio interface
    def handle_query(query):
        response, citations = query_with_advanced_scoring(query, query_engine)
        citation_texts = "\n\n".join(
            [f"Weight: {citation['weight']}%\nSource: {citation['text']}" for citation in citations]
        )
        plot_path = plot_contributions(citations)
        return response, citation_texts, plot_path
    
    # Gradio UI components
    with gr.Blocks() as ui:
        gr.Markdown("### Programming Language Evolution: Query System with Top 3 Weighted Citations")
        with gr.Row():
            input_box = gr.Textbox(label="Enter your question", placeholder="E.g., What languages focus on AI?")
        with gr.Row():
            response_box = gr.Textbox(label="Response", interactive=False)
        with gr.Row():
            citation_box = gr.Textbox(label="Top 3 Citations with Weights", interactive=False)
        with gr.Row():
            plot_box = gr.Image(label="Contribution Bar Graph")            
        with gr.Row():
            submit_btn = gr.Button("Submit")
        
        # Link input to output
        submit_btn.click(handle_query, inputs=[input_box], outputs=[response_box, citation_box,plot_box])
    
    return ui

if __name__ == "__main__":
    ui = gradio_interface()
    ui.launch()
