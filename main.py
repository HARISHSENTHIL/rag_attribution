from src.config import Config
from src.data_manager import DataManager
from src.index_manager import IndexManager
from src.query_engine import QueryProcessor
from src.interface import Interface

def main():
    # Initialize configuration
    config = Config()
    Config.setup()

    # Setup paths
    json_dir = "data2" 
    output_dir = "data/trading_signals"

    # Initialize data manager
    data_manager = DataManager(json_dir, output_dir)  

    # Process and validate data
    if not data_manager.validate_dataset():
        print("Dataset validation failed.")
        return

    processed_dir = data_manager.process_signals()
    if not processed_dir:
        print("Failed to process signals.")
        return

    # Initialize and setup components
    index_manager = IndexManager(processed_dir, config)
    if not index_manager.create_or_load_index():
        print("Failed to create/load index.")
        return

    query_processor = QueryProcessor(index_manager, config)
    interface = Interface(query_processor, json_dir)  # Changed from json_path to json_dir

    # Launch interface
    ui = interface.create_interface()
    ui.launch(server_name="0.0.0.0", server_port=7862, share=True, debug=True)

if __name__ == "__main__":
    main()