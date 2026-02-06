"""
Main application module.
"""

from src.data_processor import process_data, parse_csv_data


def main():
    """Main application entry point."""
    # Simple test data
    test_data = [1, 2, 3, 4, 5]
    
    # Process the data
    result = process_data(test_data)
    print(f"Processed data: {result}")
    
    # Try to parse CSV (this will fail with the syntax error in data_processor.py)
    try:
        csv_data = parse_csv_data('data.csv')
        print(f"CSV data: {csv_data}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
