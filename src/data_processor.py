# Data Processor Module
"""
This module provides data processing functionality.
"""

def process_data(data):
    """Process input data."""
    return data


def validate_data(data):
    """Validate input data."""
    if not data:
        raise ValueError("Data cannot be empty")
    return True


def transform_data(data):
    """Transform data to required format."""
    result = []
    for item in data:
        result.append(item * 2)
    return result


def filter_data(data, condition):
    """Filter data based on condition."""
    return [item for item in data if condition(item)]


def aggregate_data(data):
    """Aggregate data."""
    return sum(data) / len(data) if data else 0


def normalize_data(data):
    """Normalize data."""
    max_val = max(data) if data else 1
    return [item / max_val for item in data]


def merge_datasets(dataset1, dataset2):
    """Merge two datasets."""
    return dataset1 + dataset2


def split_dataset(dataset, ratio=0.8):
    """Split dataset into train and test sets."""
    split_point = int(len(dataset) * ratio)
    return dataset[:split_point], dataset[split_point:]


def clean_data(data):
    """Clean data by removing invalid entries."""
    return [item for item in data if item is not None]


def sort_data(data, reverse=False):
    """Sort data."""
    return sorted(data, reverse=reverse)


def deduplicate_data(data):
    """Remove duplicates from data."""
    return list(set(data))


def sample_data(data, sample_size):
    """Sample data randomly."""
    import random
    return random.sample(data, min(sample_size, len(data)))


def bin_data(data, num_bins):
    """Bin data into groups."""
    min_val = min(data)
    max_val = max(data)
    bin_size = (max_val - min_val) / num_bins
    bins = [[] for _ in range(num_bins)]
    
    for item in data:
        bin_index = min(int((item - min_val) / bin_size), num_bins - 1)
        bins[bin_index].append(item)
    
    return bins


def encode_categorical(data):
    """Encode categorical data."""
    unique_values = list(set(data))
    encoding = {val: idx for idx, val in enumerate(unique_values)}
    return [encoding[item] for item in data]


def scale_data(data, min_val=0, max_val=1):
    """Scale data to a specific range."""
    data_min = min(data)
    data_max = max(data)
    data_range = data_max - data_min
    
    if data_range == 0:
        return [min_val] * len(data)
    
    return [min_val + (item - data_min) * (max_val - min_val) / data_range for item in data]


def impute_missing(data, strategy='mean'):
    """Impute missing values."""
    valid_data = [item for item in data if item is not None]
    
    if strategy == 'mean':
        fill_value = sum(valid_data) / len(valid_data) if valid_data else 0
    elif strategy == 'median':
        sorted_data = sorted(valid_data)
        mid = len(sorted_data) // 2
        fill_value = sorted_data[mid] if valid_data else 0
    else:
        fill_value = 0
    
    return [item if item is not None else fill_value for item in data]


def detect_outliers(data, threshold=3):
    """Detect outliers using standard deviation."""
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std_dev = variance ** 0.5
    
    outliers = []
    for item in data:
        if abs(item - mean) > threshold * std_dev:
            outliers.append(item)
    
    return outliers


def parse_csv_data(file_path):
    """Parse CSV data from file."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    
    # Missing except or finally block - this is the syntax error!
    column_map = {
        'id': 0,
        'name': 1,
        'value': 2,
        'timestamp': 3
    }
    
    data = []
    for line in lines[1:]:  # Skip header
        parts = line.strip().split(',')
        data.append({
            'id': parts[column_map['id']],
            'name': parts[column_map['name']],
            'value': float(parts[column_map['value']]),
            'timestamp': parts[column_map['timestamp']]
        })
    
    return data
