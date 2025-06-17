"""General program to save data to a file that can be read and plotted later
Written by Jackson Butler, 3-14-2025"""
import numpy as np
import os

def saveData(arrays, headers, filename, path):
    """
    Save multiple 1D arrays as columns in a text file.
    
    Parameters:
    arrays (list): List of 1D numpy arrays to save as columns
    headers (list): List of header strings for each column
    filename (str): Name of the file
    path (str): Directory path where the file should be saved
    """
    full_path = os.path.join(path, filename)
    os.makedirs(path, exist_ok=True)
    
    # Check that all arrays have the same length
    array_lengths = [len(arr) for arr in arrays]
    if len(set(array_lengths)) > 1:
        print(f"Warning: Arrays have different lengths: {array_lengths}")
        print("Shorter arrays will be padded with empty values")
    
    # Find the maximum length
    max_length = max(array_lengths)
    
    with open(full_path, 'w') as file:
        # Write headers as a tab-separated line
        file.write('\t'.join(headers) + '\n')
        
        # Write data row by row
        for i in range(max_length):
            row_values = []
            for j, arr in enumerate(arrays):
                if i < len(arr):
                    row_values.append(str(arr[i]))
                else:
                    row_values.append('')  # Empty string for missing values
            
            file.write('\t'.join(row_values) + '\n')
    
    print(f"Successfully saved {len(arrays)} columns to {full_path}")

# Example usage
if __name__ == "__main__":
    # Example arrays with different lengths
    array1 = np.array([1, 2, 3, 4, 5])
    array2 = np.array([10, 20, 30, 40])
    array3 = np.array([100, 200, 300])
    
    # Headers for each column
    headers = ["X Values", "Y Values", "Z Values"]
    
    # Save to documents
    saveData([array1, array2, array3], headers, "example_data.txt", "/Users/jacksonbutler/Documents")
