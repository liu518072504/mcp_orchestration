import sys
import subprocess
import csv
import os

def create_python_file(file_content, file_name):
    """
    Create a Python file with the given file_content and file_name.
    The file will be created in the current working directory.
    """
    with open(file_name, "w") as f:
        f.write(file_content)
    print(f"Python file '{file_name}' created successfully.")
    return file_name

def execute_python_file(file_name):
    """
    Execute a Python file with the given file_name.
    The file should be in the current working directory.
    """
    
    try:
        result = subprocess.run([sys.executable, file_name], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error executing file: {result.stderr}"
    except Exception as e:
        print(f"Error executing Python file '{file_name}': {e}")
        return str(e)
    
def obtain_csv_header(file_name):
    """
    Obtain the header of a CSV file.
    This function assumes that the file is in the current working directory.
    """

    if not os.path.exists(file_name):
        return f"File '{file_name}' does not exist."

    with open(file_name, newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Get the first row as header
        return header
