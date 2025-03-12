import os
import chardet
from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize
import re
from langdetect import detect

def read_text_files_in_folder(folder_path):
    # List to store the contents of all text files
    text_file_contents = []
    
    try:
        # Get a list of all files in the folder
        files_in_folder = os.listdir(folder_path)
        
        # Iterate through each file in the folder
        for file_name in files_in_folder:
            file_path = os.path.join(folder_path, file_name)
            
            # Check if the file is a text file (ends with .txt)
            if os.path.isfile(file_path) and file_name.endswith(".txt"):
                # Detect the file's encoding using chardet
                with open(file_path, 'rb') as file:
                    rawdata = file.read()
                    result = chardet.detect(rawdata)
                    encoding = result['encoding']
                    print(f"Detected encoding for {file_name}: {encoding}")
                
                # Open the text file and read its contents with the detected encoding
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                    text_file_contents.append(content)
                
        # Return the list of text file contents
        return text_file_contents
    
    except FileNotFoundError as e:
        print(f"Error: The folder '{folder_path}' was not found.")
        print(f"Details: {e}")
    except PermissionError as e:
        print(f"Error: You do not have permission to access the folder or file '{folder_path}'.")
        print(f"Details: {e}")
    except OSError as e:
        print(f"Error: A general OS error occurred while accessing the folder or files.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(f"Details: {e}")
    # Return an empty list in case of an error
    return []

folder_path = r"C:\Program Files\Projects\datasets\göçteyim"
text_files = read_text_files_in_folder(folder_path)

'''
if text_files:
    first_file_content = text_files[2]
    print(first_file_content[:])  # Display only the first 200 characters
else:
    print("No text files found or there was an error.")
'''

def preprocess_documents(input_dir=None, output_dir=None):
    if input_dir and not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} not found.")
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    preprocessed_docs = []
    documentPathList = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.endswith('.txt')]

    for i, doc in enumerate(documentPathList):
        # If input_dir is provided, doc is a filename; check if it's a supported file type
        if input_dir:
            file_path = os.path.join(input_dir, doc)
            
            # Check if file is a valid text file based on extensions
            if not file_path.endswith(('.txt')):
                raise ValueError(f"File {file_path} is not a supported text file type. Only .txt is allowed.")
            
            # Read the file using detected encoding
            with open(file_path, 'rb') as f:
                rawdata = f.read()
                result = chardet.detect(rawdata)
                encoding = result['encoding']
                
            # Open the text file and read it using the detected encoding
            with open(file_path, 'r', encoding=encoding) as f:
                doc = f.read()
        
        # Detect language from the document content
        try:
            lang = detect(doc[:1000])  # Use first 1000 chars for speed
            if lang == 'de':
                lang = 'german'
            elif lang == 'tr':
                lang = 'turkish'
            else:
                lang = 'turkish'  # Default to Turkish if unknown
        except:
            lang = 'turkish'  # Fallback if detection fails
        
        # Remove page breaks and footers (based on pattern matching)
        doc = re.sub(r'- Seite \d+ von \d+ -', '', doc)
        doc = re.sub(r'Ein Service des Bundesministeriums der Justiz sowie des Bundesamts für Justiz ‒ www\.gesetze-im-internet\.de', '', doc)
        
        # Basic cleanup: Remove unnecessary newlines, spaces and replace special characters
        doc = re.sub(r'\n\s*\n+', '\n', doc.strip())
        doc = doc.replace('‒', '-')
        
        # Language-specific cleanup
        if lang == 'german':
            doc = re.sub(r'(\b[Nn]r\.|\b[Ss]\.|\b[AB]Bl\.)(\S)', r'\1 \2', doc)
            doc = re.sub(r'(§\s*\d+[a-zA-Z]?)(\s+)', r'\1\n', doc)
        elif lang == 'turkish':
            doc = re.sub(r'(\bvb\.|\bDr\.|\bProf\.)(\S)', r'\1 \2', doc)
        
        preprocessed_docs.append(doc)
        
        # Save preprocessed document if output_dir is specified
        if output_dir:
            output_filename = f"preprocessed_{os.path.basename(documentPathList[i])}.txt" if input_dir else f"doc_{i}.txt"
            with open(os.path.join(output_dir, output_filename), 'w', encoding=encoding) as f:
                f.write(doc)
    
    return preprocessed_docs

preprocess_documents("C:\Program Files\Projects\datasets\göçteyim", "C:\Program Files\Projects\datasets\göçteyim\parsedData")