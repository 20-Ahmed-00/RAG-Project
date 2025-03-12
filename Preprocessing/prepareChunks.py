import re
import os
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Path to your input text file
file_path = r"C:\Program Files\Projects\datasets\göçteyim\parsedData\preprocessed_AufenthG.txt.txt"

# Output directory for chunks
output_dir = r"C:\Program Files\Projects\datasets\göçteyim\parsedData\chunkData"

# Create the output directory if it doesn’t exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read the text file
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Split the text into lines for easier processing
lines = text.splitlines()

# Define regex patterns
kapitel_pattern = r'^Kapitel\s\d+[a-z]?$'    # Matches "Kapitel 1", "Kapitel 1a", etc.
paragraph_pattern = r'^§\s\d+[a-z]?$'        # Matches "§ 1", "§ 98a", etc.
bullet_pattern = r'^\(\d+[a-z]?\)'           # Matches "(1)", "(1a)", "(2)", etc.

# Hardcoded Abschnitt mappings (unchanged)
abschnitt_mapping = [
    # ... (your existing abschnitt_mapping list remains unchanged)
]

# Define 4-bit quantization configuration
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",  # Use NF4 quantization (normalized float 4-bit)
    bnb_4bit_compute_dtype=torch.float16,  # Compute in float16 for efficiency
    bnb_4bit_use_double_quant=True  # Optional: Use double quantization for better accuracy
)

# Load model with 4-bit quantization
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quant_config,  # Apply 4-bit quantization
    trust_remote_code=True
)
question_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

# Rest of your code remains unchanged...
# Variables to store metadata and chunks
current_kapitel = None
current_kapitel_title = None
current_paragraph = None
current_paragraph_title = None
current_chunk = []
chunk_list = []

# Function to determine Abschnitt based on paragraph and kapitel
def get_abschnitt(kapitel, paragraph):
    if not kapitel or not paragraph:
        return None, None
    for abschnitt in abschnitt_mapping:
        if abschnitt["kapitel"] == kapitel and paragraph in abschnitt["paragraphs"]:
            return abschnitt["abschnitt"], abschnitt["title"]
    return None, None

# Function to generate questions from chunk content in English
def generate_questions(content):
    prompt = (
    f"Generate three distinct, specific questions in English based on the following text. "
    f"The questions should focus on unique details within the text to help identify it, "
    f"M"
    f"And should not assume the reader knows which law or document is being referenced. So, consider the user does not know what is the law.:\n"
    f"{content}\n\nQuestions:"
)
    outputs = question_generator(
        prompt,
        max_new_tokens=200,
        num_beams=2,
        num_return_sequences=2,
        do_sample=False
    )
    questions = []
    for output in outputs:
        generated_text = output["generated_text"]
        lines = generated_text.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith(("1.", "2.", "3.", "-")) and "?" in line:
                questions.append(line.lstrip("123.- ").strip())
            elif "?" in line and line not in questions:
                questions.append(line)
    return questions[:3]  # Return up to 3 questions

# Process the text line by line (unchanged)
for line in lines:
    line = line.strip()
    if not line:  # Skip empty lines
        continue

    # Detect Kapitel
    if re.match(kapitel_pattern, line):
        if current_chunk:  # Save the previous chunk if it exists
            abschnitt, abschnitt_title = get_abschnitt(current_kapitel, current_paragraph)
            chunk_list.append((current_kapitel, current_kapitel_title, abschnitt, abschnitt_title, current_paragraph, current_paragraph_title, "\n".join(current_chunk)))
            current_chunk = []
        current_kapitel = line
        current_kapitel_title = None  # Reset title, will be set on the next line
        current_paragraph = None      # Reset paragraph when a new kapitel starts
        current_paragraph_title = None
    elif current_kapitel and not current_kapitel_title and not re.match(bullet_pattern, line) and not re.match(paragraph_pattern, line):
        current_kapitel_title = line  # The line after Kapitel is the title

    # Detect Paragraph (§)
    elif re.match(paragraph_pattern, line):
        if current_chunk:  # Save the previous chunk if it exists
            abschnitt, abschnitt_title = get_abschnitt(current_kapitel, current_paragraph)
            chunk_list.append((current_kapitel, current_kapitel_title, abschnitt, abschnitt_title, current_paragraph, current_paragraph_title, "\n".join(current_chunk)))
            current_chunk = []
        current_paragraph = line
        current_paragraph_title = None  # Reset title, will be set on the next line
    elif current_paragraph and not current_paragraph_title and not re.match(bullet_pattern, line):
        current_paragraph_title = line  # The line after Paragraph is the title

    else:
        # Handle bullet points and their content
        if re.match(bullet_pattern, line):
            if current_chunk:  # Save the previous chunk
                abschnitt, abschnitt_title = get_abschnitt(current_kapitel, current_paragraph)
                chunk_list.append((current_kapitel, current_kapitel_title, abschnitt, abschnitt_title, current_paragraph, current_paragraph_title, "\n".join(current_chunk)))
                current_chunk = []
        current_chunk.append(line)

# Append the last chunk if it exists
if current_chunk:
    abschnitt, abschnitt_title = get_abschnitt(current_kapitel, current_paragraph)
    chunk_list.append((current_kapitel, current_kapitel_title, abschnitt, abschnitt_title, current_paragraph, current_paragraph_title, "\n".join(current_chunk)))

# Save each chunk to a file with metadata and generated questions (unchanged)
for idx, (kapitel, kapitel_title, abschnitt, abschnitt_title, paragraph, paragraph_title, chunk_content) in enumerate(chunk_list):
    kapitel_safe = kapitel.replace(" ", "_") if kapitel else "NoKapitel"
    kapitel_title_safe = kapitel_title.replace(" ", "_").replace("§", "Sec")[:20] if kapitel_title else "NoKapitelTitle"
    abschnitt_safe = abschnitt.replace(" ", "_") if abschnitt else "NoAbschnitt"
    abschnitt_title_safe = abschnitt_title.replace(" ", "_").replace("§", "Sec")[:20] if abschnitt_title else "NoAbschnittTitle"
    paragraph_safe = paragraph.replace(" ", "_").replace("§", "Sec") if paragraph else "NoParagraph"
    paragraph_title_safe = paragraph_title.replace(" ", "_").replace("§", "Sec")[:20] if paragraph_title else "NoParagraphTitle"
    
    output_file = os.path.join(output_dir, f"{kapitel_safe}_{kapitel_title_safe}_{abschnitt_safe}_{abschnitt_title_safe}_{paragraph_safe}_{paragraph_title_safe}_chunk_{idx + 1}.txt")
    
    questions = generate_questions(chunk_content)
    
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(f"Kapitel: {kapitel}\n")
        file.write(f"Kapitel Title: {kapitel_title}\n")
        file.write(f"Abschnitt: {abschnitt}\n")
        file.write(f"Abschnitt Title: {abschnitt_title}\n")
        file.write(f"Paragraph: {paragraph}\n")
        file.write(f"Paragraph Title: {paragraph_title}\n")
        file.write(f"\n{chunk_content}\n")
        file.write("\nQuestions:\n")
        for i, question in enumerate(questions, 1):
            file.write(f"{i}. {question}\n")

print(f"Successfully split the text into {len(chunk_list)} chunks with Kapitel, hardcoded Abschnitt, Paragraph metadata, and English questions, saved them in {output_dir}")