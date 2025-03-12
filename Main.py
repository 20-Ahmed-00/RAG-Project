# -*- coding: utf-8 -*-
import sys
import os
import torch
import json
import numpy as np
import re
import shutil
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from huggingface_hub import login
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import FAISS
from langchain.document_loaders.base import BaseLoader
from langchain_community.document_loaders import DirectoryLoader
from charset_normalizer import detect as charset_detect

# Paths
CHUNK_DIR = os.path.abspath(r"C:\Program Files\Projects\datasets\göçteyim\parsedData\chunkData")
METADATA_PATH = os.path.abspath(r"C:\Program Files\Projects\datasets\göçteyim\parsedData\chunkData\chunk_metadata.json")
VECTOR_STORE_PATH = os.path.abspath(r"C:\Users\20ahm\Desktop\vector_store")
WORD_INDEX_PATH = os.path.abspath(r"C:\Users\20ahm\Desktop\word_index")

print(f"Initial CHUNK_DIR: {CHUNK_DIR}")
print(f"Initial METADATA_PATH: {METADATA_PATH}")
print(f"Initial VECTOR_STORE_PATH: {VECTOR_STORE_PATH}")
print(f"Initial WORD_INDEX_PATH: {WORD_INDEX_PATH}")

# Configuration
CONFIG = {
    "model_name": "meta-llama/Llama-2-7b-chat-hf",
    "embedding_model_name": "sentence-transformers/LaBSE",
    "base_dir": r"C:\Users\20ahm\Projects\ResumeModel",
    "chunk_dir": CHUNK_DIR,
    "qa_threshold": 1000,
    "max_new_tokens": 400,
    "temperature": 0.5,
    "top_p": 0.5,
    "lora_rank": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "greeting_templates": {
        'tr': "Sevgili Kullanıcı,",
        'de': "Sehr geehrter Nutzer,",
        'en': "Dear User,"
    },
    "fallback_templates": {
        'tr': "Sevgili Kullanıcı, '{terms}' hakkında yeterli bilgi bulamadım. Daha fazla detay verebilir misiniz?",
        'de': "Sehr geehrter Nutzer, ich konnte nicht genug Informationen über '{terms}' finden. Könnten Sie mehr Details geben?",
        'en': "Dear User, I couldn’t find enough information about '{terms}'. Could you provide more details?"
    },
    "max_qa_answers": 5,
    "max_relevant_questions": 5,
    "max_chunks": 5,  # Used for chunk retrieval limit
    "qa_chunk_size": 500,
    "history_similarity_threshold": 0.7,
    "max_history_messages": 5,
    "chunk_relevance_threshold": 0.4,  # Raised for stricter relevance
    "qa_similarity_threshold": 0.5    # Raised for stricter relevance
}

HF_TOKEN = os.environ.get("HF_TOKEN", "??????????????????????????")
login(HF_TOKEN)
print("Logged in to Hugging Face.")

# Device setup
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Ensure RTX 4060 drivers and CUDA are installed.")
device = torch.device("cuda")
print(f"Using device: {device} (GPU: {torch.cuda.get_device_name(0)})")
torch.cuda.empty_cache()

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name=CONFIG["embedding_model_name"], model_kwargs={"device": "cuda"})

# Directory setup for QA files
qa_base_dir = r"C:\Program Files\Projects\ResumeModel"
qa_file = os.path.join(qa_base_dir, "qa_pairs.json")
qa_faiss_dir = os.path.join(qa_base_dir, "qa_faiss_index")
qa_word_index_dir = os.path.join(qa_base_dir, "qa_word_index")

print(f"QA file path: {qa_file}")
print(f"QA FAISS dir: {qa_faiss_dir}")
print(f"QA word index dir: {qa_word_index_dir}")

# Custom loader class with lazy_load
class CustomTextLoader(BaseLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def lazy_load(self):
        with open(self.file_path, 'rb') as f:
            raw_data = f.read()
            result = charset_detect(raw_data)
            encoding = result['encoding'] if result['encoding'] else 'utf-8'
            text_content = raw_data.decode(encoding, errors='replace')
        doc = Document(page_content=text_content.lower().strip(), metadata={})
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                chunk_metadata = json.load(f)
            metadata_dict = {item["filename"]: item for item in chunk_metadata}
            filename = os.path.basename(self.file_path)
            if filename in metadata_dict:
                meta = metadata_dict[filename]
                doc.metadata["chapter"] = meta.get("chapter", "unknown")
                doc.metadata["section"] = meta.get("section", None)
                doc.metadata["subsection"] = meta.get("subsection", None)
            else:
                doc.metadata["chapter"] = "unknown"
                doc.metadata["section"] = None
                doc.metadata["subsection"] = None
        except Exception as e:
            print(f"Warning: Failed to load metadata for {filename}: {e}")
            doc.metadata["chapter"] = "unknown"
            doc.metadata["section"] = None
            doc.metadata["subsection"] = None
        yield doc

# FAISS index creation and loading
def create_faiss_indices(chunk_dir, vector_store_path, word_index_path, embedding_model):
    if not os.path.exists(chunk_dir):
        raise FileNotFoundError(f"Chunk directory '{chunk_dir}' does not exist.")
    
    print(f"Vector store path: {vector_store_path}")
    print(f"Does directory exist? {os.path.exists(vector_store_path)}")
    if os.path.exists(vector_store_path):
        print(f"Clearing existing directory: {vector_store_path}")
        shutil.rmtree(vector_store_path)
    os.makedirs(vector_store_path, exist_ok=True)
    
    loader = DirectoryLoader(chunk_dir, glob="*.txt", loader_cls=CustomTextLoader)
    documents = loader.load()
    if not documents:
        raise ValueError(f"No valid .txt files found in '{chunk_dir}'.")
    print(f"Loaded {len(documents)} chunks from {chunk_dir}.")

    vector_store = FAISS.from_documents(documents, embedding_model)
    try:
        vector_store.save_local(vector_store_path)
        print(f"FAISS vector store saved to '{vector_store_path}'.")
        print(f"Files in {vector_store_path}: {os.listdir(vector_store_path)}")
    except Exception as e:
        print(f"Failed to save FAISS index: {e}")
        raise
    
    all_words = set()
    for doc in documents:
        words = set(re.findall(r"(?u)\b\w+\b", doc.page_content))
        all_words.update(words)
    word_list = list(all_words)
    word_embeddings = embedding_model.embed_documents(word_list)
    word_index = FAISS.from_embeddings(list(zip(word_list, word_embeddings)), embedding_model)
    word_index.save_local(word_index_path)
    print(f"Word index saved to '{word_index_path}'.")
    print(f"Files in {word_index_path}: {os.listdir(word_index_path)}")

    return vector_store, word_index

def load_faiss_indices(vector_store_path, word_index_path, embedding_model):
    print(f"Attempting to load from: {vector_store_path}")
    print(f"Files in {vector_store_path}: {os.listdir(vector_store_path)}")
    vector_store = FAISS.load_local(vector_store_path, embedding_model, allow_dangerous_deserialization=True)
    print(f"Attempting to load from: {word_index_path}")
    print(f"Files in {word_index_path}: {os.listdir(word_index_path)}")
    word_index = FAISS.load_local(word_index_path, embedding_model, allow_dangerous_deserialization=True)
    print(f"Loaded FAISS vector store from '{vector_store_path}' with {vector_store.index.ntotal} vectors.")
    print(f"Loaded word index from '{word_index_path}' with {word_index.index.ntotal} words.")
    return vector_store, word_index

# Check and load FAISS indices
vector_store_index_path = os.path.join(VECTOR_STORE_PATH, "index.faiss")
word_store_index_path = os.path.join(WORD_INDEX_PATH, "index.faiss")
print(f"Checking for vector store index at: {vector_store_index_path}")
print(f"Checking for word index at: {word_store_index_path}")

if not os.path.exists(vector_store_index_path) or not os.path.exists(word_store_index_path):
    print("FAISS indices not found or incomplete. Regenerating...")
    vector_store, word_index = create_faiss_indices(CHUNK_DIR, VECTOR_STORE_PATH, WORD_INDEX_PATH, embedding_model)
else:
    vector_store, word_index = load_faiss_indices(VECTOR_STORE_PATH, WORD_INDEX_PATH, embedding_model)

# Load model with 4-bit quantization and LoRA
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained(
    CONFIG["model_name"],
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
lora_config = LoraConfig(
    r=CONFIG["lora_rank"],
    lora_alpha=CONFIG["lora_alpha"],
    target_modules=["q_proj", "v_proj"],
    lora_dropout=CONFIG["lora_dropout"],
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
print(f"Model loaded with 4-bit quantization and LoRA. GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

# LangChain memory
memory = ConversationBufferMemory(return_messages=True)

# QA pair management
def load_qa_pairs(file_path):
    print(f"DEBUG: Attempting to load QA pairs from {file_path}")
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                qa_pairs = json.load(f)
                print(f"DEBUG: Successfully loaded {len(qa_pairs)} QA pairs")
                return qa_pairs
        except Exception as e:
            print(f"DEBUG: Error loading QA pairs: {e}")
            return {}
    print(f"Warning: QA pairs file not found at {file_path}")
    return {}

def save_qa_pairs(file_path, qa_dict):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(qa_dict, f, ensure_ascii=False, indent=4)

def split_text(text, max_size):
    chunks = []
    while len(text) > max_size:
        split_point = text.rfind(". ", 0, max_size) + 1
        if split_point <= 0:
            split_point = max_size
        chunks.append(text[:split_point].strip())
        text = text[split_point:].strip()
    if text:
        chunks.append(text)
    return chunks

def initialize_qa_index(qa_file, qa_faiss_dir, qa_word_index_dir, embedding_model):
    print(f"DEBUG: Initializing QA index with file: {qa_file}")
    qa_dict = load_qa_pairs(qa_file)
    if not qa_dict:
        print("DEBUG: No QA pairs loaded, returning empty index")
        return {}, None, None
    
    # Precompute embeddings for questions and answers
    questions = list(qa_dict.keys())
    answers = list(qa_dict.values())
    question_embeddings = embedding_model.embed_documents(questions)
    answer_embeddings = embedding_model.embed_documents(answers)
    qa_embeddings = list(zip(questions, answers, question_embeddings, answer_embeddings))
    print(f"DEBUG: Precomputed embeddings for {len(qa_embeddings)} QA pairs")
    
    if len(qa_dict) >= CONFIG["qa_threshold"]:
        print(f"Q&A pairs exceed threshold ({CONFIG['qa_threshold']}). Converting to FAISS index...")
        if not os.path.exists(qa_faiss_dir) or not os.path.exists(qa_word_index_dir):
            documents = []
            for q, a in qa_dict.items():
                answer_chunks = split_text(a, CONFIG["qa_chunk_size"])
                for i, chunk in enumerate(answer_chunks):
                    documents.append(Document(
                        page_content=f"Q: {q}\nA (Part {i+1}): {chunk}",
                        metadata={"question": q, "answer": chunk, "part": i+1}
                    ))
            qa_vector_store = FAISS.from_documents(documents, embedding_model)
            qa_vector_store.save_local(qa_faiss_dir)
            qa_word_index = FAISS.from_texts(["dummy"], embedding_model)
            qa_word_index.save_local(qa_word_index_dir)
        else:
            qa_vector_store = FAISS.load_local(qa_faiss_dir, embedding_model, allow_dangerous_deserialization=True)
            qa_word_index = FAISS.load_local(qa_word_index_dir, embedding_model, allow_dangerous_deserialization=True)
        return qa_dict, qa_vector_store, qa_word_index
    
    return qa_dict, qa_embeddings, None

# Extract keywords
def extract_keywords(query):
    keywords = re.findall(r"(?u)\b\w+\b", query.lower())
    return " ".join(keywords)

# Hybrid similarity search for chunks
def hybrid_similarity_search(vector_store, word_index, query, embedding_model, k=5):
    start_time = time.time()
    
    query = query.lower().strip()
    query_words = set(re.findall(r"(?u)\b\w+\b", query))
    query_embedding = embedding_model.embed_query(query)
    
    structural_patterns = [
        r"(kapitel|chapter|bölüm)\s*(\d+)",
        r"§\s*(\d+[a-z]?)",
        r"\((\d+[a-z]?)\)"
    ]
    structural_matches = []
    for pattern in structural_patterns:
        if match := re.search(pattern, query):
            structural_matches.append(match.group(0))
    
    initial_k = min(50, vector_store.index.ntotal)
    distances, indices = vector_store.index.search(np.array([query_embedding]), initial_k)
    initial_docs = [(vector_store.docstore.search(vector_store.index_to_docstore_id[i]), distances[0][j]) 
                    for j, i in enumerate(indices[0])]
    candidates = [(doc, 1 - dist) for doc, dist in initial_docs if doc is not None]
    
    if not candidates:
        print("No matching chunks found.")
        return []
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = candidates[:k]
    
    ranked_chunks = []
    for doc, initial_score in top_candidates:
        doc_text = doc.page_content.lower()
        doc_embedding = embedding_model.embed_query(doc_text)
        semantic_score = cosine_similarity([query_embedding], [doc_embedding])[0][0]
        
        ranked_chunks.append(type('Result', (), {
            'page_content': doc_text,
            'metadata': doc.metadata,
            'score': semantic_score
        }))
    
    ranked_chunks.sort(key=lambda x: x.score, reverse=True)
    top_chunks = ranked_chunks[:k]
    
    print(f"Chunk search completed in {time.time() - start_time:.2f} seconds")
    return top_chunks

# Generate response
def generate_response(prompt, greeting):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=CONFIG["max_new_tokens"],
            temperature=CONFIG["temperature"],
            top_p=CONFIG["top_p"],
            do_sample=False
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
        if not response.startswith(greeting):
            response = f"{greeting} {response}"
        return response
    except torch.cuda.OutOfMemoryError:
        print("GPU memory exceeded. Clearing cache...")
        torch.cuda.empty_cache()
        return f"{greeting} I ran out of memory. Please try a shorter query."

# Process query with QA-first search, then chunks if needed
def process_query(query, memory, qa_dict, qa_embeddings, qa_word_index, embedding_model, lang):
    memory.save_context({"input": query}, {"output": ""})
    
    print(f"DEBUG: Processing query: '{query}'")
    query_embedding = embedding_model.embed_query(query)
    greeting = CONFIG["greeting_templates"][lang]
    qa_threshold = CONFIG["qa_similarity_threshold"]
    chunk_threshold = CONFIG["chunk_relevance_threshold"]
    
    # Step 1: Search QA embeddings (k=5 for questions and answers)
    relevant_context = []
    if qa_embeddings:
        print("DEBUG: Searching QA embeddings")
        question_similarities = []
        answer_similarities = []
        
        for q, a, q_emb, a_emb in qa_embeddings:
            q_similarity = cosine_similarity([query_embedding], [q_emb])[0][0]
            a_similarity = cosine_similarity([query_embedding], [a_emb])[0][0]
            question_similarities.append((q, a, q_similarity))
            answer_similarities.append((q, a, a_similarity))
        
        # Top 5 questions
        question_similarities.sort(key=lambda x: x[2], reverse=True)
        top_questions = question_similarities[:5]  # k=5
        print("Top 5 QA questions:")
        for i, (q, a, sim) in enumerate(top_questions, 1):
            print(f"Q{i} (Similarity: {sim:.3f}): {q}")
        
        # Top 5 answers
        answer_similarities.sort(key=lambda x: x[2], reverse=True)
        top_answers = answer_similarities[:5]  # k=5
        print("Top 5 QA answers:")
        for i, (q, a, sim) in enumerate(top_answers, 1):
            print(f"A{i} (Similarity: {sim:.3f}): {a[:50]}... (from '{q}')")
        
        # Add relevant Q&A items
        for q, a, sim in top_questions:
            if sim >= qa_threshold:
                combined_text = f"{q} {a}"
                combined_embedding = embedding_model.embed_query(combined_text)
                relevance_to_query = cosine_similarity([query_embedding], [combined_embedding])[0][0]
                if relevance_to_query >= qa_threshold:
                    relevant_context.append(f"Q: {q}\nA: {a}\n(Similarity: {sim:.3f}, Relevance: {relevance_to_query:.3f})")
        
        for q, a, sim in top_answers:
            if sim >= qa_threshold:
                combined_text = f"{q} {a}"
                combined_embedding = embedding_model.embed_query(combined_text)
                relevance_to_query = cosine_similarity([query_embedding], [combined_embedding])[0][0]
                if relevance_to_query >= qa_threshold and f"Q: {q}\nA: {a}" not in [x.split("\n(Similarity")[0] for x in relevant_context]:
                    relevant_context.append(f"A: {a}\n(from '{q}')\n(Similarity: {sim:.3f}, Relevance: {relevance_to_query:.3f})")
    
    # Step 2: If Q&A provides relevant info, generate response; otherwise, search chunks
    if relevant_context:
        print("DEBUG: Relevant Q&A context found, skipping chunk search.")
    else:
        print("DEBUG: No relevant Q&A context found, searching chunks.")
        keywords = extract_keywords(query)
        print(f"Extracted Keywords: {keywords}")
        top_chunks = hybrid_similarity_search(vector_store, word_index, query, embedding_model, k=CONFIG["max_chunks"])  # k=2
        print(f"\nTop {CONFIG['max_chunks']} chunks:")
        for i, chunk in enumerate(top_chunks, 1):
            print(f"C{i} (Similarity: {chunk.score:.3f}): {chunk.page_content[:100]}...")
        
        # Add relevant chunks
        for chunk in top_chunks:
            if chunk.score >= chunk_threshold:
                relevance_to_query = cosine_similarity([query_embedding], [embedding_model.embed_query(chunk.page_content)])[0][0]
                if relevance_to_query >= chunk_threshold:
                    relevant_context.append(f"Chunk: {chunk.page_content}\n(Similarity: {chunk.score:.3f}, Relevance: {relevance_to_query:.3f})")
    
    # Step 3: Generate response
    if relevant_context:
        context_str = "\n\n".join(relevant_context)
        prompt = (
            f"Original Query: {query}\n"
            f"Retrieved Context:\n{context_str}\n"
            f"Instructions:\n"
            f"- Answer the ORIGINAL QUERY '{query}' by searching for the answer within the context you are given.\n"
            f"- Do not go off topic and make your goal to answer the question only with only highly relevant info.\n"
            f"- If the context does not correspond to the question, then point that out only if you can't find the answer within the context.\n"
            f"- Respond in {lang.upper()}.\n"
            f"- Keep it concise and start with '{greeting}'.\n"
            f"- If the context does not directly answer the query, explicitly state that no relevant information was found and do NOT speculate.\n"
            f"Response: {greeting} "
        )
        answer = generate_response(prompt, greeting)
        print(f"\nSoru: {query}")
        print(f"Cevap: {answer}")
    else:
        answer = CONFIG["fallback_templates"][lang].format(terms=query)
        print(f"\nSoru: {query}")
        print(f"Cevap: {answer} (No relevant context found above thresholds {qa_threshold} for QA, {chunk_threshold} for chunks))")
    
    edit_prompt = {
        'tr': "Bu cevabı düzenlemek ister misiniz? (evet/hayır): ",
        'de': "Möchten Sie diese Antwort bearbeiten? (ja/nein): ",
        'en': "Do you want to edit this answer? (yes/no): "
    }
    edit = input(edit_prompt[lang]).strip().lower()
    yes_words = {'tr': 'evet', 'de': 'ja', 'en': 'yes'}
    if edit == yes_words[lang]:
        edited_answer = input("Enter your edited answer: ").strip()
        qa_dict[query] = edited_answer
        save_qa_pairs(qa_file, qa_dict)
        memory.save_context({"input": query}, {"output": edited_answer})
        print(f"Edited answer saved to {qa_file}")
        return edited_answer

    memory.save_context({"input": query}, {"output": answer})
    return answer

if __name__ == "__main__":
    qa_dict, qa_embeddings, qa_word_index = initialize_qa_index(qa_file, qa_faiss_dir, qa_word_index_dir, embedding_model)
    print(f"Loaded QA pairs: {len(qa_dict)} entries")
    memory = ConversationBufferMemory(return_messages=True)

    print("Please select a language / Lütfen bir dil seçin / Bitte wählen Sie eine Sprache:")
    print("1: Turkish (tr)")
    print("2: German (de)")
    print("3: English (en)")
    lang_choice = input("Enter 1, 2, or 3: ").strip()
    lang_map = {'1': 'tr', '2': 'de', '3': 'en'}
    lang = lang_map.get(lang_choice, 'en')
    print(f"Selected language: {lang}")

    while True:
        query_prompt = {
            'tr': "\nSorunuzu girin (veya 'çıkış' yazın): ",
            'de': "\nGeben Sie Ihre Frage ein (oder 'ausgang' zum Beenden): ",
            'en': "\nEnter your question (or 'exit' to quit): "
        }
        query = input(query_prompt[lang]).strip()
        exit_words = {'tr': 'çıkış', 'de': 'ausgang', 'en': 'exit'}
        if query.lower() == exit_words[lang]:
            break

        answer = process_query(query, memory, qa_dict, qa_embeddings, qa_word_index, embedding_model, lang)
        print(f"Final Cevap: {answer}")