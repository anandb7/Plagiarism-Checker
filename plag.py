import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv


# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")


def calculate_perplexity(text):
    encoded_input = tokenizer.encode(
        text, add_special_tokens=False, return_tensors="pt"
    )
    input_ids = encoded_input[0]

    if input_ids.numel() == 0:
        # Handle empty input text
        return None

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    perplexity = torch.exp(
        torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), input_ids.view(-1)
        )
    )
    return perplexity.item()


def calculate_burstiness(text):
    tokens = nltk.word_tokenize(text.lower())
    word_freq = FreqDist(tokens)
    repeated_count = sum(count > 1 for count in word_freq.values())
    burstiness_score = repeated_count / len(word_freq)
    return burstiness_score


def plagiarism_detector(text):

    # Search the web for documents related to the input text
    web_results = search_web(text)

    # Check if there are web search results
    if not web_results:
        return "No web search results found.", 0

    # Calculate TF-IDF vectors for the input text and each web document
    vectorizer = TfidfVectorizer()
    documents = [text] + web_results
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Check if TF-IDF matrix contains at least one sample
    if tfidf_matrix.shape[0] < 2:
        return "Insufficient documents for comparison.", 0

    # Calculate cosine similarity between the input text and each web document
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    threshold = 0.2

    # Check if any similarity score is greater than the threshold
    for idx, score in enumerate(similarity_scores):
        if score > threshold:
            matched_percentage = round(score * 100, 2)
            return (
                f"Plagiarism detected. Matched with web document {idx+1} - {web_results[idx]}",
                matched_percentage,
            )

    return "No plagiarism detected.", 0


def search_web(query):
    load_dotenv()
    api_key = os.getenv("API_KEY")
    search_engine_id = os.getenv("SEARCH_ENGINE_ID")
    params = {
        "key": api_key,
        "cx": search_engine_id,
        "q": query,
        "num": 10,  # Number of search results to return
    }
    response = requests.get("https://www.googleapis.com/customsearch/v1", params=params)
    data = response.json()
    if "items" in data:
        return [item["snippet"] for item in data["items"]]
    else:
        return []


def plagiarism_detector_with_web_search(text):
    # Search the web for documents related to the input text
    web_results = search_web(text)

    # Check if there are web search results
    if not web_results:
        return "No web search results found.", ""

    # Calculate TF-IDF vectors for the input text and each web document
    vectorizer = TfidfVectorizer()
    documents = [text] + web_results
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Check if TF-IDF matrix contains at least one sample
    if tfidf_matrix.shape[0] < 2:
        return "Insufficient documents for comparison.", ""

    # Calculate cosine similarity between the input text and each web document
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    threshold = 0.2

    # Check if any similarity score is greater than the threshold
    for idx, score in enumerate(similarity_scores):
        if score > threshold:
            return (
                f"Plagiarism detected. Web document {idx+1} - {web_results[idx]}",
                web_results[idx],
            )

    return "No plagiarism detected.", ""


def perform_plagiarism_detection():
    choice = input(
        "Choose an option:\n1. Detect plagiarism from input text\n2. Detect plagiarism from a text file\nEnter your choice (1/2): "
    )

    if choice == "1":
        text = input("Enter the text for plagiarism detection: ")
    elif choice == "2":
        file_path = input("Enter the path of the text file: ")
        if not os.path.exists(file_path):
            print("File not found.")
            return
        with open(file_path, "r") as file:
            text = file.read()
    else:
        print("Invalid choice.")
        return

    # Perform plagiarism detection
    similarity_percentage = plagiarism_detector(text)

    # Get plagiarism detection with web search results
    web_search_result, matched_web_document = plagiarism_detector_with_web_search(text)

    # Calculate perplexity
    perplexity = calculate_perplexity(text)

    # Calculate burstiness
    burstiness_score = calculate_burstiness(text)

    # Print input text
    print("\nInput Text:\n")
    print(text)

    # Print plagiarism detection with web search results
    print("\nPlagiarism Detection with Web Search:\n")
    print("Source:", web_search_result)
    print("URL:", matched_web_document)
    print("\nPlagiarism Percentage:", similarity_percentage, "%")

    print("\n\n")
    print("Perplexity:", perplexity)
    print("Burstiness Score:", burstiness_score)


if __name__ == "__main__":
    perform_plagiarism_detection()