Plagiarism Detection Tool
Overview
This repository contains a Python-based tool for detecting plagiarism in text documents. The tool utilizes Natural Language Processing (NLP) techniques such as TF-IDF vectorization, cosine similarity, and GPT-2 language model for analyzing textual content and identifying potential instances of plagiarism.

Features
Text-based Plagiarism Detection: Detects plagiarism by comparing input text with web search results and calculating similarity scores.
Web Search Integration: Utilizes Google Custom Search API to fetch relevant web documents for comparison.
TF-IDF Vectorization: Converts text documents into numerical vectors to facilitate similarity computation.
GPT-2 Language Model: Calculates perplexity and burstiness score using GPT-2 model for additional analysis.
Requirements
Python 3.x
nltk
transformers
torch
requests
scikit-learn
python-dotenv
Setup
Clone the repository:

git clone https://github.com/your_username/plagiarism-detection-tool.git
Install dependencies:

pip install -r requirements.txt
Obtain API key and search engine ID for Google Custom Search API. Refer to Google's documentation for instructions on obtaining these credentials.

Create a .env file in the root directory of the project and add your API key and search engine ID:

API_KEY=your_api_key
SEARCH_ENGINE_ID=your_search_engine_id
Usage
Run the script:

python plagiarism_detector.py
Choose an option:

Detect plagiarism from input text.
Detect plagiarism from a text file.
Enter the text or the path of the text file accordingly.

View the plagiarism detection results, including similarity percentage, web search results, perplexity, and burstiness score.

Contribution
Contributions are welcome! If you encounter any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

License
This project is licensed under the MIT License
