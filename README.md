# Reddit-Thread-Summarizer

## Overview
The **Reddit Thread Summarizer**, powered by advanced Natural Language Processing (NLP) techniques, aims to condense lengthy Reddit threads into concise, meaningful summaries. By utilizing a combination of pre-trained models and custom enhancements, the tool ensures users can quickly grasp key insights from large discussions while preserving context and nuance.

---

## Key Features
1. **Text Summarization**: Uses the BART model for generating coherent and context-aware summaries of Reddit threads.
2. **Hashtag Generation**: Automatically generates relevant hashtags based on summary content using Named Entity Recognition (NER) and keyword extraction.
3. **Sentiment Analysis**: Incorporates tools like VADER for analyzing the tone of threads and summaries (positive, neutral, or negative).
4. **Trending Topic Detection**: Identifies the most discussed themes using topic modeling with Latent Dirichlet Allocation (LDA).
5. **Streamlit-Based UI**: A user-friendly application for summarizing threads and sharing results directly on Twitter.

---

## Technologies and Tools
- **Big Data**: PySpark in Databricks for efficient handling of the TLDRHQ dataset.
- **NLP Models**: Pre-trained BART transformer with fine-tuning for text summarization.
- **Preprocessing**: Custom pipeline including text cleaning, slang replacement, and normalization.
- **Visualization**: Metrics like ROUGE, BLEU, and BERT F1 for performance evaluation.
- **Deployment**: Streamlit for the user interface and scalability for production.

---

## Methodology
### Data Preprocessing
1. **Text Cleaning**: Removes special characters, URLs, and usernames.
2. **Normalization**: Converts text to lowercase while preserving key punctuation.
3. **Slang Replacement**: Replaces internet slang with standard text.
4. **Feature Extraction**: Extracts key phrases and sentiment scores.

### Model Selection and Training
- **BART Model**: Chosen for its bidirectional encoder and autoregressive decoder, pre-trained as a denoising autoencoder.
- **Fine-Tuning**: Layers frozen selectively to retain pre-trained knowledge while adapting to Reddit-specific summarization tasks.

### Advanced Techniques
- **RAKE Integration**: Extracts key keywords to enrich model input for summarization.
- **Topic Modeling (LDA)**: Augments data with dominant topics for enhanced context.

---

## Usage Instructions
### Using the Streamlit UI
1. **Input Reddit Thread**: Paste the content of a Reddit thread into the text area.
2. **Generate Summary**: Click the "Generate Summary" button to view a concise summary.
3. **Generate Hashtags**: Click "Generate Hashtags" to get relevant hashtags.
4. **Share on Twitter**: Use the "Share on Twitter" button to post the summary and hashtags.

---

## Evaluation Metrics
- **ROUGE Scores**:
  - ROUGE-1: 0.2648
  - ROUGE-2: 0.0782
  - ROUGE-L: 0.1917
- **BLEU Score**: 0.0217
- **BERT F1 Score**: 0.2416

---

## Future Enhancements
- Live deployment for real-time summarization.
- Enhanced sentiment analysis and trend prediction.
- Integration with other social media platforms and productivity tools.


## License
This project is licensed under the [MIT License](#).

---

## Acknowledgments
Special thanks to the contributors and open-source communities behind PySpark, BART, and Streamlit for making this project possible.
