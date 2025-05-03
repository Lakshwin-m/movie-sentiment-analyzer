# Movie Sentiment Analyzer

**Movie Sentiment Analyzer** is a desktop application built with Python and Tkinter that analyzes the sentiment of movie-related text. Using NLP techniques from NLTK and spaCy, the app processes user input to classify sentiment (positive, negative) and provides visual insights using Matplotlib. It also extracts named entities from the text for further analysis, making it a useful tool for analyzing movie reviews, summaries, or any movie-related content.

## Features
- **Sentiment Analysis**: Classifies text as positive, negative, or neutral using VADER Sentiment Analysis.
- **Named Entity Recognition (NER)**: Extracts named entities from the text using spaCy’s pre-trained models.
- **Visualization**: Displays sentiment distribution using bar and pie charts.
- **Interactive UI**: A user-friendly Tkinter GUI for easy input and interaction.
- **Confidence Scoring**: Shows the confidence level of the sentiment classification.
- **Entities Extraction**: Displays entities such as names, dates, locations, etc., from movie reviews.

## Technologies Used
- **Python**: The primary programming language.
- **Tkinter**: For building the graphical user interface (GUI).
- **NLTK**: For sentiment analysis and text processing.
- **spaCy**: For Named Entity Recognition (NER).
- **Matplotlib**: For visualizing sentiment data.

## Installation

Follow the steps below to get the Movie Sentiment Analyzer up and running on your local machine:

### 1. Clone the repository
Clone the repository to your local machine using the following command:
```bash
git clone https://github.com/lakshwin-m/movie-sentiment-analyzer.git
