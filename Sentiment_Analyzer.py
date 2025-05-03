import tkinter as tk
from tkinter import ttk, messagebox
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import spacy

# Download necessary NLTK resources
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Initialize SentimentIntensityAnalyzer from VADER
sia = SentimentIntensityAnalyzer()

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Preprocess function for text tokenization
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalnum()]
    return filtered_tokens

# Function to classify sentiment using VADER
def classify_paragraph(paragraph):
    sentiment_score = sia.polarity_scores(paragraph)
    # Determine sentiment based on compound score
    if sentiment_score['compound'] >= 0.05:
        sentiment = 'pos'
    elif sentiment_score['compound'] <= -0.05:
        sentiment = 'neg'
    else:
        sentiment = 'neu'  # Neutral sentiment
    confidence = abs(sentiment_score['compound']) * 100  # Convert to percentage
    return sentiment, confidence

# NER (Named Entity Recognition) function
def extract_entities(paragraph):
    doc = nlp(paragraph)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    return entities

# GUI class
class SentimentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentiment Analyzer")
        self.root.geometry("850x650")
        self.root.configure(bg="#f4f6f8")

        self.paragraphs = []
        self.results = []
        self.confidences = []
        self.entities = []

        # Title
        title = tk.Label(root, text="🧠 Sentiment Analyzer", font=("Segoe UI", 26, "bold"),
                         fg="#333", bg="#f4f6f8")
        title.pack(pady=20)

        # Entry Frame
        entry_card = tk.Frame(root, bg="white", bd=1, relief="solid")
        entry_card.pack(pady=10, padx=20, ipadx=10, ipady=10)

        self.entry = tk.Text(entry_card, font=("Segoe UI", 14), width=60, height=6, bd=0, wrap=tk.WORD)
        self.entry.grid(row=0, column=0, padx=10, pady=10)

        self.add_btn = ttk.Button(entry_card, text="➕ Add", command=self.add_paragraph)
        self.add_btn.grid(row=0, column=1, padx=10, sticky="n")

        # Control Buttons
        btn_frame = tk.Frame(root, bg="#f4f6f8")
        btn_frame.pack(pady=10)

        style = ttk.Style()
        style.configure("TButton", font=("Segoe UI", 11), padding=6)
        style.map("TButton",
                  background=[("active", "#d6d6d6")],
                  foreground=[("active", "#222")])

        self.analyze_btn = ttk.Button(btn_frame, text="📊 Analyze & Visualize", command=self.visualize_results)
        self.analyze_btn.grid(row=0, column=0, padx=10)

        self.reset_btn = ttk.Button(btn_frame, text="🔁 Reset", command=self.reset_all)
        self.reset_btn.grid(row=0, column=1, padx=10)

        # Output Label
        self.output_label = tk.Label(root, text="", font=("Segoe UI", 12), bg="#f4f6f8", fg="#555")
        self.output_label.pack(pady=5)

        # Canvas for Matplotlib
        self.canvas_frame = tk.Frame(root, bg="#f4f6f8")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Sentiment Results Listbox
        self.results_listbox = tk.Listbox(root, width=80, height=10, font=("Segoe UI", 12))
        self.results_listbox.pack(pady=20)

    def add_paragraph(self):
        paragraph = self.entry.get("1.0", tk.END).strip()
        if paragraph:
            sentiment, confidence = classify_paragraph(paragraph)
            entities = extract_entities(paragraph)
            self.paragraphs.append(paragraph)
            self.results.append(sentiment)
            self.confidences.append(confidence)
            self.entities.append(entities)

            # Show latest sentiment and confidence
            self.output_label.config(text=f"📝 Sentiment: {sentiment.upper()} | Confidence: {confidence:.2f}%")

            # Add to Listbox
            result_text = f"Sentiment: {sentiment.upper()} | Confidence: {confidence:.2f}% | Entities: {', '.join([e[0] for e in entities])}"
            self.results_listbox.insert(tk.END, result_text)

            self.entry.delete("1.0", tk.END)
        else:
            messagebox.showwarning("Input Error", "Please enter a paragraph.")

    def visualize_results(self):
        if not self.results:
            messagebox.showinfo("No Data", "Please enter some paragraphs first.")
            return

        pos = self.results.count("pos")
        neg = self.results.count("neg")

        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4))
        fig.suptitle("Sentiment Summary", fontsize=10)

        # Bar chart
        ax1.bar(["Positive", "Negative"], [pos, neg], color=["#66bb6a", "#ef5350"])
        ax1.set_title("Sentiment Count")
        ax1.set_ylabel("Number of Paragraphs")

        # Pie chart
        ax2.pie([pos, neg], labels=["Positive", "Negative"], autopct='%1.1f%%', colors=["#81c784", "#e57373"])
        ax2.set_title("Sentiment Distribution")

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def reset_all(self):
        self.paragraphs.clear()
        self.results.clear()
        self.confidences.clear()
        self.entities.clear()
        self.output_label.config(text="")
        self.entry.delete("1.0", tk.END)
        self.results_listbox.delete(0, tk.END)
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

# Launch the GUI
root = tk.Tk()
app = SentimentApp(root)
root.mainloop()
