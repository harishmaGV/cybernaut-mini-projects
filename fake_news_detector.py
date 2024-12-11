import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re
import nltk

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords


class FakeNewsDetector:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.pipeline = None
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def preprocess_text(self, text):
        """
        Advanced text preprocessing
        """
        # Convert to lowercase and remove special characters
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Remove stopwords
        tokens = text.split()
        cleaned_tokens = [token for token in tokens if token not in self.stop_words]

        return ' '.join(cleaned_tokens)

    def load_dataset(self, filepath):
        """
        Load and preprocess dataset
        """
        try:
            # Load CSV
            self.dataset = pd.read_csv(filepath)

            # Validate dataset
            if 'text' not in self.dataset.columns or 'label' not in self.dataset.columns:
                raise ValueError("Dataset must have 'text' and 'label' columns")

            # Preprocess text
            self.dataset['processed_text'] = self.dataset['text'].apply(self.preprocess_text)

            # Display dataset info
            print("📊 Dataset Loaded Successfully!")
            print(f"Total Samples: {len(self.dataset)}")
            print(f"Real News: {sum(self.dataset['label'] == 0)}")
            print(f"Fake News: {sum(self.dataset['label'] == 1)}")

            return self.dataset

        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def prepare_data(self, test_size=0.2):
        """
        Split data into training and testing sets
        """
        if self.dataset is None:
            print("🚫 Load dataset first!")
            return False

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.dataset['processed_text'],
            self.dataset['label'],
            test_size=test_size,
            random_state=42
        )

        print("✅ Data preparation complete!")
        return True

    def train_model(self):
        """
        Train machine learning pipeline
        """
        if self.X_train is None:
            print("🚫 Prepare data first!")
            return False

        # Create ML pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('classifier', LogisticRegression(max_iter=1000))
        ])

        # Train model
        self.pipeline.fit(self.X_train, self.y_train)

        # Predictions and evaluation
        y_pred = self.pipeline.predict(self.X_test)

        print("\n🔍 Model Performance:")
        print(classification_report(self.y_test, y_pred, target_names=['Real News', 'Fake News']))

        return True

    def predict_news(self, text):
        """
        Predict news authenticity
        """
        if self.pipeline is None:
            print("🚫 Train the model first!")
            return None

        try:
            processed_text = self.preprocess_text(text)
            print(f"Processed Text: {processed_text}")
            prediction = self.pipeline.predict([processed_text])
            probability = self.pipeline.predict_proba([processed_text])

            return {
                'label': 'Fake' if prediction[0] == 1 else 'Real',
                'confidence': max(probability[0]) * 100
            }
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return None


def main():
    detector = FakeNewsDetector()

    while True:
        print("\n--- 🕵️ Fake News Detector 🕵️ ---")
        print("1. Load Dataset")
        print("2. Train Model")
        print("3. Predict News")
        print("4. Exit")

        choice = input("Enter choice (1-4): ")

        if choice == '1':
            filepath = input("Enter dataset path (default: fake_news_dataset.csv): ") or 'fake_news_dataset.csv'
            detector.load_dataset(filepath)

        elif choice == '2':
            if detector.prepare_data():
                detector.train_model()

        elif choice == '3':
            if detector.pipeline:
                text = input("Enter news text to predict: ")
                result = detector.predict_news(text)
                if result:
                    print(f"\n🔮 Prediction: {result['label']} News")
                    print(f"🎯 Confidence: {result['confidence']:.2f}%")
            else:
                print("🚫 Train the model first!")

        elif choice == '4':
            print("Exiting... Goodbye!")
            break

        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
