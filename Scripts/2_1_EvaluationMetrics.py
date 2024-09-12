from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd

# Assuming df['predicted_sentiment'] contains the predicted labels
# and df['sentiment_label'] contains the true labels

inputFilePath = r'C:\Users\tahiy\VS Code Scripts\Product-Review-Analysis\Data\Output\2Final_sentiment_emotion_analysis.csv'
df = pd.read_csv(inputFilePath)

# Calculate accuracy
accuracy = accuracy_score(df['sentiment_label'], df['predicted_sentiment'])

# Calculate precision (macro, weighted, and per class)
precision_macro = precision_score(df['sentiment_label'], df['predicted_sentiment'], average='macro')
precision_weighted = precision_score(df['sentiment_label'], df['predicted_sentiment'], average='weighted')

# Calculate recall (macro, weighted, and per class)
recall_macro = recall_score(df['sentiment_label'], df['predicted_sentiment'], average='macro')
recall_weighted = recall_score(df['sentiment_label'], df['predicted_sentiment'], average='weighted')

# Calculate F1-score (macro, weighted, and per class)
f1_macro = f1_score(df['sentiment_label'], df['predicted_sentiment'], average='macro')
f1_weighted = f1_score(df['sentiment_label'], df['predicted_sentiment'], average='weighted')

# Confusion matrix
conf_matrix = confusion_matrix(df['sentiment_label'], df['predicted_sentiment'])

# Classification report (includes precision, recall, F1-score for each class)
class_report = classification_report(df['sentiment_label'], df['predicted_sentiment'])

print("Evaluation Metrics:")
# Print all metrics
print(f"Accuracy: {accuracy}")
print(f"Macro Precision: {precision_macro}")
print(f"Weighted Precision: {precision_weighted}")
print(f"Macro Recall: {recall_macro}")
print(f"Weighted Recall: {recall_weighted}")
print(f"Macro F1-Score: {f1_macro}")
print(f"Weighted F1-Score: {f1_weighted}")

print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Path to save the file
file_path = r'C:\Users\tahiy\VS Code Scripts\Product-Review-Analysis\Data\Output\Evaluation_Metrics.txt'

# Open the file in write mode and save variables
with open(file_path, 'w') as file:
    file.write(f"Evaluation Metrics for Sentiment Analysis with 'siebert/sentiment-roberta-large-english'\n")
    file.write(f"Accuracy: {accuracy}\n")
    file.write(f"Macro Precision: {precision_macro}\n")
    file.write(f"Weighted Precision: {precision_weighted}\n")
    file.write(f"Macro Recall: {recall_macro}\n")
    file.write(f"Weighted Recall: {recall_weighted}\n")
    file.write(f"Macro F1-Score: {f1_macro}\n")
    file.write(f"Weighted F1-Score: {f1_weighted}\n")
    file.write("\nConfusion Matrix:")
    file.write(f"{conf_matrix}")
    file.write("\nClassification Report:")
    file.write(f"{class_report}")

# Inform the user the file is saved
print(f"Sentiment scores saved to {file_path}")