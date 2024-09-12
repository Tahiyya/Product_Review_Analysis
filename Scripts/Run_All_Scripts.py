import subprocess

# List of scripts to run
scripts = ['1Data_Preprocessing_and_Transformation.py', '2Sentiment_Emotion_Analysis.py', '2_1_EvaluationMetrics.py', 
           '3Summarize_Reviews.py', '4List_Pros_Cons.py', '5Final_Pros_Cons.py']

# Loop over the scripts and run them sequentially
for script in scripts:
    result = subprocess.run(['python', script])  # For Python 3, use 'python3' if needed
    if result.returncode != 0:
        print(f"{script} failed with return code {result.returncode}")
        break  # Stop the execution if any script fails
    else:
        print(f"{script} executed successfully")
