# AI-Driven Product Review Analytics for Product Improvement &amp; Actionable Business Insights

AI-Driven Product Review Analytics for Product Improvement & Actionable Business Insights automates the analysis of customer reviews to extract key insights on product strengths and areas for refinement. By leveraging AI models, it provides a comprehensive understanding of customer feedback trends, helping stakeholders make data-driven decisions.

Product reviews are a valuable source of client feedback due to the rapid growth of ecommerce. When examined, this feedback offers insights into product performance, consumer satisfaction, and areas for improvement. With AI advancements, large volumes of feedback can now be quickly mined for insights that would be difficult to handle manually. 
Manually analysing client reviews to identify user sentiment, emotions, and recurring themes is time-consuming and prone to error. Businesses often lack the resources to efficiently process customer feedback and apply it to product improvement. Without leveraging customer perspectives, it becomes harder to make data-driven decisions for enhancing user experiences and product developments. 

The objective of this project is to create an AI-powered solution to automate the evaluation of customer reviews. This includes determining sentiment and emotion, summarizing feedback, identifying strengths and weaknesses, and generating recommendations for product improvement. The goal is to provide stakeholders with AI-generated insights to guide marketing and product development. 


The dataset comprises customer reviews for the iPhone 13 from Amazon, simulating company reviews that cannot be used due to confidentiality. The review text was cleaned by removing special characters, spaces, and emojis, followed by lemmatization. Sentiment analysis is performed using SiEBERT, and emotion detection uses Emotion English DistilRoBERTa base. Phi-3, a small language model, summarizes key product features and weaknesses. The results are displayed in a Streamlit dashboard, including sentiment and emotion distribution. The language models effectively classified emotions and sentiments in the reviews. Through summarization, key product features and common strengths and weaknesses were identified. 

The Streamlit dashboard allows stakeholders to explore sentiment and emotion distribution and view word clouds of common themes. The results show that AI can efficiently process customer feedback and provide valuable insights. Positive reviews highlighted specific product features, while negative ones focused on user experience issues. These insights can help businesses prioritize product improvements
and address customer pain points more effectively. 
