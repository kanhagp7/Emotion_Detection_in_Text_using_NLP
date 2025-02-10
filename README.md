# Emotion_Detection_in_Text_using_NLP

## **Project Overview**  
This project focuses on **detecting emotions from textual data** using **Natural Language Processing (NLP) and Machine Learning (ML)/Deep Learning (DL) models**. The main goal was to **classify text into one of seven emotions**:  
1. **Anger**  
2. **Disgust**  
3. **Fear**  
4. **Joy**  
5. **Neutral**  
6. **Sadness**  
7. **Surprise**  

To achieve this, we collected a **large dataset of comments** from platforms like **YouTube and Instagram** using a **custom web scraper** built with **Selenium**. We then preprocessed the data using various NLP techniques and experimented with different ML and DL models to achieve optimal performance.  
  
## **Problem Statement**  
Understanding human emotions from text is crucial for multiple applications across various domains. However, traditional ML models often struggle to capture contextual relationships between words, making it difficult to analyze emotions accurately.  

This project aims to **bridge this gap** by leveraging **deep learning techniques (LSTM with word embeddings)** to improve contextual understanding and sentiment classification.  
  
## **Potential Applications**  
This project can be applied in several real-world scenarios, such as:  
- **Customer Sentiment Analysis**: Helps businesses understand customer feedback, product reviews, and brand perception.  
- **Mental Health Monitoring**: Identifies signs of distress or negative emotions in user-generated content on social media platforms.  
- **Chatbots & Virtual Assistants**: Enables AI-driven chatbots to respond empathetically based on user emotions.  
- **Content Moderation**: Automatically detects harmful or offensive content based on emotional tone.  
- **Market Research & Social Media Analysis**: Analyzes trends in public opinion and sentiment.  

## **Data Collection & Preprocessing**  
- **Data Scraping**: Extracted around **90,000** rows of text data from **YouTube and Instagram comments**.  
- **Data Cleaning**: Applied NLP preprocessing techniques such as:  
  - Removing **emojis, punctuations, and stopwords**  
  - **Lemmatization**  
  - Language detection to ensure only **English text** is processed  
  
## **Machine Learning Approach**  
We used **TF-IDF vectorization** to convert text into numerical representations and trained multiple ML models, including:  
- **Logistic Regression**  
- **Multinomial Naïve Bayes**  
- **Support Vector Machine (SVM)**  
- **Random Forest**  
- **XGBoost**  

**Best ML Model**: **Logistic Regression** with **highest accuracy after hyperparameter tuning**.  
However, ML models struggled to capture the **sequential relationship between words** in a sentence.  
  
## **Deep Learning Approach**  
To overcome this limitation, we built an **LSTM (Long Short-Term Memory) model** using **GloVe word embeddings**.  
- **Hyperparameter tuning** included:  
  - **Dropouts, regularization, and optimizer tuning**  
- **Final Accuracy**: **~70%**  
- **Key Improvement**: **LSTM was able to retain the context of words**, leading to **better emotion recognition**.  


## **Technologies Used**  
- **Python**  
- **Selenium** (for web scraping)  
- **NLTK & Regex** (for text preprocessing)  
- **Scikit-Learn** (for ML models)  
- **TensorFlow & Keras** (for deep learning)  
- **GloVe Word Embeddings** (for better word representations)  

## **Conclusion**  
This project successfully classifies emotions from text using a combination of **ML and DL techniques**.  
- **ML models** provided good results but lacked sequential understanding.  
- **LSTM with GloVe embeddings** improved **contextual analysis**, leading to more meaningful emotion detection.  

This model can be further improved with **larger datasets, transformer-based architectures (like BERT), and fine-tuned hyperparameters**.  

## **Future Scope**  
- Implement **transformer models like BERT or GPT** for better accuracy.
- Use **real-time streaming data** to analyze emotions dynamically.
- Deploy the model as an **API** for integration with applications like chatbots and customer support systems.  
