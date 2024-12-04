# Bangla-English Question Answering System Using Sentence Transformers

This repository contains the implementation of a Bangla-English Question Answering (QA) system using Sentence Transformers and Hugging Face Transformers. The system performs QA tasks in both Bangla and English, incorporating features like semantic retrieval and answer ranking.

---

## Notebooks

This solution is demonstrated in **two notebooks**:

1. **Bangla-English QA with Sentence Transformers**  
   - Implements the core QA system, including dataset creation, translation, context retrieval, and answering questions in both Bangla and English.  
   - [Link to Kaggle Notebook](https://www.kaggle.com/code/isratjahankhan/bangla-english-qa-with-sentence-transformers)

2. **Bangla-English QA with GUI**  
   - Extends the first notebook by adding a simple Graphical User Interface (GUI) for enhanced usability.  
   - [Link to Kaggle Notebook](https://www.kaggle.com/code/isratjahankhan/bangla-english-qa-with-sentence-transformers-with)


---

## Features

- Multilingual QA support for Bangla and English.
- Translation of English datasets to Bangla using Google Translate API.
- Context retrieval using Sentence Transformers and semantic similarity.
- Answer ranking with Direct Preference Optimization (DPO).
- Support for Retrieval-Augmented Generation (RAG) via simplified retrieval methods.

---

## Implementation Steps

### 1. Library Installation
Install necessary libraries to enable translation, semantic similarity, and QA functionality:
- `sentence-transformers`
- `transformers`
- `googletrans`
- `pandas`
- `datasets`

---

### 2. Initializing Models and Pipelines
- **Google Translator API:** Used for Bangla-English translations.
- **QA Pipelines:**
  - English: `distilbert-base-uncased-distilled-squad`  
  - Multilingual: `xlm-roberta-base`
- **SentenceTransformer:** `all-MiniLM-L6-v2` for sentence embedding and context retrieval.

---

### 3. Creating an English Dataset
- A dataset was created with contexts and questions covering diverse topics such as science, technology, and general knowledge.
- Saved as a CSV file: `large_combined_dataset.csv`.

---

### 4. Translating the Dataset to Bangla
- The English dataset was translated into Bangla using the Google Translate API.
- Output was saved as `translated_large_combined_dataset.csv`.

---

### 5. Context Retrieval Using Semantic Similarity
- Implemented a semantic similarity function to retrieve the most relevant context:
  - Questions and contexts were encoded into sentence embeddings.
  - Cosine similarity was used to select the best-matching context.

---

### 6. Ranking Answers Using Direct Preference Optimization (DPO)
- A simple heuristic was used to rank answers, selecting the longest answer as the most relevant.

---

### 7. Processing and Displaying English QA Results
- For each question:
  - Retrieved the most relevant English context using semantic similarity.
  - Used the QA pipeline to generate an answer.
  - Printed results in a structured format.

---

### 8. Processing and Displaying Bangla QA Results
- For each question in Bangla:
  - Retrieved the most relevant Bangla context.
  - Used the multilingual QA pipeline to generate an answer.
  - Printed results in a structured format.

---

### 9. Adding GUI for Interactive QA with Gradio

To enhance the usability of the Bangla-English QA system, a simple graphical user interface (GUI) was created using the **Gradio** library. This interface allows users to easily interact with the system, input questions, select a dataset index, and view the generated answers in both Bangla and English.

#### Steps for Creating the GUI:

1. **Install Gradio**  
   First, ensure that the `gradio` library is installed by running the following command:
   ```bash
   pip install gradio
   ```

2. **Creating the Interface**  
   The core of the GUI is built around the following components:
   - **Input:** A text box where users can type their questions.
   - **Dropdown:** A dropdown menu to choose the dataset index (either English or Bangla dataset).
   - **Output:** A text box that displays the generated answers in both English and Bangla after the model processes the input question.

   The interface interacts with the system’s QA pipeline and retrieves the context from the chosen dataset. Based on the input question, the system returns the most relevant context, which is then processed to generate the answer.

3. **Running the Interface**  
   After defining the interface, it is launched using the `launch()` method. This opens a web-based interface in the browser, where users can interact with the system by typing their questions and choosing the language. The system will return the most relevant answers based on the dataset and context.

4. **Example Interaction**  
   When a user types a question like “What is Blockchain?” and selects **English** or **Bangla** as the language, the system will:
   - Retrieve the context for the most relevant match from the dataset.
   - Process the question using the pre-trained QA pipeline.
   - Display the generated answer.

---

#### Advantages of Gradio Integration:
- **User-Friendly Interface:** Gradio provides a simple and intuitive web interface, making it easy for anyone, even without programming knowledge, to interact with the system.
- **Live Interactivity:** With the `live=True` option, users get instant feedback on their queries.
- **Multilingual Support:** Users can switch between Bangla and English datasets effortlessly.

By adding this GUI, the Bangla-English QA system becomes more accessible, allowing users to interact with the system without requiring any technical setup.

---

## Additional Features
- **Retrieval-Augmented Generation (RAG):** Simplified retrieval mechanism for relevant contexts.
- **Direct Preference Optimization (DPO):** Answer ranking based on defined criteria.

---

## Output
The system generates QA results in both English and Bangla, providing relevant answers for each question by integrating translation, retrieval, and QA pipelines. Below are some sample questions and answers generated after running the notebook:
### Questions (Bangla)  
1. সালোকসংশ্লেষণ কী?  
2. আইফেল টাওয়ারটি কোথায় অবস্থিত?  
3. ব্লকচেইন কী?  
4. লিওনার্দো দা ভিঞ্চি কে ছিলেন?  
5. জলবায়ু পরিবর্তন কি?  

### Answers (Bangla)  
1. সালোকসংশ্লেষণ হ'ল প্রক্রিয়া যার মাধ্যমে সবুজ উদ্ভিদ এবং কিছু অন্যান্য জীব কার্বন ডাই অক্সাইড এবং জল থেকে খাবার সংশ্লেষ করতে সূর্যের আলো ব্যবহার করে।  
2. আইফেল টাওয়ারটি ফ্রান্সের প্যারিসে অবস্থিত বিশ্বের অন্যতম আইকনিক কাঠামো।  
3. ব্লকচেইন হ'ল একটি বিকেন্দ্রীভূত ডিজিটাল লেজার যা লেনদেনগুলি নিরাপদে এবং স্বচ্ছভাবে রেকর্ড করার জন্য ব্যবহৃত হয়।  
4. লিওনার্দো দা ভিঞ্চি ছিলেন মোনা লিসা এবং দ্য লাস্ট সাপারের মতো তাঁর কাজের জন্য পরিচিত রেনেসাঁ সময়ের একটি পলিম্যাথ।  
5. জলবায়ু পরিবর্তন মূলত মানুষের ক্রিয়াকলাপের কারণে তাপমাত্রা এবং আবহাওয়ার নিদর্শনগুলিতে দীর্ঘমেয়াদী পরিবর্তনকে বোঝায়।  

> **Note:** While English answer generation works efficiently, Bangla answer generation sometimes adds incomplete extra lines after the main answers. This issue is under development and will be resolved in future updates.  

> **GUI Note:** The GUI interface can be shown and used, but it does not handle the dataset properly at this time. This issue is also under development, and I will make sure it functions correctly in future updates.

---
