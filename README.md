# Fake News Detection

## Project Overview

This project focuses on detecting fake news using deep learning techniques. The model is built using TensorFlow and Keras, with a deployment interface developed in Streamlit. The project involves several key steps: data preprocessing, model building, training, evaluation, and deployment.

## Project Structure

- **Data Preprocessing**: 
  - Dropped unnecessary columns (e.g., `id`, `author`).
  - Handled missing data by filling in with placeholders.
  - Combined `title` and `text` fields for comprehensive text analysis.
  - Performed text cleaning, including lowercasing, removing non-alphabet characters, stop words, and punctuation.
  
- **Tokenization**:
  - Used `Tokenizer` to convert text to sequences.
  - Padded sequences to ensure uniform input length.

- **Model Building**:
  - Created a Sequential model with the following architecture:
    - `Embedding` layer to convert words to dense vectors.
    - `GlobalAveragePooling1D` layer to down-sample the input.
    - `Dense` layers with ReLU activation for learning complex patterns.
    - `Softmax` output layer for binary classification.
  - Compiled the model with Adam optimizer and categorical cross-entropy loss.

- **Training**:
  - Split the data into training and testing sets.
  - Trained the model over 10 epochs with validation on the test set.

- **Evaluation**:
  - Visualized the training and validation loss/accuracy over epochs using Matplotlib.

- **Deployment**:
  - Saved the trained model and tokenizer.
  - Developed a Streamlit app for easy interaction, where users can input news articles and get predictions on whether they are real or fake.

## How to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Osama-Abo-Bakr/fake-news-detection
   cd fake-news-detection
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

4. **Input Text**: Enter any news article in the text box provided and hit 'Predict' to see the result.

## Model Performance

The model was trained on the Kaggle Fake News dataset and achieved competitive accuracy on the validation set. The use of text preprocessing techniques and a carefully designed neural network architecture contributed to the model's effectiveness.

## Future Work

- **Model Optimization**: Experiment with different model architectures, such as LSTM or GRU, to improve accuracy.
- **Dataset Expansion**: Incorporate additional datasets to enhance model robustness.
- **Advanced NLP Techniques**: Explore transformer-based models for better contextual understanding.

## Conclusion

This project demonstrates the potential of deep learning in addressing the critical issue of fake news. By following the steps outlined in this README, you can replicate the results and even extend the project further.
