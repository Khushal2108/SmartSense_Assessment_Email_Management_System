# Email Management Chatbot

## About the Project
The email management system chatbot is designed to streamline email management for HODs at a university ensuring that emails from various sources (students, corporates and researchers) are categorized, processed and handled efficiently.
.By leveraging pre-trained LLM model such as BERT and a neural network architecture custom model, this chatbot classifies incoming emails into specific categories, such as "Student Inquiry," "Academic Collaboration," and "Corporate Inquiry." It generates appropriate responses based on the classification results, allowing users to streamline their email management processes.

## Approach to Solve the Problem Statement
The project follows a two-pronged approach using both a pretrained model and a custom model for email classification. The aim is to enhance the accuracy and reliability of the email categorization process. The chatbot preprocesses the incoming email content, applies both classification models, and generates responses based on the classifications. For the pre-trained model, I have fine tuned BERT model, which is an open source model, and the exaplantion of the architecture of the custom model designed by me is provided below in the "Model Explanation" section. The chatbot is designed using the **inbuilt tkinter python** module which is a desktop based chatbot interface. It generates a response based on the classification.

## Dataset Used
I have generated a dataset of 1000 instances. The dataset generator code and the dataset can be found under the "data" folder in the repository.

## Explanation of the Models Used

### Why have I used BERT model
The project utilizes the **BERT (Bidirectional Encoder Representations from Transformers)** model, a state-of-the-art pretrained transformer model designed for natural language processing tasks. BERT was chosen for this email classification task because of its bidirectional understanding of context, which helps it grasp the full meaning of sentences more effectively than traditional models. This is crucial for emails, where understanding the intent and tone often depends on the entire sentence, not just individual words. Additionally, BERT is pretrained on large text corpora, making it well-suited for handling nuanced language, improving the accuracy of classifying emails into categories like Student Inquiry or Corporate Inquiry.

### Explanation of the Custom Model Architecture
I developed a custom email classification model that uses a simple yet effective feedforward neural network (FNN). Below is a detailed explanation of the model architecture and training process.


**1. Dataset Handling**
We start by loading the dataset from email_dataset.csv, which contains email text and corresponding categories. These are first processed as follows:

Text Vectorization: We use CountVectorizer to convert the email texts into a bag-of-words representation, transforming them into numerical vectors that the neural network can process.
Label Encoding: The email categories (labels) are encoded into integers using LabelEncoder so that they can be used as targets in our classification task.
The dataset is split into training and validation sets using an 80/20 ratio.

**2. EmailDataset Class**
The EmailDataset class is a PyTorch Dataset that helps handle the input data efficiently.

**3. Model Architecture: CustomEmailClassifier**
The CustomEmailClassifier is a feedforward neural network designed to classify emails based on their textual content. The architecture of the model is as follows:

Input Layer: The input size is determined by the number of features generated by the CountVectorizer (in this case, 84 features). These features represent the frequency of words in each email.

First Fully Connected Layer (fc1): The input is passed through a fully connected (dense) layer with 100 hidden units. This layer learns complex representations from the input data.

ReLU Activation Function: A ReLU (Rectified Linear Unit) activation function is applied to introduce non-linearity, which allows the model to learn more complex patterns from the email data. The ReLU function is defined as ReLU(x) = max(0, x).

Second Fully Connected Layer (fc2): The output from the first layer is passed to another fully connected layer, which reduces the hidden representation to the number of possible output classes (email categories). This layer generates raw scores (logits) for each class.

Output: The raw scores (logits) from the second fully connected layer are used for classification. The highest score among the classes is taken as the predicted category for an email.

**4. Training Process**
The model is trained using cross-entropy loss, a commonly used loss function for classification tasks. We optimize the model using the Adam optimizer, which is efficient and performs well in practice for most neural network training tasks.

Forward Pass: During the forward pass, input email vectors are passed through the neural network to get the predicted outputs.

Backward Pass & Optimization: The computed loss between the predicted outputs and true labels is used to adjust the model’s weights through backpropagation.

**5. Evaluation & Validation**
After each epoch, the model is evaluated on the validation set to check its performance. The validation accuracy is calculated by comparing the model's predictions to the true labels for the validation data.

**6. Saving the Model**
Once the training is completed, the following are saved:

The trained model’s weights (custom_model.pth).
The LabelEncoder used to encode the labels (custom_label_encoder.pkl).
The CountVectorizer used for text vectorization (custom_vectorizer.pkl).
By saving these components, we can later load the model and perform inference on new emails using the same encoding and vectorization as during training.

This simple feedforward model, trained on vectorized email text, provides a lightweight and efficient approach to email classification. Its modularity allows easy adaptation to different tasks with similar architectures.


## Installation Guide
**Step 1**:- Open the terminal, run "pip install -r requirements.txt".</br>
**Step 2**:- In the terminal, run "python setup.py".</br>
**Step 3**:- Navigate to the **data** folder. Run "data_generator.py".</br>
**Step 4**:- Navigate to the **models** folder. Run "pretrained_model.py" and "custom_model.py".</br>
**Step 5**:- Navigate to the **utils** folder. Run "preprocessor.py" and "response_generator.py".</br>
**Step 6**:- Navigate to the **app** folder. Run "chatbot.py" and then "main.py".</br>
**Step 7**:- Save the docker file. Run the following commands on your terminal. Ensure that docker is installed on your desktop.
            docker build -t email_management_system 
            docker run -p 8000:8000 email_management_system

## Testing the model
I have developed a **tkinter based chatbot** application, here are the steps for testing the model.</br>

**Step 1**:- Complete the installation as presented in the installation guide above.</br>
**Step 2**:- Navigate to the app folder.</br>
**Step 3**:- Run the **main.py** file. A chatbox will pop up on your screen.</br>
**Step 4**:- Provide an email to the chatbot. Based on the classification of the email, a response will be generated.</br>

**Note:-** For testing purposes, I have used the **Custom Model** for classification.



