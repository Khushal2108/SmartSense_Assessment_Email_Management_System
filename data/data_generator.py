import random
import pandas as pd

# Categories for classification
categories = ['Student Inquiry', 'Academic Collaboration', 'Corporate Inquiry']

# Sample sentences for different categories
student_inquiry_samples = [
    "Can you send me the course syllabus?", 
    "I would like to enroll in the Python course.", 
    "When will the next lecture notes be available?",
    "Can I reschedule my exam?", 
    "How can I apply for the scholarship program?",
    "Please provide the link to the course material."
]

academic_collaboration_samples = [
    "We are looking forward to collaborating on the AI research project.", 
    "Can we set up a meeting to discuss joint research?", 
    "I would like to propose a new academic partnership.",
    "Could you review my research paper on deep learning?",
    "We are exploring potential collaboration in quantum computing.", 
    "How can we organize a workshop together?"
]

corporate_inquiry_samples = [
    "We are interested in sponsoring your event.", 
    "I would like to discuss potential corporate sponsorship.", 
    "Can your team provide consulting services for our new AI initiative?",
    "What are your rates for corporate training sessions?", 
    "We are considering your team for our next product launch.", 
    "Do you offer corporate webinars?"
]

# Function to generate an email dataset
def generate_email_dataset(num_samples):
    emails = []
    
    for _ in range(num_samples):
        category = random.choice(categories)
        
        if category == 'Student Inquiry':
            text = random.choice(student_inquiry_samples)
        elif category == 'Academic Collaboration':
            text = random.choice(academic_collaboration_samples)
        elif category == 'Corporate Inquiry':
            text = random.choice(corporate_inquiry_samples)
        
        emails.append({
            'text': text,
            'category': category
        })
    
    return pd.DataFrame(emails)

# Generate dataset and save to CSV
if __name__ == "__main__":
    num_samples = 1000  # Generating 1000 email instances
    email_dataset = generate_email_dataset(num_samples)
    email_dataset.to_csv('email_dataset.csv', index=False)
    print(f"Dataset with {num_samples} samples generated and saved to data/email_dataset.csv.")
