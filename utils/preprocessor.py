import re

def preprocess_email(email_content):
    # Convert to lowercase
    email_content = email_content.lower()
    
    # Remove special characters and digits
    email_content = re.sub('[^a-zA-Z\s]', '', email_content)
    
    # Remove extra whitespace
    email_content = re.sub('\s+', ' ', email_content).strip()
    
    return email_content