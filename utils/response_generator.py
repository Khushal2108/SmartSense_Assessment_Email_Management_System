def generate_response(category, email_content):
    responses = {
        'Student Inquiry': "Thank you for your inquiry. We'll process your request and get back to you shortly with the necessary information about your academic matter.",
        'Academic Collaboration': "We appreciate your interest in academic collaboration. Our faculty is always eager to explore new opportunities. We'll review your proposal and contact you to discuss further steps.",
        'Corporate Inquiry': "Thank you for reaching out regarding corporate opportunities. We value our industry partnerships and will forward your inquiry to the relevant department for a detailed response."
    }
    
    return responses.get(category, "Thank you for your email. We'll process your request and get back to you soon.")