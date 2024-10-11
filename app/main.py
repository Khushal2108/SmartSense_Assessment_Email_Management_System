# app/chatbot_ui.py

import tkinter as tk
from chatbot import Chatbot

class ChatbotUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Email Management Chatbot")

        # Chat history text box (non-editable)
        self.chat_history = tk.Text(self.root, bg="white", height=20, width=50)
        self.chat_history.config(state=tk.DISABLED)
        self.chat_history.grid(row=0, column=0, columnspan=2)

        # Entry box for user input
        self.entry_box = tk.Entry(self.root, width=40)
        self.entry_box.grid(row=1, column=0)

        # Send button
        self.send_button = tk.Button(self.root, text="Send", width=10, command=self.send_message)
        self.send_button.grid(row=1, column=1)

        # Quit button
        self.quit_button = tk.Button(self.root, text="Quit", width=10, command=self.root.quit)
        self.quit_button.grid(row=2, column=1)

        # Initialize the chatbot
        self.chatbot = Chatbot()

    def send_message(self):
        user_input = self.entry_box.get()

        # Display the user input in chat history
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.insert(tk.END, "You: " + user_input + "\n")

        # Generate response using the chatbot logic
        response = self.chatbot.generate_response(user_input)
        
        # Display chatbot response in chat history
        self.chat_history.insert(tk.END, "Bot: " + response + "\n")

        # Clear the input box after sending the message
        self.entry_box.delete(0, tk.END)
        self.chat_history.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotUI(root)
    root.mainloop()
