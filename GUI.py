import tkinter as tk
from tkinter import scrolledtext
from tkinter import filedialog
from tkinter import messagebox
from functools import partial
import os
import json
import requests

class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Legal Chat")
        self.chat_history = []

        self.create_widgets()
        self.load_chat_history()

    def create_widgets(self):
        self.chat_display = scrolledtext.ScrolledText(self.root, state='disabled', width=60, height=20)
        self.chat_display.grid(row=0, column=0, padx=10, pady=10, columnspan=3)

        self.user_input = tk.Entry(self.root, width=50)
        self.user_input.grid(row=1, column=0, padx=10, pady=10)

        self.submit_btn = tk.Button(self.root, text="Submit", command=self.submit_query)
        self.submit_btn.grid(row=1, column=1, padx=5, pady=10)

        self.upload_btn = tk.Button(self.root, text="Upload PDF", command=self.upload_pdf)
        self.upload_btn.grid(row=1, column=2, padx=5, pady=10)

    def submit_query(self):
        query = self.user_input.get()
        if query:
            response = self.send_query(query)
            self.display_message("You:", query)
            self.display_message("Bot:", response)
            self.user_input.delete(0, 'end')
            self.chat_history.append(("You", query))
            self.chat_history.append(("Bot", response))
            self.save_chat_history()

    def upload_pdf(self):
        filepath = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if filepath:
            with open(filepath, 'rb') as file:
                files = {'file': (os.path.basename(filepath), file)}
                response = requests.post("http://localhost:8501", files=files)
                if response.status_code == 200:
                    data = response.json()
                    for message in data['messages']:
                        self.display_message(message['sender'], message['content'])
                        self.chat_history.append((message['sender'], message['content']))
                    self.save_chat_history()
                else:
                    messagebox.showerror("Error", "Failed to process the PDF file.")

    def send_query(self, query):
        # Replace this with your API endpoint to send the query to the chatbot backend
        # Example:
        response = requests.post("http://localhost:8501", json={"query": query})
        return response.json()["response"]

    def display_message(self, sender, message):
        self.chat_display.configure(state='normal')
        self.chat_display.insert(tk.END, f"{sender}: {message}\n")
        self.chat_display.configure(state='disabled')
        self.chat_display.see(tk.END)

    def save_chat_history(self):
        with open("chat_history.json", "w") as file:
            json.dump(self.chat_history, file)

    def load_chat_history(self):
        if os.path.exists("chat_history.json"):
            with open("chat_history.json", "r") as file:
                self.chat_history = json.load(file)
                for sender, message in self.chat_history:
                    self.display_message(sender, message)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatApp(root)
    root.mainloop()
