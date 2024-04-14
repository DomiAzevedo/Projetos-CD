import pandas as pd
import tkinter as tk
from tkinter import messagebox
import requests
from PIL import Image, ImageTk
from io import BytesIO

# Load CSV data
try:
    books_data = pd.read_csv("https://raw.githubusercontent.com/bernardovma/dados_livros/main/data.csv")
except Exception as e:
    messagebox.showerror("Error", f"Failed to load CSV data:\n{e}")
    exit()

# Function to search for a book by name
def search_book():
    book_name = entry.get().strip().lower()
    if book_name == '':
        messagebox.showinfo("Error", "Please enter a book name.")
        return
    
    book = books_data.loc[books_data['title'].str.lower() == book_name]
    if not book.empty:
        book_info = f"Author: {book['authors'].values[0]}\n" \
                    f"Category: {book['categories'].values[0]}\n" \
                    f"Year: {book['published_year'].values[0]}\n" \
                    f"Rating: {book['average_rating'].values[0]}\n" \
                    f"Pages: {book['num_pages'].values[0]}\n" \
                    f"Description: {book['description'].values[0]}"
        info_label.config(text=book_info)
        display_image(book['thumbnail'].values[0])
        find_similar_books(book)
    else:
        messagebox.showinfo("Book Not Found", "Book not found.")

# Function to display image
def display_image(url):
    try:
        response = requests.get(url)
        img_data = response.content
        img = Image.open(BytesIO(img_data))
        img = img.resize((200, 300))
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img
    except Exception as e:
        messagebox.showerror("Error", f"Failed to display image:\n{e}")

# Function to find similar books by author or category
def find_similar_books(book):
    author = book['authors'].values[0]
    category = book['categories'].values[0]
    similar_books = books_data[(books_data['authors'] == author) | (books_data['categories'] == category)]
    similar_books = similar_books[similar_books['title'] != book['title'].values[0]].head(5)
    if not similar_books.empty:
        similar_books_info = "\n\nSimilar Books:\n"
        for index, row in similar_books.iterrows():
            similar_books_info += f"{row['title']} by {row['authors']} - Category: {row['categories']}\n"
        info_label.config(text=info_label.cget("text") + similar_books_info)
    else:
        messagebox.showinfo("Similar Books", "No similar books found.")

# Create GUI
root = tk.Tk()
root.title("Search Books")
root.attributes('-fullscreen', True)

label = tk.Label(root, text="Enter book name:")
label.pack()

entry = tk.Entry(root)
entry.pack()

button = tk.Button(root, text="Search", command=search_book)
button.pack()

info_label = tk.Label(root, text="")
info_label.pack()

img_label = tk.Label(root)
img_label.pack()

root.mainloop()
