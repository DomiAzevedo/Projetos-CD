import pandas as pd
import tkinter as tk
from tkinter import messagebox
import requests
from PIL import Image, ImageTk
from io import BytesIO
import numpy as np

# Carregar dados dos livros
try:
    books_data = pd.read_csv("https://raw.githubusercontent.com/bernardovma/dados_livros/main/data.csv")
except Exception as e:
    messagebox.showerror("Error", f"Failed to load CSV data:\n{e}")
    exit()

# Criar função para recomendar livros
def recommend_books(book):
    # Calcular relevância de cada livro
    books_data['relevance'] = books_data['average_rating'] * books_data['ratings_count']

    # Filtrar livros similares
    similar_books = books_data[(books_data['authors'] == book['authors'].values[0]) | (books_data['categories'] == book['categories'].values[0])]
    similar_books = similar_books[similar_books['title'] != book['title'].values[0]]

    # Calcula NDCG score
    similar_books['ndcg_score'] = similar_books['relevance'].rank(ascending=False, method='max') / np.log2(similar_books.index + 2)

    # Ranking dos livros e escolha do 5 melhores
    recommended_books = similar_books.sort_values(by='ndcg_score', ascending=False).head(5)

    return recommended_books

# Função para buscar livro
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
        recommended_books = recommend_books(book)
        if not recommended_books.empty:
            recommendation_info = "\n\nRecommended Books:\n"
            for index, row in recommended_books.iterrows():
                recommendation_info += f"{row['title']} by {row['authors']} - Category: {row['categories']}\n"
            info_label.config(text=info_label.cget("text") + recommendation_info)
        else:
            messagebox.showinfo("Recommended Books", "No recommended books found.")
    else:
        messagebox.showinfo("Book Not Found", "Book not found.")

# Função para exibir imagem
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

# Criar interface gráfica
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
