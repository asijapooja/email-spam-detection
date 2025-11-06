from tkinter import *
from tkinter import messagebox
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# ---- Step 1: Training data (very small demo set) ----
texts = [
    "Congratulations you won a prize",
    "Win money now click this link",
    "Get a free coupon today",
    "Hey, are we meeting tomorrow?",
    "Please send me the project file",
    "Let's go for lunch",
    "You have been selected to win cash"
]
labels = ["spam", "spam", "spam", "ham", "ham", "ham", "spam"]

# ---- Step 2: Train a simple model ----
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
model = MultinomialNB()
model.fit(X, labels)

# ---- Step 3: Create GUI ----
root = Tk()
root.title("Email Spam Detector")
root.geometry("500x400")
root.config(bg="#e8f0fe")

Label(root, text="Enter your message below:", bg="#e8f0fe", font=("Arial", 12, "bold")).pack(pady=10)

text_box = Text(root, height=10, width=50)
text_box.pack(pady=5)

result_label = Label(root, text="", bg="#e8f0fe", font=("Arial", 12, "bold"))
result_label.pack(pady=10)

# ---- Step 4: Function to predict ----
def check_spam():
    message = text_box.get("1.0", END).strip()
    if message == "":
        messagebox.showinfo("Info", "Please type a message.")
        return

    data = vectorizer.transform([message])
    prediction = model.predict(data)[0]

    if prediction == "spam":
        result_label.config(text="🚫 This message is SPAM!", fg="red")
    else:
        result_label.config(text="✅ This message is NOT spam.", fg="green")

# ---- Step 5: Buttons ----
Button(root, text="Check Spam", command=check_spam, bg="#4CAF50", fg="white", font=("Arial", 11, "bold"), padx=10).pack(pady=10)
Button(root, text="Clear", command=lambda: text_box.delete("1.0", END), bg="#f44336", fg="white", font=("Arial", 11, "bold"), padx=10).pack(pady=5)

root.mainloop()
