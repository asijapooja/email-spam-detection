# spam_app.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import joblib
import pandas as pd
import os

MODEL_PATH = 'spam_detector.joblib'

class SpamApp:
    def __init__(self, root):
        self.root = root
        root.title("Email/SMS Spam Detector")
        root.geometry("820x600")

        # Load model pipeline
        if not os.path.exists(MODEL_PATH):
            messagebox.showwarning("Model not found", f"Model '{MODEL_PATH}' not found. Run train_model.py first.")
            self.pipeline = None
        else:
            self.pipeline = joblib.load(MODEL_PATH)

        # Controls
        frm = ttk.Frame(root, padding=10)
        frm.pack(fill='both', expand=True)

        # Text input
        ttk.Label(frm, text="Paste email / message text here:").pack(anchor='w')
        self.text_area = ScrolledText(frm, height=12)
        self.text_area.pack(fill='both', expand=False, pady=(0,10))

        btn_frame = ttk.Frame(frm)
        btn_frame.pack(fill='x', pady=(0,10))

        self.check_btn = ttk.Button(btn_frame, text="Check Spam", command=self.check_spam)
        self.check_btn.pack(side='left')

        self.clear_btn = ttk.Button(btn_frame, text="Clear", command=self.clear_text)
        self.clear_btn.pack(side='left', padx=(8,0))

        self.explain_btn = ttk.Button(btn_frame, text="Explain (top words)", command=self.show_explanation)
        self.explain_btn.pack(side='left', padx=(8,0))

        ttk.Button(btn_frame, text="Classify CSV", command=self.classify_csv).pack(side='right')

        # Result display
        self.result_var = tk.StringVar(value="Result will appear here.")
        result_label = ttk.Label(frm, textvariable=self.result_var, font=("Segoe UI", 12, "bold"))
        result_label.pack(anchor='w', pady=(10,0))

        self.prob_var = tk.StringVar(value="")
        ttk.Label(frm, textvariable=self.prob_var).pack(anchor='w')

        # Explanation area
        ttk.Label(frm, text="Explanation / logs:").pack(anchor='w', pady=(10,0))
        self.log_area = ScrolledText(frm, height=12)
        self.log_area.pack(fill='both', expand=True)

    def clear_text(self):
        self.text_area.delete('1.0', tk.END)
        self.result_var.set("Result will appear here.")
        self.prob_var.set("")

    def check_spam(self):
        text = self.text_area.get('1.0', tk.END).strip()
        if not text:
            messagebox.showinfo("No text", "Please paste an email or message to check.")
            return

        if not self.pipeline:
            messagebox.showerror("Model missing", "No trained model found. Run the training script first.")
            return

        pred = self.pipeline.predict([text])[0]
        proba = self.pipeline.predict_proba([text])[0]
        # find index of 'spam'
        try:
            spam_index = list(self.pipeline.classes_).index('spam')
            spam_prob = proba[spam_index]
        except ValueError:
            # fallback if classes are different
            spam_prob = max(proba)

        self.result_var.set(f"Prediction: {pred.upper()}")
        self.prob_var.set(f"Spam probability: {spam_prob:.3f}")
        self.log_area.insert(tk.END, f"Text: {text[:120]}...\nPrediction: {pred}, Prob: {spam_prob:.3f}\n\n")
        self.log_area.see(tk.END)

    def show_explanation(self):
        if not self.pipeline:
            messagebox.showerror("Model missing", "No trained model found. Run the training script first.")
            return

        # Attempt to compute top tokens that favor spam
        try:
            vectorizer = self.pipeline.named_steps['tfidf']
            clf = self.pipeline.named_steps['clf']  # MultinomialNB
            feature_names = vectorizer.get_feature_names_out()
            classes = list(clf.classes_)
            if 'spam' in classes and 'ham' in classes:
                idx_spam = classes.index('spam')
                idx_ham = classes.index('ham')
                # compute difference in log prob
                import numpy as np
                diff = clf.feature_log_prob_[idx_spam] - clf.feature_log_prob_[idx_ham]
                top_n = 20
                top_idx = diff.argsort()[::-1][:top_n]
                top_words = [(feature_names[i], float(diff[i])) for i in top_idx]
            else:
                # fallback: show top features for the class with highest mean probability
                cls_idx = 0
                top_idx = clf.feature_log_prob_[cls_idx].argsort()[::-1][:20]
                top_words = [(feature_names[i], float(clf.feature_log_prob_[cls_idx][i])) for i in top_idx]

            # show in a popup
            popup = tk.Toplevel(self.root)
            popup.title("Top indicative words for SPAM")
            txt = ScrolledText(popup, width=60, height=20)
            txt.pack(fill='both', expand=True)
            txt.insert(tk.END, "Word\tScore (higher => more indicative of spam)\n\n")
            for w,s in top_words:
                txt.insert(tk.END, f"{w}\t{round(s,4)}\n")
            txt.configure(state='disabled')
        except Exception as e:
            messagebox.showerror("Explain error", f"Could not compute explanation:\n{e}")

    def classify_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if not path:
            return
        try:
            df = pd.read_csv(path)
            # try common column names
            for col in ['text','message','body','content','email']:
                if col in df.columns:
                    texts = df[col].astype(str).tolist()
                    break
            else:
                # if none of the common names exist, assume first text column
                texts = df.iloc[:,0].astype(str).tolist()

            if not self.pipeline:
                messagebox.showerror("Model missing", "No trained model found. Run the training script first.")
                return

            preds = self.pipeline.predict(texts)
            probs = self.pipeline.predict_proba(texts)
            # spam index
            try:
                spam_index = list(self.pipeline.classes_).index('spam')
            except ValueError:
                spam_index = probs.shape[1]-1

            df['pred_label'] = preds
            df['spam_prob'] = [p[spam_index] for p in probs]

            out = os.path.splitext(path)[0] + "_classified.csv"
            df.to_csv(out, index=False)
            messagebox.showinfo("Done", f"Classified {len(df)} rows. Saved to {out}")
            self.log_area.insert(tk.END, f"Batch classified {len(df)} rows from {path}. Saved to {out}\n")
            self.log_area.see(tk.END)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to classify CSV: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SpamApp(root)
    root.mainloop()
