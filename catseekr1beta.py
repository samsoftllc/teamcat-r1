#!/usr/bin/env python3
# =============================================
#   CATR1 - 8B ILLUSION BEAST (DeepSeek UI Clone - CLEAN)
#   Cleaned version resembling chat.deepseek.com
# =============================================

import tkinter as tk
from tkinter import scrolledtext, ttk
import threading
import queue
import torch  # pyright: ignore[reportMissingImports]
from transformers import AutoTokenizer, AutoModelForCausalLM  # pyright: ignore[reportMissingImports]
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CatR1(torch.nn.Module):
    """CatR1 - 8B model with DeepSeek architecture."""
    def __init__(self):
        super().__init__()
        print("Loading CatR1 8B model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        print("CatR1 model loaded successfully.")

    def generate(self, prompt, max_new_tokens=1024, temperature=0.85):
        """Run inference safely across threads."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.92,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Strip redundant echo
        return response[len(prompt):].strip()

class CatR1GUI:
    """CatR1 UI - Clean DeepSeek-inspired interface."""
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CatR1 Chat")
        self.root.geometry("1200x800")
        self.root.configure(bg="#ffffff")
        self.root.resizable(True, True)

        self.catr1 = CatR1()
        self.running = False
        self.message_queue = queue.Queue()

        self._setup_ui()
        self._poll_queue()

    def _setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("DeepSeek.TButton", 
                       background="#0074e0", 
                       foreground="#ffffff", 
                       font=("Segoe UI", 11, "bold"),
                       borderwidth=0,
                       focuscolor="none")
        style.map("DeepSeek.TButton", 
                 background=[('active', '#005cb8')])

        # ── Header Bar ──
        top_bar = tk.Frame(self.root, bg="#f8f9fa", height=60)
        top_bar.pack(fill=tk.X, padx=0, pady=0)
        top_bar.pack_propagate(False)

        # Logo and title
        logo_frame = tk.Frame(top_bar, bg="#f8f9fa")
        logo_frame.pack(side=tk.LEFT, padx=20, pady=15)

        tk.Label(
            logo_frame, text="CatR1", font=("Segoe UI", 20, "bold"),
            fg="#0074e0", bg="#f8f9fa"
        ).pack(side=tk.LEFT)

        tk.Label(
            logo_frame, text="Chat",
            font=("Segoe UI", 20), fg="#333333", bg="#f8f9fa"
        ).pack(side=tk.LEFT, padx=(5, 0))

        # Model info
        tk.Label(
            top_bar, text="8B Parameter Model",
            font=("Segoe UI", 11), fg="#666666", bg="#f8f9fa"
        ).pack(side=tk.RIGHT, padx=20, pady=15)

        # ── Chat Display ──
        chat_frame = tk.Frame(self.root, bg="#ffffff")
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        self.display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD, bg="#ffffff", fg="#333333",
            font=("Segoe UI", 13),
            insertbackground="#0074e0",
            selectbackground="#e6f2ff",
            relief=tk.FLAT, bd=0,
            spacing1=12, spacing2=4, spacing3=12,
            padx=20, pady=20
        )
        self.display.pack(fill=tk.BOTH, expand=True)
        self.display.config(state=tk.DISABLED)

        # Style tags for messages
        self.display.tag_config("user_bubble", 
                               background="#0074e0", 
                               foreground="#ffffff",
                               relief=tk.FLAT,
                               lmargin1=50, lmargin2=50, rmargin=50,
                               spacing2=5)
        
        self.display.tag_config("assistant_bubble", 
                               background="#f0f4f8", 
                               foreground="#333333",
                               relief=tk.FLAT,
                               lmargin1=50, lmargin2=50, rmargin=50,
                               spacing2=5)
        
        self.display.tag_config("user_text", 
                               font=("Segoe UI", 13),
                               foreground="#ffffff")
        
        self.display.tag_config("assistant_text", 
                               font=("Segoe UI", 13),
                               foreground="#333333")

        # ── Input Area ──
        input_frame = tk.Frame(self.root, bg="#f8f9fa", height=120)
        input_frame.pack(fill=tk.X, side=tk.BOTTOM)
        input_frame.pack_propagate(False)

        input_inner_frame = tk.Frame(input_frame, bg="#f8f9fa")
        input_inner_frame.pack(fill=tk.BOTH, padx=20, pady=20)

        self.entry = tk.Text(
            input_inner_frame, 
            height=3, 
            bg="#ffffff", 
            fg="#333333",
            font=("Segoe UI", 13), 
            insertbackground="#0074e0",
            relief=tk.FLAT, 
            bd=1,
            highlightthickness=1,
            highlightbackground="#ddd",
            highlightcolor="#0074e0",
            wrap=tk.WORD,
            padx=12,
            pady=12
        )
        self.entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        send_button = ttk.Button(
            input_inner_frame, 
            text="Send", 
            style="DeepSeek.TButton", 
            command=self._on_send
        )
        send_button.pack(side=tk.RIGHT, padx=(10, 0))

        self.entry.bind("<Shift-Return>", lambda e: "break")
        self.entry.bind("<Return>", lambda e: self._on_send() or "break")
        self.entry.focus_set()

        # Welcome message
        self._add_message("assistant",
            "Hello! I'm CatR1, an AI assistant based on the DeepSeek architecture. "
            "I'm here to help you with questions, conversations, and various tasks. "
            "How can I assist you today?")

    def _add_message(self, role, content):
        """Thread-safe UI message append with bubble styling."""
        self.display.config(state=tk.NORMAL)
        
        # Add spacing between messages
        self.display.insert(tk.END, "\n\n")
        
        if role == "user":
            # User message (right-aligned bubble)
            self.display.insert(tk.END, " " * 100)  # Push to right
            start_idx = self.display.index(tk.END)
            self.display.insert(tk.END, f" {content} ", "user_bubble")
            end_idx = self.display.index(tk.END)
            # Apply text styling to the content
            self.display.tag_add("user_text", start_idx, end_idx)
            
        elif role == "assistant":
            # Assistant message (left-aligned)
            start_idx = self.display.index(tk.END)
            self.display.insert(tk.END, f" {content} ", "assistant_bubble")
            end_idx = self.display.index(tk.END)
            # Apply text styling to the content
            self.display.tag_add("assistant_text", start_idx, end_idx)
        else:
            # System message
            self.display.insert(tk.END, f"{content}\n", "system")
            
        self.display.config(state=tk.DISABLED)
        self.display.see(tk.END)

    def _poll_queue(self):
        """Check message queue periodically to update GUI safely."""
        try:
            while True:
                msg = self.message_queue.get_nowait()
                if isinstance(msg, tuple):
                    role, content = msg
                    self._add_message(role, content)
                else:
                    self._add_message("system", msg)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _on_send(self):
        """User send action."""
        if self.running:
            return
        text = self.entry.get("1.0", tk.END).strip()
        if not text:
            return
        self._add_message("user", text)
        self.entry.delete("1.0", tk.END)
        threading.Thread(target=self._generate_response, args=(text,), daemon=True).start()

    def _generate_response(self, prompt):
        self.running = True
        try:
            self.message_queue.put(("system", "Thinking..."))
            reply = self.catr1.generate(prompt)
            self.message_queue.put(("assistant", reply))
        except Exception as e:
            self.message_queue.put(("system", f"Error: {e}"))
        finally:
            self.running = False

    def run(self):
        self.root.mainloop()

def main():
    print("CatR1 Chat Interface - Clean DeepSeek Style")
    app = CatR1GUI()
    app.run()

if __name__ == "__main__":
    main()