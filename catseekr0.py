#!/usr/bin/env python3
# ============================================================
#  CatGPT R1 1.0  —  Self-Optimizing Local Reasoner
#  Inspired by DeepSeek-R1-Zero and AlphaZero evolution loops
#  Author: @ItsJustaCat00 • Platform: macOS M4 Pro / Intel Hybrid
#  (Gemini-Refactored for Responsiveness: 1.1)
# ============================================================

import os, json, random, torch, time
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
from queue import Queue, Empty  # Added for threaded GUI updates
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

BASE_PATH = os.path.expanduser("~/CatGPT_R1")
MEM_PATH   = os.path.join(BASE_PATH, "memory.json")
POLICY_PATH = os.path.join(BASE_PATH, "policy.json")
os.makedirs(BASE_PATH, exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ─────────────────────────────── Device Detection ───────────────────────────────
def detect_device():
    """Detects the best available compute device and data type."""
    if torch.backends.mps.is_available():
        print("⚙️  Using Apple Metal (M-series).")
        return "mps", torch.float16
    elif torch.cuda.is_available():
        print("⚙️  Using NVIDIA CUDA.")
        return "cuda", torch.float16
    else:
        print("⚙️  Using CPU fallback.")
        return "cpu", torch.float32

# ─────────────────────────────── Policy Manager ───────────────────────────────
class Policy:
    """Manages the generative parameters, scores, and evolution."""
    def __init__(self, path=POLICY_PATH):
        self.path = path
        self.data = {
            "temperature": 0.9,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "generation": 0,
            "scores": []
        }
        if os.path.exists(path):
            try:
                self.data.update(json.load(open(path)))
            except Exception as e:
                print(f"⚠️ Could not load policy, using defaults. Error: {e}")
                pass # Use defaults if file is corrupt

    def mutate(self, avg_score):
        """
        Slightly evolve parameters based on average performance.
        Worse performance (lower avg_score) leads to more mutation.
        """
        self.data["generation"] += 1
        drift = random.uniform(-0.05, 0.05)
        
        # Mutate more if score is bad (1-avg_score is high)
        temp_drift = drift * (1 - avg_score) 
        # Mutate more if score is good (avg_score is high)
        top_p_drift = drift * avg_score
        
        self.data["temperature"] = max(0.55, min(1.25, self.data["temperature"] + temp_drift))
        self.data["top_p"] = max(0.8, min(0.98, self.data["top_p"] + top_p_drift))
        self.data["repetition_penalty"] = max(0.9, min(1.2, self.data["repetition_penalty"] - drift * 0.3))
        
        try:
            json.dump(self.data, open(self.path, "w"), indent=2)
        except Exception as e:
            print(f"Error saving policy: {e}")


    def log_score(self, score):
        """Logs a new score and saves policy."""
        self.data["scores"].append(score)
        self.data["scores"] = self.data["scores"][-200:] # Keep last 200 scores
        try:
            json.dump(self.data, open(self.path, "w"), indent=2)
        except Exception as e:
            print(f"Error saving policy on log: {e}")

# ─────────────────────────────── Memory Engine ───────────────────────────────
class Memory:
    """Stores and retrieves prompt-response-score history."""
    def __init__(self, path=MEM_PATH):
        self.path = path
        self.data = {"runs": 0, "history": []}
        if os.path.exists(path):
            try:
                self.data = json.load(open(path))
            except Exception as e:
                print(f"⚠️ Could not load memory, starting fresh. Error: {e}")
                pass # Use defaults if file is corrupt

    def add(self, prompt, response, score):
        """Adds a new memory and saves."""
        self.data["history"].append({"p": prompt, "r": response, "s": score})
        self.data["runs"] += 1
        self.data["history"] = self.data["history"][-200:] # Keep last 200 memories
        try:
            json.dump(self.data, open(self.path, "w"), indent=2)
        except Exception as e:
            print(f"Error saving memory: {e}")


    def avg_score(self):
        """Calculates the average score of all memories."""
        if not self.data["history"]:
            return 0.5 # Default score if no history
        return sum(h["s"] for h in self.data["history"]) / len(self.data["history"])

# ─────────────────────────────── Model Core ───────────────────────────────
class CatGPT:
    """The main AI agent, handling model loading and generation."""
    def __init__(self):
        self.device, self.dtype = detect_device()
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        print(f"🐾 Loading {self.model_name} ...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,  # Fixed: Use 'torch_dtype' instead of 'dtype' for Hugging Face API compatibility
            device_map="auto" if self.device != "cpu" else None,
            trust_remote_code=True # Added for consistency
        )
        
        # Move to device if CPU is used (device_map doesn't apply)
        if self.device == "cpu":
            self.model.to(self.device)
            
        self.memory = Memory()
        self.policy = Policy()
        self.update_params()
        print("✅ Model loaded successfully.")

    def update_params(self):
        """Copies the latest parameters from policy to self."""
        d = self.policy.data
        self.temperature = d["temperature"]
        self.top_p = d["top_p"]
        self.repetition_penalty = d["repetition_penalty"]

    def evaluate(self, text):
        """
        Evaluates the quality of a generated text snippet (0.0 to 1.0).
        This is a simple heuristic.
        """
        score = 0.5 # Start neutral
        low_text = text.lower()
        
        if any(k in low_text for k in ["therefore", "because", "hence", "thus"]):
            score += 0.2
        if len(text.split()) > 40: # Reward longer, more detailed answers
            score += 0.1
        if "error" in low_text or "traceback" in low_text or "i cannot" in low_text:
            score -= 0.3
        
        return max(0, min(score, 1)) # Clamp score between 0 and 1

    def generate(self, prompt):
        """Generates a response, evaluates it, and updates memory/policy."""
        self.update_params() # Get latest policy
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_token_len = inputs["input_ids"].shape[1]

        # Generate output
        with torch.no_grad(): # Ensure no gradients are computed
            out = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id # Suppress warnings
            )
        
        # --- BUGFIX 1: Isolate generated text for evaluation ---
        # Decode only the *newly generated* tokens
        generated_tokens = out[0, input_token_len:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Evaluate *only* the new text
        score = self.evaluate(generated_text)
        
        # Format the full text for display (note: generated_text includes reasoning and any conclusion)
        full_text = f"Reasoning chain:\n{generated_text}\n\nConclusion: (CatGPT R1 output)"
        
        # Log and mutate
        self.memory.add(prompt, full_text, score)
        self.policy.log_score(score)
        self.policy.mutate(self.memory.avg_score())
        
        return full_text, score

# ─────────────────────────────── GUI Layer ───────────────────────────────
class CatGPTApp:
    """Tkinter GUI application for CatGPT."""
    def __init__(self, root):
        self.root = root
        self.root.title("CatGPT R1 1.1 — Evolving Reasoner (Threaded)")
        
        self.text = scrolledtext.ScrolledText(root, wrap="word", width=80, height=25,
                                              bg="#0d1117", fg="#e6edf3", 
                                              insertbackground="white", font=("Arial", 11))
        self.text.pack(padx=10, pady=10, fill="both", expand=True)
        
        self.entry = ttk.Entry(root, font=("Arial", 11))
        self.entry.pack(fill="x", padx=10, pady=(0, 10))
        self.entry.bind("<Return>", self.handle_prompt)
        
        self.agent = None
        self.response_queue = Queue()

        # --- BUGFIX 2: Load model *after* GUI appears ---
        self.text.insert("end", "⏳ Initializing CatGPT R1 1.1...\n")
        self.text.insert("end", "Please wait, the model is loading. This may take a moment.\n")
        self.entry.config(state="disabled")
        
        # Start the model loader in a separate thread
        threading.Thread(target=self.load_model_thread, daemon=True).start()
        
        # Start the queue checker
        self.root.after(100, self.check_queue)

    def load_model_thread(self):
        """Worker thread to load the heavy model."""
        try:
            self.agent = CatGPT()
            # Put a success message in the queue for the main thread
            self.response_queue.put(("LOAD_SUCCESS", None))
        except Exception as e:
            # Put an error message in the queue
            self.response_queue.put(("LOAD_ERROR", str(e)))

    def check_queue(self):
        """
        Polls the response queue and updates the GUI from the main thread.
        This is the only place GUI components should be modified.
        """
        try:
            # --- BUGFIX 3: Non-blocking GUI updates ---
            data = self.response_queue.get(block=False)
            
            if data[0] == "LOAD_SUCCESS":
                self.text.insert("end", "\n✅ CatGPT R1 1.1 initialized. Ready for prompts.\n")
                self.entry.config(state="normal")
                self.entry.focus()
            
            elif data[0] == "LOAD_ERROR":
                self.text.insert("end", f"\n❌ FATAL ERROR: Could not load model.\n{data[1]}\n")
            
            elif data[0] == "GEN_SUCCESS":
                t, s = data[1]
                avg = self.agent.memory.avg_score()
                
                # Update text - Fixed deletion to remove only the "Thinking..." line
                self.text.delete("end-1l", "end") # Corrected range to delete last line
                self.text.insert("end", f"CatGPT: {t}\n")
                self.text.insert("end", f"Score {round(s,2)} | Avg {round(avg,2)} | Gen {self.agent.policy.data['generation']} | Temp {self.agent.temperature:.2f}\n")
                self.text.see("end")
                
                # Re-enable entry
                self.entry.config(state="normal")
                self.entry.focus()
                
                # Update plot
                if HAVE_MPL and len(self.agent.policy.data["scores"]) > 2:
                    plt.clf()
                    plt.plot(self.agent.policy.data["scores"], color="lime")
                    plt.title("CatGPT R1 Evolution Curve")
                    plt.pause(0.001)
            
            elif data[0] == "GEN_ERROR":
                self.text.delete("end-1l", "end") # Corrected range to delete last line
                self.text.insert("end", f"❌ ERROR during generation: {data[1]}\n")
                self.entry.config(state="normal") # Re-enable
                self.entry.focus()

        except Empty:
            pass # Queue is empty, do nothing
        finally:
            # Always schedule the next check
            self.root.after(100, self.check_queue)

    def handle_prompt(self, event=None):
        """Handles the <Return> event from the entry box."""
        q = self.entry.get().strip()
        if not q or self.agent is None:
            return
            
        self.entry.delete(0, "end")
        self.text.insert("end", f"\nYou: {q}\n")
        self.text.insert("end", "CatGPT: Thinking...\n")
        self.text.see("end")
        
        # Disable entry box while processing
        self.entry.config(state="disabled")
        
        # Run generation in a thread to not block the GUI
        threading.Thread(target=self.respond_thread, args=(q,), daemon=True).start()

    def respond_thread(self, q):
        """
        Worker thread to handle the blocking model.generate() call.
        """
        try:
            t, s = self.agent.generate(q)
            self.response_queue.put(("GEN_SUCCESS", (t, s)))
        except Exception as e:
            print(f"Error during generation: {e}")
            self.response_queue.put(("GEN_ERROR", str(e)))

# ─────────────────────────────── Main ───────────────────────────────
if __name__ == "__main__":
    print("🧠 Launching CatGPT R1 1.1 — Self-Optimizing Session")
    if HAVE_MPL:
        plt.ion() # Turn on interactive mode for matplotlib
        
    root = tk.Tk()
    app = CatGPTApp(root)
    root.mainloop()
    
    if HAVE_MPL:
        print("📈 Displaying final evolution graph.")
        plt.ioff() # Turn off interactive mode 
        plt.show() # Show final plot blocking 
