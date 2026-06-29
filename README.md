# Medicobot AI 🩺

Medicobot AI is an AI-powered medical intake and diagnostic assistant. It uses a state-machine-driven pipeline to interact with patients, ask intelligent follow-up questions, retrieve relevant medical literature via RAG (Retrieval-Augmented Generation), and output a structured differential diagnosis. 

The project leverages **LLaMA-2 7B (4-bit quantized)** and the **Unsloth** library to dynamically load and switch between task-specific LoRA adapters at runtime.

> **Disclaimer:** This is an AI-generated assessment for informational purposes only. It does NOT constitute medical advice, diagnosis, or treatment. Always consult a licensed healthcare professional.

## Features

- **Dynamic Role Switching:** Uses Peft/Unsloth to switch between multiple LoRA adapters (Follow-up & Diagnostic) on top of a single base model without disk I/O overhead.
- **Intelligent Intake:** Asks focused follow-up questions based on the patient's chief complaint.
- **Symptom Extraction:** Uses the base LLaMA-2 model to distill the consultation into a concise clinical summary.
- **Medical RAG:** Automatically retrieves relevant medical literature from **PubMed** and **MedlinePlus** APIs based on extracted symptoms.
- **Structured Diagnostics:** Outputs a professional diagnostic report including:
  1. Most likely condition
  2. Other conditions to consider
  3. Recommended next steps
- **Gradio UI:** Provides a clean, patient-facing chat interface for the consultation.

## Project Structure

The project consists of three main Jupyter Notebooks designed to be run in Google Colab (or a local environment with a suitable GPU):

### 1. `finetune_followup.ipynb`
Trains the model to ask concise, relevant medical follow-up questions during patient intake.
- **Output:** Saves the `follow_up_adapter` to your Google Drive (`unsloth_training/follow_up_adapter`).

### 2. `finetune_diagnostic.ipynb`
Trains the model to produce a structured differential diagnosis from patient symptoms.
- **Datasets Used:** MedQA USMLE (clinical vignettes) and DDxPlus (probability-ranked differential structure).
- **Output:** Saves the `diagnostic_adapter` to your Google Drive (`unsloth_training/diagnostic_adapter`).

### 3. `medical_pipeline.ipynb`
The main orchestrator and inference pipeline. It brings everything together into a Gradio UI.
**Pipeline Flow:**
1. Patient Complaint
2. `[follow_up_adapter]` → Ask focused follow-up questions (up to 6 turns)
3. `[base LLM]` → Extract symptom summary for RAG search
4. `[PubMed + MedlinePlus]` → Retrieve relevant medical literature
5. `[diagnostic_adapter]` → Generate structured differential diagnosis
6. `[Gradio UI]` → Display the final assessment

## Prerequisites & Installation

1. **Environment:** These notebooks are configured for Google Colab with a T4 GPU (or better). 
2. **Google Drive:** The notebooks mount Google Drive to save and load models to `/content/drive/MyDrive/unsloth_training/`.
3. **Dependencies:** The notebooks automatically install required libraries via `pip`:
   - `unsloth[colab-new]`
   - `peft`, `accelerate`, `bitsandbytes`, `trl`
   - `datasets`, `huggingface_hub`
   - `gradio`, `httpx`

## Usage Instructions

1. **Train the Adapters:** 
   Run both `finetune_followup.ipynb` and `finetune_diagnostic.ipynb` top-to-bottom. These will train the LoRA adapters and save them to your mounted Google Drive.
2. **Launch the Agent:**
   Open `medical_pipeline.ipynb` and run all cells. The final cell will generate a public Gradio URL.
3. **Interact:**
   Click the Gradio link to open the UI. Describe your symptoms, answer the follow-up questions, and click **Generate Diagnosis Now** to see the pipeline in action.

## Model Details

- **Base Model:** `unsloth/llama-2-7b-bnb-4bit`
- **Adapters:** Task-specific LoRA adapters (Rank=16, Alpha=16).
- **Context Length:** 2048 tokens.
