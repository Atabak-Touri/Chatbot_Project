# THD Assistant Chatbot

This project is a prototype retrieval-augmented chatbot for Deggendorf Institute of Technology (DIT/THD). It searches a local collection of university web content, ranks relevant passages with sentence embeddings, generates an answer with Microsoft's Phi-2 model, and serves the result through a Gradio interface.

## How it works

1. Loads documents from `thd_content.json`.
2. Encodes document content with `all-MiniLM-L6-v2`.
3. Finds the most relevant documents using cosine similarity.
4. Sends the retrieved context and the user's question to `microsoft/phi-2`.
5. Returns the generated answer with source links in a Gradio chat interface.

Generated files:

- `thd_embeddings.npy` contains the cached document embeddings.
- `thd_metadata.json` contains document titles and URLs.

## Requirements

- Python 3.10 or newer
- A GPU is recommended, but CPU execution is supported
- Internet access on first run to download the model and Hugging Face assets
- Enough disk space for the Phi-2 model and Python dependencies

The script uses TensorFlow, sentence-transformers, scikit-learn, NumPy, Transformers, Accelerate, PyTorch, and Gradio.

## Setup and run

The source is generated from a Google Colab notebook and is intended to run in Colab or Jupyter. Colab provides the easiest setup because it supports notebook shell commands and CUDA-backed execution.

1. Open `chatbot_phi2.py` in Google Colab or upload it and `thd_content.json` to the notebook environment.
2. Run the notebook cells in order. The script installs missing packages with `pip` commands.
3. Wait while the MiniLM embeddings and Phi-2 model are downloaded and loaded.
4. Use the public Gradio link printed by `Interface.launch(share=True)`.

For a normal local Python environment, install the dependencies and remove or replace the notebook-only `!pip install ...` lines before running:

```bash
pip install tensorflow sentence-transformers scikit-learn numpy transformers accelerate torch gradio
python chatbot_phi2.py
```

TensorFlow may be optional depending on how you run the script; the current file imports it for creating a demonstration `tf.data.Dataset`.

## Data

`thd_content.json` contains an array of objects with:

- `title`: the document title
- `content`: the text used for retrieval
- `url`: the source page

Replace or update this file to add local knowledge sources. Keep the JSON structure unchanged.

## Notes

- Answers are generated from retrieved local content and may still contain hallucinations or outdated information.
- The public source links should be checked for the latest university policies and deadlines.
- Phi-2 is a gated or download-managed model in some Hugging Face environments. Sign in and accept the model's terms of use if prompted.
- Gradio's `share=True` creates a temporary public link; use it only with content you are comfortable sharing.
