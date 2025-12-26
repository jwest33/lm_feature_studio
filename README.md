<div align="center">
<h1>LM Feature Studio</h1>

<p>Local web interface for exploring Sparse Autoencoder (SAE) features on Gemma models.</p>

<a href="example.jpg"><img src="example.jpg" alt="example" width="500"></a>
</div>

## Features

1. **Activation Extraction**: Forward hooks capture residual stream activations at specified layers during inference
2. **SAE Encoding**: Activations pass through a JumpReLU sparse autoencoder, producing a sparse vector where each non-zero element corresponds to an interpretable feature
3. **Feature Lookup**: Query [Neuronpedia](https://www.neuronpedia.org/), a community database of SAE feature interpretations, to retrieve human-readable explanations, example activations, and token associations for discovered features
4. **Feature Analysis**: Compare feature activations across prompts to identify which features correlate with specific behaviors (e.g., refusal)
5. **Model Updates**: Modify generation by adding scaled SAE decoder vectors back into the residual stream, amplifying or suppressing specific features

## Quick Start

### 1. Install Dependencies (recommended to check first as this includes torch)

```bash
pip install -r requirements.txt
```

### 2. Configure Your Model

Copy the example environment file and edit it:

```bash
cp .env.example .env
```

Set `MODEL_PATH` to either:
- **HuggingFace ID** (easiest): `google/gemma-3-4b-it` (requires `huggingface-cli login`)
- **Local path**: `/path/to/your/gemma-model` or `D:\models\gemma-3-4b-it`

### 3. Set Up Neuronpedia API Key

To fetch feature explanations, get your API key from [Neuronpedia](https://www.neuronpedia.org/):

1. Create an account at neuronpedia.org
2. Go to your account settings to find your username & API key
3. Add to your `.env` file:
```
NEURONPEDIA_USER_NAME=your_username_here
NEURONPEDIA_API_KEY=your_api_key_here
```

### 4. Run the App

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## Configuration

Edit `.env` or `sae/config.py` to customize:

| Setting | Description |
|---------|-------------|
| `MODEL_PATH` | Path to Gemma model (local or HuggingFace ID) |
| `BASE_MODEL` | Model size: `"4b"` or `"12b"` (in `sae/config.py`) |
| `NEURONPEDIA_API_KEY` | API key for feature explanations (from neuronpedia.org) |

## Usage Tips

- **First request is slow** - Models load lazily on first use
- **Pre-load models** (optional): `PRELOAD_MODELS=1 python app.py`
- **GPU recommended** - CPU inference is very slow for larger models

## Sources

- [Neuronpedia](https://www.neuronpedia.org/) - Feature explanations and visualization data
- [GemmaScope](https://huggingface.co/google/gemma-scope-2-4b-it) - SAE weights by Google DeepMind
- [Gemma](https://ai.google.dev/gemma) - Base language models by Google
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) - Model loading infrastructure
