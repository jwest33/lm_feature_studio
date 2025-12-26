# LM Feature Studio

Local web interface for exploring Sparse Autoencoder (SAE) features on Gemma models.

1. **Activation Extraction**: Forward hooks capture residual stream activations at specified layers during inference
2. **SAE Encoding**: Activations pass through a JumpReLU sparse autoencoder, producing a sparse vector where each non-zero element corresponds to an interpretable feature
3. **Feature Lookup**: Query [Neuronpedia](https://www.neuronpedia.org/), a community database of SAE feature interpretations, to retrieve human-readable explanations, example activations, and token associations for discovered features
4. **Feature Analysis**: Compare feature activations across prompts to identify which features correlate with specific behaviors (e.g., refusal)
5. **Model Updates**: Modify generation by adding scaled SAE decoder vectors back into the residual stream, amplifying or suppressing specific features

## Setup

```bash
pip install -r requirements.txt
```

Configure `sae/config.py`:
- Set `MODEL_PATH` to your local Gemma model or HuggingFace ID
- Adjust `BASE_MODEL` ("4b" or "12b") to match your model

Optional: Add `NEURONPEDIA_API_KEY` to `.env` for feature explanations.

## Usage

```bash
python app.py
```

Pre-load models to avoid first-request delay:
```bash
PRELOAD_MODELS=1 python app.py
```

## Credits

- [Neuronpedia](https://www.neuronpedia.org/) - Feature explanations and visualization data
- [GemmaScope](https://huggingface.co/google/gemma-scope-2-4b-it) - SAE weights by Google DeepMind
- [Gemma](https://ai.google.dev/gemma) - Base language models by Google
- [HuggingFace Transformers](https://huggingface.co/docs/transformers) - Model loading infrastructure
