# Frozen vs. Fine-Tuned Transformers for Indonesian Short-Answer Essay Scoring

Code for comparing two paradigms for automated essay scoring (AES) on short Indonesian history essays:

* **FA-FT** (Feature-Augmented Fine-Tuning): the transformer encoder is fine-tuned end-to-end with a regression head.
* **FA-ML** (Feature-Augmented Machine Learning): the encoder is frozen, embeddings are extracted once, and an XGBoost regressor is trained.

Both paradigms use the same six handcrafted linguistic features (type-token ratio, word count, noun count, verb count, sentence count, average word length) and are evaluated with and without them (ablation).

Encoders: `denaya/indoSBERT-large`, `indobenchmark/indobert-base-p2`, `cahya/distilbert-base-indonesian`.

## Repository layout

|Path|Contents|
|-|-|
|`notebooks/01\_FA-FT\_with\_linguistic\_features.ipynb`|Fine-tuning, embeddings + linguistic features (3 encoders)|
|`notebooks/02\_FA-FT\_embedding\_only.ipynb`|Fine-tuning, embeddings only (ablation)|
|`notebooks/03\_FA-ML\_with\_and\_without\_features.ipynb`|Frozen embeddings + XGBoost, with and without features|
|`notebooks/04\_FA-ML\_CPU\_only.ipynb`|FA-ML timing on CPU only|
|`scripts/faml\_from\_saved\_embeddings.py`|Re-runs FA-ML from saved embeddings (no GPU needed)|
|`scripts/faml\_multiseed.py`|FA-ML repeated over 5 XGBoost seeds|

## Evaluation protocol

* **Outer CV:** 10-fold `KFold(shuffle=True, random\_state=42)`. Folds are identical across all notebooks, so per-fold results can be compared pairwise (e.g., Wilcoxon signed-rank test).
* **No test-fold leakage:** inside each outer fold, 10% of the training portion (stratified by score band, scores ≤ 5 merged) is held out **only for early stopping** (`inner\_split()`). The outer test fold is used once, for final evaluation. The inner split uses the same seed in every notebook.
* **Feature scaling (FA-FT):** linguistic features are z-scored using statistics from the training portion only.
* **Metrics:** QWK (predictions rounded and clipped to the 2–10 score range) and MSE.
* **Efficiency:**

  * FA-FT peak GPU memory = `torch.cuda.max\_memory\_allocated()`, reset at the start of every fold.
  * GPU memory is freed between folds (`gc.collect()` + `torch.cuda.empty\_cache()`); without this, memory accumulated by \~2.6 GB per fold.

## Data

The dataset (1,000 student responses to 10 history prompts, scored 0–10 by a teacher) is **not included**. It contains responses from secondary-school students. Contact the authors for access under appropriate ethical conditions.

The notebooks expect these files in `DATA\_DIR`:

```
scores.pkl                     # numpy array (1000,), float
question\_ids.pkl               # numpy array (1000,), int, prompt id 1-10
preprocessed\_texts.pkl         # list\[str], lowercased (IndoBERT, DistilBERT)
preprocessed\_sbert\_texts.pkl   # list\[str], case-preserved (IndoSBERT)
```

## Running on Google Colab

1. Upload the data files to a Google Drive folder and set `DATA\_DIR` / `OUTPUT\_PATH` in the paths cell.
2. Set **Runtime → Change runtime type → A100 GPU** for the FA-FT and FA-ML GPU notebooks.
3. **Runtime → Run all.** If Colab asks to restart the session after the `pip install` cell (because `numpy` was replaced), restart and run all again.
4. Each notebook writes per-fold metrics (`cv\_results\_\*.csv`), per-essay predictions (`all\_predictions\_\*.csv`) and a `experiment\_results\*.joblib` to `OUTPUT\_PATH`.

### Known environment issues (Colab, October 2026)

* `nlp-id` pins an old `huggingface\_hub` that breaks recent `transformers` (`cannot import name 'ResolvedRevision'`). The install cell upgrades `huggingface\_hub` again afterwards.
* Recent NLTK versions refuse to download through Colab's proxy unless `NLTK\_ALLOW\_PROXIED\_URLOPEN=1` is set. The NLTK cell sets it.
* The IndoSBERT FA-FT notebook saves each fold's model (\~1.3 GB, 13 GB in total). Remove the `torch.save(...)` line if you do not need the checkpoints.

## Notes on model details

* **FA-FT IndoSBERT** uses the 256-dimensional output of the Dense (tanh) projection.
* **FA-ML IndoSBERT** uses the mean-pooled 1,024-dimensional backbone embedding (without the Dense projection).

## License

Add a license before publishing (e.g., MIT for code).

