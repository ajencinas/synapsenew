# Pretraining corpora downloads

Standalone notebooks for the public corpora used by the Synapse pretraining mix.
Each notebook downloads and preprocesses one source into the shared workspace:

```
$SYNAPSE_DIR/datasets_pretrain/
  data_c4/
  data_code/
  data_finemath/
  data_fineweb/
  data_reasoning/
  data_redpajama/
  data_wikipedia/
```

| Notebook | Source |
|---|---|
| `download_c4.ipynb` | C4 / Common Crawl |
| `download_wikipedia.ipynb` | Wikipedia |
| `download_redpajama.ipynb` | RedPajama |
| `download_code.ipynb` | code corpora |
| `download_finemath.ipynb` | FineMath |
| `download_fineweb.ipynb` | FineWeb |
| `download_reasoning.ipynb` | reasoning traces |

After downloading, run the tokenizer pipeline in `tokenize/` to produce uint16
shards. The generated corpora and shards are intentionally absent from Git and
live on Google Drive under `gdrive:synapse/` (see the root README).
