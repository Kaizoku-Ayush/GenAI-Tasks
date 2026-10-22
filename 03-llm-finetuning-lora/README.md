# 03 · Fine-Tuning a Small LLM with LoRA

Fine-tunes the pre-trained **TinyLlama-1.1B-Chat** model on a custom question–answer dataset using **LoRA (Low-Rank Adaptation)**, then compares the fine-tuned model's answers with the original model's.

📓 Notebook: [`llm_finetuning_lora.ipynb`](llm_finetuning_lora.ipynb)

## Workflow

```
Custom Q&A dataset ──► Format "Question: … \nAnswer: …" ──► Tokenize
                   ──► Add LoRA adapters ──► Train (HF Trainer)
                   ──► Compare original vs fine-tuned answers
```

## Details

| Item | Value |
|---|---|
| Base model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (float16, GPU) |
| Dataset | 10 Q&A pairs on AI concepts (AI, ML, DL, LLM, NLP, learning types, neural networks, generative AI) |
| Tokenization | Max length 256, `input_ids` + `attention_mask`, pad token = EOS |
| Data collator | `DataCollatorForLanguageModeling(mlm=False)`, i.e. causal LM |

### LoRA configuration

| Parameter | Value |
|---|---|
| Rank `r` | 8 |
| `lora_alpha` | 16 |
| Target modules | `q_proj`, `v_proj` |
| Dropout | 0.05 |
| Trainable parameters | **1,126,400 / 1,101,174,784 (0.10%)** |

### Training

| Parameter | Value |
|---|---|
| Epochs | 5 |
| Batch size | 2 × gradient accumulation 2 (effective batch size 4) |
| Learning rate | 2e-4 |
| Precision | fp16 mixed precision |
| Training steps | 15 |
| Runtime | ~10 s |
| Average training loss | 1.818 |

## Results

- **Original model:** gives long, general answers that drift off-topic.
- **Fine-tuned model:** gives shorter, definition-style answers that follow the trained `Question / Answer` format, including on unseen questions such as *"Explain how computers learn from data."*
- A keyword-match score on 5 questions compares both models in a bar chart.

**Observed limitations:**
- The model sometimes continues with extra generated questions, because the training texts end without an EOS token.
- "What is an LLM?" can still be read as *Master of Laws*. Ten examples are too few to fully override pre-trained knowledge.
- More data, EOS tokens and the model's chat template would improve the results.

## Requirements

```bash
pip install transformers datasets peft accelerate
```
A GPU runtime (for example a Colab T4) is recommended.

## How to Run

1. Open the notebook in Google Colab and select a GPU runtime.
2. Run all cells in order.
3. The LoRA adapter is saved to `./fine_tuned_lora`.
