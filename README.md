# 🤖 BERT-based English-Spanish Translation Model

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)](https://huggingface.co/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

## 📋 Project Overview
This project implements an advanced neural machine translation system using BERT (Bidirectional Encoder Representations from Transformers) for English to Spanish translation. The model leverages a sophisticated encoder-decoder architecture with BERT-small as both the encoder and decoder components, demonstrating the powerful adaptation of BERT for sequence-to-sequence tasks.

<p align="center">
  <img src="https://media2.dev.to/dynamic/image/width=1000,height=420,fit=cover,gravity=auto,format=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fnt9g5x8sfvptbnu3wb1t.jpg" alt="BERT Translation Architecture">
</p>

## ✨ Key Features
- 🔄 Implements an encoder-decoder architecture using BERT-small
- 🌐 Handles translation between English and Spanish
- 🤗 Uses the Hugging Face Transformers library
- 🔍 Includes data preprocessing and tokenization
- 🔦 Implements beam search for better translation quality
- ⚡ Supports batched inference for efficient processing

## 🛠️ Technical Details

### Model Architecture
- **Encoder**: `prajjwal1/bert-small`
- **Decoder**: BERT-small with cross-attention layers
- **Tokenizer**: BertTokenizer
- **Max Sequence Length**: 128 tokens

### Training Parameters
| Parameter | Value |
|-----------|-------|
| Batch Size | 8 |
| Learning Rate | 5e-5 |
| Weight Decay | 0.01 |
| Epochs | 1 |

### Dataset
The model is trained on the `loresiensis/corpus-en-es` dataset from Hugging Face:
- 📚 Training set: 9,439 sentence pairs
- 🧪 Test set: 1,049 sentence pairs

## 📦 Requirements
```bash
pip install torch transformers datasets
```

## 💻 Usage

### Training the Model
```python
from transformers import (
    BertTokenizer,
    EncoderDecoderModel,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-small")
model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    "prajjwal1/bert-small", 
    "prajjwal1/bert-small"
)

# Configure training arguments
training_args = TrainingArguments(
    output_dir="./bert-small-translation_folder",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.01
)

# Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
)
trainer.train()
```

### Translation Example
```python
def translate(text, max_length=128, num_beams=5):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    
    generated_ids = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        decoder_start_token_id=tokenizer.cls_token_id,
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True,
    )
    
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
```

## 📊 Model Performance
The model achieves the following metrics after training:
- 📉 Training Loss: 4.04
- 📈 Validation Loss: 3.67

## 🚀 Future Improvements
- 🔄 Increase training epochs for better convergence
- 📈 Experiment with larger BERT models
- 🔍 Implement better text preprocessing
- 📊 Add data augmentation techniques
- ⚡ Implement more sophisticated beam search parameters
- 📝 Add evaluation metrics (BLEU, ROUGE)

## 📄 License

MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## 🙏 Acknowledgments
- 🤗 Hugging Face Transformers library
- 🔥 BERT-small model by prajjwal1
- 📚 loresiensis for the English-Spanish corpus

## 📫 Contact
[Add your contact information here]

---
<p align="center">
Made with ❤️ using PyTorch and Hugging Face
</p>
