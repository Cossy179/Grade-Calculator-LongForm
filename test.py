import torch
from transformers import LongformerForSequenceClassification, LongformerTokenizerFast, LongformerConfig

def split_into_chunks(text, tokenizer, max_length):
    chunks = []
    for i in range(0, len(text), max_length):
        chunk = text[i:i + max_length]
        encoding = tokenizer.encode_plus(chunk, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
        chunks.append(encoding)
    return chunks

def test(model, tokenizer, brief, assignment, device):
    chunks = split_into_chunks(brief + assignment, tokenizer, 4096)
    all_logits = []
    for chunk in chunks:
        input_ids = chunk['input_ids'].to(device)
        attention_mask = chunk['attention_mask'].to(device)
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1  # Set global attention on the first token (CLS).
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
        all_logits.append(outputs.logits)
    logits = torch.cat(all_logits, dim=1)
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    grade = torch.argmax(probabilities).item()
    return grade


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = LongformerConfig.from_pretrained('allenai/longformer-base-4096', num_labels=4)
    model = LongformerForSequenceClassification(config)
    model.load_state_dict(torch.load('model.pt'))
    model.to(device)
    tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')

    with open('guidance.txt', 'r', encoding='utf-8') as f:
        brief = f.read()

    with open('test_assignment.txt', 'r', encoding='utf-8') as f:
        assignment = f.read()

    grade = test(model, tokenizer, brief, assignment, device)
    print(['U', 'Pass', 'Merit', 'Distinction'][grade])

if __name__ == '__main__':
    main()
