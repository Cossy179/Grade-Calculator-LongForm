# test.py
from transformers import LongformerForSequenceClassification, LongformerTokenizerFast, LongformerConfig
import torch

def test(model, tokenizer, assignment, device):
    chunks = [assignment[i:i+4000] for i in range(0, len(assignment), 4000)]
    encodings = [tokenizer(chunk, truncation=True, padding='longest', max_length=4096, return_tensors='pt') 
                 for chunk in chunks]
    probabilities = []
    with torch.no_grad():
        for encoding in encodings:
            input_ids = encoding['input_ids'].squeeze().to(device)
            attention_mask = encoding['attention_mask'].squeeze().to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities.append(torch.nn.functional.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy())
    probabilities = sum(probabilities) / len(probabilities)
    grade = torch.argmax(torch.tensor(probabilities)).item()
    return grade


def main():
    # Define the model configuration
    config = LongformerConfig.from_pretrained('allenai/longformer-base-4096', num_labels=4)

    # Define the model architecture
    model = LongformerForSequenceClassification(config)

    # Load the weights
    model.load_state_dict(torch.load('LongFormer_model.pt'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')

    with open('guidance.txt', 'r', encoding='utf-8') as f:
        brief = f.read()

    with open('test_assignment.txt', 'r', encoding='utf-8') as f:
        assignment = f.read()

    grade = test(model, tokenizer, brief + ' ' + assignment, device)
    print(['U', 'Pass', 'Merit', 'Distinction'][grade])

if __name__ == '__main__':
    main()
