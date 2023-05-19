import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification

class AssignmentDataset(Dataset):
    def __init__(self, assignments, tokenizer, max_length=4096):
        self.assignments = assignments
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.assignments)

    def __getitem__(self, idx):
        assignment = self.assignments[idx]
        assignment_chunks = self.split_into_chunks(assignment['brief'] + ' ' + assignment['assignment'])
        
        encodings_and_labels = []
        for chunk in assignment_chunks:
            encodings = self.tokenizer(chunk,
                                       truncation=True,
                                       max_length=self.max_length,
                                       padding='max_length',
                                       return_tensors='pt')
            encodings_and_labels.append({
                'input_ids': encodings.input_ids.squeeze(),
                'attention_mask': encodings.attention_mask.squeeze(),
                'labels': torch.tensor([assignment['grade']])
            })
        return encodings_and_labels

    def split_into_chunks(self, text):
        words = text.split(' ')
        chunks = [' '.join(words[i:i + 4000]) for i in range(0, len(words), 4000)]
        return chunks

def train(model, data_loader, optimizer, device):
    model = model.train()

    for batch in data_loader:
        for item in batch:
            # Split the batch into its components.
            input_ids = item['input_ids'].to(device)
            attention_mask = item['attention_mask'].to(device)
            labels = item['labels'].to(device)

            # Create the global attention mask and move it to the device.
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1  # Global attention on the first token (CLS).
            global_attention_mask = global_attention_mask.to(device)

            # Reset the optimizer.
            optimizer.zero_grad()

            # Forward pass through the model.
            output = model(input_ids=input_ids, attention_mask=attention_mask, 
                        global_attention_mask=global_attention_mask, labels=labels)
            # Compute the loss.
            loss = output.loss

            # Backward pass (compute gradients).
            loss.backward()

            # Update the parameters.
            optimizer.step()

def load_data(guidance_file, assignment_dir):
    with open(guidance_file, 'r', encoding='utf-8') as f:
        guidance = f.read()

    assignments = []
    assignment_files = [f for f in os.listdir(assignment_dir) if not f.endswith('_grade.txt')]
    for assignment_file in assignment_files:
        with open(os.path.join(assignment_dir, assignment_file), 'r', encoding='utf-8') as f:
            assignment = f.read()
        
        grade_file = assignment_file.replace('.txt', '_grade.txt')
        with open(os.path.join(assignment_dir, grade_file), 'r', encoding='utf-8') as f:
            grade_text = f.read().strip()
            if grade_text == 'U':
                grade = 0
            elif grade_text == 'Pass':
                grade = 1
            elif grade_text == 'Merit':
                grade = 2
            elif grade_text == 'Distinction':
                grade = 3
            else:
                raise ValueError(f'Invalid grade: {grade_text}')

        assignments.append({
            "brief": guidance,
            "assignment": assignment,
            "grade": grade
        })
    return assignments

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    assignments = load_data('guidance.txt', 'assignments')

    tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')

    dataset = AssignmentDataset(assignments, tokenizer)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', num_labels=4)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(10):
        print(f'Starting epoch {epoch + 1}')
        train(model, data_loader, optimizer, device)

    torch.save(model.state_dict(), f'LongFormer_model.pt')

if __name__ == '__main__':
    main()
