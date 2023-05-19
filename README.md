# Grade Marker AI

This is an AI tool for grading assignments based on given guidance. It uses the Longformer model, a transformer-based model that can handle long texts, from the Hugging Face Transformers library. 

## How to use

Firstly, you need to have Python 3.7 or later installed on your system. 

1. Clone this repository.
2. Install the necessary Python packages. You can do this by running `pip install -r requirements.txt`.
3. Place your assignment guidance in a file called 'guidance.txt' and your assignments in a folder called 'assignments'. The assignments should be in '.txt' format and their corresponding grades should be in a file with the same name, appended with '_grade.txt'. For example, for an assignment 'assignment1.txt', its grade should be in 'assignment1_grade.txt'.
4. Run `python train.py` to train the model. This will create a file called 'model.pt', which is the saved trained model.
5. To test the model on a new assignment, place the assignment in a file called 'test_assignment.txt' and run `python test.py`. This will print out the grade the model predicts for the assignment.

Note: The model is currently configured to handle assignments of up to 8,000 words in length. If your assignments are longer than this, you may need to modify the `split_into_chunks` method in `train.py` and `test.py`.

## Customization

If you want to change the model, tokenizer, or any other aspects of the training/testing process, you can do so in `train.py` and `test.py`.

Enjoy grading assignments with AI!
