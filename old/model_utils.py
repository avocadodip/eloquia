import torch

def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)

def load_model(model, file_name):
    model.load_state_dict(torch.load(file_name))
    model.eval()
    return model

def calculate_accuracy(y_true, y_pred):
    predicted = torch.argmax(y_pred, 1)
    correct = (predicted == y_true).float()
    accuracy = correct.sum() / len(correct)
    return accuracy.item()
