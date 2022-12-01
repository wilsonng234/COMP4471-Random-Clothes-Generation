import torch
import os

def save_model(model, folder_dir, file_name, file_extension=".pt"):
    print("=================== Saving Model ===================")
    if not os.path.exists(folder_dir):
        os.mkdir(folder_dir)

    file_name += file_extension
    torch.save(model.state_dict(), os.path.join(folder_dir, file_name))
    print("=================== Finished Saving Model ===================")

def load_model(model, folder_dir, file_name, file_extension=".pt"):
    if not os.path.exists(folder_dir):
        raise ValueError("Checkpoint not found")
    
    file_name += file_extension
    state_dict = torch.load(os.path.join(folder_dir, file_name))
    model.load_state_dict(state_dict)
    model.eval()

def write_history(summary_writer, name, history):
    print("=================== Writing Summary ===================")
    for metric in history:
        summary_writer.add_scaler(name, metric)

    summary_writer.flush()
    print("=================== Finished Writing Model ===================")
    