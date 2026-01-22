import torch
from pathlib import Path

def main():
    MODEL_PATH = r"D:\DOCUMENTS\RAIN\AIML\Second_semester\Project\models\best_model.pth"
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    
    if 'class_names' in checkpoint:
        print("Classes found:")
        for i, name in enumerate(checkpoint['class_names']):
            print(f"{i}: {name}")
    else:
        print("Class names not found in checkpoint.")
        if 'model_state_dict' in checkpoint:
            num_classes = checkpoint['model_state_dict']['fc3.bias'].shape[0]
            print(f"Number of classes: {num_classes}")

if __name__ == "__main__":
    main()
