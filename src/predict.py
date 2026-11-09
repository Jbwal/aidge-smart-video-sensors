import sys
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import PersonDetectorCNN

def predict(image_path):
    device = torch.device("cpu")
    
    #charger modèle
    model = PersonDetectorCNN()
    model.load_state_dict(torch.load("./models/person_detector_fp32.pth", map_location=device))
    model.eval()

    image = Image.open(image_path)

    #charger et préparer l'image
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probability = torch.sigmoid(output).item()

    #affiche resultat
    print(f"{os.path.basename(image_path)}")
    print(f"confiance : {probability * 100:.2f}%")
    if probability >= 0.5:
        print("PERSONNE DÉTECTÉE")
    else:
        print("PAS DE PERSONNE")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        print("veuillez renseigner le path d'une image")