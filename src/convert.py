import sys
import os
import cv2

def setup_test_dir():
    test_dir = "./data/test"
    os.makedirs(test_dir, exist_ok=True)
    return test_dir

def convert_image(input_path):

    test_dir = setup_test_dir()
    
    #charger l'image
    img = cv2.imread(input_path)
        
    print(f"image originale : {img.shape[1]}x{img.shape[0]}")

    #conversion
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #redimensionnement format QQVGA (160x120)
    target_width = 160
    target_height = 120
    img_resized = cv2.resize(img_gray, (target_width, target_height), interpolation=cv2.INTER_AREA)

    #sauvegarde
    file_name = os.path.basename(input_path)
    base_name, _ = os.path.splitext(file_name)
    output_path = os.path.join(test_dir, f"{base_name}_ok.png")
    
    cv2.imwrite(output_path, img_resized)
    
    print(f"nouvelle image : {target_width}x{target_height}")
    print("enregistré dans data/test")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        convert_image(sys.argv[1])
    else:
        print("veuillez renseigner le chemin d'une image à convertir")