"""Constantes partagées (modèle, entraînement, post-traitement)."""

IMAGE_HEIGHT = 120
IMAGE_WIDTH = 160

# Stride de la grille de sortie. 8 -> grille 15x20 (grand champ récepteur, recommandé).
# Mettre 4 pour revenir à une grille 30x40 (champ récepteur plus faible).
STRIDE = 8
MID_LAYERS = 6        # nb de conv 3x3 stride 1 à la résolution de sortie
MID_CHANNELS = 64

GRID_H = IMAGE_HEIGHT // STRIDE
GRID_W = IMAGE_WIDTH // STRIDE

# Boîtes plus petites que ça : pas de cible positive, mais zone "ignorée" dans la perte de confiance
MIN_BOX_W = 8.0
MIN_BOX_H = 14.0

# Initialisation de la tête
LTRB_INIT = 12.0 / STRIDE          # ~ distance moyenne centre->bord (en unités de stride)
OBJ_PRIOR = 0.01                   # proba initiale (prior focal)
