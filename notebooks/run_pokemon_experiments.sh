#!/bin/bash
# Script pour lancer les expériences Pokemon (Salameche -> Strawberry)

echo "=========================================="
echo "Lancement des expériences Pokemon"
echo "Paire: Salameche -> Strawberry"
echo "=========================================="

# Activer l'environnement Python si nécessaire
# source /path/to/venv/bin/activate

# Lancer les expériences
cd /Data/janis.aiad/geodata/notebooks
python experiments_pokemon_large_scale.py

echo "=========================================="
echo "Expériences terminées"
echo "=========================================="

