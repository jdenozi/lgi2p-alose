# DOCUMENTATION OUTILS DETECTION ALOSE
# LGI2P-ALOSE
-----------
fr
-----------
Afin d'assurer le suivi de la population de poisson migrateurs, l'Association Migrateur Rhône-Méditerrannée met en place un suivi de l'Alose qui utilise la particularité de l'acte de reproduction de ceux-ci. Les aloses se manifestent en surface en effectuant des déplacements circulaires et en frappant l'eau de leur nageoire afin de créer un tourbillon qui favorisera la fécondation des oeufs.Cette phase appelée "bull" peut être particulièrement bruyante et peut durer jusqu'à 10 seconde ce qui permet sa détection automatique.

Le but de ce projet est d'alors d'optimiser les méthodes de détections de sons d'Aloses à la surface de l'eau.

en
------------
In order to monitor the migratory fish population, the Association
 Migrateur Rhône-Méditerrannée is setting up a monitoring of the Alose using the particularity of the act of reproduction of the Alose. The Alose manifests itself on the surface by moving in circles and hitting the water with their fins to create a whirlpool that will increase the fertilization of the eggs.
This phase called "bull" can be particularly noisy.

The goal of this project is to optimize the methods of detection of its
s of shad on the water surface.

## Getting Started
--------------------
These instructions will get you a copy of the project up and running on your local machine for development and testing purpose. 

### Prerequisites
- Panda
- librosa
- numpy
- matplotlib 
- sphinx
- tensorflow

# Commande de lancement

Preprocessing
--
Découpage des audios:
```
python3 lgi2p.py -p "path_of_audio"
```

Création des spectrogrammes:

- Permet de créer les spectrogrammes sous formes de png et sous format txt (argument t comme true)
    ```
    python3 lgi2p.py -s t
    ```
- Permet de créer les spectrogrammes sous format txt (argument f comme false)
    ```
    python3 lgi2p.py -s f
    ```
Modélisation
--

---
**Apprentissage du modèle CNN : **

- Chargement des données + utilisation d'un modèle CNN (argument b comme basique)
    ```
    python3 lgi2p -cnn b
    ```
    
- Chargement des données + under sampling + utilisation d'un modèle CNN

    ```
    python3 lgi2p -cnn us
    ```
    
- Chargement des données + over sampling + utilisation d'un modèle CNN
    ```
    python3 lgi2p -cnn os
   ```
   
- Chargement des données + under sampling +over sampling + utilisation d'un modèle CNN
    ```
    python3 lgi2p -cnn uos
   ```
   
- Chargement des données "on fly" à partir des images de spectrogrammes + modèle CNN (argument of comme on fly)
    ```
  python3 lgi2p -cnn of 
  
   ```
   
- Permet de sortir la matrice de confusion et le rapport du modèle
    ```
  python3 lgi2p -cnn cm
   ```
   
- Permet de prédire un audio et vérifier grâce à l'annotation si les prédictions sont bonnes
    ```
  python3 lgi2p -cnnt "path_of_audio"
  
   ```
----
Apprentissage du modèle VGG16:

- Chargement des données + utilisation d'un modèle VGG (argument b comme basique)
    ```
    python3 lgi2p -vgg b
    ```
    
- Chargement des données + under sampling + utilisation d'un modèle VGG

    ```
    python3 lgi2p -vgg us
    ```
    
- Chargement des données + over sampling + utilisation d'un modèle VGG avec la dernière couche en apprentissage
    ```
    python3 lgi2p -vgg os
   ```
   
- Chargement des données + under sampling +over sampling + utilisation d'un modèle VGG avec la dernière couche en apprentissage
    ```
    python3 lgi2p -vgg uos
   ```
   
   
 - Chargement des données + utilisation d'un modèle VGG (argument ll comme last layer) avec la dernière couche en apprentissage
    ```
    python3 lgi2p -vgg ll
    ```
    
- Chargement des données + under sampling + utilisation d'un modèle VGG avec la dernière couche en apprentissage

    ```
    python3 lgi2p -vgg llus
    ```
    
- Chargement des données + over sampling + utilisation d'un modèle VGG avec la dernière couche en apprentissage
    ```
    python3 lgi2p -vgg llos
   ```
   
- Chargement des données + under sampling +over sampling + utilisation d'un modèle VGG avec la dernière couche en apprentissage
    ```
    python3 lgi2p -vgg lluos
   ```
 ---
 **Generation de donnée on fly:**
 
 Dans le dossier :  src/models/dataGen.py
 
A améliorer, rien d'automatique.
Les données d'entraînement et de test doivent se trouver dans 2 dossiers différents et doivent comporter un dossier Bulls et un dossiers non Bulls.

A chaque epoch de nouvelle images sont crées à partir du jeu de données.

Le script peut être lancé avec : 
``` 
python3 dataGen.py
```

Exemple d'utilisation : 
--
- Découper les audios
- Faire les spectrogrammes sans les png
- Faire un under sampling puis un CNN sur les données
- Sortir les données d'apprentissages
- Tester le modèle sur un fichier audio

```
python3 lgi2p -p '/media/data_bulls_audio'
python3 lgi2p -s f
python3 lgi2p -cnn us
python3 lgi2p -cnn cm
python3 lgi2p -cnnt '/media/15juin.wav'
```


## Resultats : 

** DataGenerator + CNN **





**CNN + Binary Crossentropy**:
Informations apprentissage: 

              precision    recall  f1-score   support

    No Bulls       0.96      1.00      0.98     18100
       Bulls       0.66      0.17      0.26       955


Informations test 15juin.wav:

True positive: 0.045454545454545456
True negative: 0.9671052631578947
False positive: 0.03289473684210526
False negative: 0.9545454545454546

              precision    recall  f1-score   support

       Bulls       0.93      0.97      0.95       304
    No Bulls       0.09      0.05      0.06        22

**CNN + under sampling**

Informations apprentissage:

              precision    recall  f1-score   support

    No Bulls       0.96      0.98      0.97     18100
       Bulls       0.31      0.18      0.23       955



Informations test 15juin.wav:

True positive0.0
True negative0.9703947368421053
False positive0.029605263157894735
False negative1.0

              precision    recall  f1-score   support

       Bulls       0.93      0.97      0.95       304
    No Bulls       0.00      0.00      0.00        22

**CNN + under sampling over sampling**

Informations apprentissage: 

              precision    recall  f1-score   support

    No Bulls       0.95      0.97      0.96     18100
       Bulls       0.15      0.10      0.12       955
       
Informations test 15juin.wav

True positive: 0.045454545454545456
True negative: 0.9210526315789473
False positive: 0.07894736842105263
False negative: 0.9545454545454546

              precision    recall  f1-score   support

       Bulls       0.93      0.92      0.93       304
    No Bulls       0.04      0.05      0.04        22

