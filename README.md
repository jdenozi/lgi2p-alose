# DOCUMENTATION OUTILS DETECTION ALOSE
# ALOSE
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
Jinja2                       2.10.1             
jsonpickle                   1.4.1              
jsonschema                   3.2.0              
Keras                        2.3.1              
Keras-Applications           1.0.8              
launchpadlib                 1.10.13            
librosa                      0.7.2              
Markdown                     3.2.2              
matplotlib                   3.2.1              
mlxtend                      0.17.3             
numba                        0.48.0             
numpy                        1.18.4             
pandas                       1.0.3              
Pillow                       7.0.0              
progressbar                  2.5                
scikit-learn                 0.23.1             
scipy                        1.4.1              
seaborn                      0.10.1             
setuptools                   45.2.0             
simplejson                   3.16.0             
SoundFile                    0.10.3.post1       
tb-nightly                   2.3.0a20200529     
tensorboard                  2.3.0              
tensorboard-plugin-wit       1.6.0.post3        
tensorflow                   2.3.0              
tensorflow-addons            0.11.1             
tensorflow-estimator         2.3.0              
terminado                    0.8.2              
tf-estimator-nightly         2.3.0.dev2020052901
tf-nightly-gpu               2.3.0.dev20200529  
Theano                       1.0.5              
# Commande de lancement

Preprocessing
--
Découpage des audios:
```
python3 alose.py.py -p "path_of_audio"
```

Création des spectrogrammes:

- Permet de créer les spectrogrammes sous formes de png et sous format txt (argument t comme true)
    ```
    python3 alose.py.py -s t
    ```
- Permet de créer les spectrogrammes sous format txt (argument f comme false)
    ```
    python3 alose.py.py -s f
    ```
Modélisation
--

---
**Apprentissage du modèle CNN : **

- Chargement des données + utilisation d'un modèle CNN (argument b comme basique)
    ```
    python3 alose.py -cnn b
    ```

- Chargement des données + under sampling + utilisation d'un modèle CNN

    ```
    python3 alose.py -cnn us
    ```

- Chargement des données + over sampling + utilisation d'un modèle CNN
    ```
    python3 alose.py -cnn os
   ```

- Chargement des données + under sampling +over sampling + utilisation d'un modèle CNN
    ```
    python3 alose.py -cnn uos
   ```

- Chargement des données "on fly" à partir des images de spectrogrammes + modèle CNN (argument of comme on fly)
    ```
  python3 alose.py -cnn of

   ```

- Permet de sortir la matrice de confusion et le rapport du modèle
    ```
  python3 alose.py -cnn cm
   ```

- Permet de prédire un audio et vérifier grâce à l'annotation si les prédictions sont bonnes
    ```
  python3 alose.py -cnnt "path_of_audio"

   ```
----
Apprentissage du modèle VGG16:

- Chargement des données + utilisation d'un modèle VGG (argument b comme basique)
    ```
    python3 alose.py -vgg b
    ```

- Chargement des données + under sampling + utilisation d'un modèle VGG

    ```
    python3 alose.py -vgg us
    ```

- Chargement des données + over sampling + utilisation d'un modèle VGG avec la dernière couche en apprentissage
    ```
    python3 alose.py -vgg os
   ```

- Chargement des données + under sampling +over sampling + utilisation d'un modèle VGG avec la dernière couche en apprentissage
    ```
    python3 alose.py -vgg uos
   ```


 - Chargement des données + utilisation d'un modèle VGG (argument ll comme last layer) avec la dernière couche en apprentissage
    ```
    python3 alose.py -vgg ll
    ```

- Chargement des données + under sampling + utilisation d'un modèle VGG avec la dernière couche en apprentissage

    ```
    python3 alose.py -vgg llus
    ```

- Chargement des données + over sampling + utilisation d'un modèle VGG avec la dernière couche en apprentissage
    ```
    python3 alose.py -vgg llos
   ```

- Chargement des données + under sampling +over sampling + utilisation d'un modèle VGG avec la dernière couche en apprentissage
    ```
    python3 alose.py -vgg lluos
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
python3 alose.py -p '/media/data_bulls_audio'
python3 alose.py -s f
python3 alose.py -cnn us
python3 alose.py -cnn cm
python3 alose.py -cnnt '/media/15juin.wav'
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

**CNN + Changement des poids avec nouvelles méthodes**

Informations sur les données d'entraînements:

True positive0.8659685863874346
True negative0.2574585635359116
False positive0.7425414364640884
False negative0.13403141361256546


              precision    recall  f1-score   support

    No Bulls       0.96      0.94      0.95     18100
       Bulls       0.14      0.19      0.16       955

Informations sur le fichier test 15juin.wav

True positive0.9545454545454546
True negative0.02631578947368421
False positive0.9736842105263158
False negative0.045454545454545456

              precision    recall  f1-score   support

    No Bulls       0.89      0.03      0.05       304
      iBulls       0.07      0.95      0.12        22
