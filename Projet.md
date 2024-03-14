# Projet : Attaque et Défense dans l'apprentissage fédéré
Date de rendu : 14/03/2024

Format : Rapport

N'hésitez pas à utiliser des graphiques pour illustrer vos résultats obtenus !
## Testez directement le code
* Dans Terminal 1:
  ```bash
  > python server.py --rounds 10
  ```
* Dans Terminal 2:
  ```bash
  > python client.py --node-id 0 
  ```
* Dans Terminal 3:
  ```bash
  > python client_mal.py --node-id 1 
  ```
Question : Quel est le retour du client malveillant au serveur ? Comparez avec le cas où
les deux clients sont honnêtes. Quelle est votre observation sur le modèle obtenu ?


## Attaque active : 
### Attaque active :  Inversion d'étiquettes (Binôme I) 
1. Dans le fichier client_mal.py qui présente les clients malveillants,
implémentez l'attaque "inversion d'étiquettes" dans la fonction train. Par exemple, pour CIFAR10, tous les "labels" seront décalés d'un.

### Attaque active :  Altération du modèle (Binôme II)
1. Dans le fichier client_mal.py qui présente les clients malveillants,
implémentez l'attaque "altération du modèle" dans la fonction train. Au lieu d'appliquer la descente de gradient,
le client appliquera une montée de gradient.


**Chacun répond les questions suivantes sur leur attaque implémentée** :
1. Testez votre code sur un scénario avec quatre clients. Augmentez le nombre de clients malveillants de 1 à 3.
Quelle est votre observation sur le modèle obtenu et pourquoi ? Affichez un graphique où l'axe des abscisses représente le nombre de clients malveillants et l'axe des ordonnées représente la précision du modèle final.

2. Retestez le scénario précédent, mais cette fois avec l'option "--data_split non_iid_class". Cette attaque est-elle plus efficace
dans cette situation et pourquoi ? Utilisez des graphiques pour montrer vos résultats.

3. Testez des scénarios plus réalistes avec 10 clients (sur NEF). Vous pouvez utiliser le script run.sh pour lancer les clients et le serveur.
Attention, vous devez modifier le script pour ajouter les clients malveillants. Étudiez les cas où il y a 1, 3, 5, 7 clients malveillants.
Utilisez des graphiques pour montrer vos résultats.

### Comparaison des performances de ces deux attaques (Ensemble)
1. Quelle attaque est plus efficace et dans quel scénario ? Pourquoi ?

## Défense 
Appliquer la défense "Médiane par coordonnées" et "Moyenne tronquée" sur le serveur,
en utilisant la stratégie fournie par flwr: class [`FedMedian`](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedmedian.py)
and class [`FedTrimmedAvg`](https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedtrimmedavg.py). 

Répondez aux questions suivantes pour l'attaque d'inversion d'étiquettes (Binôme I) et pour l'attaque d'altération du modèle (Binôme II) :
N'hésitez pas à utiliser des graphiques pour montrer les résultats !

1. Quelle défense est plus efficace contre l'attaque ?
2. À partir de combien de clients malveillants la défense échoue-t-elle totalement ?
3. Comparez les cas de "--data_split iid" et "--data_split non_iid_class". La défense est-elle plus efficace
    dans quelle situation et pourquoi ?


*Remarque finale : Assurez-vous d'inclure votre nom pour l'attaque que vous avez choisi de travailler dans le rapport.*
