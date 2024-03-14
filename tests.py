import matplotlib.pyplot as plt
import pandas as pd

# Lire les résultats
results = pd.read_csv("results.txt")

# Générer le graphique
plt.figure(figsize=(8, 6))
plt.plot(results['ClientsMalveillants'], results['Précision'], marker='o', linestyle='-', color='b')
plt.title("Impact des Clients Malveillants sur la Précision du Modèle")
plt.xlabel("Nombre de Clients Malveillants")
plt.ylabel("Précision du Modèle")
plt.grid(True)
plt.savefig("precision_impact.png")
plt.show()
