


![GitHub Logo](/Images/accueille.png) 
# Strategy Trading

Nous avons dans notre porfeuille 3 actions (AMAZON, MICROSOFT et GOOGLE). 

Le But de ce projet est de prédire le prix de fermeture avec le modèle LSTM et d'implémenter une stratégie de trading afin de réaliser un gain. 

## Collecte data
Pour ce faire, nous allons collecter les données :
 
1. l'historique de prix des 10 dernières années, depuis l'API yahoo finance. 
2. les news sur nos 3 sociétés 


# Stockage des données dans Mongodb 

Nous allons stocker nos données dans Mongodb. 


## Modèle LSTM 

est une architecture de réseau neuronal récurrent artificiel (RNN) utilisée dans le domaine de l’apprentissage profond.
LSTM va nous permettre de faire la  prediction des prix fermermetures. 

## Modèle NLP 
Recueillir des news sur nos 3 sociétés en analysant le sentiment de chaque news. 

## Heroku Deployment

 [Cliquez ici](). 
 
## Web app
Nous avons utilisé Streamlit, pour visualiser l'historiques des prix, la prediction des prix de fermetures et la stratégie trading. 

## Local install
pipreqs
pip install backtrader

pip install backtrader[plotting]
    
L'api backtrader marche qu'avec la version Matplotlib >= 1.4.1. 

## Run streamlit
streamlit run app.py

## Run backtrader
python  backtrader

python bt_main.py  

# Team

[Bintou](https://github.com/bintou579)  
[Malika](https://github.com/malikaO)  
[Celine](https://github.com/CelineD75)  
 
