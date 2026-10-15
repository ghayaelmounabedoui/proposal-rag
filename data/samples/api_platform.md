# Plateforme API (DOCUMENT D'EXEMPLE FICTIF)

## API REST

La plateforme expose une API REST documentée avec OpenAPI 3.
Elle permet de créer, consulter, modifier et supprimer les fiches clients.

## Recherche de commandes

Les commandes clients peuvent être recherchées par identifiant de commande,
par client ou par période.
La recherche par identifiant de commande répond en moins de 200 millisecondes.

# Intégration

## Authentification

L'API utilise OAuth 2.0 avec des jetons à durée de vie limitée.
