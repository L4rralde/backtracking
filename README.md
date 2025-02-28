# Regresión logística con descenso de gradiente (y backtracking)

## Resumen

En este trabajo vemos como maximizar/minimizar una función puede funcionar para generar una función de clasificación. Más particularmente, se maximiza una función de costo relacionada con la entropía cruzada de una regresión logística utilizada para clasificar `1s` y `0s` del conjunto de datos de números escritos a mano `MNIST`, y usando `Steepest gradient descent` con tamaño de paso fijo y `Backtracking` para resolver el problema de optimización.

## Instalación

Si ya cuentas con python y las librerías de numpy y matplotlib, omite esa sección.

1. (Recomendado pero no necesario) Crea un ambiente virtual:
   
```sh
pip install virtualenv    
python -m venv .env
source .env/bin/activate
```

2. Instala las librerías

```sh
pip install -r requirements.txt
```
