# TP2 Métodos Numéricos - Reconocimiento de dígitos

## Integrantes

* F. Galileo Cappella Lewi, 653/20, galileocapp@gmail.com
* Juan Pablo Anachure, 99/16, janachure@gmail.com

## Info

### Compilar

Para compilar primero hay que descomprimir `./src/eigen.tar.gz` en el mismo directorio y corriendo el script `compilar.sh` o el comando `g++ -I ./src/eigen ./src/main.cc -o tp2`

### Preparar datos

El programa tiene dos archivos de entrada y uno de salida:
* El primero de entrada son los datos de entrenamiento, un csv con la primera columna siendo eel label y el resto siendo los pixeles 
* El segundo de entrada son los datos para adivinar, tiene el mismo formato que el de entrenamiento, sin la columna label
* El de salida es un archivo con dos lineas, una de los tiempos que tomó calcular pca (si aplica) y adivinar cada vector de prueba

### Corriendo el programa

El programa tiene tres parámetros obligatorios y otros tres opcionales:
* Path al archivo de entrenamiento [OBLIGATORIO]
* Path al archivo de test [OBLIGATORIO]
* Path al archivo de salida [OBLIGATORIO]
* Cantidad de vecinos cercanos que revisar
* Cantidad de autovectores para calcular en el PCA
* Cantidad de iteraciones para calcular autovectores en el PCA
Para usar PCA hay que pasar la cantidad de autovectores para calcular  
NOTA: Los parámetros opcionales tienen que ser agregados en órden

Ejemplos:
~~~
./tp2 ./data/kaggle/train.csv ./data/kaggle/test.csv ./data/kaggle/results.csv
./tp2 ./data/kaggle/train.csv ./data/kaggle/test.csv ./data/kaggle/results.csv 50 
./tp2 ./data/kaggle/train.csv ./data/kaggle/test.csv ./data/kaggle/results.csv 50 100 
./tp2 ./data/kaggle/train.csv ./data/kaggle/test.csv ./data/kaggle/results.csv 50 100 1000 
~~~

