# Notes

## 20 de febrero de 2023

***

* Total de nodos:  19902
* Se tienen  274 nodos tienen una suma total de enlaces de 199403.5 dando una media de 724.5 enlaces por nodo.
* Se tiene 19628 nodos tiene una suma total de enlace de 3386929 dando una media de 172.5 enlaces por nodo.
* Nodos con mayor centralidad por su degree:  `('Tdc2-F-000040', 0.18084518365911262), ('TH-F-300050', 0.18129742224008843), ('TH-F-200014', 0.18632229536204212), ('TH-F-200047', 0.18632229536204212), ('TH-F-300044', 0.1871262750615547), ('TH-F-300002', 0.19893472689814581), ('TH-F-300046', 0.20295462539570877)]`

***

* Nodos con mayor grado: `'TH-F-300046': 4039` y `'TH-F-300002': 3959` 
* No hay nodos sin enlaces ni con uno solo.
* Nodos con mayor peso: `('GH298-F-300012', 88164), ('VT8167-F-200005', 88337), ('104198-F-000012', 89594), ('Gad1-F-400376', 93511), ('Trh-F-100107', 102453), ('Gad1-F-800089', 108239), ('Tdc2-F-000013', 110584)]`

***

* Nodos con menos peso: `('VGlut-F-000127', 4), ('fru-F-800087', 3), ('Gad1-F-600441', 3), ('Gad1-F-800468', 3), ('VGlut-F-600233', 3), ('VGlut-F-700592', 3), ('VGlut-F-400107', 2), ('VGlut-F-400399', 2)]`

***

* Hay muchos nodos con poco peso, pero no hay nodos con peso 0.

## 27 de febrero de 2023

***

* El algoritmo de Louvain es consistente en cuanto al número de comunidades y la cantidad de nodos en la mismas si se utiliza un mismo seed.
* El tiempo de ejecución del algoritmo de Louvain es de 57.5 segundos en promedio para el el grafo completo.
* El algoritmo de Louvain se ejecuta en un solo core del CPU.
* El algoritmo por defecto `si` evalua el peso de la arista, solo el grado de los nodos.
* Weighted `[17, 44, 85, 98, 102, 116, 116, 371, 499, 527, 531, 541, 598, 610, 623, 716, 827, 1096, 1512, 1625, 1718, 1866, 2767, 2897]`
* Unweighted `[102, 211, 397, 431, 454, 455, 561, 595, 1141, 1339, 1399, 1436, 2379, 2613, 2881, 3508]`

***

* El algoritmo de LPA (version asynchrone) es consistente en cuanto al número de comunidades y la cantidad de nodos en la mismas si se utiliza un mismo seed.
* El tiempo de ejecución del algoritmo de LPA es de 70 segundos en promedio para el el grafo completo. Se observan tambien variaciones del tiempo según el seed.
* El algoritmo de LPA se ejecuta en un solo core del CPU.
* El algoritmo por defecto `no` evalua el peso de la arista, solo el grado de los nodos.
* Weighted `[19, 59, 97, 2039, 2768, 3648, 11272]`
* Unweighted `[96, 2054, 2097, 4913, 10742]`

***

* El algoritmo de Greedy Modularity Optimization no es consistente en cuanto al número de comunidades y la cantidad de nodos. No cuenta con un seed.
* El tiempo de ejecución del algoritmo de Greedy Modularity Optimization es de `2100` segundos (`35min`) en promedio para el el grafo completo.
* El algoritmo de Greedy Modularity Optimization se ejecuta en un solo core del CPU.
* El algoritmo por defecto `no` evalua el peso de la arista, solo el grado de los nodos.
* Weighted `[1011, 1962, 2337, 6141, 8451]`
* Unweighted `[2214, 6443, 11245]` 

***

* El algoritmo de InfoMap es consistente en cuanto al número de comunidades y la cantidad de nodos en la mismas siempre.
* El tiempo de ejecución del algoritmo de InfoMap es de 54 segundos en promedio para el el grafo completo.
* El algoritmo de InfoMap se ejecuta en un solo core del CPU.
* Las comunidades de solo un nodos en corridas con el mismo seed son las mismas.
* Analizar el tratamiento de los pesos de las aristas.
* Por defecto el algoritmo evalua el peso de la aristas, pero el atributo tiene que estar normbrado `weight`.
* Para mejores resultados se recomiendo que los valores de los pesos de las aristas esten normalizados entre 0 y 1. Lo cual no es el caso, por lo tanto hay que hacer una normalización manual.
 