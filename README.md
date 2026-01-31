VISION

este es un sistema de vision basado en la levtrua de HSV (para detectar colores) y de letras (atraves de la deteccion de
contornos).

este es uno de mis sistemas para la deteccion de victumas en la competencia de rescue maze a continuacion explicare que hace
cada codigo y como funcionan.

Vision_comparador.py

Este código es básicamente el "cerebro" que reconoce las letras o señas mediante contornos. Aquí explico cómo funciona cada
parte, aunque a veces me doy vueltas en lo mismo porque así es como lo fui probando:

  1. Base de Datos y Normalización (Cargar Base)
Primero, el sistema se conecta a una base de datos de SQLite (contours.db) donde tengo guardados los moldes de las
letras que ya registré antes. Lo que hago aquí es sacar los contornos que están guardados como "pickles" (o sea,
comprimidos).

Redundancia: Aquí lo que hago es normalizar el contorno. O sea, lo muevo al punto (0,0) para que no importe en qué parte
de la cámara esté la mano, siempre lo vea igual. Luego lo escalo a un tamaño de 64 píxeles porque si es más grande o más
chico el sistema se confunde.

Momento de Hu: Uso los "Momentos de Hu" que son como el ADN del contorno. Les aplico un logaritmo porque los números son
muy chiquitos y así es más fácil que la computadora los compare sin que explote el cálculo.

  2. Procesamiento de Imagen (Extraer Contorno)
Esta es la parte donde la cámara "limpia" la imagen para ver qué hay:

Filtros: Convierto todo a gris y luego le meto un Desenfoque Gaussiano (Blur) para quitar el ruido. Uso un
adaptiveThreshold que es como un filtro que decide qué es blanco y qué es negro dependiendo de la luz.

Dilatación y Erosión: Aquí aplico una dilatación para que las líneas se hagan más gordas y luego una erosión para que se
limpien. O sea, trato de que el contorno se vea sólido y no con huecos.

Filtros de Área: Puse unos límites (AREA_MIN y AREA_MAX) porque si algo es muy chiquito es basura y si es muy grande es el
fondo. También checo el "Ratio" para que la forma no sea demasiado larga o demasiado ancha, porque las letras suelen ser
cuadraditas.

  3. Inteligencia de Clasificación (KNN)
Para saber qué letra es, uso el algoritmo de KNN (K-Nearest Neighbors):

El sistema mide la "distancia" entre la forma que ve la cámara y las 5 formas más parecidas que tengo en la base de datos.

Error de explicación: Si la distancia es mayor a 4, el código dice "ERROR" porque significa que lo que está viendo no se
parece a nada que yo conozca.

El Buffer: Tengo un buffer_letras que guarda las últimas 15 predicciones. Esto lo hice porque a veces la cámara parpadea y
cambia de letra rápido, así que mejor saco el "promedio" (la moda) de las últimas 15 para que la letra en pantalla se
quede quieta y no esté saltando como loca.

  4. Visualización y Ciclo Principal
En el while True, la cámara está leyendo todo el tiempo. Si encuentra un contorno válido, le dibuja un rectángulo verde y
pone el nombre de la letra con su porcentaje de seguridad. Si el porcentaje es bajo o la distancia es mucha, simplemente
marca error para que yo sepa que tengo que acomodar mejor la mano o el objeto.

VISIÓN_LETRAS_GUADAR_CONTORNOS.py

Este programa lo hice para no tener que estar escribiendo a mano los valores de los colores. Es básicamente una herramienta
que "aprende" los colores en tiempo real y los guarda en una base de datos de SQLite para que luego el robot o el sistema
los reconozca.

  1. El Almacén (Base de Datos)
Aquí usé sqlite3 para crear una tabla que se llama colores. Lo que hace es guardar la etiqueta (por ejemplo, "rojo") y
sus valores H, S y V.

Redundancia: Cada vez que le pico al botón, el sistema guarda el valor exacto del píxel central. Lo configuré para que tome
200 muestras seguidas. ¿Por qué 200? Porque la luz cambia mucho y así tengo un promedio real de cómo se ve el color en
diferentes micro-segundos. Si guardara solo uno, el sistema fallaría si pasa una mosca o cambia una sombra.

  2. Localización del Objeto (Detector de Color)
Antes de guardar los datos, tengo que estar seguro de que estoy viendo el objeto y no la pared.

Máscara de Saturación: Puse un SAT_MIN y VAL_MIN de 40. Esto es porque si no hay suficiente color (saturación), el sistema
empieza a detectar blancos, grises o negros que no me sirven.

El Contorno más grande: El código busca todos los bultos de color que ve la cámara, pero con max(contours,
key=cv2.contourArea) me quedo solo con el más grande. O sea, asumo que lo que estoy acercando a la cámara es lo que quiero
entrenar. Si el área es menor a 400 píxeles, el código lo ignora porque seguro es ruido de la cámara.

  3. El Píxel Central (Obtener HSV)
Esta es la parte "mañosa" del código. En lugar de sacar el promedio de todo el objeto (que consume mucho procesador),
simplemente calculo el centro geométrico (cx, cy) del rectángulo que encierra al objeto.

Explicación redundante: Saco el color de ese único punto central. Convierto ese píxel de BGR (que es como lee OpenCV) a HSV,
porque el HSV es mucho más estable para nosotros los humanos y para programar.

  4. Interfaz con Tkinter (La Ventana)
Como no quería estar usando la terminal para todo, le armé una ventanita con Tkinter.

El Combobox: Ahí elijo si voy a entrenar el "rojo", "verde" o "amarillo".

Actualización constante: Uso self.root.after(10, ...) para que el video se vea fluido. Es un truco porque Tkinter y OpenCV a
veces se pelean por quién manda en la pantalla. Al convertir la imagen de OpenCV a una que PIL (Pillow) entienda, puedo ver
el video dentro de la ventana de Windows sin que se trabe.

Por: Nestor Tec
