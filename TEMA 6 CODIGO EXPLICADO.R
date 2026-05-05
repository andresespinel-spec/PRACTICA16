# Andrés Lorenzo Espienl Mesa
# Tue May  5 12:55:06 2026 ------------------------------

#TEMA 6: Técnicas avanzadas de trabajo con datos

# ==========================================
# 1. REGRESIÓN LINEAL
# ==========================================

library(haven)

# 1.1 Carga y limpieza
pisa <- read_sav("PISA22SP.sav")
pisa <- na.omit(pisa)  # Elimina las filas con al menos un valor perdido (NA)

# 1.2 Regresión Lineal Simple
# Comprobamos visualmente la relación entre lectura (eje X) y matemáticas (eje Y)
plot(pisa$lectu, pisa$mates)
# Creamos el modelo: ¿Cómo afecta la nota de lectura a la de matemáticas?
reg <- lm(mates ~ lectu, data = pisa)
summary(reg) # Muestra los resultados (coeficientes, R-cuadrado, p-valor)

# 1.3 Regresión Lineal Múltiple
# Preparamos la variable género como factor y le asignamos etiquetas
pisa$genero <- as.factor(pisa$genero)
levels(pisa$genero) <- c("mujer", "hombre") # Corrección: quité el "|" que tenías en "hombre"
# Vemos un resumen de las variables independientes
indepen <- c("genero", "heduc", "lectu")
summary(pisa[, indepen])
# Creamos el modelo múltiple: ¿Cómo afectan el género, la educación de los padres y la lectura a las matemáticas?
reg_mult <- lm(mates ~ genero + heduc + lectu, data = pisa)
summary(reg_mult)

# 1.4 Cálculo manual de bondad de ajuste (Opcional, asumiendo que 'reg_mult' es tu modelo)
# Error Estándar Residual (RSE): Mide la desviación típica de los residuos
RSE <- sqrt(sum(residuals(reg_mult)^2) / reg_mult$df.residual)
# R-cuadrado: Porcentaje de varianza explicada por el modelo
R2 <- summary(reg_mult)$r.squared

# ==========================================
# 2. REGRESIÓN LOGÍSTICA
# ==========================================

# 2.1 Transformar la variable dependiente a binaria (0 y 1)
summary(pisa$mates)
# Calculamos el corte: el percentil 75 (el 25% de los alumnos con mejores notas)
tercer_cuartil <- quantile(pisa$mates, 0.75)
# Creamos una nueva variable: 1 si supera el 3er cuartil (Bueno), 0 si no (Normal)
pisa$mates_r <- ifelse(pisa$mates > tercer_cuartil, 1, 0)
pisa$mates_r <- as.factor(pisa$mates_r) # Es importante que sea factor
levels(pisa$mates_r) <- c("Normales", "Buenos")
summary(pisa$mates_r)

# 2.2 Modelo Logístico Simple
# Predice la probabilidad de ser "Bueno" en matemáticas según la nota de lectura
logit <- glm(mates_r ~ lectu, family = binomial, data = pisa)
summary(logit)

# 2.3 Modelo Logístico Múltiple
# Predice la probabilidad de ser "Bueno" en mates según género, educación de padres y nota en lectura
logit2 <- glm(mates_r ~ genero + heduc + lectu, family = binomial, data = pisa)
summary(logit2)

# ==========================================
# 3. ÁRBOLES DE DECISIÓN (SEGMENTACIÓN)
# ==========================================

# Instalación y carga del paquete
if (!require(tree)) install.packages("tree")
library(tree)

# 3.1 Árbol de Regresión (La variable a predecir es numérica: 'mates')
arbolr <- tree(mates ~ genero + heduc + lectu, data = pisa)
plot(arbolr)            # Dibuja la estructura del árbol
text(arbolr, pretty=0)  # Pone los textos y condiciones en las ramas
arbolr                  # Imprime las reglas del árbol en la consola

# 3.2 Árbol de Clasificación (La variable a predecir es categórica: 'mates_r')
# Nos aseguramos de que es un factor (ya lo hicimos en el paso anterior)
table(pisa$mates_r)

arbolc <- tree(mates_r ~ genero + heduc + lectu, data = pisa)
plot(arbolc)            # Dibuja el árbol
text(arbolc, pretty=0)  # Pone los textos
arbolc                  # Imprime las reglas

# ==========================================
# 4. TÉCNICAS NO SUPERVISADAS
# ==========================================

# 4.1 Analisis factorial
#Preparación de datos (Solo numéricos)
datos <- na.omit(pisa[, c("heduc", "hisei", "status", "mates", "lectu", "cienc")])
# Estadísticas básicas por columna (2 significa columnas)
apply(datos, 2, mean) # Medias
apply(datos, 2, sd)   # Desviaciones típicas
cor(datos)            # Matriz de correlaciones

# 4.2 Análisis de Componentes Principales (PCA)
# Busca reducir muchas variables correlacionadas en unos pocos "factores" o "componentes" latentes.
require("psych")
# Diferentes formas de extraer componentes:
principal(datos)                                     # Por defecto extrae 1 componente
principal(datos, nfactors = ncol(datos))             # Extrae tantos como variables haya
principal(datos, nfactors = ncol(datos), rotate = "none") # Sin rotación (difícil de interpretar)
principal(datos, nfactors = 3, rotate = "varimax")   # Extrae 3 componentes rotados (más fácil de interpretar)
# Análisis Factorial puro (similar al PCA pero con matices teóricos)
fa(datos, 3, fm = "pa", rotate = "varimax")
# PCA con el paquete FactoMineR (muy visual y completo)
require("FactoMineR")
PCA(datos, ncp = 2) # Extrae 2 componentes

# 4.3 Analisis de conglomerados
# Agrupa a los estudiantes en K grupos según sus similitudes en las variables dadas.
# Primero estandarizamos los datos (Z-scores) para que todas las variables pesen igual
sdatos <- as.data.frame(scale(datos))
# Probamos creando 2 clusters (K=2)
set.seed(3) # Fijar semilla para que el resultado sea siempre el mismo
K2 <- kmeans(sdatos, centers = 2)
# Guardamos las inercias (variabilidad) para ver si la agrupación es buena
# totss = inercia total (si hubiera 1 solo grupo). tot.withinss = inercia dentro de los grupos
Ks <- c(K2$totss, K2$tot.withinss) 
# Muestra los centroides (medias de cada grupo) y cuántos alumnos hay en cada grupo
round(cbind(K2$centers, K2$size), 1)
# Muestra el perfil medio de cada grupo con los datos originales (sin estandarizar)
cbind(round(aggregate(. ~ K2$cluster, data = cbind(datos, K2$cluster), FUN = mean), 1)[,-1], Tamaño = K2$size)
# Bucle manual: Repite el proceso para K=3, K=4, K=5 y K=6
# K=3
set.seed(3)
K3 <- kmeans(sdatos, centers = 3)
Ks <- c(Ks, K3$tot.withinss)
cbind(round(aggregate(. ~ K3$cluster, data = cbind(datos, K3$cluster), FUN = mean), 1)[,-1], Tamaño = K3$size)
# K=4
set.seed(3)
K4 <- kmeans(sdatos, centers = 4)
Ks <- c(Ks, K4$tot.withinss)
cbind(round(aggregate(. ~ K4$cluster, data = cbind(datos, K4$cluster), FUN = mean), 1)[,-1], Tamaño = K4$size)
# K=5
set.seed(3)
K5 <- kmeans(sdatos, centers = 5)
Ks <- c(Ks, K5$tot.withinss)
cbind(round(aggregate(. ~ K5$cluster, data = cbind(datos, K5$cluster), FUN = mean), 1)[,-1], Tamaño = K5$size)
# K=6
set.seed(3)
K6 <- kmeans(sdatos, centers = 6)
Ks <- c(Ks, K6$tot.withinss)
cbind(round(aggregate(. ~ K6$cluster, data = cbind(datos, K6$cluster), FUN = mean), 1)[,-1], Tamaño = K6$size)
# Gráfico del codo (Elbow plot): Ayuda a decidir cuántos clusters son los ideales.
# Se busca el punto donde la curva hace un "codo" y deja de bajar drásticamente.
plot(Ks, type = "b", ylim = c(0, max(Ks)), main="Método del Codo", ylab="Inercia intra-grupos")
text(Ks, c("K1", "K2", "K3", "K4", "K5", "K6"), pos = 1)

# 4.4 Factorial más conglomerados (K-means sobre PCA)
# En lugar de usar las 6 variables originales, usamos los 2 factores resumen creados por el PCA.
pc <- principal(datos, nfactors = 2) # Extraer 2 componentes
pc$loadings                          # Ver qué peso tiene cada variable en cada componente
# K-means sobre los "scores" (puntuaciones) de los 2 componentes con K=2
set.seed(4)
k2_pc <- kmeans(pc$scores, 2)
round(cbind(k2_pc$centers, k2_pc$size), 1)
cbind(round(aggregate(. ~ k2_pc$cluster, data = cbind(datos, k2_pc$cluster), FUN = mean), 1)[,-1], k2_pc$size)
# K-means sobre los scores con K=6
set.seed(3)
k6_pc <- kmeans(pc$scores, 6)
round(cbind(k6_pc$centers, k6_pc$size), 1)
cbind(round(aggregate(. ~ k6_pc$cluster, data = cbind(datos, k6_pc$cluster), FUN = mean), 1)[,-1], k6_pc$size)
# Visualización de los Clusters en el plano PCA
# Dibujamos a los alumnos en un plano 2D usando los dos componentes principales, 
# y los coloreamos según el grupo (cluster) al que han sido asignados.
library(tidyverse)
Grupo <- as_factor(k6_pc$cluster) # Convertimos el resultado de los 6 clusters en factor
# pc$scores contiene RC1 (Rendimiento) y RC2 (Origen socioeconómico)
ggplot(as.data.frame(pc$scores), mapping = aes(x = RC1, y = RC2, colour = Grupo)) +
  geom_point(alpha = 0.5) +  # Alpha añade algo de transparencia para ver los puntos superpuestos
  ggtitle("Grupos conformados por rendimiento y origen") +
  labs(x = "Factor 1: Rendimiento (RC1)", y = "Factor 2: Origen Social (RC2)") +
  theme_minimal()

