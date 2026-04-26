# Autonomous-Drone

# Control de seguimiento de trayectoria con DJI Tello

## 📌 Descripción del proyecto

Este proyecto implementa un sistema de control de seguimiento de trayectoria para el dron DJI Tello en Python. Se utiliza un controlador proporcional, una odometría simulada mediante un modelo integrador y la conversión de velocidades a comandos RC del dron.

El objetivo es analizar el comportamiento dinámico del sistema, su estabilidad y su respuesta ante diferentes referencias de posición.

---

# 🧠 Modelo del sistema

## 🔹 Controlador

El sistema de control utiliza una ley proporcional:

u = K (q_d - q)

Donde:
- q: posición actual estimada
- q_d: posición deseada
- K: matriz de ganancias

---

## 🔹 Conversión a RC

El vector de control u se convierte a comandos del dron:

- FB → adelante / atrás  
- LR → izquierda / derecha  
- UD → arriba / abajo  

Los valores se saturan al rango permitido por el Tello.

---

## 🔹 Odometría (modelo integrador)

El movimiento del dron se modela como:

q(k+1) = q(k) + v_est · dt

La velocidad estimada se calcula como:

v_est = α v_est + (1 - α) v_meas

---

# 📊 Test 1 — Control a un solo punto

## 🎯 Descripción

Este código implementa el seguimiento de un único punto objetivo fijo.

### Funcionamiento:
- Se define una posición deseada constante q_d
- Se calcula el error respecto a la posición actual
- Se genera una acción de control proporcional
- Se convierte a comandos RC
- Se actualiza el estado mediante el modelo de odometría
- El sistema termina al alcanzar la tolerancia definida

---

## 📌 Características

- Control de un solo setpoint
- Análisis de estabilidad del sistema
- Evaluación del error global
- Observación de saturación del actuador (RC)
- Registro de variables para análisis gráfico

---

## 📈 Variables registradas

- q: posición estimada
- q_d: posición deseada
- error total
- u: señal de control
- RC: comandos del dron
- v_est: velocidad estimada
- v_meas: velocidad medida

---

## 🎯 Objetivo del sistema

Demostrar la convergencia del sistema de control hacia un punto fijo y analizar su comportamiento dinámico.

---

# 📊 Test 2 — Control por trayectoria segmentada (X → Y → Z)

## 🎯 Descripción

Este segundo código implementa un control por etapas, donde el dron sigue una secuencia de puntos en el espacio.

### Trayectoria:
- Fase 1: X
- Fase 2: Y
- Fase 3: Z

---

## 🔁 Funcionamiento

- Se define una lista de objetivos q_d_list
- El controlador sigue cada punto hasta cumplir la tolerancia
- Al alcanzar un objetivo, se cambia automáticamente al siguiente
- El sistema registra todo el proceso dinámico

---

## 📌 Características

- Control por múltiples referencias
- Transiciones entre objetivos
- Análisis del comportamiento transitorio
- Estudio de estabilidad en cambios de setpoint
- Evaluación de respuesta dinámica del sistema

---

## 📈 Variables registradas

- q: posición estimada
- q_d: referencia activa
- error por eje (x, y, z)
- error total
- u: señal de control
- RC: comandos del dron
- v_est: velocidad estimada
- v_meas: velocidad medida
- fase del sistema (X, Y, Z)

---

## 🎯 Objetivo del sistema

Evaluar la capacidad del controlador para seguir trayectorias secuenciales y analizar su comportamiento durante cambios de referencia.

---

# 🧪 Conclusión general

Ambos sistemas permiten analizar:

- Estabilidad del control
- Respuesta dinámica del dron
- Saturación del actuador
- Seguimiento de trayectoria
- Efectos de cambios de referencia

Estos modelos sirven como base para una futura implementación en ROS.
