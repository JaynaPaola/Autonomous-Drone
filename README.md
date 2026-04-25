# Autonomous-Drone
# 🚁 Control de Trayectoria para Drone Tello (X → Y → Z)

Este proyecto implementa un sistema de control de posición para el dron DJI Tello utilizando un controlador proporcional y una odometría simple basada en integración de velocidad estimada.

El objetivo del sistema es que el dron siga una trayectoria secuencial en tres etapas:
1. Movimiento en X
2. Movimiento en Y
3. Movimiento en Z

---

## 📌 Características del sistema

- Control de posición tipo proporcional (P control)
- Odometría simplificada mediante integración numérica
- Ejecución secuencial de trayectoria (X → Y → Z)
- Comunicación directa con el dron Tello mediante `djitellopy`
- Control en tiempo real con comandos RC
- Transiciones automáticas entre objetivos

---

## 📐 Modelo del sistema

### 🔹 Control

El control se basa en:

\[
u = K (q_d - q)
\]

Donde:
- `q` = posición actual estimada
- `q_d` = posición deseada
- `K` = matriz de ganancia proporcional

---

### 🔹 Odometría

Se utiliza un modelo simple:

1. Estimación de velocidad a partir de comandos RC  
2. Filtrado exponencial  
3. Integración numérica:

\[
q_{k+1} = q_k + v \cdot dt
\]

---

## 🧭 Trayectoria implementada

El dron sigue tres etapas secuenciales:

### 1. Movimiento en X
\[
[0, 0, 110] \rightarrow [50, 0, 110]
\]

### 2. Movimiento en Y
\[
[50, 0, 110] \rightarrow [50, 50, 110]
\]

### 3. Movimiento en Z
\[
[50, 50, 110] \rightarrow [50, 50, 150]
\]

---

## ⚙️ Requisitos

- Python 3.8+
- Drone DJI Tello
- Librerías necesarias:

```bash
pip install djitellopy numpy
