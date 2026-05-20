# tello_controller — ROS2 Humble

Nodo ROS2 que migra el script Python original de control de un DJI Tello e
integra la posición real proveniente del nodo **optitrack_client** (C++/NatNet).

---

## Arquitectura de nodos y topics

```
┌─────────────────────────────┐        optitrack/rigid_body         ┌──────────────────────────┐
│   optitrack_client (C++)    │  ──── geometry_msgs/PoseStamped ──► │  tello_controller (Py)   │
│                             │        (SensorDataQoS / BE)         │                          │
│  NatNet → RigidBody pose    │                                      │  Control proporcional    │
│  x,y,z en METROS            │                                      │  K = diag(1.2,1.2,1.2)  │
└─────────────────────────────┘                                      │  → djitellopy RC cmds    │
                                                                     └──────────┬───────────────┘
                                                                                │ publica
                                                               ┌────────────────▼────────────────┐
                                                               │  tello/estimated_state  Float32 │
                                                               │  tello/control_error    Float32 │
                                                               │  tello/goal_reached     Bool    │
                                                               └─────────────────────────────────┘
```

---

## Match con optitrack_client.cpp

| Campo C++ (DrainNetworkQueue)              | Topic ROS2                  | Uso en tello_controller       |
|--------------------------------------------|-----------------------------|-------------------------------|
| `data->RigidBodies[i].x/y/z` (metros)     | `optitrack/rigid_body`      | `q = [x,y,z] * 100` (→ cm)   |
| `msg.header.frame_id` = nombre rigid body  | `optitrack/rigid_body`      | (disponible, no usado aún)    |
| `msg.header.stamp`                         | `optitrack/rigid_body`      | (disponible, no usado aún)    |
| QoS: `rclcpp::SensorDataQoS()`            | BEST_EFFORT / KEEP_LAST(10) | Idéntico en el subscriber Py  |

---

## Variables y lógica original preservadas

```python
DT            = 0.1          # Período del timer de control [s]
RC_LIMIT      = 40           # Saturación comandos RC
MAX_SPEED_CM_S = 40.0        # Velocidad máxima estimada [cm/s]
TOL_CM        = 2            # Tolerancia de parada [cm]
ALPHA         = 0.6          # Coeficiente del filtro de odometría
K             = diag(1.2, 1.2, 1.2)  # Ganancia proporcional
q_d           = [50, 0, 150]          # Objetivo por defecto [cm]
```

- `control()`, `velocity_to_rc()`, `odometry()` → **sin cambios**.
- Condición de parada por precisión (`dist < TOL_CM`) y por estancamiento
  (`rc == 0` tras 10 iteraciones) → **sin cambios**.
- La odometría se usa como **fallback** si OptiTrack aún no entrega datos.
  En cuanto llega el primer `PoseStamped`, `self.q` se actualiza con la
  posición real y la odometría deja de integrarse.

---

## Instalación

```bash
# 1. Clonar en tu workspace ROS2
cd ~/ros2_ws/src
cp -r tello_controller .

# 2. Instalar dependencia Python (djitellopy)
pip install djitellopy

# 3. Compilar
cd ~/ros2_ws
colcon build --packages-select tello_controller
source install/setup.bash
```

---

## Ejecución

### Solo el nodo Python (optitrack_client ya corriendo aparte)
```bash
ros2 run tello_controller tello_controller_node
```

### Con parámetros personalizados
```bash
ros2 run tello_controller tello_controller_node \
  --ros-args \
  -p target_x_cm:=100.0 \
  -p target_z_cm:=120.0 \
  -p k_gain:=1.5
```

### Launch completo (optitrack_client + tello_controller)
```bash
ros2 launch tello_controller tello_with_optitrack.launch.py \
  target_x_cm:=80.0 target_z_cm:=100.0
```

---

## Topics publicados

| Topic                    | Tipo                      | Contenido                        |
|--------------------------|---------------------------|----------------------------------|
| `tello/estimated_state`  | `std_msgs/Float32MultiArray` | `[x, y, z, dist_to_goal]` cm    |
| `tello/control_error`    | `std_msgs/Float32MultiArray` | `[ex, ey, ez]` cm               |
| `tello/goal_reached`     | `std_msgs/Bool`           | `true` al alcanzar el objetivo   |

## Topics suscritos

| Topic                    | Tipo                            | Fuente              |
|--------------------------|---------------------------------|---------------------|
| `optitrack/rigid_body`   | `geometry_msgs/PoseStamped`     | optitrack_client.cpp |
