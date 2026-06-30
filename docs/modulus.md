# NVIDIA Modulus en SOOG: Simulación Acústica Neuronal

Este documento detalla la integración de **NVIDIA Modulus** en el ecosistema SOOG (Speculative Organology Organogram Generator), la teoría física subyacente, la implementación actual mediante "Surrogates" y la visión especulativa hacia el futuro de la acústica computacional.

---

## 1. Teoría Dura: PINNs (Physics-Informed Neural Networks)

NVIDIA Modulus se basa en el paradigma de **PINNs**, que representa un cambio fundamental respecto a los métodos tradicionales de simulación como FEM (Finite Element Method) o BEM (Boundary Element Method).

### El Concepto de PINN
A diferencia de un solver tradicional que discretiza el espacio en una malla (mesh) y resuelve sistemas de ecuaciones lineales masivos, una PINN es una **función continua** aproximada por una red neuronal profunda $f(x, y, z, t, \theta) \approx p$, donde:
- $(x, y, z, t)$ son las coordenadas espacio-temporales.
- $\theta$ son los parámetros de entrada (frecuencia, posición de obstáculos, material).
- $p$ es la presión acústica resultante.

### La Función de Pérdida Física (Physics Loss)
La red no se entrena solo con datos, sino con la **Física**. La función de pérdida incluye el residuo de la ecuación diferencial:
$$L = L_{data} + L_{physics} + L_{boundary}$$
Donde $L_{physics}$ obliga a la red a cumplir, por ejemplo, la **Ecuación de Helmholtz**:
$$\nabla^2 p + k^2 p = 0$$
Esto significa que la red aprende a ser un "solver" universal para una geometría parametrizada.

---

## 2. Implementación Actual: El "Acoustic Surrogate"

### El Desafío CUDA
NVIDIA Modulus requiere hardware NVIDIA con soporte CUDA para el entrenamiento y la ejecución de sus kernels especializados (como `Signed Distance Functions` aceleradas). Debido a las restricciones del entorno local (macOS/Apple Silicon), hemos implementado un **Surrogate Model** en PyTorch nativo.

### El Reemplazo por Surrogate
En `modeltrainer/acoustic_modulus.py`, la clase `AcousticPINN` actúa como un puente:
1.  **Arquitectura:** Un Perceptrón Multicapa (MLP) con funciones de activación $Tanh$ (esenciales para capturar la naturaleza oscilatoria de las ondas).
2.  **Inferencia Determinista:** En lugar de una malla iterativa, realizamos un **forward pass** único sobre una rejilla de 50x50 puntos.
3.  **Hibridación:** El modelo actual utiliza una aproximación analítica de la Ecuación de Helmholtz perturbada por obstáculos, emulando lo que una PINN entrenada devolvería tras converger.

---

## 3. Operaciones Posibles (Estado Actual)

1.  **Resolución de Helmholtz 2D:** Simulación de propagación de ondas senoidales en un dominio cerrado.
2.  **Interferencia Dinámica:** Modificación instantánea del campo de presión al mover coordenadas de obstáculos (`obs_x`, `obs_y`).
3.  **Virtual Mic Response:** Extracción de la amplitud de presión en puntos específicos para generar feedback sonoro.
4.  **Generación de Mapas de Calor:** Visualización de fases y nodos de presión mediante `ModulusHeatmap.vue`.

---

## 4. Pasos Futuros e Integración DDSP

El roadmap de SOOG contempla la transición de simulaciones visuales a síntesis neuronal física:

1.  **Cálculo de IR (Impulse Response):** En lugar de una sola frecuencia, simular un pulso para obtener la respuesta espectral de una cavidad.
2.  **Integración DDSP (Differentiable Digital Signal Processing):**
    *   Usar la respuesta de Modulus como los parámetros de un filtro diferencial.
    *   Hacer que el motor de audio de SOOG "resuene" literalmente según la geometría generada.
3.  **Inferencia en 3D:** Pasar de mapas 2D a volúmenes STL reales, calculando la resonancia interna de los instrumentos de viento generados.

---

## 5. Especulación: Modelos de Inferencia del Futuro

Más allá de lo que permite la tecnología actual, SOOG apunta a conceptos de **"Acoustic Latent Spaces"**:

### Difusión de Campos de Presión (Acoustic Diffusion)
Imaginar modelos de difusión que no generen imágenes, sino **campos vectoriales de sonido**. En lugar de resolver la física, el modelo "imagina" cómo se desplazaría el aire en una geometría arbitraria, entrenado en millones de simulaciones previas de Modulus.

### Geometría Generativa Co-Dependiente
Modelos donde la pérdida (loss) no sea estética, sino **acústica**. El LLM propone una geometría, Modulus la evalúa en tiempo real, y el gradiente de la simulación "empuja" la geometría para que el instrumento suene exactamente como el usuario describió (ej: "un clarinete con armónicos metálicos").

### Inferencia Multi-Modal Extrema
Modelos de inferencia donde el peso de la red sea el **material mismo**. Una red neuronal que predice la elasticidad y la densidad de un material inexistente para lograr una resonancia específica, permitiendo la fabricación de instrumentos con "metamateriales acústicos" diseñados por IA.

---
*Documento generado para el proyecto SOOG - Mayo 2026*
