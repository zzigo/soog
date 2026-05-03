SOOG Project Roadmap
# 1. Dec 2024: Conceptual Inception

Initial design of SOOG as the pioneering generative system for
musical instrument notation based on Mantle Hood’s organogram technique.
Implementation of large-scale models via GPT-4o and DeepSeek-V3 APIs to translate natural language into structured organological representations.

# 2. April 2025: Prototyping & 3D Synthesis

Development of the first physical-digital prototype, Phosphorbone (https://zztt.org/research/phosphorbone/). Refinement of prompt engineering and expansion of the organological dictionary. Introduction of the 3D prototyping module using Midjourney v6 and SDXL for visual aesthetics, establishing the foundation for future digital fabrication workflows.

# 3. Dec 2025: Local Intelligence (Current Phase)

Transition to a hybrid infrastructure utilizing local, open-source
models (Ollama,Qwen 2.5) for logic and Matplotlib generation. Integration of
specialized creative pipelines:Stable Diffusion (SDXL-Turbo) for industrial design sketches and Stable Audio Open for timbral synthesis.

# 4. July 2026: SoMap & Generative CAD

Completion of SoMap, a comprehensive Markdown database of the MOIAE (Material, Object, Agent, Interface, Environment) system. Transitioning to direct 3D output via Unique3D or InstantMesh for high-fidelity STL mesh synthesis from 2D sketches. Implementation of generative hardware design using Flux.ai and DeepPCB for AI-assisted routing of embedded controller PCBs.
# 5.2027: Physical Realization & Neural Simulation

Construction of the first generation of SOOG-built physical
instruments. Integration of real-time acoustical BEM/FEM simulations using
NVIDIA Modulus (Physics-Informed Neural Networks) for near-instant sonic feedback. Implementation of DDSP (Differentiable Digital Signal Processing) to bridge physical resonance data with real-time neural synthesis models.


---
Puntos clave de los nuevos modelos propuestos:
*
Unique3D / InstantMesh (Punto 4):
Son los modelos punteros para pasar de un sketch
2D a un objeto 3D real (malla STL) manteniendo la fidelidad del diseño industrial.
*
Flux.ai / DeepPCB (Punto 4):
Permiten que la IA diseñe el ruteo de las placas de
circuito (PCB) para la electrónica embebida del instrumento automáticamente.
*
NVIDIA Modulus (Punto 5):
Es el estándar de "Physics-Informed Neural Networks"
(PINNs). Permite simular cómo vibrará el objeto (FEM/BEM) mil veces más rápido que
el software de ingeniería tradicional.
*
DDSP (Punto 5):
Desarrollado por Google Magenta, permite "mapear" la física real
de un objeto resonante a un modelo de síntesis de audio, haciendo que el sonido
virtual y el físico sean indistinguible