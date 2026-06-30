# SOOG v2: Technical Architecture & Design Patterns

This document details the new technologies, architectural improvements, and design patterns introduced during the implementation of the **Colored LRM Engine** and the **3D Concert Simulator**.

## 1. Frontend: Shared API Resolution Pattern

To handle the transition between local development (`localhost`) and remote VPS access (`domain/IP`) without manual configuration, a centralized API resolution pattern was implemented.

### `useApi` Composable
*   **Technology:** Vue 3 Composition API + Nuxt Composables.
*   **Pattern:** Adaptive Singleton.
*   **Location:** `frontend/composables/useApi.ts`.
*   **Logic:** 
    *   Reads a default `apiBase` from `runtimeConfig`.
    *   Detects if the current `window.location.hostname` is a remote IP/domain while the API is still pointing to `localhost`.
    *   Dynamically rewrites the `apiBase` to match the actual host.
    *   Ensures consistent API access across all components (`HelpModal`, `GalleryModal`, `ConcertSimulator`, etc.) without redundant logic.

## 2. 3D Pipeline: Neural-to-GLB Workflow

The 3D generation was upgraded from simple STL geometry to high-fidelity colored GLB models.

### Colored LRM Engine (TripoSR)
*   **Technology:** StabilityAI TripoSR (Large Reconstruction Model).
*   **Model Format:** Switched from `.stl` to `.glb` to support native vertex coloring.
*   **Memory Optimization Pattern:**
    *   **Resolution Scaling:** Marching Cubes resolution was reduced from 256 to 128 to fit within VPS RAM limits (approx. 4-6GB peak usage).
    *   **Garbage Collection:** Explicit `gc.collect()` and `torch.cuda.empty_cache()` (if applicable) are called after every inference to prevent OOM (Out Of Memory) crashes.
    *   **Background Blending:** RGBA sketches are blended with neutral gray before neural processing to avoid channel mismatch errors.

### Dynamic Loader Pattern
*   **Location:** `StlViewer.vue` and `ConcertSimulator.vue`.
*   **Logic:** Implemented a file-extension-aware loader that switches between `STLLoader` and `GLTFLoader` at runtime, ensuring backward compatibility with legacy models.

## 3. Immersive Audio: Spatial & Convolution System

The Concert Simulator introduces a sophisticated audio pipeline using the Web Audio API integrated with Three.js.

### Spatial Audio
*   **Technology:** `THREE.AudioListener` + `THREE.PositionalAudio`.
*   **Pattern:** Panner-Node Mesh Parenting.
*   **Details:** Audio sources are parented to moving 3D meshes. The volume and panning are dynamically calculated based on the camera's position and orientation relative to the instrument's bounce and spin.

### Convolution Reverb
*   **Technology:** `ConvolverNode` + Synthetic Impulse Response.
*   **Pattern:** Parallel Effects Bus.
*   **Algorithm:** Generates a 3.5s exponentially decaying white noise buffer to simulate the "acoustic fingerprint" of a large concert hall.
*   **Routing:** Dry signal (direct) + Wet signal (reverb gain node) are summed before the final listener output, allowing real-time "Reverb Mix" control in the HUD.

## 4. Interaction: Reactive HUD Pattern

For real-time control of the 3D environment, a "Head-Up Display" (HUD) pattern was developed.

### Parametric Control
*   **Pattern:** Reactive Parameter Proxy.
*   **Implementation:** A Vue `reactive` object (`params`) binds HUD sliders (Volume, Spacing, Speed, Light Intensity) directly to Three.js scene properties.
*   **Key Controls:**
    *   **Neighbor Spacing:** Implements a repulsion-force physics loop in the `animate()` function.
    *   **Cinematic Lighting:** Controls 20 multi-colored spotlights (5000 lumens) and a global Godray backlight.
    *   **Fullscreen Mode:** Custom SVG cursor logic (`[ ]`) for immersive interaction.

## 5. Deployment & Persistence

### PM2 Process Management
*   **Pattern:** Standardized Working Directory.
*   **Configuration:** Services are configured with explicit `cwd` (Current Working Directory) paths in PM2. This ensures the Flask backend correctly resolves the `/offload/gallery` relative paths regardless of how the system boots.
*   **Authentication:** `HF_TOKEN` environment variable management for gated model access (Stable Audio Open).
