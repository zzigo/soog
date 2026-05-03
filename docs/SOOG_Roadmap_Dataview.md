# SOOG Roadmap (DataviewJS)

```dataviewjs
const roadmapData = [
  {
    date: 'Dec 2024',
    title: 'Conceptual Inception',
    synopsis: 'Initial design of SOOG as the pioneering generative system for musical instrument notation based on Mantle Hood’s organogram technique. Implementation of large-scale models via GPT-4o and DeepSeek-V3 APIs to translate natural language into structured organological representations.'
  },
  {
    date: 'April 2025',
    title: 'Prototyping & 3D Synthesis',
    synopsis: 'Development of the first physical-digital prototype, Phosphorbone. Refinement of prompt engineering and expansion of the organological dictionary. Introduction of the 3D prototyping module using Midjourney v6 and SDXL for visual aesthetics, establishing the foundation for future digital fabrication workflows.'
  },
  {
    date: 'Dec 2025',
    title: 'Local Intelligence',
    current: true,
    synopsis: 'Transition to a hybrid infrastructure utilizing local, open-source models (Ollama, Qwen 2.5) for logic and Matplotlib generation. Integration of specialized creative pipelines: Stable Diffusion (SDXL-Turbo) for industrial design sketches and Stable Audio Open for timbral synthesis.'
  },
  {
    date: 'July 2026',
    title: 'SoMap & Generative CAD',
    synopsis: 'Completion of SoMap, a comprehensive Markdown database of the MOIAE system. Transitioning to direct 3D output via Unique3D or InstantMesh for high-fidelity STL mesh synthesis from 2D sketches. Implementation of generative hardware design using Flux.ai and DeepPCB for AI-assisted routing.'
  },
  {
    date: '2027',
    title: 'Physical Realization',
    synopsis: 'Construction of the first generation of SOOG-built physical instruments. Integration of real-time acoustical BEM/FEM simulations using NVIDIA Modulus (Physics-Informed Neural Networks). Implementation of DDSP to bridge physical resonance data with real-time neural synthesis models.'
  }
];

// Inyectar Estilos
const style = document.createElement('style');
style.innerHTML = `
  .soog-roadmap-wrapper {
    padding: 60px 20px 40px 20px;
    font-family: var(--font-interface, ui-monospace, 'IBM Plex Mono', monospace);
    background: #000;
    border-radius: 8px;
    margin: 20px 0;
  }
  .roadmap-container {
    position: relative;
    width: 100%;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
  }
  .roadmap-line {
    position: absolute;
    top: 6px;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #4CAF50 0%, #2196F3 50%, #9C27B0 100%);
    opacity: 0.3;
    z-index: 0;
  }
  .roadmap-point {
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 18%;
    z-index: 1;
    cursor: help;
  }
  .point-dot {
    width: 12px;
    height: 12px;
    background: #1a1a1a;
    border: 2px solid var(--dot-color);
    border-radius: 50%;
    transition: all 0.3s ease;
    margin-bottom: 15px;
  }
  .roadmap-point:hover .point-dot {
    background: #fff;
    transform: scale(1.3);
    box-shadow: 0 0 15px var(--dot-color);
  }
  .roadmap-point.is-current .point-dot {
    background: #fff;
    box-shadow: 0 0 12px #2196F3;
    transform: scale(1.1);
  }
  .point-info {
    text-align: center;
  }
  .point-date {
    font-size: 0.65rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
  }
  .point-title {
    font-size: 0.75rem;
    color: #eee;
    font-weight: 600;
    line-height: 1.2;
  }
  /* Tooltip logic */
  .point-synopsis {
    position: absolute;
    bottom: 130%;
    left: 50%;
    transform: translateX(-50%) translateY(10px);
    width: 280px;
    background: rgba(20, 20, 20, 0.95);
    border: 1px solid #444;
    border-radius: 6px;
    padding: 12px;
    color: #ccc;
    font-size: 0.8rem;
    line-height: 1.5;
    opacity: 0;
    visibility: hidden;
    transition: all 0.3s ease;
    z-index: 10;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    pointer-events: none;
    backdrop-filter: blur(5px);
  }
  .roadmap-point:hover .point-synopsis {
    opacity: 1;
    visibility: visible;
    transform: translateX(-50%) translateY(0);
  }
  .point-synopsis b {
    color: var(--dot-color);
    display: block;
    margin-bottom: 6px;
    font-size: 0.85rem;
  }
`;
dv.container.appendChild(style);

// Colores del degradado para los puntos
const colors = ['#4CAF50', '#38a595', '#2196F3', '#5e5ebf', '#9C27B0'];

// Renderizar HTML
const root = dv.el('div', '', { cls: 'soog-roadmap-wrapper' });
const container = root.createEl('div', { cls: 'roadmap-container' });
container.createEl('div', { cls: 'roadmap-line' });

roadmapData.forEach((item, i) => {
  const point = container.createEl('div', { cls: `roadmap-point ${item.current ? 'is-current' : ''}` });
  point.style.setProperty('--dot-color', colors[i]);
  
  // Dot
  point.createEl('div', { cls: 'point-dot' });
  
  // Labels
  const info = point.createEl('div', { cls: 'point-info' });
  info.createEl('div', { cls: 'point-date', text: item.date });
  info.createEl('div', { cls: 'point-title', text: item.title });
  
  // Tooltip
  const tooltip = point.createEl('div', { cls: 'point-synopsis' });
  tooltip.createEl('b', { text: item.title });
  tooltip.createEl('span', { text: item.synopsis });
});
```
