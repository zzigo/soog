<template>
  <div ref="containerRef" class="iso-container"></div>
</template>

<script setup>
import { onMounted, onUnmounted, ref, watch } from 'vue';
import { sampleAcousticPalette } from '~/utils/acousticPalette';
import { phaseDistance3d, temporalBandWeight, temporalFieldValue, wrapPhase } from '~/utils/acousticTemporal';

const props = defineProps({
  volume: {
    type: Array,
    required: true,
  },
  markers: {
    type: Array,
    default: () => [],
  },
  primitive: {
    type: String,
    default: 'sphere',
  },
  phase: {
    type: Number,
    default: 0,
  },
  pointSize: {
    type: Number,
    default: 0.074,
  },
});

const containerRef = ref(null);

let scene;
let camera;
let renderer;
let controls;
let shellCloud;
let markerGroup;
let outlineMesh;
let frameId = null;
let resizeHandler = null;
let blobTexture = null;

function disposeObject(object3d) {
  if (!object3d) return;
  object3d.traverse?.((child) => {
    child.geometry?.dispose?.();
    if (Array.isArray(child.material)) {
      child.material.forEach((material) => material?.dispose?.());
    } else {
      child.material?.dispose?.();
    }
  });
}

function stopLoop() {
  if (frameId) cancelAnimationFrame(frameId);
  frameId = null;
}

function clearScene() {
  if (shellCloud) {
    scene?.remove(shellCloud);
    disposeObject(shellCloud);
    shellCloud = null;
  }
  if (markerGroup) {
    scene?.remove(markerGroup);
    disposeObject(markerGroup);
    markerGroup = null;
  }
  if (outlineMesh) {
    scene?.remove(outlineMesh);
    disposeObject(outlineMesh);
    outlineMesh = null;
  }
}

async function ensureThree() {
  const THREE = await import('three');
  const { OrbitControls } = await import('three/examples/jsm/controls/OrbitControls.js');
  return { THREE, OrbitControls };
}

function makeBlobTexture() {
  if (blobTexture || typeof document === 'undefined') return blobTexture;
  const canvas = document.createElement('canvas');
  canvas.width = 96;
  canvas.height = 96;
  const ctx = canvas.getContext('2d');
  const gradient = ctx.createRadialGradient(48, 48, 6, 48, 48, 48);
  gradient.addColorStop(0, 'rgba(255,255,255,1)');
  gradient.addColorStop(0.28, 'rgba(255,255,255,0.92)');
  gradient.addColorStop(0.62, 'rgba(255,255,255,0.36)');
  gradient.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 96, 96);
  blobTexture = canvas;
  return blobTexture;
}

async function buildPrimitiveOutline(THREE) {
  let geometry;
  if (props.primitive === 'cube') {
    geometry = new THREE.BoxGeometry(1.6, 1.6, 1.6, 1, 1, 1);
  } else if (props.primitive === 'cylinder') {
    geometry = new THREE.CylinderGeometry(0.72, 0.72, 1.76, 30, 1, true);
  } else {
    geometry = new THREE.SphereGeometry(0.84, 26, 18);
  }

  const wireGeometry = new THREE.WireframeGeometry(geometry);
  const material = new THREE.LineBasicMaterial({
    color: 0x8da8c6,
    transparent: true,
    opacity: 0.16,
  });
  outlineMesh = new THREE.LineSegments(wireGeometry, material);
  if (props.primitive === 'cylinder') {
    outlineMesh.rotation.x = Math.PI / 2;
  }
  scene.add(outlineMesh);
  geometry.dispose?.();
}

async function renderShell() {
  if (!scene || !Array.isArray(props.volume) || !props.volume.length || !Array.isArray(props.volume[0])) return;
  const { THREE } = await ensureThree();
  clearScene();
  await buildPrimitiveOutline(THREE);

  const depth = props.volume.length;
  const rows = props.volume[0]?.length || 0;
  const cols = props.volume[0]?.[0]?.length || 0;
  if (!depth || !rows || !cols) return;

  let maxAbs = 0;
  for (let z = 0; z < depth; z += 1) {
    for (let y = 0; y < rows; y += 1) {
      for (let x = 0; x < cols; x += 1) {
        maxAbs = Math.max(maxAbs, Math.abs(Number(props.volume[z]?.[y]?.[x] || 0)));
      }
    }
  }
  maxAbs = maxAbs || 1;

  const positions = [];
  const colors = [];
  const phases = [];
  const cellCount = depth * rows * cols;
  const stride = cellCount > 140000 ? 3 : cellCount > 72000 ? 2 : 1;
  const phase = wrapPhase(props.phase);
  const bandCenter = 0.36 + 0.14 * Math.sin(phase * 0.5);
  const bandWidth = 0.07;

  for (let z = 0; z < depth; z += 1) {
    for (let y = 0; y < rows; y += 1) {
      for (let x = 0; x < cols; x += 1) {
        if (((x + y + z) % stride) !== 0) continue;
        const value = Number(props.volume[z]?.[y]?.[x] || 0);
        if (!value) continue;
        const px = ((x / Math.max(cols - 1, 1)) * 2 - 1) * 1.05;
        const py = ((y / Math.max(rows - 1, 1)) * 2 - 1) * 1.05;
        const pz = ((z / Math.max(depth - 1, 1)) * 2 - 1) * 1.05;
        const distance = phaseDistance3d(px, py, pz, props.markers);
        const bandValue = temporalBandWeight(value, phase, distance) / maxAbs;
        if (bandValue < 0.12) continue;
        if (Math.abs(bandValue - bandCenter) > bandWidth) continue;

        const signedValue = temporalFieldValue(value, phase, distance) / maxAbs;
        const color = sampleAcousticPalette(signedValue, 0.94);
        positions.push(px, py, pz);
        colors.push(color.r / 255, color.g / 255, color.b / 255);
        phases.push(distance);
      }
    }
  }

  const spriteTexture = new THREE.CanvasTexture(makeBlobTexture());
  spriteTexture.colorSpace = THREE.SRGBColorSpace;

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.setAttribute('phaseOffset', new THREE.Float32BufferAttribute(phases, 1));

  const material = new THREE.PointsMaterial({
    size: props.pointSize * (stride === 1 ? 1.08 : 1.22),
    vertexColors: true,
    transparent: true,
    opacity: 0.84,
    sizeAttenuation: true,
    depthWrite: false,
    map: spriteTexture,
    alphaMap: spriteTexture,
    alphaTest: 0.05,
    blending: THREE.AdditiveBlending,
  });

  shellCloud = new THREE.Points(geometry, material);
  scene.add(shellCloud);

  markerGroup = new THREE.Group();
  const sphereGeometry = new THREE.SphereGeometry(0.038, 18, 18);
  for (const marker of props.markers || []) {
    const x = Number(marker?.x);
    const y = Number(marker?.y);
    const z = Number(marker?.z || 0);
    if (![x, y, z].every(Number.isFinite)) continue;

    const palette = {
      source: 0xffb24c,
      probe: 0xcde7ff,
      obstacle: 0xf5f7fb,
    };
    const color = palette[marker?.type] || 0xb9c4ff;
    const markerMaterial = new THREE.MeshStandardMaterial({
      color,
      emissive: color,
      emissiveIntensity: 0.92,
      transparent: true,
      opacity: 0.94,
    });
    const mesh = new THREE.Mesh(sphereGeometry, markerMaterial);
    mesh.position.set(x * 1.05, y * 1.05, z * 1.05);
    markerGroup.add(mesh);
  }
  scene.add(markerGroup);
}

async function buildScene() {
  if (!containerRef.value) return;
  const { THREE, OrbitControls } = await ensureThree();

  scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x09111d, 0.16);
  camera = new THREE.PerspectiveCamera(44, 1, 0.1, 100);
  camera.position.set(2.12, 1.88, 2.78);

  renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
    powerPreference: 'high-performance',
  });
  renderer.setClearColor(0x000000, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.02;
  containerRef.value.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.22;
  controls.minDistance = 1.5;
  controls.maxDistance = 6.5;
  controls.target.set(0, 0, 0);

  scene.add(new THREE.AmbientLight(0xa2b6d8, 0.4));

  const coolKey = new THREE.PointLight(0x80c7ff, 1.44, 14, 2);
  coolKey.position.set(-1.6, 1.6, 2.2);
  scene.add(coolKey);

  const warmKey = new THREE.PointLight(0xffb65a, 1.28, 12, 2);
  warmKey.position.set(2.2, 1.54, -1.14);
  scene.add(warmKey);

  const rim = new THREE.PointLight(0x59d4c0, 1.08, 10, 2);
  rim.position.set(0.1, -1.12, 1.88);
  scene.add(rim);

  resizeHandler = () => {
    if (!containerRef.value || !renderer || !camera) return;
    const width = containerRef.value.clientWidth || 320;
    const height = containerRef.value.clientHeight || 240;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2.25));
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };

  window.addEventListener('resize', resizeHandler);
  resizeHandler();

  await renderShell();

  const animate = () => {
    frameId = requestAnimationFrame(animate);
    const now = performance.now() * 0.001;
    const phase = wrapPhase(props.phase);
    if (shellCloud?.material) {
      shellCloud.material.opacity = 0.64 + (0.5 + 0.5 * Math.sin(phase + now * 0.4)) * 0.2;
      shellCloud.material.size = props.pointSize * (1.08 + Math.sin(phase + now * 0.8) * 0.08);
    }
    if (outlineMesh) {
      outlineMesh.rotation.y = Math.sin(phase * 0.6) * 0.1;
      outlineMesh.rotation.x = props.primitive === 'cylinder' ? (Math.PI / 2) : Math.cos(phase * 0.36) * 0.05;
    }
    controls?.update();
    renderer?.render(scene, camera);
  };
  animate();
}

onMounted(() => {
  if (process.server) return;
  buildScene();
});

watch(() => props.volume, () => {
  if (!process.server) renderShell();
}, { deep: true });

watch(() => props.markers, () => {
  if (!process.server) renderShell();
}, { deep: true });

watch(() => props.primitive, () => {
  if (!process.server) renderShell();
});

watch(() => props.phase, () => {
  if (!process.server) renderShell();
});

onUnmounted(() => {
  stopLoop();
  if (resizeHandler) window.removeEventListener('resize', resizeHandler);
  resizeHandler = null;
  controls?.dispose?.();
  clearScene();
  if (renderer) {
    renderer.dispose?.();
    renderer.forceContextLoss?.();
    renderer.domElement?.remove?.();
  }
  scene = null;
  camera = null;
  renderer = null;
  controls = null;
});
</script>

<style scoped>
.iso-container {
  width: 100%;
  height: 100%;
  min-height: 260px;
  background: transparent;
}

.iso-container :deep(canvas) {
  display: block;
  width: 100%;
  height: 100%;
}
</style>
