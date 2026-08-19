<template>
  <div ref="containerRef" class="surface-container"></div>
</template>

<script setup>
import { onMounted, onUnmounted, ref, watch } from 'vue';
import { sampleAcousticPalette } from '~/utils/acousticPalette';
import {
  normalizePlaneKey,
  phaseDistance2d,
  projectMarkerToPlane,
  temporalFieldValue,
} from '~/utils/acousticTemporal';

const props = defineProps({
  data: {
    type: Array,
    required: true,
  },
  markers: {
    type: Array,
    default: () => [],
  },
  plane: {
    type: String,
    default: 'xy',
  },
  phase: {
    type: Number,
    default: 0,
  },
  heightScale: {
    type: Number,
    default: 0.4,
  },
});

const containerRef = ref(null);

let scene;
let camera;
let renderer;
let controls;
let surfaceMesh;
let wireMesh;
let markerGroup;
let frameId = null;
let resizeHandler = null;

function disposeObject(object3d) {
  if (!object3d) return;
  object3d.traverse?.((child) => {
    if (child.geometry) child.geometry.dispose?.();
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
  if (surfaceMesh) {
    scene?.remove(surfaceMesh);
    disposeObject(surfaceMesh);
    surfaceMesh = null;
  }
  if (wireMesh) {
    scene?.remove(wireMesh);
    disposeObject(wireMesh);
    wireMesh = null;
  }
  if (markerGroup) {
    scene?.remove(markerGroup);
    disposeObject(markerGroup);
    markerGroup = null;
  }
}

async function ensureThree() {
  const THREE = await import('three');
  const { OrbitControls } = await import('three/examples/jsm/controls/OrbitControls.js');
  return { THREE, OrbitControls };
}

function markerHeightAt(data, x, y, heightScale, phase, plane, markers) {
  if (!Array.isArray(data) || !data.length || !Array.isArray(data[0]) || !data[0].length) return 0;
  const rows = data.length;
  const cols = data[0].length;
  const col = Math.min(cols - 1, Math.max(0, Math.round(((x + 1) * 0.5) * (cols - 1))));
  const row = Math.min(rows - 1, Math.max(0, Math.round((1 - ((y + 1) * 0.5)) * (rows - 1))));
  const distance = phaseDistance2d(row, col, rows, cols, markers, plane);
  return temporalFieldValue(data[row]?.[col], phase, distance) * heightScale;
}

async function buildScene() {
  if (!containerRef.value) return;
  const { THREE, OrbitControls } = await ensureThree();

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
  camera.position.set(0, 1.45, 2.65);

  renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
    powerPreference: 'high-performance',
  });
  renderer.setClearColor(0x000000, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.18;

  containerRef.value.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.5;
  controls.minDistance = 1.3;
  controls.maxDistance = 5.5;
  controls.target.set(0, 0.05, 0);

  const ambient = new THREE.AmbientLight(0x6a7dff, 0.6);
  scene.add(ambient);

  const cyanLight = new THREE.PointLight(0x00f6ff, 2.2, 12, 2);
  cyanLight.position.set(-1.8, 1.4, 1.9);
  scene.add(cyanLight);

  const magentaLight = new THREE.PointLight(0xff41f3, 2.4, 12, 2);
  magentaLight.position.set(2.1, 1.8, -0.7);
  scene.add(magentaLight);

  const orangeLight = new THREE.PointLight(0xff8a2b, 1.4, 9, 2);
  orangeLight.position.set(0.1, 0.4, 2.5);
  scene.add(orangeLight);

  const grid = new THREE.GridHelper(2.8, 12, 0x2ff3ff, 0x1c2b58);
  grid.position.y = -0.44;
  grid.material.transparent = true;
  grid.material.opacity = 0.18;
  scene.add(grid);

  resizeHandler = () => {
    if (!containerRef.value || !renderer || !camera) return;
    const width = containerRef.value.clientWidth || 320;
    const height = containerRef.value.clientHeight || 240;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };

  window.addEventListener('resize', resizeHandler);
  resizeHandler();

  renderSurface();

  const animate = () => {
    frameId = requestAnimationFrame(animate);
    controls?.update();
    renderer?.render(scene, camera);
  };
  animate();
}

async function renderSurface() {
  if (!scene || !Array.isArray(props.data) || !props.data.length || !Array.isArray(props.data[0])) return;
  const { THREE } = await ensureThree();
  clearScene();

  const rows = props.data.length;
  const cols = props.data[0].length;
  const planeKey = normalizePlaneKey(props.plane);
  const geometry = new THREE.PlaneGeometry(2.2, 2.2, cols - 1, rows - 1);
  const position = geometry.attributes.position;
  const colors = [];
  const values = [];
  let maxAbs = 0;

  for (let row = 0; row < rows; row += 1) {
    values[row] = [];
    for (let col = 0; col < cols; col += 1) {
      const distance = phaseDistance2d(row, col, rows, cols, props.markers, planeKey);
      const value = temporalFieldValue(props.data[row]?.[col], props.phase, distance);
      values[row][col] = value;
      maxAbs = Math.max(maxAbs, Math.abs(value));
    }
  }
  maxAbs = maxAbs || 1;

  let pointer = 0;
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const value = Number(values[row]?.[col] || 0);
      position.setZ(pointer, value * props.heightScale);
      const color = sampleAcousticPalette(value / maxAbs, 1);
      colors.push(color.r / 255, color.g / 255, color.b / 255);
      pointer += 1;
    }
  }

  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.computeVertexNormals();

  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    metalness: 0.16,
    roughness: 0.28,
    transparent: true,
    opacity: 0.92,
    side: THREE.DoubleSide,
    emissive: new THREE.Color(0x0b1028),
    emissiveIntensity: 0.36,
  });

  surfaceMesh = new THREE.Mesh(geometry, material);
  surfaceMesh.rotation.x = -Math.PI / 2.1;
  surfaceMesh.position.y = -0.05;
  scene.add(surfaceMesh);

  const wireGeometry = new THREE.WireframeGeometry(geometry);
  const wireMaterial = new THREE.LineBasicMaterial({
    color: 0x6af7ff,
    transparent: true,
    opacity: 0.18,
  });
  wireMesh = new THREE.LineSegments(wireGeometry, wireMaterial);
  wireMesh.rotation.copy(surfaceMesh.rotation);
  wireMesh.position.copy(surfaceMesh.position);
  scene.add(wireMesh);

  markerGroup = new THREE.Group();
  const sphereGeometry = new THREE.SphereGeometry(0.028, 16, 16);
  for (const marker of props.markers || []) {
    const projected = projectMarkerToPlane(marker, planeKey);
    const x = Number(projected.u);
    const y = Number(projected.v);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;

    const palette = {
      source: 0xff9a1f,
      probe: 0x00f6ff,
      obstacle: 0xffffff,
    };
    const markerMaterial = new THREE.MeshStandardMaterial({
      color: palette[marker?.type] || 0xb9c4ff,
      emissive: palette[marker?.type] || 0x5162ff,
      emissiveIntensity: 1.1,
      transparent: true,
      opacity: 0.96,
    });

    const mesh = new THREE.Mesh(sphereGeometry, markerMaterial);
    mesh.position.set(
      x * 1.1,
      markerHeightAt(props.data, x, y, props.heightScale, props.phase, planeKey, props.markers) + 0.08,
      y * 1.1
    );
    markerGroup.add(mesh);
  }
  scene.add(markerGroup);
}

onMounted(() => {
  if (process.server) return;
  buildScene();
});

watch(() => props.data, () => {
  if (!process.server) renderSurface();
}, { deep: true });

watch(() => props.markers, () => {
  if (!process.server) renderSurface();
}, { deep: true });

watch(() => props.plane, () => {
  if (!process.server) renderSurface();
});

watch(() => props.phase, () => {
  if (!process.server) renderSurface();
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
.surface-container {
  width: 100%;
  height: 100%;
  min-height: 240px;
  background: transparent;
}

.surface-container :deep(canvas) {
  display: block;
  width: 100%;
  height: 100%;
}
</style>
