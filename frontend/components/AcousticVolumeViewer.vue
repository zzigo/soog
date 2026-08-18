<template>
  <div ref="containerRef" class="volume-container"></div>
</template>

<script setup>
import { onMounted, onUnmounted, ref, watch } from 'vue';
import { sampleAcousticPalette } from '~/utils/acousticPalette';

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
  pointSize: {
    type: Number,
    default: 0.042,
  },
});

const containerRef = ref(null);

let scene;
let camera;
let renderer;
let controls;
let pointCloud;
let markerGroup;
let outlineMesh;
let frameId = null;
let resizeHandler = null;

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
  if (pointCloud) {
    scene?.remove(pointCloud);
    disposeObject(pointCloud);
    pointCloud = null;
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

async function buildScene() {
  if (!containerRef.value) return;
  const { THREE, OrbitControls } = await ensureThree();

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(44, 1, 0.1, 100);
  camera.position.set(2.3, 1.95, 2.8);

  renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
    powerPreference: 'high-performance',
  });
  renderer.setClearColor(0x000000, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.08;
  containerRef.value.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.42;
  controls.minDistance = 1.5;
  controls.maxDistance = 6.5;
  controls.target.set(0, 0, 0);

  scene.add(new THREE.AmbientLight(0x7486ff, 0.54));

  const cyan = new THREE.PointLight(0x00f6ff, 1.9, 12, 2);
  cyan.position.set(-1.8, 1.6, 2.2);
  scene.add(cyan);

  const magenta = new THREE.PointLight(0xff41f3, 2.15, 12, 2);
  magenta.position.set(2.2, 1.9, -1.1);
  scene.add(magenta);

  const amber = new THREE.PointLight(0xff8a2b, 1.45, 10, 2);
  amber.position.set(0.2, -1.2, 2.2);
  scene.add(amber);

  const box = new THREE.Box3Helper(new THREE.Box3(
    new THREE.Vector3(-1.1, -1.1, -1.1),
    new THREE.Vector3(1.1, 1.1, 1.1)
  ), 0x24386f);
  box.material.transparent = true;
  box.material.opacity = 0.3;
  scene.add(box);

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

  await renderVolume();

  const animate = () => {
    frameId = requestAnimationFrame(animate);
    controls?.update();
    renderer?.render(scene, camera);
  };
  animate();
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
    color: 0x6af7ff,
    transparent: true,
    opacity: 0.18,
  });
  outlineMesh = new THREE.LineSegments(wireGeometry, material);
  if (props.primitive === 'cylinder') {
    outlineMesh.rotation.x = Math.PI / 2;
  }
  scene.add(outlineMesh);
  geometry.dispose?.();
}

async function renderVolume() {
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
  const threshold = maxAbs * 0.05;
  const stride = depth > 28 ? 2 : 1;

  for (let z = 0; z < depth; z += 1) {
    for (let y = 0; y < rows; y += 1) {
      for (let x = 0; x < cols; x += 1) {
        if (((x + y + z) % stride) !== 0) continue;
        const value = Number(props.volume[z]?.[y]?.[x] || 0);
        if (Math.abs(value) < threshold) continue;
        const px = ((x / Math.max(cols - 1, 1)) * 2 - 1) * 1.05;
        const py = ((y / Math.max(rows - 1, 1)) * 2 - 1) * 1.05;
        const pz = ((z / Math.max(depth - 1, 1)) * 2 - 1) * 1.05;
        const color = sampleAcousticPalette(value / maxAbs, 0.88);
        positions.push(px, py, pz);
        colors.push(color.r / 255, color.g / 255, color.b / 255);
      }
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

  const material = new THREE.PointsMaterial({
    size: props.pointSize,
    vertexColors: true,
    transparent: true,
    opacity: 0.8,
    sizeAttenuation: true,
    depthWrite: false,
  });

  pointCloud = new THREE.Points(geometry, material);
  scene.add(pointCloud);

  markerGroup = new THREE.Group();
  const sphereGeometry = new THREE.SphereGeometry(0.043, 18, 18);
  for (const marker of props.markers || []) {
    const x = Number(marker?.x);
    const y = Number(marker?.y);
    const z = Number(marker?.z || 0);
    if (![x, y, z].every(Number.isFinite)) continue;

    const palette = {
      source: 0xff9a1f,
      probe: 0x00f6ff,
      obstacle: 0xffffff,
    };
    const color = palette[marker?.type] || 0xb9c4ff;
    const markerMaterial = new THREE.MeshStandardMaterial({
      color,
      emissive: color,
      emissiveIntensity: 1.05,
      transparent: true,
      opacity: 0.95,
    });
    const mesh = new THREE.Mesh(sphereGeometry, markerMaterial);
    mesh.position.set(x * 1.05, y * 1.05, z * 1.05);
    markerGroup.add(mesh);
  }
  scene.add(markerGroup);
}

onMounted(() => {
  if (process.server) return;
  buildScene();
});

watch(() => props.volume, () => {
  if (!process.server) renderVolume();
}, { deep: true });

watch(() => props.markers, () => {
  if (!process.server) renderVolume();
}, { deep: true });

watch(() => props.primitive, () => {
  if (!process.server) renderVolume();
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
.volume-container {
  width: 100%;
  height: 100%;
  min-height: 320px;
  background:
    radial-gradient(circle at top left, rgba(26, 36, 84, 0.24), rgba(5, 8, 20, 0.04) 60%),
    transparent;
}
</style>
