<template>
  <div 
    class="app-container" 
    :class="{ 
      'sorganoid-active': sorganoidMode,
      'ui-hidden': !uiVisible,
      'workspace-background-mode': workspaceLayout === 'background',
      'workspace-split-mode': workspaceLayout === 'split'
    }"
  >
    <!-- Left column: resizable editor -->
    <div 
      class="left-column"
      :class="{ 'left-column--background': workspaceLayout === 'background' }"
      :style="leftColumnStyle"
    >
      <div class="editor-wrapper">
        <AceEditor 
          ref="editorRef" 
          @evaluate="handleEvaluate"
          :terminal-mode="sorganoidMode"
        />
      </div>
    </div>
    <!-- Draggable divider -->
    <div 
      v-if="!sorganoidMode && uiVisible && workspaceLayout === 'split'"
      class="divider" 
      @mousedown="startDrag" 
      @touchstart.prevent="startDragTouch"
    ></div>
    <!-- Right column hosts workspace HUD and results -->
    <div
      v-if="!sorganoidMode && uiVisible"
      class="right-column"
      :class="{ 'right-column--background': workspaceLayout === 'background' }"
      :style="rightColumnStyle"
    >
      <div class="hud">
        <div class="hud-group hud-group--workspace">
          <button
            class="layout-pill"
            :class="{ active: workspaceLayout === 'background' }"
            @click="setWorkspaceLayout('background')"
            title="Background workspace"
          >
            BG
          </button>
          <button
            class="layout-pill"
            :class="{ active: workspaceLayout === 'split' }"
            @click="setWorkspaceLayout('split')"
            title="Split workspace"
          >
            SPLIT
          </button>
          <button
            class="icon-button"
            :class="{ active: workspaceSidebarOpen }"
            @click="toggleWorkspaceSidebar"
            title="Workspace sidebar (⌘/Ctrl + Shift + \\)"
          >
            <svg class="icon" viewBox="0 0 24 24">
              <path fill="currentColor" d="M3 5H21V7H3V5M3 11H21V13H3V11M3 17H9V19H3V17M13 17H21V19H13V17Z" />
            </svg>
          </button>
        </div>

        <div v-if="workspaceLayout === 'background'" class="hud-group hud-group--views">
          <button
            v-for="item in availableBackgroundViews"
            :key="item.key"
            class="layout-pill layout-pill--view"
            :class="{ active: activeBackgroundView === item.key }"
            :disabled="!item.enabled"
            @click="setBackgroundView(item.key)"
            :title="item.label"
          >
            {{ item.label }}
          </button>
        </div>

        <div class="hud-group hud-group--main">
          <span class="model-name" :title="currentModel || ''">{{ shortModelLabel }}</span>
          <button
            @click="cycleOllamaModel"
            class="icon-button model-cycle-button"
            :disabled="modelSwitching || ollamaModels.length < 2"
            :title="`Model: ${currentModel || 'loading...'} (click to cycle)`"
          >
            <svg class="icon" viewBox="0 0 24 24">
              <path fill="currentColor" d="M12,4V1L8,5L12,9V6A6,6 0 0,1 18,12C18,13 17.75,13.96 17.3,14.8L18.76,16.26C19.53,15.05 20,13.57 20,12A8,8 0 0,0 12,4M6.7,9.2L5.24,7.74C4.47,8.95 4,10.43 4,12A8,8 0 0,0 12,20V23L16,19L12,15V18A6,6 0 0,1 6,12C6,11 6.25,10.04 6.7,9.2Z"/>
            </svg>
          </button>
          <button @click="toggleShowCode" class="icon-button" :title="showCode ? 'Hide Code' : 'Show Code'">
            <svg v-if="showCode" class="icon" viewBox="0 0 24 24">
              <path fill="currentColor" d="M12,9A3,3 0 0,1 15,12A3,3 0 0,1 12,15A3,3 0 0,1 9,12A3,3 0 0,1 12,9M12,4.5C17,4.5 21.27,7.61 23,12C21.27,16.39 17,19.5 12,19.5C7,19.5 2.73,16.39 1,12C2.73,7.61 7,4.5 12,4.5M3.18,12C4.83,15.36 8.24,17.5 12,17.5C15.76,17.5 19.17,15.36 20.82,12C19.17,8.64 15.76,6.5 12,6.5C8.24,6.5 4.83,8.64 3.18,12Z" />
            </svg>
            <svg v-else class="icon" viewBox="0 0 24 24">
              <path fill="currentColor" d="M11.83,9L15,12.16C15,12.11 15,12.05 15,12A3,3 0 0,0 12,9C11.94,9 11.89,9 11.83,9M7.53,9.8L9.08,11.35C9.03,11.56 9,11.77 9,12A3,3 0 0,0 12,15C12.22,15 12.44,14.97 12.65,14.92L14.2,16.47C13.53,16.8 12.79,17 12,17A5,5 0 0,1 7,12C7,11.21 7.2,10.47 7.53,9.8M2,4.27L4.28,6.55L4.73,7C3.08,8.3 1.78,10 1,12C2.73,16.39 7,19.5 12,19.5C13.55,19.5 15.03,19.2 16.38,18.66L16.81,19.08L19.73,22L21,20.73L3.27,3M12,7A5,5 0 0,1 17,12C17,12.64 16.87,13.26 16.64,13.82L19.57,16.75C21.07,15.5 22.27,13.86 23,12C21.27,7.61 17,4.5 12,4.5C10.6,4.5 9.26,4.75 8,5.2L10.17,7.35C10.74,7.13 11.35,7 12,7Z" />
            </svg>
          </button>
          <button @click="handleMobileEvaluate" class="icon-button" title="Evaluate selected text or all if no selection">
            <svg class="icon" viewBox="0 0 24 24">
              <path fill="currentColor" d="M8,5.14V19.14L19,12.14L8,5.14Z" />
            </svg>
          </button>
          <button @click="handleClear" class="icon-button" title="Clear Editor (Ctrl+H)">
            <svg class="icon" viewBox="0 0 24 24">
              <path fill="currentColor" d="M9,3V4H4V6H5V19A2,2 0 0,0 7,21H17A2,2 0 0,0 19,19V6H20V4H15V3H9M7,6H17V19H7V6M9,8V17H11V8H9M13,8V17H15V8H13Z" />
            </svg>
          </button>
          <button @click="handleRandomPrompt" class="icon-button" title="Random Prompt">
            <svg class="icon" viewBox="0 0 24 24">
              <path fill="currentColor" d="M14.83,13.41L13.42,14.82L16.55,17.95L14.5,20H20V14.5L17.96,16.54L14.83,13.41M14.5,4L16.54,6.04L4,18.59L5.41,20L17.96,7.46L20,9.5V4M10.59,9.17L5.41,4L4,5.41L9.17,10.58L10.59,9.17Z" />
            </svg>
          </button>
          <button @click="showHelp = true" class="icon-button" title="Help">
            <svg class="icon" viewBox="0 0 24 24">
              <path fill="currentColor" d="M11,18H13V16H11V18M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M12,20C7.59,20 4,16.41 4,12C4,7.59 7.59,4 12,4C16.41,4 20,7.59 20,12C20,16.41 16.41,20 12,20M12,6A4,4 0 0,0 8,10H10A2,2 0 0,1 12,8A2,2 0 0,1 14,10C14,12 11,11.75 11,15H13C13,12.75 16,12.5 16,10A4,4 0 0,0 12,6Z" />
            </svg>
          </button>
          <button @click="showGallery = true" class="icon-button" title="Gallery (Alt+↑/↓)">
            <svg class="icon" viewBox="0 0 24 24">
              <path fill="currentColor" d="M21 19V5C21 3.89 20.1 3 19 3H5C3.89 3 3 3.89 3 5V19C3 20.1 3.89 21 5 21H19C20.1 21 21 20.1 21 19M8.5 13.5L11 16.51L14.5 12L19 18H5L8.5 13.5Z"/>
            </svg>
          </button>
          <button
            @click="toggleSomap"
            class="icon-button"
            :class="{ active: $route.path === '/somap' }"
            title="Somap (Alt+2)"
          >
            <svg class="icon" viewBox="0 0 24 24">
              <circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="2" fill="none"/>
              <circle cx="12" cy="12" r="4" fill="currentColor" opacity="0.8"/>
              <path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1" stroke="currentColor" stroke-width="1.2" fill="none"/>
            </svg>
          </button>
          <button
            @click="$router.push('/concert')"
            class="icon-button"
            title="3D Concert Hall (Alt+3)"
          >
            <svg class="icon" viewBox="0 0 24 24">
              <path fill="currentColor" d="M12,3L2,12H5V20H19V12H22L12,3M11.5,18V14H12.5V18H11.5M9,18V16H10V18H9M14,18V16H15V18H14M11.5,13V12H12.5V13H11.5M9,15V14H10V15H9M14,15V14H15V15H14Z" />
            </svg>
          </button>
          <NuxtLink
            v-if="profile?.is_admin"
            to="/usage"
            class="auth-pill admin-pill"
            title="Daily and weekly usage"
          >
            USAGE
          </NuxtLink>
          <span v-if="isAuthenticated && quota" class="quota-pill" :title="quotaTitle">{{ quotaLabel }}</span>
          <button
            v-if="isAuthenticated"
            @click="handleSignOut"
            class="auth-pill"
            :title="`${userLabel} · sign out`"
          >
            {{ userLabel }}
          </button>
          <button v-else @click="showAuthGate = true" class="auth-pill" title="Sign in">SIGN IN</button>
        </div>
      </div>

      <Transition
        enter-active-class="fadeIn"
        leave-active-class="fadeOut"
        :duration="300"
        mode="out-in"
      >
        <div v-if="hasResults && workspaceLayout === 'split'" class="results-panel" :key="transitionKey">
          <div class="results-split">
            <section class="panel panel-organogram">
              <h3 class="section-title">ORGANOGRAM</h3>
              <img
                v-if="plotImage"
                :src="plotImage"
                alt="Organogram"
                @click="openLightbox(plotImage, 'Organogram')"
                class="plot-image"
              />
              <div v-else class="panel-placeholder">Live acoustic previews do not generate an organogram image.</div>
            </section>

            <section class="panel panel-text">
              <div class="section-header">
                <h3 class="section-title">CONCEPTUAL SUMMARY</h3>
                <span v-if="responseMetaText" class="section-meta">{{ responseMetaText }}</span>
              </div>
              <div v-if="summaryHtml" class="summary-content" v-html="summaryHtml"></div>
              <div v-else class="panel-placeholder">Evaluate a generation to populate conceptual summary and materials.</div>
              <div class="section-header materials-title">
                <h3 class="section-title">MATERIALS</h3>
                <span v-if="responseMetaText" class="section-meta">{{ responseMetaText }}</span>
              </div>
              <pre v-if="materialsText" class="materials-list">{{ materialsText }}</pre>
              <div v-else class="panel-placeholder">No material list yet.</div>
            </section>

            <section class="panel panel-visualizer">
              <div class="section-header tab-header">
                <div class="tabs">
                  <button
                    v-for="item in splitTabOptions"
                    :key="item.key"
                    class="tab-btn"
                    :class="{ active: viewMode === item.key }"
                    :disabled="!item.enabled"
                    @click="viewMode = item.key"
                  >
                    {{ item.label }}
                  </button>
                </div>
                <div class="section-meta-group">
                  <button v-if="viewMode === 'stl' && stlUrl" @click="downloadCurrentStl" class="download-btn">
                    Download STL
                  </button>
                  <span v-if="viewMode === 'sketch' && sketchModel" class="section-meta">{{ sketchModel }}</span>
                  <button
                    v-if="viewMode === 'sketch' && hasResults"
                    @click="remakeSketch"
                    class="remake-btn-small"
                    :disabled="loading"
                  >
                    {{ loading ? '...' : 'GENERATE' }}
                  </button>
                  <template v-if="viewMode === 'modulus' && activeModulusData">
                    <div class="modulus-mode-toggle">
                      <button
                        class="modulus-mode-btn"
                        :class="{ active: modulusRenderMode === '2d' }"
                        @click="modulusRenderMode = '2d'"
                      >
                        2D
                      </button>
                      <button
                        class="modulus-mode-btn"
                        :class="{ active: modulusRenderMode === '3d' }"
                        @click="modulusRenderMode = '3d'"
                      >
                        3D
                      </button>
                    </div>
                    <span class="section-meta">
                      {{ activeModulusData.method || 'SOT-A' }}
                    </span>
                  </template>
                </div>
              </div>

              <div class="tab-content">
                <div v-show="viewMode === 'stl'" class="tab-pane">
                  <ClientOnly>
                    <div v-if="stlUrl" class="stl-viewer-container">
                      <StlViewer :url="stlUrl" />
                    </div>
                    <div v-else class="stl-placeholder">No STL geometry generated for this response.</div>
                  </ClientOnly>
                </div>
                <div v-show="viewMode === 'sketch'" class="tab-pane">
                  <img
                    v-if="sketchImage"
                    :src="sketchImage"
                    alt="Sketch render"
                    @click="openLightbox(sketchImage, 'Sketch')"
                    class="plot-image"
                  />
                  <div v-else class="sketch-placeholder">No diffusion sketch generated for this response.</div>
                </div>
                <div v-show="viewMode === 'modulus'" class="tab-pane">
                  <div v-if="activeModulusData" class="modulus-results">
                    <div v-if="activeModulusData.results && activeModulusData.results.pressure_map" class="modulus-stack">
                      <ModulusHeatmap
                        v-if="modulusRenderMode === '2d'"
                        :data="activeModulusData.results.pressure_map"
                        :markers="modulusMarkers"
                      />
                      <ClientOnly v-else>
                        <div class="modulus-surface-shell">
                          <AcousticFieldSurface
                            :data="activeModulusData.results.pressure_map"
                            :markers="modulusMarkers"
                          />
                        </div>
                      </ClientOnly>
                      <div class="modulus-readout">
                        <span>Probe {{ acousticProbeResponseText }}</span>
                        <span>Peaks {{ acousticPeakSummary }}</span>
                        <span v-if="liveAcousticCommand">Live code-linked preview</span>
                      </div>
                    </div>
                    <pre v-else class="modulus-json">{{ JSON.stringify(activeModulusData.results || activeModulusData, null, 2) }}</pre>
                  </div>
                  <div v-else class="sketch-placeholder">No acoustical simulation data for this response.</div>
                </div>
              </div>
            </section>
          </div>
        </div>

        <div v-else-if="hasResults && workspaceLayout === 'background'" class="background-stage" :key="`${transitionKey}-${activeBackgroundView}`">
          <div class="background-stage__meta">
            <span class="background-stage__label">{{ backgroundStageLabel }}</span>
            <span v-if="responseMetaText" class="background-stage__sub">{{ responseMetaText }}</span>
          </div>

          <img
            v-if="activeBackgroundView === 'plot' && plotImage"
            :src="plotImage"
            alt="Organogram background"
            class="background-stage__image"
            @click="openLightbox(plotImage, 'Organogram')"
          />
          <ClientOnly v-else-if="activeBackgroundView === 'stl'">
            <div v-if="stlUrl" class="background-stage__viewer">
              <StlViewer :url="stlUrl" />
            </div>
            <div v-else class="stl-placeholder">No STL geometry generated for this response.</div>
          </ClientOnly>
          <img
            v-else-if="activeBackgroundView === 'sketch' && sketchImage"
            :src="sketchImage"
            alt="Sketch background"
            class="background-stage__image"
            @click="openLightbox(sketchImage, 'Sketch')"
          />
          <div v-else-if="activeBackgroundView === 'modulus'" class="background-stage__heatmap">
            <div v-if="activeModulusData?.results?.pressure_map" class="background-stage__acoustic">
              <div class="background-stage__acoustic-hud">
                <button
                  class="modulus-mode-btn"
                  :class="{ active: modulusRenderMode === '2d' }"
                  @click="modulusRenderMode = '2d'"
                >
                  2D
                </button>
                <button
                  class="modulus-mode-btn"
                  :class="{ active: modulusRenderMode === '3d' }"
                  @click="modulusRenderMode = '3d'"
                >
                  3D
                </button>
              </div>
              <ModulusHeatmap
                v-if="modulusRenderMode === '2d'"
                :data="activeModulusData.results.pressure_map"
                :size="720"
                :markers="modulusMarkers"
              />
              <ClientOnly v-else>
                <div class="background-stage__surface">
                  <AcousticFieldSurface
                    :data="activeModulusData.results.pressure_map"
                    :markers="modulusMarkers"
                  />
                </div>
              </ClientOnly>
            </div>
            <div v-else class="sketch-placeholder">No acoustical simulation data for this response.</div>
          </div>
        </div>
      </Transition>
    </div>

    <aside
      v-if="!sorganoidMode && uiVisible"
      class="workspace-sidebar"
      :class="{ 'workspace-sidebar--open': workspaceSidebarOpen }"
    >
      <div class="workspace-sidebar__rail">
        <button
          class="workspace-sidebar__tab"
          :class="{ active: workspaceSidebarTab === 'help' }"
          @click="workspaceSidebarOpen = true; workspaceSidebarTab = 'help'"
        >
          HELP
        </button>
        <button
          class="workspace-sidebar__tab"
          :class="{ active: workspaceSidebarTab === 'session' }"
          @click="workspaceSidebarOpen = true; workspaceSidebarTab = 'session'"
        >
          SESSION
        </button>
        <button
          class="workspace-sidebar__tab"
          :class="{ active: workspaceSidebarTab === 'acoustic' }"
          @click="workspaceSidebarOpen = true; workspaceSidebarTab = 'acoustic'"
        >
          SOT-A
        </button>
      </div>

      <div class="workspace-sidebar__panel">
        <div class="workspace-sidebar__header">
          <span>{{ workspaceSidebarTab.toUpperCase() }}</span>
          <button class="icon-button workspace-sidebar__close" @click="toggleWorkspaceSidebar" title="Collapse sidebar">
            <svg class="icon" viewBox="0 0 24 24">
              <path fill="currentColor" d="M19,13H5V11H19V13Z" />
            </svg>
          </button>
        </div>

        <div v-if="workspaceSidebarTab === 'help'" class="workspace-sidebar__body">
          <div v-for="domain in commandDomains" :key="domain.key" class="command-domain">
            <div class="command-domain__header">
              <span class="command-domain__badge" :style="{ '--domain-color': domain.color }">{{ domain.label }}</span>
            </div>
            <div v-for="command in domain.commands" :key="command.syntax" class="command-domain__row">
              <code>{{ command.syntax }}</code>
              <p>{{ command.detail }}</p>
            </div>
          </div>
        </div>

        <div v-else-if="workspaceSidebarTab === 'session'" class="workspace-sidebar__body">
          <div class="session-facts">
            <div v-for="fact in sessionFacts" :key="fact.label" class="session-fact">
              <span>{{ fact.label }}</span>
              <strong>{{ fact.value }}</strong>
            </div>
          </div>
          <div class="session-summary">
            <h4>Summary</h4>
            <div class="summary-content" v-html="summaryHtml || '<p>No summary yet.</p>'"></div>
          </div>
        </div>

        <div v-else class="workspace-sidebar__body">
          <div class="acoustic-summary">
            <h4>Live Command</h4>
            <p>{{ acousticCommandSummary }}</p>
          </div>
          <div v-if="!liveAcousticCommand" class="acoustic-summary">
            <h4>First Primitive</h4>
            <p>Use <code>@acoustic primitive=circle freq=440 source=-0.55,0 probe=0.55,0 obstacle=0.15,0</code> and evaluate it directly from the editor.</p>
            <button class="sidebar-action" @click="insertStarterAcousticCommand">Insert starter command</button>
          </div>
          <template v-else>
            <div class="acoustic-actions">
              <button class="sidebar-action" @click="evaluateCurrentAcousticCommand">Evaluate SOT-A</button>
              <button class="sidebar-action" @click="modulusRenderMode = modulusRenderMode === '2d' ? '3d' : '2d'">
                {{ modulusRenderMode === '2d' ? 'Switch to 3D' : 'Switch to 2D' }}
              </button>
            </div>
            <div class="acoustic-primitive-row">
              <button
                v-for="primitive in ['circle', 'square', 'triangle', 'hexagon']"
                :key="primitive"
                class="primitive-chip"
                :class="{ active: acousticSliderState.primitive === primitive }"
                @click="updateAcousticCommandInEditor({ primitive })"
              >
                {{ primitive }}
              </button>
            </div>
            <div class="acoustic-slider-grid">
              <label class="slider-unit slider-unit--wide">
                <span>Frequency</span>
                <strong>{{ acousticSliderState.freq }} Hz</strong>
                <input
                  type="range"
                  min="40"
                  max="2400"
                  step="10"
                  :value="acousticSliderState.freq"
                  @input="updateAcousticCommandInEditor({ freq: Number($event.target.value) })"
                >
              </label>
              <label class="slider-unit">
                <span>Source X</span>
                <strong>{{ acousticSliderState.sourceX.toFixed(2) }}</strong>
                <input type="range" min="-0.95" max="0.95" step="0.01" :value="acousticSliderState.sourceX" @input="updateAcousticSliderValue('source', 0, Number($event.target.value))">
              </label>
              <label class="slider-unit">
                <span>Source Y</span>
                <strong>{{ acousticSliderState.sourceY.toFixed(2) }}</strong>
                <input type="range" min="-0.95" max="0.95" step="0.01" :value="acousticSliderState.sourceY" @input="updateAcousticSliderValue('source', 1, Number($event.target.value))">
              </label>
              <label class="slider-unit">
                <span>Probe X</span>
                <strong>{{ acousticSliderState.probeX.toFixed(2) }}</strong>
                <input type="range" min="-0.95" max="0.95" step="0.01" :value="acousticSliderState.probeX" @input="updateAcousticSliderValue('probe', 0, Number($event.target.value))">
              </label>
              <label class="slider-unit">
                <span>Probe Y</span>
                <strong>{{ acousticSliderState.probeY.toFixed(2) }}</strong>
                <input type="range" min="-0.95" max="0.95" step="0.01" :value="acousticSliderState.probeY" @input="updateAcousticSliderValue('probe', 1, Number($event.target.value))">
              </label>
              <label class="slider-unit">
                <span>Obstacle X</span>
                <strong>{{ acousticSliderState.obstacleX.toFixed(2) }}</strong>
                <input type="range" min="-0.95" max="0.95" step="0.01" :value="acousticSliderState.obstacleX" @input="updateAcousticSliderValue('obstacle', 0, Number($event.target.value))">
              </label>
              <label class="slider-unit">
                <span>Obstacle Y</span>
                <strong>{{ acousticSliderState.obstacleY.toFixed(2) }}</strong>
                <input type="range" min="-0.95" max="0.95" step="0.01" :value="acousticSliderState.obstacleY" @input="updateAcousticSliderValue('obstacle', 1, Number($event.target.value))">
              </label>
            </div>
            <div class="acoustic-preview-shell">
              <ModulusHeatmap
                v-if="modulusRenderMode === '2d' && activeModulusData?.results?.pressure_map"
                :data="activeModulusData.results.pressure_map"
                :size="280"
                :markers="modulusMarkers"
              />
              <ClientOnly v-else-if="activeModulusData?.results?.pressure_map">
                <div class="acoustic-preview-surface">
                  <AcousticFieldSurface
                    :data="activeModulusData.results.pressure_map"
                    :markers="modulusMarkers"
                  />
                </div>
              </ClientOnly>
            </div>
          </template>
          <div v-if="activeModulusData" class="session-facts">
            <div class="session-fact">
              <span>Method</span>
              <strong>{{ activeModulusData.method || 'SOT-A' }}</strong>
            </div>
            <div class="session-fact">
              <span>Probe response</span>
              <strong>{{ acousticProbeResponseText }}</strong>
            </div>
            <div class="session-fact">
              <span>Peaks</span>
              <strong>{{ acousticPeakSummary }}</strong>
            </div>
          </div>
        </div>
      </div>
    </aside>

    <Transition
      enter-active-class="fadeIn"
      leave-active-class="fadeOut"
      :duration="300"
    >
      <div v-if="showLightbox" class="lightbox" @click="closeLightbox">
        <button class="close-button" @click.stop="closeLightbox">
          <svg class="icon" viewBox="0 0 24 24">
            <path fill="currentColor" d="M19,6.41L17.59,5L12,10.59L6.41,5L5,6.41L10.59,12L5,17.59L6.41,19L12,13.41L17.59,19L19,17.59L13.41,12L19,6.41Z" />
          </svg>
        </button>
        <img 
          :src="lightboxImage || plotImage" 
          :alt="lightboxAlt"
          class="lightbox-image"
          @click.stop
        />
      </div>
    </Transition>
    <div v-if="uiVisible" class="footer">
      <div v-if="loading" class="loading">
        <div class="loading-status">{{ loadingStatus }}</div>
        <div v-if="progressStage" class="progress-stage">{{ formatProgressStage(progressStage) }}</div>
        <div v-if="reasoningPreview" class="reasoning-preview" :title="progressStage || 'reasoning'">
          {{ reasoningPreview }}
        </div>
      </div>
      <button 
        v-if="isMobileOrTablet" 
        @click="handleMobileEvaluate"
        class="mobile-evaluate-btn"
        title="Alt+Enter"
      >
        Evaluate
      </button>
    </div>
    <div v-if="error && uiVisible" class="error">{{ error }}</div>
    <HelpModal 
      v-if="uiVisible" v-model="showHelp" 
      @select-featured="handleSelectFeatured"
    />
    <GalleryModal
      v-if="uiVisible" v-model="showGallery"
      :initial-basename="targetBasename"
      @load-code="loadCodeFromGallery"
    />
    <AuthGateModal
      :model-value="showAuthGate"
      :loading="authActionLoading"
      :configured="authConfigured"
      :error="authGateError"
      @close="showAuthGate = false"
      @google="beginSignIn(true)"
      @logto="beginSignIn(false)"
    />
    <!-- Sorganoid Layer -->
    <ClientOnly>
      <SorganoidWorld 
        :active="sorganoidMode" 
        :command="lastSorganoidCommand" 
        :result="lastSorganoidResult"
        :loading="loading"
        :api-base="apiBase"
        :ui-visible="uiVisible"
        @print-log="(p) => editorRef?.addToEditor(p.text, p.type)"
        @exit="sorganoidMode = false"
      />
    </ClientOnly>
    </div>
    </template>
<script setup>
import { ref, onMounted, onUnmounted, computed, nextTick, watch } from 'vue';
import { useRuntimeConfig } from '#app';
import { marked } from 'marked';
import AceEditor from '~/components/AceEditor.vue';
import HelpModal from '~/components/HelpModal.vue';
import GalleryModal from '~/components/GalleryModal.vue';
import StlViewer from '~/components/StlViewer.vue';
import ModulusHeatmap from '~/components/ModulusHeatmap.vue';
import AcousticFieldSurface from '~/components/AcousticFieldSurface.vue';
import SorganoidWorld from '~/components/SorganoidWorld.vue';
import AuthGateModal from '~/components/AuthGateModal.vue';
import UsageDashboard from '~/components/UsageDashboard.vue';
import { useRandomPrompt } from '~/composables/useRandomPrompt';
import { useFavicon } from '~/composables/useFavicon';
import { useAcousticSonification } from '~/composables/useAcousticSonification';
import { useApi } from '~/composables/useApi';
import { useSoogAuth } from '~/composables/useSoogAuth';
import { runAcousticSimulation } from '~/utils/acousticSimulation';

// Configure marked
marked.setOptions({
  breaks: true,
  gfm: true,
});

// State variables
const { apiBase } = useApi();
const {
  isAuthenticated,
  isLoading: authLoading,
  configured: authConfigured,
  profile,
  quota,
  authHeaders,
  refreshProfile,
  signIn,
  signOut,
} = useSoogAuth();
const { startProcessing, completeProcessing } = useFavicon();
const { playResponse } = useAcousticSonification();
const editorRef = ref(null);
const leftWidth = ref(50);
let dragging = false;
let startX = 0;
let startLeft = 50;
const loading = ref(false);
const progress = ref(0);
const elapsedMs = ref(0);
const responseTimesMs = ref([]);
const isReversioning = ref(false);
const error = ref(null);
const plotImage = ref(null);
const sketchImage = ref(null);
const summary = ref(null);
const organogramCode = ref('');
const geometryCode = ref('');
const stlUrl = ref(null);
const modulusData = ref(null);
const materialsText = ref('');
const ollamaModels = ref([]);
const currentModel = ref('');
const modelSwitching = ref(false);
const responseModel = ref('');
const responseElapsedMs = ref(0);
const sorganoidMode = ref(false);
const lastSorganoidCommand = ref('');
const lastSorganoidResult = ref(null);
const uiVisible = ref(true);
const sketchModel = ref('');
const lightboxImage = ref(null);
const lightboxAlt = ref('Preview');
const generationRequestId = ref('');
const reasoningPreview = ref('');
const progressStage = ref('');
const viewMode = ref('stl'); // 'stl', 'sketch', or 'modulus'
const backgroundViewMode = ref('plot'); // 'plot', 'stl', 'sketch', or 'modulus'
const modulusRenderMode = ref('2d');
const workspaceLayout = ref('background'); // 'split' or 'background'
const workspaceSidebarOpen = ref(true);
const workspaceSidebarTab = ref('help'); // 'help', 'session', 'acoustic'
const lastAcousticCommand = ref(null);
const editorContent = ref('');
const liveModulusData = ref(null);
const targetBasename = ref('');
let progressPollInterval = null;
let editorSyncBound = false;
let editorSessionChangeHandler = null;
let livePreviewTimer = null;
let editorSyncTimer = null;

const summaryHtml = computed(() => {
  if (!summary.value) return '';
  return marked(summary.value);
});
const showCode = ref(true);
const showHelp = ref(false);
const showGallery = ref(false);
const showAuthGate = ref(false);
const authActionLoading = ref(false);
const authGateError = ref('');
const pendingRenderPrompt = ref('');
const profileRefreshAttempted = ref(false);
const transitionKey = ref(0);
const isMobileOrTablet = ref(false);
const showLightbox = ref(false);
const RESPONSE_TIMES_KEY = 'soog_response_times_ms';
const MAX_RESPONSE_SAMPLES = 20;
const PENDING_RENDER_KEY = 'soog.pending-render.v1';

const userLabel = computed(() => {
  const value = profile.value?.name || profile.value?.email || profile.value?.subject || 'ACCOUNT';
  return String(value).length > 16 ? `${String(value).slice(0, 13)}…` : String(value);
});
const quotaLabel = computed(() => {
  const daily = quota.value?.daily;
  if (!daily) return '';
  return `${daily.used}/${daily.limit === null ? '∞' : daily.limit} D`;
});
const quotaTitle = computed(() => {
  if (!quota.value) return '';
  const daily = quota.value.daily;
  const weekly = quota.value.weekly;
  return `Daily ${daily.used}/${daily.limit ?? 'unlimited'} · Weekly ${weekly.used}/${weekly.limit ?? 'unlimited'}`;
});

const averageResponseMs = computed(() => {
  if (!responseTimesMs.value.length) return 20000;
  const total = responseTimesMs.value.reduce((sum, ms) => sum + ms, 0);
  return total / responseTimesMs.value.length;
});

const loadingStatus = computed(() => {
  const pct = Math.round(progress.value);
  const elapsedSec = (elapsedMs.value / 1000).toFixed(1);
  const avgSec = (averageResponseMs.value / 1000).toFixed(1);
  const etaMs = Math.max(0, averageResponseMs.value - elapsedMs.value);
  const etaSec = (etaMs / 1000).toFixed(1);
  const prefix = isReversioning.value ? '[reversion] ' : '';
  return `${prefix}Processing... ${pct}% | ${elapsedSec}s elapsed | avg ${avgSec}s | ETA ${etaSec}s`;
});

const shortModelLabel = computed(() => {
  const model = currentModel.value || 'model?';
  return model.length > 22 ? `${model.slice(0, 22)}...` : model;
});

const responseMetaText = computed(() => {
  const parts = [];
  if (responseModel.value) parts.push(`model: ${responseModel.value}`);
  if (responseElapsedMs.value > 0) parts.push(`elapsed: ${(responseElapsedMs.value / 1000).toFixed(1)}s`);
  return parts.join(' | ');
});

const liveAcousticCommand = computed(() => parseAcousticCommand(editorContent.value));
const currentAcousticMeta = computed(() => liveAcousticCommand.value?.meta || lastAcousticCommand.value);
const activeModulusData = computed(() => liveModulusData.value || modulusData.value);
const hasResults = computed(() => !!(plotImage.value || sketchImage.value || summary.value || materialsText.value || stlUrl.value || activeModulusData.value));
const workspaceModeLabel = computed(() => workspaceLayout.value === 'background' ? 'background' : 'split');
const leftColumnStyle = computed(() => {
  if (sorganoidMode.value) {
    return { width: '100%' };
  }
  if (workspaceLayout.value === 'background') {
    const clamped = Math.max(34, Math.min(leftWidth.value, 62));
    return { width: `min(${clamped}vw, 760px)` };
  }
  return { width: `${leftWidth.value}%` };
});
const rightColumnStyle = computed(() => (
  workspaceLayout.value === 'background'
    ? {}
    : { width: `${100 - leftWidth.value}%` }
));
const availableBackgroundViews = computed(() => ([
  { key: 'plot', label: 'ORGANOGRAM', enabled: Boolean(plotImage.value) },
  { key: 'stl', label: 'GEOMETRY', enabled: Boolean(stlUrl.value) },
  { key: 'sketch', label: 'SKETCH', enabled: Boolean(sketchImage.value) },
  { key: 'modulus', label: 'ACOUSTIC', enabled: Boolean(activeModulusData.value) }
]));
const activeBackgroundView = computed(() => {
  const preferred = availableBackgroundViews.value.find((item) => item.key === backgroundViewMode.value && item.enabled);
  if (preferred) return preferred.key;
  return availableBackgroundViews.value.find((item) => item.enabled)?.key || 'plot';
});
const backgroundStageLabel = computed(() => {
  const current = availableBackgroundViews.value.find((item) => item.key === activeBackgroundView.value);
  return current?.label || 'RESULT';
});
const splitTabOptions = computed(() => ([
  { key: 'stl', label: 'GEOMETRY (3D)', enabled: Boolean(stlUrl.value) },
  { key: 'sketch', label: 'SKETCH (INFERRED)', enabled: Boolean(sketchImage.value) },
  { key: 'modulus', label: 'ACOUSTICS (SOT-A)', enabled: Boolean(activeModulusData.value) }
]));
const acousticCommandSummary = computed(() => {
  const meta = currentAcousticMeta.value;
  if (!meta) return 'No acoustic command evaluated yet.';
  return `${meta.primitive.toUpperCase()} · ${meta.freq} Hz · src ${meta.source.join(', ')} · probe ${meta.probe.join(', ')}`;
});
const modulusMarkers = computed(() => {
  const params = activeModulusData.value?.params || {};
  const markers = [];
  const sources = Array.isArray(params.sources) ? params.sources : [];
  for (const src of sources) {
    if (Array.isArray(src?.pos) && src.pos.length >= 2) {
      markers.push({
        type: 'source',
        x: Number(src.pos[0]),
        y: Number(src.pos[1]),
        label: src.freq ? `${Math.round(Number(src.freq))}Hz` : 'src'
      });
    }
  }
  if (Array.isArray(params.probe) && params.probe.length >= 2) {
    markers.push({
      type: 'probe',
      x: Number(params.probe[0]),
      y: Number(params.probe[1]),
      label: 'probe'
    });
  }
  if (Array.isArray(params.obstacle) && params.obstacle.length >= 2) {
    markers.push({
      type: 'obstacle',
      x: Number(params.obstacle[0]),
      y: Number(params.obstacle[1]),
      label: params.primitive || 'obs'
    });
  }
  return markers.filter((marker) => Number.isFinite(marker.x) && Number.isFinite(marker.y));
});
const acousticProbeResponseText = computed(() => {
  const value = Number(activeModulusData.value?.results?.probe_response ?? activeModulusData.value?.results?.mic_response);
  return Number.isFinite(value) ? value.toFixed(3) : '—';
});
const acousticPeakSummary = computed(() => (
  (activeModulusData.value?.results?.resonance_peaks_hz || []).slice(0, 4).join(', ') || '—'
));
const acousticSliderState = computed(() => ({
  primitive: currentAcousticMeta.value?.primitive || 'circle',
  freq: Number(currentAcousticMeta.value?.freq || 440),
  sourceX: Number(currentAcousticMeta.value?.source?.[0] ?? -0.55),
  sourceY: Number(currentAcousticMeta.value?.source?.[1] ?? 0),
  probeX: Number(currentAcousticMeta.value?.probe?.[0] ?? 0.55),
  probeY: Number(currentAcousticMeta.value?.probe?.[1] ?? 0),
  obstacleX: Number(currentAcousticMeta.value?.obstacle?.[0] ?? 0.15),
  obstacleY: Number(currentAcousticMeta.value?.obstacle?.[1] ?? 0),
}));

const commandDomains = [
  {
    key: 'render',
    label: 'RENDER',
    color: '#8bc34a',
    commands: [
      { syntax: 'Alt+Enter', detail: 'Evaluate current selection or the full editor buffer.' },
      { syntax: 'BG / SPLIT', detail: 'Switch between livecoding-over-background and split workspace.' },
      { syntax: 'Alt+1 · Alt+2 · Alt+3', detail: 'Jump between SOOG, SOMAP, and Concert Room.' }
    ]
  },
  {
    key: 'acoustic',
    label: 'SOT-A',
    color: '#ff9800',
    commands: [
      { syntax: '@acoustic primitive=circle freq=440 source=-0.55,0 probe=0.55,0 obstacle=0.15,0', detail: 'Run the first acoustic command against the online field solver.' },
      { syntax: 'primitive=<circle|square|triangle|hexagon>', detail: 'Choose a cavity family for the current acoustic test.' },
      { syntax: 'SOT-A sliders', detail: 'The right sidebar now edits the @acoustic command directly and refreshes a local preview.' },
      { syntax: '2D / 3D', detail: 'Switch the acoustic field between cyberpunk heatmap and 3D surface preview.' },
      { syntax: 'follow-up lines', detail: 'Any lines after the command are passed as contextual notes to the solver.' }
    ]
  },
  {
    key: 'reversion',
    label: 'REVERSION',
    color: '#42a5f5',
    commands: [
      { syntax: '* correction notes…', detail: 'Fast reversion on top of the current instrument lineage.' },
      { syntax: '+ addon notes…', detail: 'Alternative shorthand for iterative variants.' },
      { syntax: '[REFACT source=<basename> group=<group_id> title=<name>]', detail: 'Explicit reversion header for precise lineage control.' }
    ]
  },
  {
    key: 'world',
    label: 'WORLD',
    color: '#ab47bc',
    commands: [
      { syntax: '§ ...', detail: 'Enter Sorganoid / world commands without touching the default workspace.' },
      { syntax: '⌘/Ctrl + Shift + \\', detail: 'Toggle this command sidebar.' },
      { syntax: '?', detail: 'Legacy help stays intact; command docs now live here.' }
    ]
  }
];

const sessionFacts = computed(() => ([
  { label: 'Workspace', value: workspaceModeLabel.value.toUpperCase() },
  { label: 'Model', value: responseModel.value || currentModel.value || '—' },
  { label: 'Last render', value: responseElapsedMs.value ? `${(responseElapsedMs.value / 1000).toFixed(1)}s` : '—' },
  { label: 'Rolling avg', value: `${(averageResponseMs.value / 1000).toFixed(1)}s` },
  { label: 'Code append', value: showCode.value ? 'ON' : 'OFF' }
]));

const checkDevice = () => {
  isMobileOrTablet.value = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
};

const handleEscapeKey = (e) => {
  if (e.key === 'Escape' && showLightbox.value) {
    closeLightbox();
  }
};

function openLightbox(src, alt = 'Preview') {
  if (!src) return;
  lightboxImage.value = src;
  lightboxAlt.value = alt;
  showLightbox.value = true;
}

function closeLightbox() {
  showLightbox.value = false;
  lightboxImage.value = null;
  lightboxAlt.value = 'Preview';
}

function createRequestId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `soog-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function stopReasoningPolling() {
  if (progressPollInterval) {
    clearInterval(progressPollInterval);
    progressPollInterval = null;
  }
}

async function fetchReasoningProgress() {
  if (!generationRequestId.value) return;
  try {
    const response = await fetch(`${apiBase.value}/generate/progress/${generationRequestId.value}`, {
      headers: { Accept: 'application/json' }
    });
    if (!response.ok) return;
    const payload = await response.json();
    if (!payload?.ok) return;
    reasoningPreview.value = String(payload.reasoning_preview || '').trim();
    progressStage.value = String(payload.stage || '').trim();
    if (payload.status === 'completed' || payload.status === 'error') {
      stopReasoningPolling();
    }
  } catch {
    // keep progress polling quiet
  }
}

function startReasoningPolling() {
  stopReasoningPolling();
  reasoningPreview.value = '';
  progressStage.value = '';
  progressPollInterval = setInterval(fetchReasoningProgress, 700);
}

function formatProgressStage(stage) {
  return String(stage || '')
    .replace(/[_-]+/g, ' ')
    .trim()
    .replace(/\b\w/g, (ch) => ch.toUpperCase());
}

function clampUnitCoord(value, fallback = 0) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return Math.max(-0.95, Math.min(0.95, numeric));
}

function formatCommandCoord(value) {
  return String(Number(value).toFixed(2)).replace(/\.?0+$/, '');
}

function parsePointToken(rawValue, fallbackX, fallbackY) {
  if (typeof rawValue !== 'string' || !rawValue.trim()) {
    return [clampUnitCoord(fallbackX), clampUnitCoord(fallbackY)];
  }
  const parts = rawValue
    .split(/[,:/]/)
    .map((part) => part.trim())
    .filter(Boolean);
  return [
    clampUnitCoord(parts[0], fallbackX),
    clampUnitCoord(parts[1], fallbackY)
  ];
}

function parseAcousticCommand(text) {
  const lines = String(text || '').split('\n');
  const firstIdx = lines.findIndex((line) => line.trim().toLowerCase().startsWith('@acoustic'));
  if (firstIdx === -1) return null;

  const header = lines[firstIdx].trim();
  const rawTokens = header.slice('@acoustic'.length).trim().split(/\s+/).filter(Boolean);
  const params = {};
  for (const token of rawTokens) {
    const eqIdx = token.indexOf('=');
    if (eqIdx === -1) continue;
    const key = token.slice(0, eqIdx).trim().toLowerCase();
    const value = token.slice(eqIdx + 1).trim();
    if (key) params[key] = value;
  }

  const primitive = ['circle', 'square', 'triangle', 'hexagon'].includes(String(params.primitive || '').toLowerCase())
    ? String(params.primitive).toLowerCase()
    : 'circle';
  const freq = Math.max(20, Math.min(4000, Math.round(Number(params.freq || params.hz || 440) || 440)));
  const source = parsePointToken(params.source, -0.55, 0);
  const probe = parsePointToken(params.probe, 0.55, 0);
  const obstacle = parsePointToken(params.obstacle || params.obs, 0.15, 0);

  const remainder = lines
    .slice(firstIdx + 1)
    .join('\n')
    .trim();

  const body = remainder || [
    `Acoustic primitive study for a ${primitive} cavity.`,
    `Use a sine generator at ${freq} Hz.`,
    `Source at (${formatCommandCoord(source[0])}, ${formatCommandCoord(source[1])}), probe at (${formatCommandCoord(probe[0])}, ${formatCommandCoord(probe[1])}), obstacle at (${formatCommandCoord(obstacle[0])}, ${formatCommandCoord(obstacle[1])}).`
  ].join(' ');

  return {
    normalizedPrompt: [
      `[MODULUS freq=${freq} primitive=${primitive} source_x=${formatCommandCoord(source[0])} source_y=${formatCommandCoord(source[1])} probe_x=${formatCommandCoord(probe[0])} probe_y=${formatCommandCoord(probe[1])} obs_x=${formatCommandCoord(obstacle[0])} obs_y=${formatCommandCoord(obstacle[1])}]`,
      body
    ].join('\n'),
    meta: {
      primitive,
      freq,
      source,
      probe,
      obstacle,
      notes: body
    }
  };
}

function buildAcousticHeader(meta) {
  if (!meta) return '';
  return [
    '@acoustic',
    `primitive=${meta.primitive}`,
    `freq=${Math.round(Number(meta.freq) || 440)}`,
    `source=${formatCommandCoord(meta.source[0])},${formatCommandCoord(meta.source[1])}`,
    `probe=${formatCommandCoord(meta.probe[0])},${formatCommandCoord(meta.probe[1])}`,
    `obstacle=${formatCommandCoord(meta.obstacle[0])},${formatCommandCoord(meta.obstacle[1])}`,
  ].join(' ');
}

function findAcousticHeaderLine(lines) {
  return lines.findIndex((line) => line.trim().toLowerCase().startsWith('@acoustic'));
}

function setEditorValuePreservingViewport(nextValue) {
  const editor = editorRef.value?.aceEditor?.();
  if (!editor) return;
  const cursor = editor.getCursorPosition();
  const scrollTop = editor.session.getScrollTop();
  const scrollLeft = editor.session.getScrollLeft();
  editor.session.setValue(nextValue);
  editor.moveCursorToPosition(cursor);
  editor.session.setScrollTop(scrollTop);
  editor.session.setScrollLeft(scrollLeft);
  editor.clearSelection();
}

function updateAcousticCommandInEditor(patch) {
  const editor = editorRef.value?.aceEditor?.();
  const meta = liveAcousticCommand.value?.meta;
  if (!editor || !meta) return;

  const nextMeta = {
    primitive: patch.primitive ?? meta.primitive,
    freq: patch.freq ?? meta.freq,
    source: patch.source ?? meta.source,
    probe: patch.probe ?? meta.probe,
    obstacle: patch.obstacle ?? meta.obstacle,
    notes: meta.notes,
  };

  const lines = String(editor.getValue() || '').split('\n');
  const headerIndex = findAcousticHeaderLine(lines);
  if (headerIndex === -1) return;
  const indent = (lines[headerIndex].match(/^\s*/) || [''])[0];
  lines[headerIndex] = `${indent}${buildAcousticHeader(nextMeta)}`;
  setEditorValuePreservingViewport(lines.join('\n'));
}

function updateAcousticSliderValue(channel, axis, rawValue) {
  const meta = liveAcousticCommand.value?.meta;
  if (!meta) return;
  const safe = clampUnitCoord(rawValue, 0);
  const next = [...meta[channel]];
  next[axis] = safe;
  updateAcousticCommandInEditor({ [channel]: next });
}

function insertStarterAcousticCommand() {
  const editor = editorRef.value?.aceEditor?.();
  if (!editor) return;
  const current = String(editor.getValue() || '');
  if (findAcousticHeaderLine(current.split('\n')) !== -1) return;

  const starter = [
    '@acoustic primitive=circle freq=440 source=-0.55,0 probe=0.55,0 obstacle=0.15,0',
    'Primitive cavity study for a first live acoustic test.',
    '',
  ].join('\n');

  const lines = current.split('\n');
  let insertAt = 0;
  while (insertAt < lines.length) {
    const trimmed = lines[insertAt].trim();
    if (!trimmed || trimmed.startsWith('#')) {
      insertAt += 1;
      continue;
    }
    break;
  }

  const nextLines = [
    ...lines.slice(0, insertAt),
    starter,
    ...lines.slice(insertAt),
  ];
  setEditorValuePreservingViewport(nextLines.join('\n'));
}

function evaluateCurrentAcousticCommand() {
  const editor = editorRef.value?.aceEditor?.();
  if (!editor) return;
  const content = String(editor.getValue() || '').trim();
  if (!content) return;
  handleEvaluate(content);
}

function syncEditorContent() {
  const editor = editorRef.value?.aceEditor?.();
  if (!editor) return false;
  editorContent.value = editor.getValue() || '';

  if (!editorSyncBound) {
    editorSessionChangeHandler = () => {
      editorContent.value = editor.getValue() || '';
    };
    editor.session.on('change', editorSessionChangeHandler);
    editorSyncBound = true;
  }
  return true;
}

function pickPreferredSplitView() {
  if (stlUrl.value) return 'stl';
  if (sketchImage.value) return 'sketch';
  if (activeModulusData.value) return 'modulus';
  return 'stl';
}

function pickPreferredBackgroundView() {
  if (plotImage.value) return 'plot';
  if (stlUrl.value) return 'stl';
  if (sketchImage.value) return 'sketch';
  if (activeModulusData.value) return 'modulus';
  return 'plot';
}

function toggleWorkspaceSidebar() {
  workspaceSidebarOpen.value = !workspaceSidebarOpen.value;
}

function setWorkspaceLayout(layout) {
  workspaceLayout.value = layout === 'background' ? 'background' : 'split';
  if (workspaceLayout.value === 'background') {
    backgroundViewMode.value = pickPreferredBackgroundView();
  } else {
    viewMode.value = pickPreferredSplitView();
  }
}

function setBackgroundView(mode) {
  backgroundViewMode.value = mode;
}

const handleKeyDown = (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'm') {
    e.preventDefault();
    uiVisible.value = !uiVisible.value;
  }
  if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.code === 'Backslash') {
    e.preventDefault();
    toggleWorkspaceSidebar();
  }
};

onMounted(() => {
  loadResponseTimeHistory();
  fetchOllamaModels();
  checkDevice();
  editorSyncTimer = window.setInterval(() => {
    if (syncEditorContent()) {
      window.clearInterval(editorSyncTimer);
      editorSyncTimer = null;
    }
  }, 120);
  window.addEventListener('resize', checkDevice);
  window.addEventListener('keydown', handleEscapeKey);
  window.addEventListener('keydown', handleKeyDown);
  window.addEventListener('mousemove', onDrag);
  window.addEventListener('mouseup', stopDrag);
  window.addEventListener('touchmove', onDragTouch, { passive: false });
  window.addEventListener('touchend', stopDrag);
});

onUnmounted(() => {
  clearInterval(progressInterval);
  stopReasoningPolling();
  if (editorSyncTimer) {
    clearInterval(editorSyncTimer);
    editorSyncTimer = null;
  }
  if (livePreviewTimer) {
    clearTimeout(livePreviewTimer);
    livePreviewTimer = null;
  }
  const editor = editorRef.value?.aceEditor?.();
  if (editor && editorSessionChangeHandler) {
    editor.session.off?.('change', editorSessionChangeHandler);
  }
  editorSessionChangeHandler = null;
  editorSyncBound = false;
  window.removeEventListener('resize', checkDevice);
  window.removeEventListener('keydown', handleEscapeKey);
  window.removeEventListener('keydown', handleKeyDown);
  window.removeEventListener('mousemove', onDrag);
  window.removeEventListener('mouseup', stopDrag);
  window.removeEventListener('touchmove', onDragTouch);
  window.removeEventListener('touchend', stopDrag);
});

watch(editorRef, () => {
  nextTick(() => {
    syncEditorContent();
  });
});

watch(
  () => {
    const meta = liveAcousticCommand.value?.meta;
    if (!meta) return '';
    return JSON.stringify({
      primitive: meta.primitive,
      freq: meta.freq,
      source: meta.source,
      probe: meta.probe,
      obstacle: meta.obstacle,
      notes: meta.notes,
    });
  },
  () => {
    if (livePreviewTimer) clearTimeout(livePreviewTimer);
    const parsed = liveAcousticCommand.value;
    if (!parsed?.meta) {
      liveModulusData.value = null;
      return;
    }
    livePreviewTimer = setTimeout(() => {
      const meta = parsed.meta;
      liveModulusData.value = runAcousticSimulation({
        prompt: meta.notes,
        primitive: meta.primitive,
        freq: meta.freq,
        source_x: meta.source[0],
        source_y: meta.source[1],
        probe_x: meta.probe[0],
        probe_y: meta.probe[1],
        obs_x: meta.obstacle[0],
        obs_y: meta.obstacle[1],
      });
    }, 36);
  },
  { immediate: true }
);

const handleClear = () => {
  if (editorRef.value) {
    editorRef.value.clearEditor();
  }
};

const toggleShowCode = () => {
  showCode.value = !showCode.value;
};

function startDrag(e) {
  dragging = true;
  startX = e.clientX;
  startLeft = leftWidth.value;
}
function onDrag(e) {
  if (!dragging) return;
  const dx = e.clientX - startX;
  const vw = window.innerWidth;
  const deltaPct = (dx / vw) * 100;
  leftWidth.value = Math.min(80, Math.max(20, startLeft + deltaPct));
}
function stopDrag() { dragging = false; }
function startDragTouch(e) {
  dragging = true;
  startX = e.touches[0].clientX;
  startLeft = leftWidth.value;
}
function onDragTouch(e) {
  if (!dragging) return;
  const dx = e.touches[0].clientX - startX;
  const vw = window.innerWidth;
  const deltaPct = (dx / vw) * 100;
  leftWidth.value = Math.min(80, Math.max(20, startLeft + deltaPct));
}

const handleRandomPrompt = async () => {
  if (editorRef.value) {
    const { getRandomPrompt } = useRandomPrompt();
    const prompt = await getRandomPrompt();
    editorRef.value.clearEditor();
    editorRef.value.addToEditor(prompt);
    // Add random prompt to command history
    editorRef.value.addToHistory(prompt);
  }
};

const handleMobileEvaluate = () => {
  if (editorRef.value) {
    const editor = editorRef.value.aceEditor();
    if (editor) {
      const selectedText = editor.getSelectedText();
      const textToEvaluate = selectedText || editor.getValue();
      if (textToEvaluate.trim()) {
        startProcessing();
        handleEvaluate(textToEvaluate);
      } else {
        error.value = 'Please enter some text to evaluate.';
      }
    }
  }
};

const rememberPendingRender = (prompt = pendingRenderPrompt.value) => {
  if (typeof window === 'undefined' || !String(prompt || '').trim()) return;
  window.sessionStorage.setItem(PENDING_RENDER_KEY, JSON.stringify({ prompt: String(prompt).trim() }));
};

const beginSignIn = async (google) => {
  authActionLoading.value = true;
  authGateError.value = '';
  rememberPendingRender();
  try {
    await signIn(Boolean(google));
  } catch (cause) {
    authGateError.value = cause?.message || 'Unable to start sign-in.';
    authActionLoading.value = false;
  }
};

const handleSignOut = async () => {
  try {
    await signOut();
  } catch (cause) {
    error.value = cause?.message || 'Unable to sign out.';
  }
};

const resumePendingRender = async () => {
  if (typeof window === 'undefined' || !isAuthenticated.value || loading.value) return;
  const raw = window.sessionStorage.getItem(PENDING_RENDER_KEY);
  if (!raw) return;
  window.sessionStorage.removeItem(PENDING_RENDER_KEY);
  try {
    const payload = JSON.parse(raw);
    const prompt = String(payload?.prompt || '').trim();
    if (!prompt) return;
    showHelp.value = false;
    showAuthGate.value = false;
    pendingRenderPrompt.value = '';
    window.history.replaceState({}, '', window.location.pathname);
    await nextTick();
    await handleEvaluate(prompt);
  } catch {
    // Ignore malformed or stale pending intent.
  }
};

function handleSelectFeatured(basename) {
  targetBasename.value = basename;
  showHelp.value = false;
  showGallery.value = true;
}

const loadResponseTimeHistory = () => {
  if (typeof window === 'undefined') return;
  try {
    const raw = window.localStorage.getItem(RESPONSE_TIMES_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return;
    responseTimesMs.value = parsed
      .filter((n) => Number.isFinite(n) && n > 0 && n < 10 * 60 * 1000)
      .slice(-MAX_RESPONSE_SAMPLES);
  } catch {
    responseTimesMs.value = [];
  }
};

const saveResponseTimeHistory = () => {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(RESPONSE_TIMES_KEY, JSON.stringify(responseTimesMs.value.slice(-MAX_RESPONSE_SAMPLES)));
  } catch {
    // ignore storage write errors
  }
};

const registerResponseTime = (ms) => {
  if (!Number.isFinite(ms) || ms <= 0) return;
  responseTimesMs.value.push(ms);
  if (responseTimesMs.value.length > MAX_RESPONSE_SAMPLES) {
    responseTimesMs.value = responseTimesMs.value.slice(-MAX_RESPONSE_SAMPLES);
  }
  saveResponseTimeHistory();
};

// Progress based on real elapsed time and rolling average
let progressInterval;
let requestStartAt = 0;
const startProgress = () => {
  clearInterval(progressInterval);
  requestStartAt = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
  elapsedMs.value = 0;
  progress.value = 0;
  progressInterval = setInterval(() => {
    const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    elapsedMs.value = Math.max(0, now - requestStartAt);

    // Keep progress realistic: approach ~92% by average time, then slowly creep to 98%.
    const avg = Math.max(3000, averageResponseMs.value);
    const ratio = elapsedMs.value / avg;
    if (ratio <= 1) {
      progress.value = Math.min(92, ratio * 92);
    } else {
      const over = (elapsedMs.value - avg) / avg;
      progress.value = Math.min(98, 92 + Math.log1p(over) * 6);
    }
  }, 200);
};

const completeProgress = () => {
  const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
  if (requestStartAt > 0) {
    elapsedMs.value = Math.max(0, now - requestStartAt);
    registerResponseTime(elapsedMs.value);
  }
  clearInterval(progressInterval);
  progress.value = 100;
  setTimeout(() => {
    progress.value = 0;
    elapsedMs.value = 0;
    requestStartAt = 0;
  }, 500);
};

// Runtime configuration
const config = useRuntimeConfig();

const getGenerateFetchTimeoutMs = () => {
  const value = Number(config.public.generateTimeoutMs);
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.floor(value));
};

function formatGeneratedCode(plotCode, stlCode) {
  const sections = [];
  if (plotCode && plotCode.trim()) {
    sections.push([
      "## organogram (matplotlib)",
      "```python",
      plotCode.trim(),
      "```"
    ].join('\n'));
  }
  if (stlCode && stlCode.trim()) {
    sections.push([
      "## geometry (trimesh)",
      "```python",
      stlCode.trim(),
      "```"
    ].join('\n'));
  }
  return sections.join('\n\n');
}

function safeToken(value) {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

function inferTitleToken(item) {
  const explicit = safeToken(item?.title_slug || item?.title);
  if (explicit) return explicit;
  const basename = String(item?.basename || '');
  if (!basename) return '';
  const rawTail = basename.split('_').slice(1).join('_');
  if (!rawTail) return '';
  const withoutVersion = rawTail.replace(/_v\d+(?:_\d+)?(?:_\d+)?$/i, '');
  return safeToken(withoutVersion || rawTail);
}

function resolveAssetUrl(url) {
  if (!url) return '';
  if (url.startsWith('http')) return url;
  if (apiBase.value.endsWith('/api') && url.startsWith('/api/')) {
    return apiBase.value + url.substring(4);
  }
  return apiBase.value + url;
}

function stlFilenameFromUrl(url, fallback = 'model.stl') {
  try {
    if (!url) return fallback;
    const pathname = new URL(resolveAssetUrl(url), window.location.origin).pathname;
    const last = pathname.split('/').pop() || '';
    if (last.toLowerCase().endsWith('.stl')) return last;
    return fallback;
  } catch {
    return fallback;
  }
}

async function remakeSketch() {
  if (loading.value || !hasResults.value) return
  loading.value = true
  progressStage.value = 'regenerating sketch'
  try {
    const prompt = editorRef.value?.aceEditor()?.getValue() || ''
    const summaryText = summary.value || ''
    const materials_text = materialsText.value || ''
    const plot_code = organogramCode.value || ''

    const response = await fetch(`${apiBase.value}/generate/sketch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...(await authHeaders()) },
      body: JSON.stringify({
        prompt,
        summary: summaryText,
        materials: materials_text,
        plot_code,
        image: plotImage.value?.split(',')[1] // send base64 if available
      })
    })
    const data = await response.json()
    if (!response.ok) throw new Error(data.error || 'Failed to remake sketch')
    
    if (data.sketch) {
      sketchImage.value = `data:image/png;base64,${data.sketch}`
    } else if (data.sketch_url) {
      sketchImage.value = resolveAssetUrl(data.sketch_url)
    }
    sketchModel.value = data.sketch_model || ''
  } catch (e) {
    error.value = `Sketch remake failed: ${e.message}`
  } finally {
    loading.value = false
    progressStage.value = ''
  }
}

async function downloadCurrentStl() {
  if (!stlUrl.value) return;
  try {
    const response = await fetch(stlUrl.value);
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload?.error || `STL download failed (${response.status})`);
    }
    const blob = await response.blob();
    const blobUrl = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = blobUrl;
    link.download = stlFilenameFromUrl(stlUrl.value);
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(blobUrl);
  } catch (e) {
    error.value = e?.message || 'STL download failed';
  }
}

function buildRefactorDraft(item) {
  if (!item || typeof item !== 'object') return '';

  const source = safeToken(item.basename);
  const group = safeToken(item.group_id || item.basename || item.title_slug || item.title);
  const title = inferTitleToken(item);

  const headerTokens = ['REFACT'];
  if (source) headerTokens.push(`source=${source}`);
  if (group) headerTokens.push(`group=${group}`);
  if (title) headerTokens.push(`title=${title}`);
  const header = `[${headerTokens.join(' ')}]`;

  const summaryText = (item.summary || item.answer || '').trim();
  const materials = (item.materials_text || '').trim();
  const plotCode = (item.plot_code || item.code || '').trim();
  const stlCode = (item.stl_code || '').trim();

  const lines = [
    header,
    'Refine this existing organogram and geometry as a new version.',
    'Keep the same instrument identity and improve only what is requested.',
    '',
    'Change Request:',
    '- describe the corrections/additions for this next version',
    '',
    'BASE CONTEXT (for iteration):',
    '',
    '## Base Prompt',
    (item.prompt || '(no stored prompt)').trim(),
    '',
    '## Base Conceptual Summary',
    summaryText || '(no stored summary)',
    '',
    '## Base Materials',
    materials || '(no stored materials)',
    ''
  ];

  if (plotCode) {
    lines.push('## Base Organogram Code (matplotlib)');
    lines.push('```python');
    lines.push(plotCode);
    lines.push('```');
    lines.push('');
  }

  if (stlCode) {
    lines.push('## Base Geometry Code (trimesh)');
    lines.push('```python');
    lines.push(stlCode);
    lines.push('```');
    lines.push('');
  }

  return `${lines.join('\n').trim()}\n`;
}

async function loadCodeFromGallery(item) {
  if (!item || !editorRef.value) return;

  const draft = buildRefactorDraft(item);
  if (editorRef.value.setEditorContent) {
    editorRef.value.setEditorContent(draft);
  } else {
    editorRef.value.clearEditor();
    editorRef.value.addToEditor(draft, 'code');
  }
  if (editorRef.value.addToHistory) {
    editorRef.value.addToHistory(draft);
  }

  summary.value = item.summary || item.answer || null;
  materialsText.value = item.materials_text || '';
  organogramCode.value = (item.plot_code || item.code || '').trim();
  geometryCode.value = (item.stl_code || '').trim();
  responseModel.value = (item.llm_model || '').trim();
  sketchModel.value = (item.sketch_model || '').trim();
  responseElapsedMs.value = Number.isFinite(Number(item.elapsed_ms))
    ? Number(item.elapsed_ms)
    : 0;

  stlUrl.value = item.stl_url ? resolveAssetUrl(item.stl_url) : null;
  plotImage.value = item.image_url ? resolveAssetUrl(item.image_url) : null;
  sketchImage.value = item.sketch_url ? resolveAssetUrl(item.sketch_url) : null;
  modulusData.value = item.modulus || null;
  viewMode.value = pickPreferredSplitView();
  backgroundViewMode.value = pickPreferredBackgroundView();
  showGallery.value = false;
}

async function fetchOllamaModels() {
  try {
    const response = await fetch(`${apiBase.value}/ollama/models`, {
      headers: { 'Accept': 'application/json' }
    });
    const payload = await response.json();
    if (!response.ok || !payload?.ok) {
      return;
    }
    ollamaModels.value = Array.isArray(payload.models) ? payload.models : [];
    currentModel.value = payload.current_model || '';
  } catch {
    // keep UI quiet when models endpoint isn't available
  }
}

async function cycleOllamaModel() {
  if (modelSwitching.value) return;
  modelSwitching.value = true;
  try {
    if (!ollamaModels.value.length) {
      await fetchOllamaModels();
    }
    if (ollamaModels.value.length < 2) {
      error.value = 'Need at least 2 installed Ollama models to cycle.';
      return;
    }
    const currentIdx = Math.max(0, ollamaModels.value.indexOf(currentModel.value));
    const nextIdx = (currentIdx + 1) % ollamaModels.value.length;
    const nextModel = ollamaModels.value[nextIdx];

    const response = await fetch(`${apiBase.value}/ollama/model`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify({ model: nextModel })
    });
    const payload = await response.json();
    if (!response.ok || !payload?.ok) {
      throw new Error(payload?.error || `Failed to switch model (${response.status})`);
    }
    currentModel.value = payload.model || nextModel;
    error.value = null;
  } catch (e) {
    error.value = e?.message || 'Model switch failed.';
  } finally {
    modelSwitching.value = false;
  }
}

function isRefactorPrompt(text) {
  const trimmed = String(text || '').trim();
  if (!trimmed) return false;
  const first = trimmed.split('\n')[0].trim();
  if (!first) return false;
  return first.startsWith('[REFACT') || first.startsWith('*') || first.startsWith('+');
}

// Handle evaluation of selected text
const handleEvaluate = async (selectedText) => {
  if (!selectedText.trim()) {
    error.value = 'Please select some text to evaluate.';
    return;
  }

  const acousticCommand = parseAcousticCommand(selectedText);
  const requestPrompt = acousticCommand?.normalizedPrompt || selectedText;
  lastAcousticCommand.value = acousticCommand?.meta || null;

  // Sorganoid Command Detection
  if (selectedText.trim().startsWith('§')) {
    const cmd = selectedText.trim();
    sorganoidMode.value = true;
    lastSorganoidCommand.value = cmd;

    // If it's a 'create' or 'mutate' command that might require neural assets
    if (cmd.includes('species') || cmd.includes('mutate') || cmd.includes('evolve')) {
      // Proceed with neural generation loop but don't reset sorganoidMode
    } else {
      // Local procedural command (like spawn or gravity) handled by SorganoidWorld component
      return;
    }
  } else {
    // Normal evaluation resets sorganoid mode for now
    sorganoidMode.value = false;
  }

  if (!isAuthenticated.value) {
    pendingRenderPrompt.value = selectedText.trim();
    authGateError.value = '';
    showAuthGate.value = true;
    return;
  }

  const requestStartedAt = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
  loading.value = true;
  isReversioning.value = isRefactorPrompt(selectedText);
  error.value = null;
  lastSorganoidResult.value = null;
  generationRequestId.value = createRequestId();
  reasoningPreview.value = '';
  progressStage.value = '';
  startProgress();
  startProcessing();
  startReasoningPolling();

    async function callOnce() {
      const timeoutMs = getGenerateFetchTimeoutMs();
      const controller = timeoutMs > 0 ? new AbortController() : null;
      const timeoutId = timeoutMs > 0 ? setTimeout(() => controller?.abort(), timeoutMs) : null;
      let response;
      try {
        const secureHeaders = await authHeaders();
        response = await fetch(`${apiBase.value}/generate`, {
          method: 'POST',
          headers: { 
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            ...secureHeaders,
          },
          body: JSON.stringify({ prompt: requestPrompt, request_id: generationRequestId.value }),
          signal: controller?.signal
        });
      } catch (fetchErr) {
        if (fetchErr?.name === 'AbortError') {
          if (timeoutMs > 0) {
            throw new Error(`Generation timed out after ${Math.round(timeoutMs / 1000)}s. Try a shorter prompt or lighter model.`);
          }
          throw new Error('Generation request was aborted.');
        }
        throw fetchErr;
      } finally {
        if (timeoutId) clearTimeout(timeoutId);
      }
      
      // Read body only once as text
      const text = await response.text();
      
      if (!response.ok) {
        let errorData = {};
        try {
          errorData = JSON.parse(text);
        } catch {
          errorData = {};
        }
        if (response.status === 401) {
          pendingRenderPrompt.value = selectedText.trim();
          showAuthGate.value = true;
          throw new Error(errorData.error || 'Sign in to render with SOOG.');
        }
        if (response.status === 429) {
          quota.value = errorData.quota || quota.value;
          throw new Error(errorData.error || 'Your render quota has been reached.');
        }
        // Handle 502 Bad Gateway and similar errors
        if (response.status === 502 || response.status === 503) {
          throw new Error('Backend service unavailable (502/503). Please check server status.');
        }
        throw new Error(errorData.error || text || `Server error ${response.status}`);
      }
      
      // Validate response has content
      if (!text || text.trim().length === 0) {
        throw new Error("Empty response from server");
      }
      
      const contentType = response.headers.get("content-type");
      if (!contentType || !contentType.includes("application/json")) {
        throw new Error("Invalid response type from server");
      }
      
      return JSON.parse(text);
    }

  function extractMaterials(text) {
    if (!text) return [];
    const physical = [
      'spruce','maple','rosewood','ebony','mahogany','cedar','pine','oak','bamboo','reed','brass','bronze','copper','steel','aluminum','nickel','silver','gold','titanium','carbon fiber','fiberglass','plastic','acrylic','rubber','leather','gut','nylon','silk','ceramic','clay','glass','cork','felt'
    ];
    const virtual = [
      'texture','shader','sample','sampling','synthesis','granular','wavetable','fm','additive','subtractive','midi','vst','plugin','max/msp','pure data','supercollider','osc','convolution','impulse response','ir','reverb','impulse','unity','unreal','game engine','shader graph','material graph'
    ];
    const found = new Map();
    const lower = text.toLowerCase();
    function add(name, type) {
      const key = name.toLowerCase();
      if (!found.has(key)) found.set(key, { name, type });
    }
    physical.forEach(w => { if (lower.includes(w)) add(w, 'physical'); });
    virtual.forEach(w => { if (lower.includes(w)) add(w, 'virtual'); });
    return Array.from(found.values());
  }

  try {
    const data = await callOnce();
    void refreshProfile(apiBase.value).catch(() => {});

    // Reset results
    plotImage.value = null;
    sketchImage.value = null;
    summary.value = null;
    organogramCode.value = '';
    geometryCode.value = '';
    stlUrl.value = null;
    modulusData.value = null;
    materialsText.value = '';
    responseModel.value = '';
    responseElapsedMs.value = 0;
    sketchModel.value = '';

    // Handle special Modulus response (if only modulus was requested)
    if (data.type === 'modulus') {
      modulusData.value = data.modulus;
      summary.value = data.summary;
      materialsText.value = data.materials;
      viewMode.value = 'modulus';
      backgroundViewMode.value = 'modulus';

      // Play sonic feedback
      if (modulusData.value && modulusData.value.results) {
        const freq = Number(modulusData.value.params?.freq) || 440;
        const amp = Number(modulusData.value.results.mic_response) || 0.5;
        playResponse(freq, amp);
      }

      loading.value = false;
      completeProgress();
      completeProcessing();
      return;
    }

    if (sorganoidMode.value) {
      lastSorganoidResult.value = data;
    }

    const imageUrl = data.image_url || data.gallery?.image_url || null;
    const sketchUrl = data.sketch_url || data.gallery?.sketch_url || null;
    if (data.image) {
      plotImage.value = `data:image/png;base64,${data.image}`;
    } else if (imageUrl) {
      plotImage.value = resolveAssetUrl(imageUrl);
    } else {
      throw new Error('Backend did not return an organogram image. Generation aborted.');
    }
    if (data.sketch) {
      sketchImage.value = `data:image/png;base64,${data.sketch}`;
    } else if (sketchUrl) {
      sketchImage.value = resolveAssetUrl(sketchUrl);
    }

    if (data.summary) summary.value = data.summary;
    modulusData.value = data.modulus || null;
    responseModel.value = (data.llm_model || currentModel.value || '').trim();
    sketchModel.value = (data.sketch_model || data.gallery?.sketch_model || '').trim();
    viewMode.value = pickPreferredSplitView();
    backgroundViewMode.value = pickPreferredBackgroundView();

    // Play sonic feedback for standard generation if modulus data exists
    if (modulusData.value && modulusData.value.results) {
      const freq = Number(modulusData.value.params?.freq) || 440;
      const amp = Number(modulusData.value.results.mic_response) || 0.5;
      playResponse(freq, amp);
    }

    const requestEndedAt = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    const clientElapsed = Math.max(0, requestEndedAt - requestStartedAt);
    responseElapsedMs.value = Number.isFinite(Number(data.elapsed_ms))
      ? Number(data.elapsed_ms)
      : Math.round(clientElapsed);

    organogramCode.value = (data.plot_code || data.content || '').trim();
    geometryCode.value = (data.stl_code || '').trim();

    stlUrl.value = data.gallery?.stl_url ? resolveAssetUrl(data.gallery.stl_url) : null;

    if (typeof data.materials === 'string' && data.materials.trim()) {
      materialsText.value = data.materials.trim();
    } else if (Array.isArray(data.materials) && data.materials.length) {
      materialsText.value = data.materials.map(x => (typeof x === 'string' ? x : (x.name || ''))).filter(Boolean).join('\n');
    } else if (summary.value) {
      const list = extractMaterials(summary.value).map(m => `- ${m.name}`);
      materialsText.value = list.join('\n');
    }

    // Append structured code sections in editor (left panel)
    const codeBundle = formatGeneratedCode(organogramCode.value, geometryCode.value);
    if (showCode.value && codeBundle && editorRef.value) {
      editorRef.value.addToEditor(codeBundle, 'code');
    }

    transitionKey.value++;
  } catch (err) {
    console.error(err);
    error.value = err.message;
  } finally {
    await fetchReasoningProgress();
    stopReasoningPolling();
    generationRequestId.value = '';
    completeProgress();
    loading.value = false;
    isReversioning.value = false;
    completeProcessing();
  }
};

import { useRouter, useRoute } from 'vue-router'

const router = useRouter()
const route = useRoute()

function toggleSomap() {
  if (route.path === '/somap') {
    router.push('/')
  } else {
    router.push('/somap')
  }
}
function goToSoog() {
  if (route.path !== '/') router.push('/')
}
function goToSomap() {
  if (route.path !== '/somap') router.push('/somap')
}

function isAltDigitShortcut(event, digit) {
  return event.altKey && (event.code === `Digit${digit}` || event.key === String(digit))
}

// Shortcuts Alt+1 (Soog), Alt+2 (Somap)
function handleToggleShortcuts(e) {
  if (isAltDigitShortcut(e, 1)) {
    goToSoog()
    e.preventDefault()
    return
  }
  if (isAltDigitShortcut(e, 2)) {
    goToSomap()
    e.preventDefault()
  }
}

onMounted(() => {
  window.addEventListener('keydown', handleToggleShortcuts)
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleToggleShortcuts)
})

// Alt+ArrowUp/Down to navigate saved gallery outputs and load code
let galleryItems = []
let galleryIndex = -1

async function ensureGalleryLoaded() {
  if (galleryItems.length === 0) {
    try {
      const res = await fetch(`${apiBase.value}/gallery/list`)
      const data = await res.json()
      galleryItems = data.items || []
      galleryIndex = galleryItems.length > 0 ? 0 : -1
    } catch (e) {
      console.error('Failed to load gallery', e)
    }
  }
}

async function handleGalleryArrows(e) {
  if (!e.altKey) return
  if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return
  e.preventDefault()
  await ensureGalleryLoaded()
  if (galleryIndex === -1) return
  if (e.key === 'ArrowUp') {
    galleryIndex = Math.max(0, galleryIndex - 1)
  } else if (e.key === 'ArrowDown') {
    galleryIndex = Math.min(galleryItems.length - 1, galleryIndex + 1)
  }
  const item = galleryItems[galleryIndex]
  if (item && editorRef.value) {
    await loadCodeFromGallery(item)
  }
}

onMounted(() => window.addEventListener('keydown', handleGalleryArrows))
onUnmounted(() => window.removeEventListener('keydown', handleGalleryArrows))

watch(
  [isAuthenticated, authLoading],
  async ([authenticated, authBusy]) => {
    if (!authenticated) {
      profileRefreshAttempted.value = false;
      profile.value = null;
      quota.value = null;
      return;
    }
    if (authBusy || profileRefreshAttempted.value) return;
    profileRefreshAttempted.value = true;
    try {
      await refreshProfile(apiBase.value);
      authGateError.value = '';
      await resumePendingRender();
    } catch (cause) {
      authGateError.value = cause?.message || 'Unable to validate the SOOG session.';
    }
  },
  { immediate: true }
)

</script>

<style scoped>
.app-container {
  font-family: 'IBM Plex Mono', monospace;
  display: flex;
  flex-direction: row;
  height: 100vh;
  width: 100vw;
  overflow: hidden;
  position: relative;
  gap: 0;
  padding: 0 !important;
  margin: 0;
  background: #000;
  transition: background-color 0.5s ease;
}

.app-container.workspace-background-mode {
  isolation: isolate;
}

.app-container.sorganoid-active {
  background: transparent;
}

.app-container.sorganoid-active .left-column {
  background: transparent;
  height: 100vh !important;
}

.app-container.sorganoid-active :deep(.ace_scroller),
.app-container.sorganoid-active :deep(.ace_content),
.app-container.sorganoid-active :deep(.ace_gutter) {
  background: transparent !important;
  pointer-events: auto;
}

.app-container.sorganoid-active :deep(.ace-editor) {
  background: transparent !important;
  pointer-events: none; /* Let clicks pass through empty areas */
}

.app-container.ui-hidden {
  background: transparent;
}

.app-container.ui-hidden .left-column {
  opacity: 0;
  pointer-events: none;
}

.left-column {
  width: 50%;
  height: 100vh;
  display: block;
  min-width: 0;
  overflow: hidden;
  z-index: 20;
  transition: opacity 0.5s ease, width 0.5s ease;
}

.left-column--background {
  position: relative;
  z-index: 30;
  border-right: 1px solid rgba(255, 255, 255, 0.14);
}

.right-column {
  width: 50%;
  height: 100vh;
  display: flex;
  flex-direction: column;
  min-width: 0;
  z-index: 10;
  transition: opacity 0.5s ease;
}

.right-column--background {
  position: absolute;
  inset: 0;
  width: 100% !important;
  z-index: 0;
  border-left: none;
}

.app-container.workspace-background-mode .editor-wrapper {
  background:
    linear-gradient(90deg, rgba(0, 0, 0, 0.94) 0%, rgba(0, 0, 0, 0.86) 48%, rgba(0, 0, 0, 0.28) 100%);
}

.app-container.workspace-background-mode .left-column :deep(.ace_editor),
.app-container.workspace-background-mode .left-column :deep(.ace_scroller),
.app-container.workspace-background-mode .left-column :deep(.ace_content),
.app-container.workspace-background-mode .left-column :deep(.ace_gutter) {
  background: transparent !important;
}

.divider {
  width: 1px;
  cursor: col-resize;
  background: #111;
  z-index: 5;
  flex-shrink: 0;
  position: relative;
}

.divider::before {
  content: '';
  position: absolute;
  top: 0;
  bottom: 0;
  left: -4px;
  right: -4px;
  background: transparent;
}

.editor-wrapper {
  flex: 1;
  width: 100%;
  height: 100%;
  position: relative;
  overflow: hidden;
  margin: 0;
  padding: 0;
}

.left-column :deep(.ace_editor),
.left-column :deep(.ace_scroller),
.left-column :deep(.ace_content) {
  margin: 0 !important;
  padding: 0 !important;
}

.hud {
  padding: 4px 12px;
  display: flex;
  gap: 8px;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  border-bottom: 1px solid rgba(255, 255, 255, 0.12);
  position: relative;
  z-index: 12;
}

.right-column--background .hud {
  position: absolute;
  top: 0;
  right: 0;
  left: 0;
  border-bottom: none;
  padding: 8px 14px;
  background: linear-gradient(180deg, rgba(0, 0, 0, 0.86), rgba(0, 0, 0, 0));
}

.hud-group {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
}

.hud-group--main {
  margin-left: auto;
  flex-wrap: wrap;
  justify-content: flex-end;
}

.layout-pill {
  background: transparent;
  border: 1px solid rgba(255, 255, 255, 0.16);
  color: rgba(255, 255, 255, 0.52);
  padding: 2px 7px;
  font-size: 10px;
  letter-spacing: 0.08em;
  cursor: pointer;
  transition: color 0.2s ease, border-color 0.2s ease;
}

.layout-pill:hover,
.layout-pill.active {
  color: rgba(255, 255, 255, 0.88);
  border-color: rgba(255, 255, 255, 0.44);
}

.layout-pill:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.layout-pill--view.active {
  border-color: rgba(255, 179, 0, 0.9);
  color: #ffb300;
}

.icon-button {
  background: transparent;
  border: none;
  color: #666;
  padding: 4px;
  cursor: pointer;
  border-radius: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: opacity 0.2s;
}

.icon-button:hover {
  opacity: 0.7;
}

.icon-button.active,
.icon-button:hover {
  opacity: 1;
  filter: none;
}

.icon-button:disabled {
  opacity: 0.35;
  cursor: not-allowed;
}

.model-name {
  font-size: 11px;
  color: rgba(255, 255, 255, 0.7);
  margin-right: 2px;
  letter-spacing: 0.03em;
}

.model-cycle-button {
  margin-right: 2px;
}

.auth-pill,
.quota-pill {
  min-height: 22px;
  padding: 3px 7px;
  border: 1px solid rgba(255, 255, 255, 0.14);
  border-radius: 0;
  background: transparent;
  color: rgba(255, 255, 255, 0.48);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 8px;
  line-height: 1;
  letter-spacing: 0.08em;
  white-space: nowrap;
}

.auth-pill {
  cursor: pointer;
}

.auth-pill:hover {
  border-color: rgba(76, 175, 80, 0.65);
  color: #81c784;
}

.admin-pill {
  display: inline-flex;
  align-items: center;
  text-decoration: none;
}

.admin-pill,
.admin-pill:hover {
  border-color: rgba(255, 255, 255, 0.14);
  color: rgba(255, 255, 255, 0.48);
}

.quota-pill {
  display: inline-flex;
  align-items: center;
}

.icon {
  width: 18px;
  height: 18px;
}

.plot-image {
  cursor: pointer;
  transition: transform 0.3s;
}

.plot-image:hover {
  transform: scale(1.02);
}

.lightbox {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
  background: rgba(0, 0, 0, 0.9);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2000;
}

.lightbox-image {
  max-width: 80vw;
  max-height: 80vh;
  object-fit: contain;
}

.close-button {
  position: absolute;
  top: 20px;
  right: 20px;
  background: transparent;
  border: none;
  color: white;
  cursor: pointer;
  padding: 8px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: background-color 0.3s;
}

.close-button:hover {
  background: rgba(255, 255, 255, 0.1);
}

.footer {
  border-top: 1px solid #111;
  position: fixed;
  bottom: 0;
  right: 0;
  left: 0;
  display: flex;
  justify-content: flex-end;
  align-items: center;
  padding: 0.5rem 1rem;
  gap: 1rem;
  background: black !important;
  z-index: 1000;
}

.loading {
  margin-right: auto;
}

.loading-status {
  color: rgba(255, 255, 255, 0.9);
}

.progress-stage {
  margin-top: 3px;
  color: rgba(255, 255, 255, 0.46);
  font-size: 11px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.reasoning-preview {
  margin-top: 4px;
  max-width: min(62vw, 860px);
  color: rgba(255, 255, 255, 0.3);
  font-size: 11px;
  line-height: 1.3;
  white-space: normal;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.mobile-evaluate-btn {
  background: #4CAF50;
  color: white;
  border: none;
  padding: 4px 12px;
  border-radius: 4px;
  font-size: 14px;
  cursor: pointer;
  box-shadow: 0 2px 4px rgba(0,0,0,0.2);
  transition: background-color 0.3s;
}

.mobile-evaluate-btn:hover {
  background: #45a049;
}

.mobile-evaluate-btn:active {
  background: #3d8b40;
  transform: translateY(1px);
}

.error {
  position: fixed;
  bottom: 60px;
  left: 50%;
  transform: translateX(-50%);
  background: #ff5252;
  color: white;
  padding: 4px 12px;
  border-radius: 4px;
  z-index: 1000;
}

.results-panel {
  height: 100%;
  background: #000;
  display: flex;
  flex-direction: column;
  position: relative;
  overflow: hidden;
}

.results-split {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  background: transparent;
  display: grid;
  grid-template-rows: minmax(240px, 1.1fr) minmax(220px, 0.9fr) minmax(300px, 1fr);
  gap: 0;
  padding: 0 0 88px 0;
  box-sizing: border-box;
}

.panel {
  background: transparent;
  border: none;
  border-radius: 0;
  padding: 12px 16px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.panel + .panel {
  border-top: 1px solid rgba(255, 255, 255, 0.2);
}

.panel-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  flex: 1;
  min-height: 120px;
  color: rgba(194, 214, 255, 0.48);
  font-size: 12px;
  letter-spacing: 0.03em;
  text-align: center;
}

.section-title {
  margin: 0 0 10px 0;
  font-size: 12px;
  letter-spacing: 0.12em;
  color: rgba(255, 255, 255, 0.78);
  font-weight: 400;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 10px;
}

.section-meta {
  font-size: 11px;
  color: rgba(255, 255, 255, 0.58);
  white-space: nowrap;
}

.panel-organogram {
  justify-content: center;
}

.panel-visualizer {
  min-height: 380px;
}

.tab-header {
  margin-bottom: 12px;
}

.tabs {
  display: flex;
  gap: 16px;
}

.modulus-mode-toggle {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.modulus-mode-btn {
  border: 1px solid rgba(0, 246, 255, 0.16);
  background: rgba(7, 12, 30, 0.35);
  color: rgba(192, 228, 255, 0.56);
  padding: 2px 7px;
  font-size: 10px;
  letter-spacing: 0.12em;
  cursor: pointer;
}

.modulus-mode-btn.active,
.modulus-mode-btn:hover {
  border-color: rgba(0, 246, 255, 0.72);
  color: #00f6ff;
  box-shadow: 0 0 14px rgba(0, 246, 255, 0.12);
}

.tab-btn {
  background: transparent;
  border: none;
  color: rgba(255, 255, 255, 0.4);
  font-size: 11px;
  letter-spacing: 0.12em;
  padding: 0 0 4px 0;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: all 0.2s;
}

.tab-btn:hover {
  color: rgba(255, 255, 255, 0.8);
}

.tab-btn.active {
  color: #4CAF50;
  border-bottom-color: #4CAF50;
}

.remake-btn-small {
  background: rgba(255, 255, 255, 0.1);
  color: rgba(255, 255, 255, 0.7);
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 4px;
  padding: 2px 8px;
  font-size: 10px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  margin-left: 8px;
}

.remake-btn-small:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.2);
  color: white;
  border-color: rgba(255, 255, 255, 0.4);
}

.remake-btn-small:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.tab-content {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.tab-pane {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.section-meta-group {
  display: flex;
  align-items: center;
  gap: 12px;
}

.panel-text {
  overflow: auto;
}

.materials-title {
  margin-top: 14px;
}

.materials-list {
  white-space: pre-wrap;
  margin: 0;
  color: rgba(255, 255, 255, 0.86);
  background: transparent;
  border: none;
  border-radius: 0;
  padding: 0;
  font-size: 13px;
  line-height: 1.5;
}

.stl-viewer-container {
  width: 100%;
  flex: 1;
  min-height: 240px;
}

.modulus-results {
  flex: 1;
  overflow: auto;
  background: transparent;
  padding: 0;
  border-radius: 0;
}

.modulus-stack {
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-height: 100%;
}

.modulus-surface-shell {
  min-height: 320px;
  border: 1px solid rgba(0, 246, 255, 0.16);
  background:
    radial-gradient(circle at top left, rgba(26, 36, 84, 0.32), rgba(5, 8, 20, 0.08) 60%),
    transparent;
}

.modulus-readout {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  font-size: 11px;
  color: rgba(194, 214, 255, 0.56);
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.modulus-readout span {
  border-top: 1px solid rgba(0, 246, 255, 0.16);
  padding-top: 6px;
}

.modulus-json {
  margin: 0;
  font-size: 11px;
  color: #4CAF50;
  font-family: 'Courier New', monospace;
  line-height: 1.4;
}

.stl-placeholder {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #9e9e9e;
  font-size: 13px;
  border: none;
  border-radius: 0;
}

.sketch-placeholder {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #9e9e9e;
  font-size: 13px;
  border: none;
  border-radius: 0;
}

.download-btn {
  background: transparent;
  color: rgba(255, 255, 255, 0.86);
  border: none;
  padding: 0;
  border-radius: 0;
  font-size: 12px;
  cursor: pointer;
  text-decoration: none;
  transition: opacity 0.2s;
}

.download-btn:hover {
  opacity: 0.7;
}

.plot-image {
  width: 100%;
  height: 100%;
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  display: block;
  border-radius: 0;
}

.summary-content {
  color: #eee;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  font-size: 14px;
  line-height: 1.7;
  width: 100%;
  text-align: left;
}

.summary-content :deep(h1),
.summary-content :deep(h2),
.summary-content :deep(h3),
.summary-content :deep(h4),
.summary-content :deep(h5),
.summary-content :deep(h6) {
  color: #4CAF50;
  margin-top: 1.5em;
  margin-bottom: 0.5em;
  font-weight: 600;
}

.summary-content :deep(h1) { font-size: 1.8em; }
.summary-content :deep(h2) { font-size: 1.5em; }
.summary-content :deep(h3) { font-size: 1.3em; }

.summary-content :deep(p) {
  margin: 0.8em 0;
}

.summary-content :deep(ul),
.summary-content :deep(ol) {
  margin: 1em 0;
  padding-left: 2em;
}

.summary-content :deep(li) {
  margin: 0.4em 0;
}

.summary-content :deep(code) {
  background: rgba(255, 255, 255, 0.1);
  padding: 2px 6px;
  border-radius: 3px;
  font-family: 'Courier New', monospace;
  font-size: 0.9em;
}

.summary-content :deep(pre) {
  background: rgba(255, 255, 255, 0.05);
  padding: 12px;
  border-radius: 4px;
  overflow-x: auto;
  margin: 1em 0;
}

.summary-content :deep(pre code) {
  background: none;
  padding: 0;
}

.summary-content :deep(a) {
  color: #4CAF50;
  text-decoration: none;
}

.summary-content :deep(a:hover) {
  text-decoration: underline;
}

.summary-content :deep(strong) {
  font-weight: 600;
  color: #fff;
}

.summary-content :deep(em) {
  font-style: italic;
  color: #ddd;
}

.summary-content :deep(blockquote) {
  border-left: 3px solid #4CAF50;
  padding-left: 1em;
  margin: 1em 0;
  color: #aaa;
}

.background-stage {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 48px 40px 72px 40px;
  overflow: hidden;
}

.background-stage__meta {
  position: absolute;
  top: 16px;
  left: 16px;
  display: flex;
  flex-direction: column;
  gap: 2px;
  z-index: 2;
}

.background-stage__label {
  font-size: 11px;
  letter-spacing: 0.14em;
  color: rgba(255, 255, 255, 0.78);
}

.background-stage__sub {
  font-size: 10px;
  color: rgba(255, 255, 255, 0.44);
}

.background-stage__image,
.background-stage__viewer,
.background-stage__heatmap {
  width: 100%;
  height: 100%;
  max-width: 100%;
  max-height: 100%;
}

.background-stage__image {
  object-fit: contain;
}

.background-stage__viewer {
  min-height: 0;
}

.background-stage__heatmap {
  display: flex;
  align-items: center;
  justify-content: center;
}

.background-stage__acoustic {
  position: relative;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.background-stage__acoustic-hud {
  position: absolute;
  top: 12px;
  right: 12px;
  display: inline-flex;
  gap: 4px;
  z-index: 3;
}

.background-stage__surface {
  width: min(92vw, 1040px);
  height: min(70vh, 760px);
}

.workspace-sidebar {
  position: fixed;
  top: 0;
  right: 0;
  bottom: 44px;
  width: 360px;
  display: flex;
  transform: translateX(316px);
  transition: transform 0.24s ease;
  z-index: 80;
  border-left: 1px solid rgba(255, 255, 255, 0.14);
  background: rgba(0, 0, 0, 0.18);
  backdrop-filter: blur(14px);
}

.workspace-sidebar--open {
  transform: translateX(0);
}

.workspace-sidebar__rail {
  width: 44px;
  border-right: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  flex-direction: column;
  align-items: stretch;
  background: rgba(0, 0, 0, 0.5);
}

.workspace-sidebar__tab {
  flex: 1;
  background: transparent;
  border: none;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  color: rgba(255, 255, 255, 0.46);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.18em;
  cursor: pointer;
  writing-mode: vertical-rl;
  transform: rotate(180deg);
  padding: 12px 0;
}

.workspace-sidebar__tab.active {
  color: rgba(255, 255, 255, 0.92);
}

.workspace-sidebar__panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.workspace-sidebar__header {
  height: 38px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 14px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  font-size: 11px;
  letter-spacing: 0.16em;
  color: rgba(255, 255, 255, 0.82);
}

.workspace-sidebar__close {
  color: rgba(255, 255, 255, 0.46);
}

.workspace-sidebar__body {
  flex: 1;
  overflow: auto;
  padding: 14px 16px 68px;
}

.acoustic-actions {
  display: flex;
  gap: 8px;
  margin-top: 18px;
  flex-wrap: wrap;
}

.sidebar-action {
  border: 1px solid rgba(0, 246, 255, 0.18);
  background: rgba(7, 12, 30, 0.42);
  color: rgba(202, 232, 255, 0.72);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.12em;
  padding: 7px 10px;
  cursor: pointer;
  text-transform: uppercase;
}

.sidebar-action:hover {
  border-color: rgba(255, 65, 243, 0.46);
  color: #fff;
}

.acoustic-primitive-row {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 6px;
  margin-top: 18px;
}

.primitive-chip {
  border: 1px solid rgba(0, 246, 255, 0.12);
  background: transparent;
  color: rgba(191, 227, 255, 0.5);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  padding: 7px 4px;
  cursor: pointer;
}

.primitive-chip.active,
.primitive-chip:hover {
  border-color: rgba(255, 65, 243, 0.56);
  color: #ff41f3;
}

.acoustic-slider-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  margin-top: 18px;
}

.slider-unit {
  display: grid;
  gap: 4px;
  padding-top: 8px;
  border-top: 1px solid rgba(0, 246, 255, 0.1);
}

.slider-unit--wide {
  grid-column: 1 / -1;
}

.slider-unit span {
  font-size: 10px;
  color: rgba(194, 214, 255, 0.56);
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.slider-unit strong {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.86);
  font-weight: 400;
}

.slider-unit input[type=range] {
  width: 100%;
  background: transparent;
}

.slider-unit input[type=range]::-webkit-slider-runnable-track {
  height: 2px;
  background: linear-gradient(90deg, rgba(0, 246, 255, 0.4), rgba(255, 65, 243, 0.4));
}

.slider-unit input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #00f6ff;
  margin-top: -4px;
  box-shadow: 0 0 12px rgba(0, 246, 255, 0.32);
}

.acoustic-preview-shell {
  margin-top: 18px;
  border-top: 1px solid rgba(255, 65, 243, 0.12);
  padding-top: 12px;
}

.acoustic-preview-surface {
  min-height: 280px;
  border: 1px solid rgba(0, 246, 255, 0.16);
  background: radial-gradient(circle at top left, rgba(26, 36, 84, 0.28), rgba(4, 8, 18, 0.06) 60%);
}

.command-domain + .command-domain,
.acoustic-summary + .acoustic-summary,
.session-summary,
.session-facts + .session-facts {
  margin-top: 18px;
}

.command-domain__header {
  margin-bottom: 8px;
}

.command-domain__badge {
  display: inline-block;
  padding-bottom: 3px;
  border-bottom: 1px solid var(--domain-color);
  color: var(--domain-color);
  font-size: 10px;
  letter-spacing: 0.14em;
}

.command-domain__row + .command-domain__row {
  margin-top: 10px;
}

.command-domain__row code,
.acoustic-summary code {
  display: block;
  color: rgba(255, 255, 255, 0.9);
  background: transparent;
  padding: 0;
  font-size: 11px;
  line-height: 1.4;
}

.command-domain__row p,
.acoustic-summary p,
.session-summary :deep(p) {
  margin: 4px 0 0;
  color: rgba(255, 255, 255, 0.68);
  font-size: 12px;
  line-height: 1.55;
}

.session-facts {
  display: grid;
  gap: 10px;
}

.session-fact {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-size: 11px;
  color: rgba(255, 255, 255, 0.52);
}

.session-fact strong {
  color: rgba(255, 255, 255, 0.86);
  font-weight: 400;
  text-align: right;
}

.session-summary h4,
.acoustic-summary h4 {
  margin: 0 0 6px;
  font-size: 10px;
  letter-spacing: 0.14em;
  color: rgba(255, 255, 255, 0.82);
  font-weight: 400;
}

@media (max-width: 768px) {
  .hud { width: 100% !important; box-sizing: border-box; position: absolute; top: 0; left: 0; z-index: 100; background: rgba(0,0,0,0.8); }
  .left-column { padding-top: 50px !important; }

  .app-container {
  font-family: 'IBM Plex Mono', monospace;
    flex-direction: column;
  }
  
  .left-column, .right-column {
    width: 100% !important;
  }
  .left-column { height: 50vh !important; }
  .right-column { height: 50vh !important; }
  .results-split {
    grid-template-rows: minmax(180px, 1fr) minmax(180px, 0.95fr) minmax(220px, 1fr);
    padding-bottom: 110px;
  }
  .workspace-sidebar {
    width: 100vw;
    transform: translateX(calc(100vw - 44px));
  }
  .background-stage {
    padding-left: 12px;
    padding-right: 12px;
  }
  .acoustic-slider-grid,
  .acoustic-primitive-row {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
