<template>
  <div 
    class="app-container" 
    :class="{ 
      'sorganoid-active': sorganoidMode,
      'ui-hidden': !uiVisible,
      'has-rail': !sorganoidMode && uiVisible,
      'gallery-open': showGallery && !sorganoidMode && uiVisible,
      'workspace-background-mode': workspaceLayout === 'background',
      'workspace-split-mode': workspaceLayout === 'split'
    }"
  >
    <!-- Thin tools rail (⌘/Ctrl + \ expands it) -->
<NavRail
      v-if="!sorganoidMode && uiVisible"
      ref="railRef"
      :expanded="railExpanded"
      :bottom="17"
    >
      <template #head>
        <button
          class="rail-item"
          :title="railExpanded ? 'Collapse tools (⌘/Ctrl + \\)' : 'Expand tools (⌘/Ctrl + \\)'"
          :aria-expanded="String(railExpanded)"
          @click="toggleRail"
        >
          <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" :d="railExpanded ? 'M14,7L9,12L14,17V7Z' : 'M10,17L15,12L10,7V17Z'" /></svg>
          <span class="rail-label">Tools</span>
        </button>
      </template>
      <template #tools>
    <div ref="workspacePresetMenuRef" class="workspace-preset-menu">
      <button
        class="rail-item workspace-preset-trigger"
        :class="{ active: workspacePresetMenuOpen }"
        @click="toggleWorkspacePresetMenu"
        title="Workspace presets"
      >
        <svg class="icon workspace-preset-icon" viewBox="0 0 24 24" aria-hidden="true">
          <rect x="4" y="4" width="16" height="16" stroke="currentColor" stroke-width="1.5" fill="none" />
          <template v-if="workspacePreset === 'splitVertical'">
            <path d="M12 4V20" stroke="currentColor" stroke-width="1.5" />
          </template>
          <template v-else-if="workspacePreset === 'splitThree'">
            <path d="M12 4V20" stroke="currentColor" stroke-width="1.5" />
            <path d="M12 12H20" stroke="currentColor" stroke-width="1.5" />
          </template>
          <template v-else-if="workspacePreset === 'splitFour'">
            <path d="M12 4V20" stroke="currentColor" stroke-width="1.5" />
            <path d="M4 12H20" stroke="currentColor" stroke-width="1.5" />
          </template>
        </svg>
        <span class="rail-label">{{ workspaceModeLabel }}</span>
      </button>
      <div v-if="workspacePresetMenuOpen" class="workspace-preset-dropdown">
        <button
          v-for="preset in workspacePresetOptions"
          :key="preset.key"
          class="workspace-preset-option"
          :class="{ active: workspacePreset === preset.key }"
          @click="setWorkspacePreset(preset.key)"
        >
          <svg class="workspace-preset-option__icon" viewBox="0 0 24 24" aria-hidden="true">
            <rect x="4" y="4" width="16" height="16" stroke="currentColor" stroke-width="1.5" fill="none" />
            <template v-if="preset.key === 'splitVertical'">
              <path d="M12 4V20" stroke="currentColor" stroke-width="1.5" />
            </template>
            <template v-else-if="preset.key === 'splitThree'">
              <path d="M12 4V20" stroke="currentColor" stroke-width="1.5" />
              <path d="M12 12H20" stroke="currentColor" stroke-width="1.5" />
            </template>
            <template v-else-if="preset.key === 'splitFour'">
              <path d="M12 4V20" stroke="currentColor" stroke-width="1.5" />
              <path d="M4 12H20" stroke="currentColor" stroke-width="1.5" />
            </template>
          </svg>
          <span class="workspace-preset-option__copy">
            <strong>{{ preset.label }}</strong>
            <small>{{ preset.summary }}</small>
          </span>
        </button>
      </div>
    </div>
        <button class="rail-item" :title="showCode ? 'Hide Code' : 'Show Code'" @click="toggleShowCode">
          <svg v-if="showCode" class="icon" viewBox="0 0 24 24" aria-hidden="true">
            <path fill="currentColor" d="M12,9A3,3 0 0,1 15,12A3,3 0 0,1 12,15A3,3 0 0,1 9,12A3,3 0 0,1 12,9M12,4.5C17,4.5 21.27,7.61 23,12C21.27,16.39 17,19.5 12,19.5C7,19.5 2.73,16.39 1,12C2.73,7.61 7,4.5 12,4.5M3.18,12C4.83,15.36 8.24,17.5 12,17.5C15.76,17.5 19.17,15.36 20.82,12C19.17,8.64 15.76,6.5 12,6.5C8.24,6.5 4.83,8.64 3.18,12Z" />
          </svg>
          <svg v-else class="icon" viewBox="0 0 24 24" aria-hidden="true">
            <path fill="currentColor" d="M11.83,9L15,12.16C15,12.11 15,12.05 15,12A3,3 0 0,0 12,9C11.94,9 11.89,9 11.83,9M7.53,9.8L9.08,11.35C9.03,11.56 9,11.77 9,12A3,3 0 0,0 12,15C12.22,15 12.44,14.97 12.65,14.92L14.2,16.47C13.53,16.8 12.79,17 12,17A5,5 0 0,1 7,12C7,11.21 7.2,10.47 7.53,9.8M2,4.27L4.28,6.55L4.73,7C3.08,8.3 1.78,10 1,12C2.73,16.39 7,19.5 12,19.5C13.55,19.5 15.03,19.2 16.38,18.66L16.81,19.08L19.73,22L21,20.73L3.27,3M12,7A5,5 0 0,1 17,12C17,12.64 16.87,13.26 16.64,13.82L19.57,16.75C21.07,15.5 22.27,13.86 23,12C21.27,7.61 17,4.5 12,4.5C10.6,4.5 9.26,4.75 8,5.2L10.17,7.35C10.74,7.13 11.35,7 12,7Z" />
          </svg>
          <span class="rail-label">{{ showCode ? 'Hide code' : 'Show code' }}</span>
        </button>
        <button class="rail-item" title="Evaluate selected text or all if no selection" @click="handleMobileEvaluate">
          <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M8,5.14V19.14L19,12.14L8,5.14Z" /></svg>
          <span class="rail-label">Evaluate</span>
        </button>
        <button class="rail-item" title="Clear Editor (Ctrl+H)" @click="handleClear">
          <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M9,3V4H4V6H5V19A2,2 0 0,0 7,21H17A2,2 0 0,0 19,19V6H20V4H15V3H9M7,6H17V19H7V6M9,8V17H11V8H9M13,8V17H15V8H13Z" /></svg>
          <span class="rail-label">Clear</span>
        </button>
        <button class="rail-item" title="Random Prompt" @click="handleRandomPrompt">
          <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M14.83,13.41L13.42,14.82L16.55,17.95L14.5,20H20V14.5L17.96,16.54L14.83,13.41M14.5,4L16.54,6.04L4,18.59L5.41,20L17.96,7.46L20,9.5V4M10.59,9.17L5.41,4L4,5.41L9.17,10.58L10.59,9.17Z" /></svg>
          <span class="rail-label">Random prompt</span>
        </button>
      
      </template>
      <template #bottom>
        <button class="rail-item" title="Gallery (Alt+↑/↓)" @click="showGallery = true">
          <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M21 19V5C21 3.89 20.1 3 19 3H5C3.89 3 3 3.89 3 5V19C3 20.1 3.89 21 5 21H19C20.1 21 21 20.1 21 19M8.5 13.5L11 16.51L14.5 12L19 18H5L8.5 13.5Z" /></svg>
          <span class="rail-label">Gallery</span>
        </button>
        
        <button class="rail-item" title="Stage 3D pane (Alt+3 opens the full page)" :class="{ active: workspacePaneShows('stage') }" @click="openWorkspaceView('stage')">
          <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3l8 4.5v9L12 21l-8-4.5v-9zM12 12l8-4.5M12 12v9M12 12L4 7.5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" /></svg>
          <span class="rail-label">Concert hall</span>
        </button>
        <button class="rail-item" :class="{ active: workspacePaneShows('sota') }" title="SOT-A acoustic pane (/sot-a)" @click="openWorkspaceView('sota')">
          <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M2 12q2.5-9 5 0t5 0 5 0 5 0" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" /></svg>
          <span class="rail-label">SOT-A</span>
        </button>
        <button class="rail-item" :class="{ active: workspaceSidebarOpen && workspaceSidebarTab === 'session' }" title="Session (/session)" @click="openSidebarTab('session')">
          <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12h4l3-8 4 16 3-8h4" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" /></svg>
          <span class="rail-label">Session</span>
        </button>
        <button class="rail-item" :class="{ active: workspaceSidebarOpen && workspaceSidebarTab === 'help' }" title="Command help (⌘/Ctrl + Shift + \\)" @click="openSidebarTab('help')">
          <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M11,18H13V16H11V18M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M12,20C7.59,20 4,16.41 4,12C4,7.59 7.59,4 12,4C16.41,4 20,7.59 20,12C20,16.41 16.41,20 12,20M12,6A4,4 0 0,0 8,10H10A2,2 0 0,1 12,8A2,2 0 0,1 14,10C14,12 11,11.75 11,15H13C13,12.75 16,12.5 16,10A4,4 0 0,0 12,6Z" /></svg>
          <span class="rail-label">Help</span>
        </button>
        <button class="rail-item" title="Welcome to SOOG" @click="showHelp = true">
          <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M11,9H13V7H11M12,20C7.59,20 4,16.41 4,12C4,7.59 7.59,4 12,4C16.41,4 20,7.59 20,12C20,16.41 16.41,20 12,20M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M11,17H13V11H11V17Z" /></svg>
          <span class="rail-label">Welcome</span>
        </button>
        <NuxtLink v-if="profile?.is_admin" to="/usage" class="rail-item" title="Daily and weekly usage">
          <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M5,9.2H8V19H5V9.2M10.6,5H13.4V19H10.6V5M16.2,13H19V19H16.2V13" /></svg>
          <span class="rail-label">Usage</span>
        </NuxtLink>

        <div class="rail-flyout-wrap">
          <button
            class="rail-item"
            :class="{ active: railFlyout === 'model' }"
            :title="`Model: ${currentModel || 'loading...'}`"
            aria-haspopup="menu"
            :aria-expanded="String(railFlyout === 'model')"
            @click="toggleRailFlyout('model')"
          >
            <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M12,4V1L8,5L12,9V6A6,6 0 0,1 18,12C18,13 17.75,13.96 17.3,14.8L18.76,16.26C19.53,15.05 20,13.57 20,12A8,8 0 0,0 12,4M6.7,9.2L5.24,7.74C4.47,8.95 4,10.43 4,12A8,8 0 0,0 12,20V23L16,19L12,15V18A6,6 0 0,1 6,12C6,11 6.25,10.04 6.7,9.2Z" /></svg>
            <span class="rail-label">{{ shortModelLabel }}</span>
          </button>
          <div v-if="railFlyout === 'model'" class="rail-flyout" role="menu" aria-label="Model">
            <div class="rail-flyout__title">Model</div>
            <button
              v-for="m in ollamaModels"
              :key="m"
              class="rail-flyout__option"
              :class="{ active: m === currentModel }"
              :disabled="modelSwitching"
              role="menuitemradio"
              :aria-checked="String(m === currentModel)"
              @click="selectOllamaModel(m)"
            >{{ m }}</button>
            <p v-if="!ollamaModels.length" class="rail-flyout__empty">{{ currentModel || 'No model list available.' }}</p>
          </div>
        </div>

        <div class="rail-flyout-wrap">
          <button
            class="rail-item"
            :class="{ active: railFlyout === 'account' }"
            :title="isAuthenticated ? `${userLabel} · account` : 'Sign in'"
            aria-haspopup="menu"
            :aria-expanded="String(railFlyout === 'account')"
            @click="toggleRailFlyout('account')"
          >
            <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M12,4A4,4 0 0,1 16,8A4,4 0 0,1 12,12A4,4 0 0,1 8,8A4,4 0 0,1 12,4M12,14C16.42,14 20,15.79 20,18V20H4V18C4,15.79 7.58,14 12,14Z" /></svg>
            <span class="rail-label">{{ isAuthenticated ? userLabel : 'Sign in' }}</span>
          </button>
          <div v-if="railFlyout === 'account'" class="rail-flyout" role="menu" aria-label="Account">
            <template v-if="isAuthenticated">
              <div class="rail-flyout__title">{{ userLabel }}</div>
              <span v-if="quota" class="quota-pill" :title="quotaTitle">{{ quotaLabel }}</span>
              <button class="rail-flyout__option" role="menuitem" @click="railFlyout = ''; handleSignOut()">Sign out</button>
            </template>
            <button v-else class="rail-flyout__option" role="menuitem" @click="railFlyout = ''; showAuthGate = true">Sign in</button>
          </div>
        </div>
      
      </template>
    </NavRail>

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
      <Transition
        enter-active-class="fadeIn"
        leave-active-class="fadeOut"
        :duration="300"
        mode="out-in"
      >
        <div
          class="results-panel"
          :class="{
            'results-panel--idle': !hasResults,
            'results-panel--background': workspaceLayout === 'background'
          }"
          :key="workspaceLayout"
        >
          <div
            ref="workspaceStageRef"
            class="workspace-stage"
            :class="[`workspace-stage--${workspacePreset}`, { 'workspace-stage--dragging': workspaceDrag }]"
          >
            <section
              v-for="pane in activeWorkspacePanes"
              :key="pane.id"
              class="workspace-pane"
              :class="{ 'workspace-pane--moving': workspaceDrag?.fromId === pane.id }"
              :style="{ '--x': `${pane.x}%`, '--y': `${pane.y}%`, '--w': `${pane.w}%`, '--h': `${pane.h}%` }"
            >
              <div
                class="workspace-pane__header"
                :class="{ 'workspace-pane__header--movable': activeWorkspacePanes.length > 1 }"
                @pointerdown="onPaneHeaderDown(pane, $event)"
              >
                <PaneViewMenu
                  :model-value="pane.assigned"
                  :options="workspaceViewOptions"
                  @update:model-value="setWorkspacePaneView(pane.index, $event)"
                />
                <div class="workspace-pane__controls">
                  <button
                    v-if="pane.view === 'geometry' && stlUrl"
                    @click="downloadCurrentStl"
                    class="download-btn"
                  >
                    Download STL
                  </button>
                  <button
                    v-if="pane.view === 'geometry' && itemTarget"
                    class="pane-action"
                    title="Reconstruct a high-resolution 3D mesh for this item"
                    :disabled="remakingLRM"
                    @click="buildMesh(itemTarget)"
                  >
                    {{ remakingLRM ? '…' : 'BUILD 3D' }}
                  </button>
                  <button
                    v-if="pane.view === 'sketch' && hasResults"
                    class="pane-action"
                    title="Regenerate the diffusion sketch"
                    :disabled="loading || remakingSketch"
                    @click="regenerateSketch"
                  >
                    {{ loading || remakingSketch ? '…' : 'REGEN SKETCH' }}
                  </button>
                  <button
                    v-if="pane.view === 'acoustic' && itemTarget"
                    class="pane-action"
                    title="Run the acoustic simulation for this item"
                    :disabled="remakingModulus"
                    @click="calcAcoustics(itemTarget)"
                  >
                    {{ remakingModulus ? '…' : 'CALC ACOUSTICS' }}
                  </button>
                  <template v-if="pane.view === 'acoustic' && acousticViewUsesPlane && activeAcousticPlaneData">
                    <div class="modulus-mode-toggle">
                      <button
                        class="modulus-mode-btn"
                        :class="{ active: modulusRenderMode === '2d' }"
                        @click="modulusRenderMode = '2d'"
                      >
                        MAP
                      </button>
                      <button
                        class="modulus-mode-btn"
                        :class="{ active: modulusRenderMode === '3d' }"
                        @click="modulusRenderMode = '3d'"
                      >
                        SURF
                      </button>
                    </div>
                  </template>
                  <template v-if="pane.view === 'acoustic' && activeModulusData">
                    <div class="acoustic-view-toggle">
                      <button
                        v-for="option in acousticViewModeOptions"
                        :key="option.key"
                        class="acoustic-view-btn"
                        :class="{ active: activeAcousticViewMode === option.key }"
                        :title="option.detail"
                        @click="setAcousticViewMode(option.key)"
                      >
                        {{ option.label }}
                      </button>
                    </div>
                  </template>
                  <button
                    v-if="pane.view === 'stage'"
                    class="pane-action"
                    title="Toggle stage fullscreen (browser)"
                    @click="toggleStageBrowserFullscreen"
                  >
                    FULL
                  </button>
                  <button
                    v-if="activeWorkspacePanes.length < 8"
                    class="pane-icon"
                    title="Add a pod: split this one to the right (Shift: below)"
                    aria-label="Split pod"
                    @click="splitWorkspacePane(pane.index, $event.shiftKey ? 'col' : 'row')"
                  >
                    <svg class="pane-svg" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5v14M5 12h14" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" /></svg>
                  </button>
                  <button
                    v-if="workspacePreset !== 'fullscreen'"
                    class="pane-icon"
                    title="Show this pod alone"
                    aria-label="Maximize pane"
                    @click="maximizeWorkspacePane(pane.index)"
                  >
                    <svg class="pane-svg" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" /></svg>
                  </button>
                  <button
                    v-else-if="canRestoreWorkspace"
                    class="pane-icon"
                    title="Back to the previous layout"
                    aria-label="Restore layout"
                    @click="restoreWorkspacePanes"
                  >
                    <svg class="pane-svg" viewBox="0 0 24 24" aria-hidden="true"><path d="M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" /></svg>
                  </button>
                  <button
                    v-if="activeWorkspacePanes.length > 1"
                    class="pane-icon pane-close"
                    title="Close this pod; the others fill the space"
                    aria-label="Close pod"
                    @click="closeWorkspacePane(pane.index)"
                  >
                    <svg class="pane-svg" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" /></svg>
                  </button>
                </div>
              </div>

              <div class="workspace-pane__body">
                <template v-if="pane.view === 'organogram'">
                  <img
                    v-if="plotImage"
                    :src="plotImage"
                    alt="Organogram"
                    @click="openLightbox(plotImage, 'Organogram')"
                    class="plot-image"
                  />
                  <div v-else class="panel-placeholder">No organogram image generated yet.</div>
                </template>

                <template v-else-if="pane.view === 'sound'">
                  <div class="sound-pane">
                    <div v-if="!itemTarget" class="panel-placeholder">Sound samples belong to a gallery item. Render or load one first.</div>
                    <template v-else>
                      <div v-if="soundSamples.length" class="sound-list">
                        <div v-for="(sample, sampleIdx) in soundSamples" :key="sampleIdx" class="sound-row" :class="{ playing: playingIdx === sampleIdx }">
                          <button class="pane-action sample-btn" :title="`Play sample ${sampleIdx + 1}`" @click="toggleAudio(sample, sampleIdx)">
                            {{ playingIdx === sampleIdx ? '❚❚' : '▶' }}{{ sampleIdx + 1 }}
                          </button>
                          <span class="sound-row__prompt" :title="sample.prompt">{{ sample.prompt || 'no prompt stored' }}</span>
                          <span v-if="playingIdx === sampleIdx" class="pane-timer">{{ formatAudioTime(currentTime) }}/{{ formatAudioTime(duration) }}</span>
                          <a class="pane-action" :href="assetHref(sample.url)" download title="Download WAV">WAV</a>
                        </div>
                      </div>
                      <div v-else class="panel-placeholder">No sound samples yet for this item.</div>
                      <button class="pane-action sound-pane__generate" :disabled="generatingSound" @click="openSoundGenerator(itemTarget)">
                        {{ generatingSound ? 'SYNTHESIZING…' : '+ SOUND' }}
                      </button>
                    </template>
                  </div>
                </template>

                <template v-else-if="pane.view === 'summary'">
                  <div class="sound-pane">
                    <div v-if="soundPaneFacts.length" class="modulus-readout sound-pane__facts">
                      <span v-for="fact in soundPaneFacts" :key="fact">{{ fact }}</span>
                    </div>
                    <div v-if="summaryHtml" class="summary-content" v-html="summaryHtml"></div>
                    <div v-else class="panel-placeholder">Evaluate a generation to populate conceptual summary and materials.</div>
                    <div class="section-header materials-title">
                      <h3 class="section-title">MATERIALS</h3>
                      <span v-if="responseMetaText" class="section-meta">{{ responseMetaText }}</span>
                    </div>
                    <pre v-if="materialsText" class="materials-list">{{ materialsText }}</pre>
                    <div v-else class="panel-placeholder">No material list yet.</div>
                  </div>
                </template>

                <template v-else-if="pane.view === 'geometry'">
                  <ClientOnly>
                    <div v-if="stlUrl" class="stl-viewer-container">
                      <StlViewer :url="stlUrl" />
                    </div>
                    <div v-else class="stl-placeholder">No STL geometry generated for this response.</div>
                  </ClientOnly>
                </template>

                <template v-else-if="pane.view === 'acoustic'">
                  <div v-if="activeModulusData" class="modulus-results">
                    <div class="modulus-stack">
                      <div class="acoustic-temporal-bar">
                        <div class="acoustic-temporal-group">
                          <button class="modulus-mode-btn" @click="toggleAcousticTemporalPlayback">
                            {{ acousticTemporalPlaying ? 'PAUSE' : 'PLAY' }}
                          </button>
                          <span class="acoustic-temporal-readout">{{ activeAcousticViewDetail }} · {{ acousticTemporalPhaseLabel }}</span>
                        </div>
                        <label class="acoustic-temporal-slider">
                          <span>Phase</span>
                          <input
                            type="range"
                            min="0"
                            :max="acousticTemporalMax"
                            step="0.01"
                            :value="acousticTemporalPhase"
                            @input="acousticTemporalPhase = Number($event.target.value)"
                          >
                        </label>
                        <label class="acoustic-temporal-slider acoustic-temporal-slider--speed">
                          <span>Speed {{ acousticTemporalSpeedLabel }}</span>
                          <input
                            type="range"
                            min="0.2"
                            max="2.4"
                            step="0.05"
                            :value="acousticTemporalSpeed"
                            @input="acousticTemporalSpeed = Number($event.target.value)"
                          >
                        </label>
                      </div>

                      <ClientOnly v-if="activeAcousticViewMode === 'volume' && acousticHasVolume">
                        <div class="modulus-volume-shell">
                          <AcousticVolumeViewer
                            :volume="activeModulusData.results.pressure_volume"
                            :markers="modulusMarkers"
                            :primitive="activeModulusData.params?.primitive || 'sphere'"
                            :phase="acousticTemporalPhase"
                          />
                        </div>
                      </ClientOnly>
                      <ClientOnly v-else-if="activeAcousticViewMode === 'iso' && acousticHasVolume">
                        <div class="modulus-volume-shell modulus-volume-shell--iso">
                          <AcousticIsoSurfaceViewer
                            :volume="activeModulusData.results.pressure_volume"
                            :markers="modulusMarkers"
                            :primitive="activeModulusData.params?.primitive || 'sphere'"
                            :phase="acousticTemporalPhase"
                          />
                        </div>
                      </ClientOnly>
                      <template v-else-if="activeAcousticPlaneData">
                        <ModulusHeatmap
                          v-if="modulusRenderMode === '2d'"
                          :data="activeAcousticPlaneData"
                          :markers="modulusMarkers"
                          :plane="activeAcousticPlaneKey"
                          :phase="acousticTemporalPhase"
                        />
                        <ClientOnly v-else>
                          <div class="modulus-surface-shell">
                            <AcousticFieldSurface
                              :data="activeAcousticPlaneData"
                              :markers="modulusMarkers"
                              :plane="activeAcousticPlaneKey"
                              :phase="acousticTemporalPhase"
                            />
                          </div>
                        </ClientOnly>
                      </template>
                      <pre v-else class="modulus-json">{{ JSON.stringify(activeModulusData.results || activeModulusData, null, 2) }}</pre>

                      <div class="modulus-readout">
                        <span>Probe {{ acousticProbeResponseText }}</span>
                        <span>Peaks {{ acousticPeakSummary }}</span>
                        <span>{{ activeAcousticViewDetail }}</span>
                        <span>{{ activeModulusSource === 'preview' ? 'Live surrogate preview' : 'Evaluated phase 2 field' }}</span>
                      </div>
                    </div>
                  </div>
                  <div v-else class="panel-placeholder">No acoustical simulation data for this response.</div>
                </template>

                <template v-else-if="pane.view === 'sota'">
                  <div class="sota-pane">
              <div class="acoustic-summary">
                <h4>Live Command</h4>
                <p>{{ acousticCommandSummary }}</p>
              </div>
              <div v-if="!liveAcousticCommand" class="acoustic-summary">
                <h4>First Primitive</h4>
                <p>Use <code>@acoustic mode=2d primitive=circle freq=440 source=-0.55,0 probe=0.55,0 obstacle=0.15,0</code> or <code>@acoustic mode=3d primitive=sphere freq=440 source=-0.45,0,-0.28 probe=0.42,0.08,0.32 obstacle=0.08,0.12,0</code>.</p>
                <button class="sidebar-action" @click="insertStarterAcousticCommand">Insert starter command</button>
              </div>
              <template v-else>
                <div class="acoustic-actions">
                  <button class="sidebar-action" @click="evaluateCurrentAcousticCommand">Evaluate SOT-A</button>
                  <div v-if="acousticViewUsesPlane && activeAcousticPlaneData" class="modulus-mode-toggle">
                    <button
                      class="modulus-mode-btn"
                      :class="{ active: modulusRenderMode === '2d' }"
                      @click="modulusRenderMode = '2d'"
                    >
                      MAP
                    </button>
                    <button
                      class="modulus-mode-btn"
                      :class="{ active: modulusRenderMode === '3d' }"
                      @click="modulusRenderMode = '3d'"
                    >
                      SURF
                    </button>
                  </div>
                </div>
                <div class="acoustic-view-toggle acoustic-view-toggle--sidebar">
                  <button
                    v-for="option in acousticViewModeOptions"
                    :key="option.key"
                    class="acoustic-view-btn"
                    :class="{ active: activeAcousticViewMode === option.key }"
                    :title="option.detail"
                    @click="setAcousticViewMode(option.key)"
                  >
                    {{ option.label }}
                  </button>
                </div>
                <div class="acoustic-mode-row">
                  <button
                    class="primitive-chip"
                    :class="{ active: acousticSliderState.mode === '2d' }"
                    @click="setAcousticMode('2d')"
                  >
                    planar
                  </button>
                  <button
                    class="primitive-chip"
                    :class="{ active: acousticSliderState.mode === '3d' }"
                    @click="setAcousticMode('3d')"
                  >
                    volumetric
                  </button>
                </div>
                <div class="acoustic-primitive-row">
                  <button
                    v-for="primitive in acousticPrimitiveOptions"
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
                  <label v-if="acousticSliderState.mode === '3d'" class="slider-unit">
                    <span>Source Z</span>
                    <strong>{{ acousticSliderState.sourceZ.toFixed(2) }}</strong>
                    <input type="range" min="-0.95" max="0.95" step="0.01" :value="acousticSliderState.sourceZ" @input="updateAcousticSliderValue('source', 2, Number($event.target.value))">
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
                  <label v-if="acousticSliderState.mode === '3d'" class="slider-unit">
                    <span>Probe Z</span>
                    <strong>{{ acousticSliderState.probeZ.toFixed(2) }}</strong>
                    <input type="range" min="-0.95" max="0.95" step="0.01" :value="acousticSliderState.probeZ" @input="updateAcousticSliderValue('probe', 2, Number($event.target.value))">
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
                  <label v-if="acousticSliderState.mode === '3d'" class="slider-unit">
                    <span>Obstacle Z</span>
                    <strong>{{ acousticSliderState.obstacleZ.toFixed(2) }}</strong>
                    <input type="range" min="-0.95" max="0.95" step="0.01" :value="acousticSliderState.obstacleZ" @input="updateAcousticSliderValue('obstacle', 2, Number($event.target.value))">
                  </label>
                </div>
                <div class="acoustic-temporal-bar acoustic-temporal-bar--sidebar">
                  <div class="acoustic-temporal-group">
                    <button class="modulus-mode-btn" @click="toggleAcousticTemporalPlayback">
                      {{ acousticTemporalPlaying ? 'PAUSE' : 'PLAY' }}
                    </button>
                    <span class="acoustic-temporal-readout">{{ acousticTemporalPhaseLabel }}</span>
                  </div>
                  <label class="acoustic-temporal-slider">
                    <span>Phase</span>
                    <input
                      type="range"
                      min="0"
                      :max="acousticTemporalMax"
                      step="0.01"
                      :value="acousticTemporalPhase"
                      @input="acousticTemporalPhase = Number($event.target.value)"
                    >
                  </label>
                  <label class="acoustic-temporal-slider acoustic-temporal-slider--speed">
                    <span>Speed {{ acousticTemporalSpeedLabel }}</span>
                    <input
                      type="range"
                      min="0.2"
                      max="2.4"
                      step="0.05"
                      :value="acousticTemporalSpeed"
                      @input="acousticTemporalSpeed = Number($event.target.value)"
                    >
                  </label>
                </div>
                <div class="acoustic-preview-shell">
                  <ClientOnly v-if="activeAcousticViewMode === 'volume' && acousticHasVolume">
                    <div class="acoustic-preview-surface acoustic-preview-volume">
                      <AcousticVolumeViewer
                        :volume="activeModulusData.results.pressure_volume"
                        :markers="modulusMarkers"
                        :primitive="activeModulusData.params?.primitive || 'sphere'"
                        :phase="acousticTemporalPhase"
                      />
                    </div>
                  </ClientOnly>
                  <ClientOnly v-else-if="activeAcousticViewMode === 'iso' && acousticHasVolume">
                    <div class="acoustic-preview-surface acoustic-preview-volume">
                      <AcousticIsoSurfaceViewer
                        :volume="activeModulusData.results.pressure_volume"
                        :markers="modulusMarkers"
                        :primitive="activeModulusData.params?.primitive || 'sphere'"
                        :phase="acousticTemporalPhase"
                      />
                    </div>
                  </ClientOnly>
                  <ModulusHeatmap
                    v-else-if="modulusRenderMode === '2d' && activeAcousticPlaneData"
                    :data="activeAcousticPlaneData"
                    :size="280"
                    :markers="modulusMarkers"
                    :plane="activeAcousticPlaneKey"
                    :phase="acousticTemporalPhase"
                  />
                  <ClientOnly v-else-if="activeAcousticPlaneData">
                    <div class="acoustic-preview-surface">
                      <AcousticFieldSurface
                        :data="activeAcousticPlaneData"
                        :markers="modulusMarkers"
                        :plane="activeAcousticPlaneKey"
                        :phase="acousticTemporalPhase"
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
                </template>

                <template v-else-if="pane.view === 'stage'">
                  <ClientOnly>
                    <ConcertSimulator embedded />
                  </ClientOnly>
                </template>

                <template v-else-if="pane.view === 'sketch'">
                  <img
                    v-if="sketchImage"
                    :src="sketchImage"
                    alt="Sketch render"
                    @click="openLightbox(sketchImage, 'Sketch')"
                    class="plot-image"
                  />
                  <div v-else class="sketch-placeholder">No diffusion sketch generated for this response.</div>
                </template>

                <template v-else>
                  <div class="panel-placeholder">No routed content for this pane yet.</div>
                </template>
              </div>
            </section>
            <div
              v-for="handle in workspaceHandles"
              :key="handle.id"
              class="pane-handle"
              :class="handle.dir === 'row' ? 'pane-handle--x' : 'pane-handle--y'"
              :style="handleStyle(handle)"
              role="separator"
              :aria-orientation="handle.dir === 'row' ? 'vertical' : 'horizontal'"
              title="Drag to resize (double-click to reset)"
              @pointerdown.prevent="startHandleDrag(handle, $event)"
              @dblclick="resetHandle(handle)"
            ></div>
            <div
              v-if="workspaceDropPreview"
              class="pane-drop"
              :style="{ left: `${workspaceDropPreview.x}%`, top: `${workspaceDropPreview.y}%`, width: `${workspaceDropPreview.w}%`, height: `${workspaceDropPreview.h}%` }"
            >
              <span>{{ workspaceDropPreview.zone === 'center' ? 'SWAP' : `DOCK ${workspaceDropPreview.zone.toUpperCase()}` }}</span>
            </div>
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
            <button
              class="command-domain__header"
              :aria-expanded="String(isHelpOpen(domain.key))"
              @click="toggleHelpDomain(domain.key)"
            >
              <span class="command-domain__badge" :style="{ '--domain-color': domain.color }">{{ domain.label }}</span>
              <span class="command-domain__count">{{ domain.commands.length }}</span>
              <span class="command-domain__chevron" aria-hidden="true">{{ isHelpOpen(domain.key) ? '−' : '+' }}</span>
            </button>
            <template v-if="isHelpOpen(domain.key)">
              <div v-for="command in domain.commands" :key="command.syntax" class="command-domain__row">
                <div class="command-domain__line">
                  <code>{{ command.syntax }}</code>
                  <button
                    v-if="command.example"
                    class="pane-action"
                    :title="`Insert in the editor:\n${command.example}`"
                    @click="insertHelpExample(command.example)"
                  >
                    INSERT
                  </button>
                </div>
                <p>{{ command.detail }}</p>
              </div>
            </template>
          </div>
        </div>

        <div v-else-if="workspaceSidebarTab === 'session'" class="workspace-sidebar__body">
          <div class="session-facts">
            <div v-for="fact in sessionFacts" :key="fact.label" class="session-fact">
              <span>{{ fact.label }}</span>
              <strong>{{ fact.value }}</strong>
            </div>
          </div>

          <div class="session-section">
            <h4>Stages</h4>
            <div v-if="stageRows.length" class="stage-list">
              <div
                v-for="row in stageRows"
                :key="row.key"
                class="stage-row"
                :class="{ 'stage-row--running': row.running }"
              >
                <span class="stage-row__name">{{ row.label }}</span>
                <span class="stage-row__time">{{ row.duration }}</span>
                <span
                  v-if="isOperator && row.cpu !== null"
                  class="stage-row__cpu"
                  :title="`Server CPU ${row.cpu}% when the stage began`"
                ><i :style="{ width: `${row.cpu}%` }"></i></span>
              </div>
            </div>
            <p v-else class="session-empty">No render in this session yet.</p>
          </div>

          <div v-if="isOperator" class="session-section">
            <h4>Server</h4>
            <template v-if="serverMetrics">
              <div v-if="serverMetrics.cpu_pct != null" class="cpu-meter" :title="`CPU ${serverMetrics.cpu_pct}%`">
                <i :style="{ width: `${serverMetrics.cpu_pct}%` }"></i>
              </div>
              <div class="session-facts">
                <div v-for="fact in serverFacts" :key="fact.label" class="session-fact">
                  <span>{{ fact.label }}</span>
                  <strong>{{ fact.value }}</strong>
                </div>
              </div>
            </template>
            <p v-else class="session-empty">Reading server load…</p>
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
    <ProcessDock v-if="uiVisible" :inset="railExpanded ? 210 : 44" :bottom="17" />
    <div v-if="uiVisible" class="footer">
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
    <div v-if="soundModalOpen" class="sound-modal-overlay" @click.self="closeSoundGenerator">
      <div class="sound-modal">
        <header class="sound-modal__header">
          <h4>CUSTOM SOUND SYNTHESIS</h4>
          <button @click="closeSoundGenerator">×</button>
        </header>
        <p class="sound-modal__desc">Describe the sound; Stable Audio generates samples for this item.</p>
        <textarea v-model="soundPrompt" class="sound-modal__textarea" placeholder="Describe the sound..."></textarea>
        <div class="sound-modal__actions">
          <button class="cancel-btn" @click="closeSoundGenerator">Cancel</button>
          <button class="confirm-btn" :disabled="generatingSound" @click="runSoundGenerator">
            {{ generatingSound ? 'GENERATING...' : 'GENERATE SAMPLES' }}
          </button>
        </div>
      </div>
    </div>
    <HelpModal 
      v-if="uiVisible" v-model="showHelp" 
      @select-featured="handleSelectFeatured"
    />
    <GalleryDock
      v-if="uiVisible && !sorganoidMode" v-model="showGallery"
      :initial-basename="targetBasename"
      @load-code="loadCodeFromGallery"
      @preview="previewGalleryItem"
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
import { ref, reactive, onMounted, onUnmounted, computed, nextTick, watch } from 'vue';
import { useRuntimeConfig } from '#app';
import { marked } from 'marked';
import AceEditor from '~/components/AceEditor.vue';
import HelpModal from '~/components/HelpModal.vue';
import GalleryDock from '~/components/GalleryDock.vue';
import NavRail from '~/components/NavRail.vue';
import StlViewer from '~/components/StlViewer.vue';
import ConcertSimulator from '~/components/ConcertSimulator.vue';
import ModulusHeatmap from '~/components/ModulusHeatmap.vue';
import AcousticFieldSurface from '~/components/AcousticFieldSurface.vue';
import AcousticVolumeViewer from '~/components/AcousticVolumeViewer.vue';
import AcousticIsoSurfaceViewer from '~/components/AcousticIsoSurfaceViewer.vue';
import SorganoidWorld from '~/components/SorganoidWorld.vue';
import AuthGateModal from '~/components/AuthGateModal.vue';
import UsageDashboard from '~/components/UsageDashboard.vue';
import { useRandomPrompt } from '~/composables/useRandomPrompt';
import { useFavicon } from '~/composables/useFavicon';
import { useAcousticSonification } from '~/composables/useAcousticSonification';
import { useApi } from '~/composables/useApi';
import { useWorkspace } from '~/composables/useWorkspace';
import { useItemTools } from '~/composables/useItemTools';
import { useProcesses } from '~/composables/useProcesses';
import { shortLabel } from '~/composables/processMath';
import { buildServerFacts, buildStageRows } from '~/composables/sessionTelemetry';
import { useSoogAuth } from '~/composables/useSoogAuth';
import { runAcousticSimulation } from '~/utils/acousticSimulation';
import { phaseTurnsLabel, wrapPhase } from '~/utils/acousticTemporal';

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
const leftWidth = ref(38);
let dragging = false;
let startX = 0;
let startLeft = 50;
const {
  preset: workspacePreset,
  layout: workspaceLayout,
  presetMenuOpen: workspacePresetMenuOpen,
  presetMenuRef: workspacePresetMenuRef,
  autoView: workspaceAutoView,
  stageRef: workspaceStageRef,
  drag: workspaceDrag,
  dropPreview: workspaceDropPreview,
  handles: workspaceHandles,
  presetOptions: workspacePresetOptions,
  viewOptions: workspaceViewOptions,
  modeLabel: workspaceModeLabel,
  panes: activeWorkspacePanes,
  viewLabel: workspaceViewLabel,
  toggleMenu: toggleWorkspacePresetMenu,
  setPreset: setWorkspacePreset,
  setPaneView: setWorkspacePaneView,
  showView: showWorkspaceView,
  maximizePane: maximizeWorkspacePane,
  restorePanes: restoreWorkspacePanes,
  canRestore: canRestoreWorkspace,
  closePane: closeWorkspacePane,
  splitPane: splitWorkspacePane,
  startHandleDrag,
  resetHandle,
  startPaneMove,
  restoreLayout: restoreWorkspaceLayout,
  snapshotLayout: snapshotWorkspaceLayout,
  restoreSnapshot: restoreWorkspaceSnapshot,
  tree: workspaceTree,
} = useWorkspace();
const {
  remakingSketch, remakingLRM, remakingModulus, generatingSound,
  soundModalOpen, soundPrompt, openSoundGenerator, closeSoundGenerator, runSoundGenerator,
  playingIdx, currentTime, duration, toggleAudio, toolError,
  remakeSketch: remakeItemSketch, buildMesh, calcAcoustics, onItemUpdated, assetHref,
} = useItemTools();
const { startProcess, finishProcess } = useProcesses();
const loading = ref(false);
const progress = ref(0);
const elapsedMs = ref(0);
const responseTimesByModel = ref({});
const progressModel = ref('');
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
const modulusRenderMode = ref('2d');
const acousticViewMode = ref('auto');
const acousticTemporalPlaying = ref(true);
const acousticTemporalPhase = ref(0);
const acousticTemporalSpeed = ref(1);
const workspaceSidebarOpen = ref(false);
const railRef = ref(null);
const railExpanded = ref(false);
const railFlyout = ref(''); // '' | 'model' | 'account'
const RAIL_KEY = 'soog.rail.expanded';
const workspaceSidebarTab = ref('help'); // 'help', 'session'
const lastAcousticCommand = ref(null);
const evaluatedAcousticSignature = ref('');
const editorContent = ref('');
const liveModulusData = ref(null);
const targetBasename = ref('');
const soundSamples = ref([]);
const itemPrompt = ref('');
// The gallery item the pods are showing; the pod tools (sketch, 3D, sound, acoustics) act on it.
const itemTarget = computed(() => (targetBasename.value ? { basename: targetBasename.value, prompt: itemPrompt.value } : null));
const acousticTemporalMax = Math.PI * 2;
let progressPollInterval = null;
let editorSyncBound = false;
let editorSessionChangeHandler = null;
let livePreviewTimer = null;
let editorSyncTimer = null;
let acousticTemporalInterval = null;
let acousticTemporalLastTick = 0;

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
const isMobileOrTablet = ref(false);
const showLightbox = ref(false);
const RESPONSE_TIMES_KEY = 'soog_response_times_by_model.v1';
const DEFAULT_RESPONSE_MS = 180000;
const MAX_RESPONSE_MS = 60 * 60 * 1000;
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

const median = (values) => {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

// Typical duration for the model of the running request: median of its own successful runs.
// Without history for that model, fall back to the median of all models, then a conservative default.
const typicalResponse = computed(() => {
  const byModel = responseTimesByModel.value;
  const own = byModel[progressModel.value || currentModel.value] || [];
  if (own.length) return { ms: median(own), basis: `${own.length} run${own.length > 1 ? 's' : ''}` };
  const all = Object.values(byModel).flat();
  if (all.length) return { ms: median(all), basis: 'other models' };
  return { ms: DEFAULT_RESPONSE_MS, basis: 'no history' };
});
const averageResponseMs = computed(() => typicalResponse.value.ms);


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
const liveAcousticSignature = computed(() => acousticSignatureFromMeta(liveAcousticCommand.value?.meta));
const currentAcousticMeta = computed(() => liveAcousticCommand.value?.meta || lastAcousticCommand.value);
const activeModulusData = computed(() => {
  if (modulusData.value) {
    if (evaluatedAcousticSignature.value && evaluatedAcousticSignature.value === liveAcousticSignature.value) {
      return modulusData.value;
    }
    if (!liveModulusData.value) {
      return modulusData.value;
    }
  }
  return liveModulusData.value || modulusData.value;
});
const activeModulusSource = computed(() => {
  if (activeModulusData.value && modulusData.value && activeModulusData.value === modulusData.value) return 'evaluated';
  if (activeModulusData.value && liveModulusData.value && activeModulusData.value === liveModulusData.value) return 'preview';
  return 'none';
});
const activeAcousticMode = computed(() => (
  String(activeModulusData.value?.params?.mode || currentAcousticMeta.value?.mode || '2d').toLowerCase() === '3d'
    ? '3d'
    : '2d'
));
const acousticPrimitiveOptions = computed(() => (
  activeAcousticMode.value === '3d'
    ? ['sphere', 'cube', 'cylinder']
    : ['circle', 'square', 'triangle', 'hexagon']
));
const acousticHasPlane = computed(() => Boolean(activeModulusData.value?.results?.pressure_map));
const acousticHasVolume = computed(() => Boolean(activeModulusData.value?.results?.pressure_volume));
const acousticViewModeOptions = computed(() => (
  activeAcousticMode.value === '3d'
    ? [
        { key: 'volume', label: 'VOL', detail: 'Volumetric cloud' },
        { key: 'iso', label: 'ISO', detail: 'Iso-shell' },
        { key: 'xy', label: 'XY', detail: 'Probe slice XY' },
        { key: 'xz', label: 'XZ', detail: 'Probe slice XZ' },
        { key: 'yz', label: 'YZ', detail: 'Probe slice YZ' },
      ]
    : [
        { key: 'xy', label: 'XY', detail: 'Planar field' },
      ]
));
const activeAcousticViewMode = computed(() => {
  const allowed = acousticViewModeOptions.value.map((option) => option.key);
  if (allowed.includes(acousticViewMode.value)) return acousticViewMode.value;
  return activeAcousticMode.value === '3d' ? 'volume' : 'xy';
});
const acousticViewUsesPlane = computed(() => ['xy', 'xz', 'yz'].includes(activeAcousticViewMode.value));
const activeAcousticPlaneKey = computed(() => (
  activeAcousticViewMode.value === 'xz' || activeAcousticViewMode.value === 'yz'
    ? activeAcousticViewMode.value
    : 'xy'
));
const activeAcousticPlaneData = computed(() => {
  const results = activeModulusData.value?.results || {};
  if (activeAcousticPlaneKey.value === 'xz') return results.slice_xz || null;
  if (activeAcousticPlaneKey.value === 'yz') return results.slice_yz || null;
  return results.slice_xy || results.pressure_map || null;
});
const activeAcousticViewDetail = computed(() => {
  const entry = acousticViewModeOptions.value.find((option) => option.key === activeAcousticViewMode.value);
  return entry?.detail || 'Acoustic field';
});
const acousticTemporalPhaseLabel = computed(() => phaseTurnsLabel(acousticTemporalPhase.value));
const acousticTemporalSpeedLabel = computed(() => `${acousticTemporalSpeed.value.toFixed(2)}x`);
const hasResults = computed(() => !!(plotImage.value || sketchImage.value || summary.value || materialsText.value || stlUrl.value || activeModulusData.value));
const rightColumnStyle = computed(() => (
  workspaceLayout.value === 'background'
    ? {}
    : { width: 'auto', flex: '1 1 auto', minWidth: '0' }
));
const acousticCommandSummary = computed(() => {
  const meta = currentAcousticMeta.value;
  if (!meta) return 'No acoustic command evaluated yet.';
  const formatPoint = (point = []) => point.map((value) => Number(value).toFixed(2)).join(', ');
  return `${String(meta.mode || '2d').toUpperCase()} · ${meta.primitive.toUpperCase()} · ${meta.freq} Hz · src ${formatPoint(meta.source)} · probe ${formatPoint(meta.probe)}`;
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
        z: Number(src.pos[2] ?? 0),
        label: src.freq ? `${Math.round(Number(src.freq))}Hz` : 'src'
      });
    }
  }
  if (Array.isArray(params.probe) && params.probe.length >= 2) {
    markers.push({
      type: 'probe',
      x: Number(params.probe[0]),
      y: Number(params.probe[1]),
      z: Number(params.probe[2] ?? 0),
      label: 'probe'
    });
  }
  if (Array.isArray(params.obstacle) && params.obstacle.length >= 2) {
    markers.push({
      type: 'obstacle',
      x: Number(params.obstacle[0]),
      y: Number(params.obstacle[1]),
      z: Number(params.obstacle[2] ?? 0),
      label: params.primitive || 'obs'
    });
  }
  return markers.filter((marker) => Number.isFinite(marker.x) && Number.isFinite(marker.y) && Number.isFinite(marker.z ?? 0));
});
const acousticProbeResponseText = computed(() => {
  const value = Number(activeModulusData.value?.results?.probe_response ?? activeModulusData.value?.results?.mic_response);
  return Number.isFinite(value) ? value.toFixed(3) : '—';
});
const acousticPeakSummary = computed(() => (
  (activeModulusData.value?.results?.resonance_peaks_hz || []).slice(0, 4).join(', ') || '—'
));
const soundPaneFacts = computed(() => {
  const facts = [];
  if (responseMetaText.value) facts.push(responseMetaText.value);
  if (activeModulusData.value?.method) facts.push(activeModulusData.value.method);
  if (activeModulusSource.value === 'preview') facts.push('live preview');
  if (activeModulusSource.value === 'evaluated') facts.push('evaluated field');
  if (activeModulusData.value?.results?.probe_response !== undefined) {
    facts.push(`probe ${acousticProbeResponseText.value}`);
  }
  if (activeModulusData.value?.results?.resonance_peaks_hz?.length) {
    facts.push(`peaks ${acousticPeakSummary.value}`);
  }
  return facts;
});
const acousticSliderState = computed(() => ({
  mode: currentAcousticMeta.value?.mode || '2d',
  primitive: currentAcousticMeta.value?.primitive || (currentAcousticMeta.value?.mode === '3d' ? 'sphere' : 'circle'),
  freq: Number(currentAcousticMeta.value?.freq || 440),
  sourceX: Number(currentAcousticMeta.value?.source?.[0] ?? -0.55),
  sourceY: Number(currentAcousticMeta.value?.source?.[1] ?? 0),
  sourceZ: Number(currentAcousticMeta.value?.source?.[2] ?? -0.28),
  probeX: Number(currentAcousticMeta.value?.probe?.[0] ?? 0.55),
  probeY: Number(currentAcousticMeta.value?.probe?.[1] ?? 0),
  probeZ: Number(currentAcousticMeta.value?.probe?.[2] ?? 0.32),
  obstacleX: Number(currentAcousticMeta.value?.obstacle?.[0] ?? 0.15),
  obstacleY: Number(currentAcousticMeta.value?.obstacle?.[1] ?? 0),
  obstacleZ: Number(currentAcousticMeta.value?.obstacle?.[2] ?? 0),
}));

const commandDomains = [
  {
    key: 'render',
    label: 'RENDER',
    color: '#8bc34a',
    commands: [
      { syntax: 'Alt+Enter', detail: 'Evaluate current selection or the full editor buffer.', example: 'A glass harp bowed by two performers on a rotating table' },
      { syntax: 'workspace menu', detail: 'Choose fullscreen, split vertical, split 3, or split 4 from the HUD preset icon.' },
      { syntax: 'pane routing', detail: 'Each pane can point to auto, organogram, sound, 3D object, acoustic, sketch, summary, SOT-A, or Stage 3D. ⤢ shows a pane alone, ⤡ returns.' },
      { syntax: 'pane docking', detail: 'Drag a pane header onto another pane: an edge docks beside it, the centre swaps the two. + splits a pane (Shift: below), × closes it and the others fill the space, dividers resize (double-click resets). The arrangement is remembered; the layout menu resets it.' },
      { syntax: '/sot-a · /stage · /session · /help', detail: 'Open that pane or sidebar tab without calling the model. Select the line and press Alt+Enter.', example: '/sot-a' },
      { syntax: 'Alt+1 · Alt+2 · Alt+3', detail: 'Jump between SOOG, SOMAP, and Concert Room.' }
    ]
  },
  {
    key: 'acoustic',
    label: 'SOT-A',
    color: '#ff9800',
    commands: [
      { syntax: '@acoustic mode=2d primitive=circle freq=440 source=-0.55,0 probe=0.55,0 obstacle=0.15,0', detail: 'Run the planar cavity study against the online field solver.', example: '@acoustic mode=2d primitive=circle freq=440 source=-0.55,0 probe=0.55,0 obstacle=0.15,0\nPlanar cavity study.' },
      { syntax: '@acoustic mode=3d primitive=sphere freq=440 source=-0.45,0,-0.28 probe=0.42,0.08,0.32 obstacle=0.08,0.12,0', detail: 'Run the volumetric cavity study with real XYZ coordinates.', example: '@acoustic mode=3d primitive=sphere freq=440 source=-0.45,0,-0.28 probe=0.42,0.08,0.32 obstacle=0.08,0.12,0\nVolumetric cavity study.' },
      { syntax: 'solver=<fd|surrogate>', detail: 'Optional override when you want to force the backend phase 2 field or the lighter preview-style solver.', example: '@acoustic mode=2d primitive=square freq=330 source=-0.5,0 probe=0.5,0 obstacle=0.1,0 solver=surrogate' },
      { syntax: 'preview vs evaluate', detail: 'Sliders and edits keep a local surrogate preview; Evaluate swaps in the backend phase 2 finite-difference field when the command matches.' },
      { syntax: 'primitive=<circle|square|triangle|hexagon|sphere|cube|cylinder>', detail: 'Choose planar or volumetric primitive families for the current acoustic test.', example: '@acoustic mode=3d primitive=cylinder freq=220 source=-0.45,0,-0.28 probe=0.42,0.08,0.32 obstacle=0.08,0.12,0' },
      { syntax: 'SOT-A sliders', detail: 'The SOT-A pane (rail icon, or the /sot-a command) edits the @acoustic command directly, including XYZ when mode=3d.' },
      { syntax: 'map / surf / vol / iso / xy-xz-yz', detail: 'Plane fields can switch between heatmap and relief surface; 3D studies can flip between cloud, iso-shell, and probe slices.' },
      { syntax: 'follow-up lines', detail: 'Any lines after the command are passed as contextual notes to the solver.', example: '@acoustic mode=2d primitive=hexagon freq=523 source=-0.5,0 probe=0.5,0 obstacle=0.12,0\nThe walls are fired clay; the source is a struck bowl.' }
    ]
  },
  {
    key: 'reversion',
    label: 'REVERSION',
    color: '#42a5f5',
    commands: [
      { syntax: '* correction notes…', detail: 'Fast reversion on top of the current instrument lineage.', example: '* make the resonator longer and add a second reed' },
      { syntax: '+ addon notes…', detail: 'Alternative shorthand for iterative variants.', example: '+ add a bowed drone string under the main body' },
      { syntax: '[REFACT source=<basename> group=<group_id> title=<name>]', detail: 'Explicit reversion header for precise lineage control. Replace the placeholders with the item basename and group from the gallery.', example: '[REFACT source=<basename> group=<group_id> title=<name>]\nReplace the glass bowls with ceramic ones.' }
    ]
  },
  {
    key: 'world',
    label: 'WORLD',
    color: '#ab47bc',
    commands: [
      { syntax: '§ list instruments', detail: 'Enter the Sorganoid world and list the instruments you can load.', example: '§ list instruments' },
      { syntax: '§ spawn 3 individuals of rectangulata', detail: 'Place individuals of an instrument in the world.', example: '§ spawn 3 individuals of rectangulata' },
      { syntax: '§ evolve speed for 3 generations', detail: 'Run evolution cycles toward speed, survival or efficiency.', example: '§ evolve speed for 3 generations' },
      { syntax: '§ mutate gravity 0.5', detail: 'Change the world gravity as a factor of 9.82.', example: '§ mutate gravity 0.5' },
      { syntax: '⌘/Ctrl + Shift + \\', detail: 'Toggle this command sidebar.' },
      { syntax: '?', detail: 'Legacy help stays intact; command docs now live here.' }
    ]
  }
];

const HELP_OPEN_KEY = 'soog.help.open';
const helpOpen = ref({ render: true });

function loadHelpOpenState() {
  try {
    const raw = window.localStorage.getItem(HELP_OPEN_KEY);
    if (raw) helpOpen.value = { ...helpOpen.value, ...JSON.parse(raw) };
  } catch {
    // keep the default fold state
  }
}

const isHelpOpen = (key) => !!helpOpen.value[key];

function toggleHelpDomain(key) {
  helpOpen.value = { ...helpOpen.value, [key]: !helpOpen.value[key] };
  try {
    window.localStorage.setItem(HELP_OPEN_KEY, JSON.stringify(helpOpen.value));
  } catch {
    // the fold state just won't persist
  }
}

// Append a help example to the editor and select it, so Alt+Enter evaluates exactly that example.
function insertHelpExample(text) {
  const editor = editorRef.value?.aceEditor?.();
  if (!editor || !text) return;
  const before = String(editor.getValue() || '');
  const gap = !before.trim() ? '' : before.endsWith('\n\n') ? '' : before.endsWith('\n') ? '\n' : '\n\n';
  editor.navigateFileEnd();
  const start = editor.getCursorPosition();
  editor.insert(`${gap}${text}\n`);
  const lines = text.split('\n');
  const startRow = start.row + (gap.match(/\n/g) || []).length;
  const startColumn = gap ? 0 : start.column;
  editor.selection.setSelectionRange({
    start: { row: startRow, column: startColumn },
    end: { row: startRow + lines.length - 1, column: lines[lines.length - 1].length },
  });
  editor.focus();
}

const isOperator = computed(() => !!profile.value?.is_admin);
const stageEvents = ref([]);
const progressInfo = ref({ status: '', started_at: 0, updated_at: 0 });
const serverMetrics = ref(null);

// One row per stage of the last render: how long it ran and, for operators, the server CPU when it began.
const stageRows = computed(() => buildStageRows(stageEvents.value, progressInfo.value));
const serverFacts = computed(() => buildServerFacts(serverMetrics.value));

const sessionFacts = computed(() => ([
  { label: 'Workspace', value: workspaceModeLabel.value.toUpperCase() },
  { label: 'Model', value: responseModel.value || currentModel.value || '—' },
  { label: 'Last render', value: responseElapsedMs.value ? `${(responseElapsedMs.value / 1000).toFixed(1)}s` : '—' },
  { label: 'Rolling avg', value: `${(averageResponseMs.value / 1000).toFixed(1)}s` },
  { label: 'Acoustic field', value: activeModulusSource.value === 'preview' ? 'PREVIEW' : activeModulusSource.value === 'evaluated' ? 'EVALUATED' : '—' },
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
    // Operators send their token so the response carries stage CPU and server load.
    const headers = { Accept: 'application/json', ...(isOperator.value ? await authHeaders().catch(() => ({})) : {}) };
    const response = await fetch(`${apiBase.value}/generate/progress/${generationRequestId.value}`, { headers });
    if (!response.ok) return;
    const payload = await response.json();
    if (!payload?.ok) return;
    reasoningPreview.value = String(payload.reasoning_preview || '').trim();
    progressStage.value = String(payload.stage || '').trim();
    stageEvents.value = Array.isArray(payload.events) ? payload.events : [];
    progressInfo.value = {
      status: String(payload.status || ''),
      started_at: Number(payload.started_at) || 0,
      updated_at: Number(payload.updated_at) || 0,
    };
    if (payload.server) serverMetrics.value = payload.server;
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
  stageEvents.value = [];
  progressInfo.value = { status: 'running', started_at: 0, updated_at: 0 };
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

function roundAcousticCoord(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? Number(numeric.toFixed(3)) : 0;
}

function acousticSignatureFromMeta(meta) {
  if (!meta) return '';
  const mode = String(meta.mode || '2d').toLowerCase() === '3d' ? '3d' : '2d';
  const defaults = acousticPointDefaults(mode);
  const size = mode === '3d' ? 3 : 2;
  const primitive = normalizeAcousticPrimitive(mode, meta.primitive);
  const freq = Math.round(Number(meta.freq) || 440);
  const source = ensurePointSize(meta.source, size, defaults.source).map(roundAcousticCoord).join(',');
  const probe = ensurePointSize(meta.probe, size, defaults.probe).map(roundAcousticCoord).join(',');
  const obstacle = ensurePointSize(meta.obstacle, size, defaults.obstacle).map(roundAcousticCoord).join(',');
  return [mode, primitive, freq, source, probe, obstacle].join('|');
}

function acousticSignatureFromSimulation(simulation) {
  const params = simulation?.params || {};
  const sources = Array.isArray(params.sources) ? params.sources : [];
  const sourcePoint = sources[0]?.pos || params.source || params.sources?.[0]?.pos;
  return acousticSignatureFromMeta({
    mode: params.mode,
    primitive: params.primitive,
    freq: params.freq || params.frequencies_hz?.[0],
    source: sourcePoint,
    probe: params.probe,
    obstacle: params.obstacle,
  });
}

function acousticPointDefaults(mode = '2d') {
  return mode === '3d'
    ? {
        source: [-0.45, 0, -0.28],
        probe: [0.42, 0.08, 0.32],
        obstacle: [0.08, 0.12, 0],
      }
    : {
        source: [-0.55, 0],
        probe: [0.55, 0],
        obstacle: [0.15, 0],
      };
}

function normalizeAcousticPrimitive(mode, primitive) {
  const nextMode = String(mode || '2d').toLowerCase() === '3d' ? '3d' : '2d';
  const value = String(primitive || '').trim().toLowerCase();
  const aliases = nextMode === '3d'
    ? { round: 'sphere', circular: 'sphere', ball: 'sphere', box: 'cube', pipe: 'cylinder', tube: 'cylinder' }
    : { round: 'circle', circular: 'circle', box: 'square', triangular: 'triangle', honeycomb: 'hexagon' };
  const resolved = aliases[value] || value;
  const allowed = nextMode === '3d'
    ? ['sphere', 'cube', 'cylinder']
    : ['circle', 'square', 'triangle', 'hexagon'];
  return allowed.includes(resolved) ? resolved : (nextMode === '3d' ? 'sphere' : 'circle');
}

function ensurePointSize(point, size, fallbacks) {
  return Array.from({ length: size }, (_, index) => {
    const raw = Array.isArray(point) ? point[index] : undefined;
    return clampUnitCoord(raw, fallbacks[index] ?? 0);
  });
}

function parsePointToken(rawValue, fallbacks) {
  if (typeof rawValue !== 'string' || !rawValue.trim()) {
    return ensurePointSize([], fallbacks.length, fallbacks);
  }
  const parts = rawValue
    .split(/[,:/]/)
    .map((part) => part.trim())
    .filter(Boolean);
  return ensurePointSize(parts, fallbacks.length, fallbacks);
}

function parseAcousticCommand(text) {
  const lines = String(text || '').split('\n');
  const firstIdx = lines.findIndex((line) => {
    const trimmed = line.trim().toLowerCase();
    return trimmed.startsWith('@acoustic');
  });
  if (firstIdx === -1) return null;

  const header = lines[firstIdx].trim();
  const headerLower = header.toLowerCase();
  const headerPrefix = headerLower.startsWith('@acoustic3d')
    ? '@acoustic3d'
    : headerLower.startsWith('@acoustic2d')
      ? '@acoustic2d'
      : '@acoustic';
  const rawTokens = header.slice(headerPrefix.length).trim().split(/\s+/).filter(Boolean);
  const params = {};
  for (const token of rawTokens) {
    const eqIdx = token.indexOf('=');
    if (eqIdx === -1) continue;
    const key = token.slice(0, eqIdx).trim().toLowerCase();
    const value = token.slice(eqIdx + 1).trim();
    if (key) params[key] = value;
  }

  const mode = (
    String(params.mode || '').toLowerCase() === '3d'
    || headerPrefix === '@acoustic3d'
    || ['sphere', 'cube', 'cylinder'].includes(String(params.primitive || '').toLowerCase())
  ) ? '3d' : '2d';
  const primitive = normalizeAcousticPrimitive(mode, params.primitive);
  const solver = String(params.solver || '').trim().toLowerCase();
  const defaults = acousticPointDefaults(mode);
  const freq = Math.max(20, Math.min(4000, Math.round(Number(params.freq || params.hz || 440) || 440)));
  const source = parsePointToken(params.source, defaults.source);
  const probe = parsePointToken(params.probe, defaults.probe);
  const obstacle = parsePointToken(params.obstacle || params.obs, defaults.obstacle);

  const remainder = lines
    .slice(firstIdx + 1)
    .join('\n')
    .trim();

  const formatPoint = (point) => point.map((value) => formatCommandCoord(value)).join(', ');
  const body = remainder || [
    `${mode === '3d' ? 'Volumetric' : 'Planar'} acoustic primitive study for a ${primitive} cavity.`,
    `Use a sine generator at ${freq} Hz.`,
    `Source at (${formatPoint(source)}), probe at (${formatPoint(probe)}), obstacle at (${formatPoint(obstacle)}).`
  ].join(' ');

  const modulusArgs = [
    `mode=${mode}`,
    `freq=${freq}`,
    `primitive=${primitive}`,
    `source_x=${formatCommandCoord(source[0])}`,
    `source_y=${formatCommandCoord(source[1])}`,
    `probe_x=${formatCommandCoord(probe[0])}`,
    `probe_y=${formatCommandCoord(probe[1])}`,
    `obs_x=${formatCommandCoord(obstacle[0])}`,
    `obs_y=${formatCommandCoord(obstacle[1])}`,
  ];
  if (solver) {
    modulusArgs.push(`solver=${solver}`);
  }
  if (mode === '3d') {
    modulusArgs.push(
      `source_z=${formatCommandCoord(source[2])}`,
      `probe_z=${formatCommandCoord(probe[2])}`,
      `obs_z=${formatCommandCoord(obstacle[2])}`,
    );
  }

  return {
    normalizedPrompt: [
      `[MODULUS ${modulusArgs.join(' ')}]`,
      body
    ].join('\n'),
    meta: {
      mode,
      primitive,
      solver,
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
  const mode = String(meta.mode || '2d').toLowerCase() === '3d' ? '3d' : '2d';
  const defaults = acousticPointDefaults(mode);
  const source = ensurePointSize(meta.source, mode === '3d' ? 3 : 2, defaults.source);
  const probe = ensurePointSize(meta.probe, mode === '3d' ? 3 : 2, defaults.probe);
  const obstacle = ensurePointSize(meta.obstacle, mode === '3d' ? 3 : 2, defaults.obstacle);
  return [
    '@acoustic',
    `mode=${mode}`,
    `primitive=${normalizeAcousticPrimitive(mode, meta.primitive)}`,
    `freq=${Math.round(Number(meta.freq) || 440)}`,
    `source=${source.map((value) => formatCommandCoord(value)).join(',')}`,
    `probe=${probe.map((value) => formatCommandCoord(value)).join(',')}`,
    `obstacle=${obstacle.map((value) => formatCommandCoord(value)).join(',')}`,
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
  const mode = String(patch.mode ?? meta.mode ?? '2d').toLowerCase() === '3d' ? '3d' : '2d';
  const defaults = acousticPointDefaults(mode);
  const size = mode === '3d' ? 3 : 2;

  const nextMeta = {
    mode,
    primitive: normalizeAcousticPrimitive(mode, patch.primitive ?? meta.primitive),
    freq: patch.freq ?? meta.freq,
    source: ensurePointSize(patch.source ?? meta.source, size, defaults.source),
    probe: ensurePointSize(patch.probe ?? meta.probe, size, defaults.probe),
    obstacle: ensurePointSize(patch.obstacle ?? meta.obstacle, size, defaults.obstacle),
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
  const defaults = acousticPointDefaults(meta.mode || '2d');
  const size = String(meta.mode || '2d').toLowerCase() === '3d' ? 3 : 2;
  const next = ensurePointSize(meta[channel], size, defaults[channel]);
  next[axis] = safe;
  updateAcousticCommandInEditor({ [channel]: next });
}

function setAcousticMode(mode) {
  updateAcousticCommandInEditor({ mode });
}

function setAcousticViewMode(mode) {
  acousticViewMode.value = mode;
}

function stopAcousticTemporalLoop() {
  if (acousticTemporalInterval) {
    clearInterval(acousticTemporalInterval);
    acousticTemporalInterval = null;
  }
  acousticTemporalLastTick = 0;
}

function startAcousticTemporalLoop() {
  stopAcousticTemporalLoop();
  if (typeof window === 'undefined' || !acousticTemporalPlaying.value) return;
  acousticTemporalLastTick = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
  acousticTemporalInterval = window.setInterval(() => {
    const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    const deltaSec = Math.max(0.05, (now - acousticTemporalLastTick) / 1000);
    acousticTemporalLastTick = now;
    acousticTemporalPhase.value = wrapPhase(acousticTemporalPhase.value + (deltaSec * acousticTemporalSpeed.value * 2.15));
  }, 85);
}

function toggleAcousticTemporalPlayback() {
  acousticTemporalPlaying.value = !acousticTemporalPlaying.value;
}

function insertStarterAcousticCommand() {
  const editor = editorRef.value?.aceEditor?.();
  if (!editor) return;
  const current = String(editor.getValue() || '');
  if (findAcousticHeaderLine(current.split('\n')) !== -1) return;

  const starter = [
    '@acoustic mode=2d primitive=circle freq=440 source=-0.55,0 probe=0.55,0 obstacle=0.15,0',
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

function pickPreferredWorkspaceView() {
  if (plotImage.value) return 'organogram';
  if (activeModulusData.value) return 'acoustic';
  if (stlUrl.value) return 'geometry';
  if (sketchImage.value) return 'sketch';
  if (summary.value || materialsText.value) return 'summary';
  return 'organogram';
}

function syncWorkspaceLayoutFromPreset() {
  workspaceLayout.value = workspacePreset.value === 'fullscreen' ? 'background' : 'split';
}

function syncWorkspaceAutoView() {
  workspaceAutoView.value = pickPreferredWorkspaceView();
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

let serverMetricsTimer = null;

async function fetchServerMetrics() {
  try {
    const response = await fetch(`${apiBase.value}/admin/server-metrics`, {
      headers: { Accept: 'application/json', ...(await authHeaders()) },
    });
    if (!response.ok) return;
    const payload = await response.json();
    if (payload?.ok && payload.server) serverMetrics.value = payload.server;
  } catch {
    // the SESSION panel keeps its last reading
  }
}

function stopServerMetricsPolling() {
  if (serverMetricsTimer) clearInterval(serverMetricsTimer);
  serverMetricsTimer = null;
}

// While SESSION is open an operator sees the server load even when nothing is rendering.
watch(
  [workspaceSidebarOpen, workspaceSidebarTab, isOperator, loading],
  ([open, tab, operator, busy]) => {
    stopServerMetricsPolling();
    if (!open || tab !== 'session' || !operator || busy) return;
    fetchServerMetrics();
    serverMetricsTimer = setInterval(fetchServerMetrics, 3000);
  },
  { immediate: true },
);
onUnmounted(stopServerMetricsPolling);

function toggleWorkspaceSidebar() {
  workspaceSidebarOpen.value = !workspaceSidebarOpen.value;
}

// Rail icons for HELP and SESSION: open the tab, or close the sidebar when that tab is already showing.
function openSidebarTab(tab) {
  if (workspaceSidebarOpen.value && workspaceSidebarTab.value === tab) {
    workspaceSidebarOpen.value = false;
    return;
  }
  workspaceSidebarTab.value = tab;
  workspaceSidebarOpen.value = true;
}

// Drag a pod by its header, but never from its buttons or the view selector.
function onPaneHeaderDown(pane, event) {
  if (event.target?.closest?.('button, select, a, input')) return;
  startPaneMove(pane.id, event);
}

function handleStyle(handle) {
  return handle.dir === 'row'
    ? { left: `${handle.x}%`, top: `${handle.y}%`, height: `${handle.length}%` }
    : { top: `${handle.y}%`, left: `${handle.x}%`, width: `${handle.length}%` };
}

function workspacePaneShows(viewKey) {
  return workspacePreset.value !== 'fullscreen' && activeWorkspacePanes.value.some((pane) => pane.view === viewKey);
}

// Make a registered view visible: leave fullscreen auto-routing for a split so the pane sits beside the rest.
function openWorkspaceView(viewKey) {
  if (workspacePreset.value === 'fullscreen') setWorkspacePreset('splitFour');
  showWorkspaceView(viewKey);
  syncWorkspaceLayoutFromPreset();
}

function toggleStageBrowserFullscreen() {
  if (typeof document === 'undefined') return;
  if (document.fullscreenElement) {
    document.exitFullscreen?.();
  } else {
    document.documentElement.requestFullscreen?.();
  }
}

// Commands that arrange the workspace and never reach the model.
const WORKSPACE_COMMANDS = {
  '/sot-a': { view: 'sota' },
  '/sota': { view: 'sota' },
  '/stage': { view: 'stage' },
  '/session': { sidebar: 'session' },
  '/help': { sidebar: 'help' },
};

function runWorkspaceCommand(text) {
  // A command alone in the evaluated text; blank lines and # comments around it are ignored.
  const meaningful = String(text || '').split('\n').map((line) => line.trim()).filter((line) => line && !line.startsWith('#'));
  if (meaningful.length !== 1) return false;
  const command = WORKSPACE_COMMANDS[meaningful[0].toLowerCase()];
  if (!command) return false;
  if (command.view) {
    openWorkspaceView(command.view);
  } else if (command.sidebar) {
    workspaceSidebarTab.value = command.sidebar;
    workspaceSidebarOpen.value = true;
  }
  return true;
}

function toggleRail() {
  railExpanded.value = !railExpanded.value;
  railFlyout.value = '';
  try {
    window.localStorage.setItem(RAIL_KEY, railExpanded.value ? '1' : '0');
  } catch (_) {
    // storage may be unavailable; the rail still works for this session
  }
}

function restoreRailState() {
  try {
    railExpanded.value = window.localStorage.getItem(RAIL_KEY) === '1';
  } catch (_) {
    railExpanded.value = false;
  }
}

function toggleRailFlyout(name) {
  railFlyout.value = railFlyout.value === name ? '' : name;
  if (railFlyout.value === 'model') fetchOllamaModels(true);
}

// On Swiss and other ISO layouts the backslash is not on the physical Backslash key.
const isBackslashKey = (e) => e.code === 'Backslash' || e.code === 'IntlBackslash' || e.key === '\\';

const handleKeyDown = (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'm') {
    e.preventDefault();
    uiVisible.value = !uiVisible.value;
  }
  if ((e.metaKey || e.ctrlKey) && e.shiftKey && isBackslashKey(e)) {
    e.preventDefault();
    toggleWorkspaceSidebar();
  } else if ((e.metaKey || e.ctrlKey) && !e.shiftKey && !e.altKey && isBackslashKey(e)) {
    e.preventDefault();
    toggleRail();
  }
  if (e.key === 'Escape' && railFlyout.value) {
    railFlyout.value = '';
  }
  if (e.key === 'Escape' && workspacePresetMenuOpen.value) {
    workspacePresetMenuOpen.value = false;
  }
};

const handleDocumentPointerDown = (event) => {
  if (railFlyout.value && !railRef.value?.$el?.contains?.(event.target)) railFlyout.value = '';
  if (!workspacePresetMenuOpen.value) return;
  if (workspacePresetMenuRef.value?.contains?.(event.target)) return;
  workspacePresetMenuOpen.value = false;
};

onMounted(() => {
  loadHelpOpenState();
  restoreRailState();
  loadResponseTimeHistory();
  fetchOllamaModels();
  checkDevice();
  restoreWorkspaceLayout();
  syncWorkspaceLayoutFromPreset();
  syncWorkspaceAutoView();
  startAcousticTemporalLoop();
  editorSyncTimer = window.setInterval(() => {
    if (syncEditorContent()) {
      window.clearInterval(editorSyncTimer);
      editorSyncTimer = null;
    }
  }, 120);
  window.addEventListener('resize', checkDevice);
  window.addEventListener('keydown', handleEscapeKey);
  window.addEventListener('keydown', handleKeyDown);
  window.addEventListener('pointerdown', handleDocumentPointerDown);
  window.addEventListener('mousemove', onDrag);
  window.addEventListener('mouseup', stopDrag);
  window.addEventListener('touchmove', onDragTouch, { passive: false });
  window.addEventListener('touchend', stopDrag);
});

onUnmounted(() => {
  clearInterval(progressInterval);
  stopReasoningPolling();
  stopAcousticTemporalLoop();
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
  window.removeEventListener('pointerdown', handleDocumentPointerDown);
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

watch(acousticTemporalPlaying, (playing) => {
  if (playing) {
    startAcousticTemporalLoop();
  } else {
    stopAcousticTemporalLoop();
  }
});

watch(
  () => [activeAcousticMode.value, acousticViewModeOptions.value.map((option) => option.key).join('|')],
  () => {
    const allowed = acousticViewModeOptions.value.map((option) => option.key);
    if (!allowed.includes(acousticViewMode.value)) {
      acousticViewMode.value = activeAcousticMode.value === '3d' ? 'volume' : 'xy';
    }
    if (activeAcousticMode.value !== '3d' && modulusRenderMode.value !== '2d' && !activeAcousticPlaneData.value) {
      modulusRenderMode.value = '2d';
    }
  },
  { immediate: true }
);

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
        mode: meta.mode,
        primitive: meta.primitive,
        freq: meta.freq,
        source_x: meta.source[0],
        source_y: meta.source[1],
        source_z: meta.source[2],
        probe_x: meta.probe[0],
        probe_y: meta.probe[1],
        probe_z: meta.probe[2],
        obs_x: meta.obstacle[0],
        obs_y: meta.obstacle[1],
        obs_z: meta.obstacle[2],
      });
    }, 36);
  },
  { immediate: true }
);

watch(
  [plotImage, stlUrl, sketchImage, activeModulusData, summary, materialsText],
  () => {
    syncWorkspaceAutoView();
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
  leftWidth.value = Math.min(58, Math.max(24, startLeft + deltaPct));
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
  leftWidth.value = Math.min(58, Math.max(24, startLeft + deltaPct));
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
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return;
    const clean = {};
    for (const [model, values] of Object.entries(parsed)) {
      if (!Array.isArray(values)) continue;
      const kept = values
        .filter((n) => Number.isFinite(n) && n > 0 && n < MAX_RESPONSE_MS)
        .slice(-MAX_RESPONSE_SAMPLES);
      if (kept.length) clean[model] = kept;
    }
    responseTimesByModel.value = clean;
  } catch {
    responseTimesByModel.value = {};
  }
};

const saveResponseTimeHistory = () => {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(RESPONSE_TIMES_KEY, JSON.stringify(responseTimesByModel.value));
  } catch {
    // ignore storage write errors
  }
};

const registerResponseTime = (model, ms) => {
  const key = String(model || '').trim();
  if (!key || !Number.isFinite(ms) || ms <= 0 || ms >= MAX_RESPONSE_MS) return;
  const next = [...(responseTimesByModel.value[key] || []), ms].slice(-MAX_RESPONSE_SAMPLES);
  responseTimesByModel.value = { ...responseTimesByModel.value, [key]: next };
  saveResponseTimeHistory();
};

// Progress based on real elapsed time and rolling average
let progressInterval;
let requestStartAt = 0;
let progressSucceeded = false;
let progressResultModel = '';
const startProgress = () => {
  clearInterval(progressInterval);
  progressSucceeded = false;
  progressResultModel = '';
  progressModel.value = currentModel.value;
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
    // Only successful generations teach the estimator; failures would drag the average down.
    if (progressSucceeded) registerResponseTime(progressResultModel || progressModel.value, elapsedMs.value);
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
  const sketchProcessId = createRequestId()
  startProcess({ id: sketchProcessId, kind: 'sketch', label: 'sketch (unsaved render)', cancellable: false })
  let sketchOutcome = 'completed'
  let sketchError = ''
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
    sketchOutcome = 'error'
    sketchError = e.message
  } finally {
    finishProcess(sketchProcessId, sketchOutcome, sketchError)
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

  generationSnapshot = null; // LOAD commits the item, closing the gallery keeps it
  applyGalleryItem(item);
}

// Show a gallery item in the workspace panes without touching the editor.
function applyGalleryItem(item) {
  targetBasename.value = item.basename || '';
  soundSamples.value = Array.isArray(item.sound_samples) ? item.sound_samples : [];
  itemPrompt.value = item.prompt || '';
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
  evaluatedAcousticSignature.value = acousticSignatureFromSimulation(item.modulus);
  syncWorkspaceAutoView();
}

// What the panes showed before the gallery started previewing items; restored when the gallery closes.
let generationSnapshot = null;

function captureGenerationState() {
  if (generationSnapshot) return;
  generationSnapshot = {
    targetBasename: targetBasename.value,
    soundSamples: soundSamples.value,
    itemPrompt: itemPrompt.value,
    summary: summary.value,
    materialsText: materialsText.value,
    organogramCode: organogramCode.value,
    geometryCode: geometryCode.value,
    responseModel: responseModel.value,
    sketchModel: sketchModel.value,
    responseElapsedMs: responseElapsedMs.value,
    stlUrl: stlUrl.value,
    plotImage: plotImage.value,
    sketchImage: sketchImage.value,
    modulusData: modulusData.value,
    evaluatedAcousticSignature: evaluatedAcousticSignature.value,
  };
}

function restoreGenerationState() {
  const saved = generationSnapshot;
  generationSnapshot = null;
  if (!saved) return;
  targetBasename.value = saved.targetBasename;
  soundSamples.value = saved.soundSamples;
  itemPrompt.value = saved.itemPrompt;
  summary.value = saved.summary;
  materialsText.value = saved.materialsText;
  organogramCode.value = saved.organogramCode;
  geometryCode.value = saved.geometryCode;
  responseModel.value = saved.responseModel;
  sketchModel.value = saved.sketchModel;
  responseElapsedMs.value = saved.responseElapsedMs;
  stlUrl.value = saved.stlUrl;
  plotImage.value = saved.plotImage;
  sketchImage.value = saved.sketchImage;
  modulusData.value = saved.modulusData;
  evaluatedAcousticSignature.value = saved.evaluatedAcousticSignature;
  syncWorkspaceAutoView();
}

// The gallery takes the left of the stage: the editor steps aside and a single pod becomes the
// four-pod item view. Closing the gallery puts the previous layout back unless it was rearranged meanwhile.
let galleryLayout = null;
watch(showGallery, (open) => {
  if (open) {
    galleryLayout = null;
    if (workspacePreset.value === 'fullscreen') {
      const before = snapshotWorkspaceLayout();
      setWorkspacePreset('splitFour');
      galleryLayout = { before, applied: workspaceTree.value };
    }
    return;
  }
  restoreGenerationState();
  if (galleryLayout && workspaceTree.value === galleryLayout.applied) restoreWorkspaceSnapshot(galleryLayout.before);
  galleryLayout = null;
  // The editor was hidden: let Ace measure itself again.
  nextTick(() => window.dispatchEvent(new Event('resize')));
});

// A pod tool finished: show the refreshed item if it is still the one on screen.
const stopItemUpdates = onItemUpdated((item) => {
  if (item.basename === targetBasename.value) applyGalleryItem(item);
});
onUnmounted(stopItemUpdates);
watch(toolError, (message) => { if (message) error.value = message; });

function formatAudioTime(seconds) {
  if (!Number.isFinite(seconds)) return '0:00';
  return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
}

// REGEN SKETCH: a saved item goes through the gallery endpoint, a result not yet saved through /generate/sketch.
function regenerateSketch() {
  if (itemTarget.value) return remakeItemSketch(itemTarget.value);
  return remakeSketch();
}

function previewGalleryItem(item) {
  if (!item) return;
  captureGenerationState();
  applyGalleryItem(item);
  if (workspacePreset.value === 'fullscreen') {
    setWorkspacePreset('splitFour');
    syncWorkspaceLayoutFromPreset();
  }
}

async function fetchOllamaModels(refresh = false) {
  try {
    const response = await fetch(`${apiBase.value}/ollama/models${refresh ? '?refresh=1' : ''}`, {
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

async function applyOllamaModel(nextModel) {
  const response = await fetch(`${apiBase.value}/ollama/model`, {
    method: 'POST',
    headers: {
      ...(await authHeaders()),
      'Content-Type': 'application/json',
      'Accept': 'application/json'
    },
    body: JSON.stringify({ model: nextModel })
  });
  const payload = await response.json().catch(() => ({}));
  if (response.status === 401 || response.status === 403) {
    throw new Error('Changing the active model is restricted to admins.');
  }
  if (!response.ok || !payload?.ok) {
    throw new Error(payload?.error || `Failed to switch model (${response.status})`);
  }
  currentModel.value = payload.model || nextModel;
}

async function selectOllamaModel(name) {
  if (modelSwitching.value || name === currentModel.value) {
    railFlyout.value = '';
    return;
  }
  modelSwitching.value = true;
  try {
    await applyOllamaModel(name);
    error.value = null;
    railFlyout.value = '';
  } catch (e) {
    error.value = e?.message || 'Model switch failed.';
  } finally {
    modelSwitching.value = false;
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
    await applyOllamaModel(ollamaModels.value[nextIdx]);
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

  if (runWorkspaceCommand(selectedText)) return;

  generationSnapshot = null; // a new generation replaces whatever the gallery was previewing
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
  const processId = generationRequestId.value;
  let processOutcome = 'completed';
  let processError = '';
  startProcess({
    id: processId,
    kind: 'render',
    label: isReversioning.value ? `reversion ${shortLabel(selectedText, 30)}` : shortLabel(selectedText),
    model: currentModel.value,
    estimateMs: averageResponseMs.value,
  });

    async function pollGenerateResult(requestId) {
      const limitMs = Math.max(getGenerateFetchTimeoutMs(), 45 * 60 * 1000);
      const started = Date.now();
      while (Date.now() - started < limitMs) {
        await new Promise((resolve) => setTimeout(resolve, 3000));
        try {
          const res = await fetch(`${apiBase.value}/generate/result/${requestId}`, {
            headers: { 'Accept': 'application/json' }
          });
          if (res.status !== 202) return res;
        } catch {
          // transient network error: keep polling
        }
      }
      throw new Error('Generation timed out while waiting for the result.');
    }

    async function callOnce() {
      const timeoutMs = getGenerateFetchTimeoutMs();
      const controller = timeoutMs > 0 ? new AbortController() : null;
      const timeoutId = timeoutMs > 0 ? setTimeout(() => controller?.abort(), timeoutMs) : null;
      let response;
      try {
        const secureHeaders = await authHeaders();
        response = await fetch(`${apiBase.value}/generate/start`, {
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

      // The backend runs the generation as a background job (202); poll for the result so the
      // proxy's 120 s read timeout (HTTP 524) never applies to a slow model.
      if (response.status === 202) {
        response = await pollGenerateResult(generationRequestId.value);
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
        if (response.status === 499) {
          const cancelled = new Error('Cancelled');
          cancelled.cancelled = true;
          throw cancelled;
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

  function deriveMaterialsText(materialsValue, summaryValue) {
    if (typeof materialsValue === 'string' && materialsValue.trim()) {
      return materialsValue.trim();
    }
    if (Array.isArray(materialsValue) && materialsValue.length) {
      return materialsValue.map(x => (typeof x === 'string' ? x : (x.name || ''))).filter(Boolean).join('\n');
    }
    if (summaryValue) {
      return extractMaterials(summaryValue).map(m => `- ${m.name}`).join('\n');
    }
    return '';
  }

  function applyGenerationState(nextState) {
    targetBasename.value = nextState.basename || '';
    soundSamples.value = nextState.soundSamples || [];
    itemPrompt.value = nextState.itemPrompt || '';
    plotImage.value = nextState.plotImage;
    sketchImage.value = nextState.sketchImage;
    summary.value = nextState.summary;
    organogramCode.value = nextState.organogramCode;
    geometryCode.value = nextState.geometryCode;
    stlUrl.value = nextState.stlUrl;
    modulusData.value = nextState.modulusData;
    evaluatedAcousticSignature.value = nextState.evaluatedAcousticSignature;
    materialsText.value = nextState.materialsText;
    responseModel.value = nextState.responseModel;
    responseElapsedMs.value = nextState.responseElapsedMs;
    sketchModel.value = nextState.sketchModel;
    syncWorkspaceAutoView();
  }

  try {
    const data = await callOnce();
    progressSucceeded = true;
    progressResultModel = String(data?.llm_model || '').trim();
    void refreshProfile(apiBase.value).catch(() => {});
    const requestEndedAt = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
    const clientElapsed = Math.max(0, requestEndedAt - requestStartedAt);
    const stagedElapsedMs = Number.isFinite(Number(data.elapsed_ms))
      ? Number(data.elapsed_ms)
      : Math.round(clientElapsed);
    const stagedResponseModel = (data.llm_model || currentModel.value || '').trim();

    // Handle special Modulus response (if only modulus was requested)
    if (data.type === 'modulus') {
      const stagedSummary = data.summary || summary.value || '';
      const stagedMaterials = deriveMaterialsText(data.materials, stagedSummary);
      const stagedModulus = data.modulus || null;
      applyGenerationState({
        plotImage: plotImage.value,
        sketchImage: sketchImage.value,
        summary: stagedSummary,
        organogramCode: organogramCode.value,
        geometryCode: geometryCode.value,
        stlUrl: stlUrl.value,
        modulusData: stagedModulus,
        evaluatedAcousticSignature: acousticSignatureFromSimulation(stagedModulus),
        materialsText: stagedMaterials,
        responseModel: stagedResponseModel,
        responseElapsedMs: stagedElapsedMs,
        sketchModel: sketchModel.value,
      });

      // Play sonic feedback
      if (stagedModulus?.results) {
        const freq = Number(stagedModulus.params?.freq) || 440;
        const amp = Number(stagedModulus.results.mic_response) || 0.5;
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
    let stagedPlotImage = null;
    if (data.image) {
      stagedPlotImage = `data:image/png;base64,${data.image}`;
    } else if (imageUrl) {
      stagedPlotImage = resolveAssetUrl(imageUrl);
    } else {
      throw new Error('Backend did not return an organogram image. Generation aborted.');
    }
    let stagedSketchImage = null;
    if (data.sketch) {
      stagedSketchImage = `data:image/png;base64,${data.sketch}`;
    } else if (sketchUrl) {
      stagedSketchImage = resolveAssetUrl(sketchUrl);
    }

    const stagedSummary = data.summary || '';
    const stagedModulus = data.modulus || null;
    const stagedOrganogramCode = (data.plot_code || data.content || '').trim();
    const stagedGeometryCode = (data.stl_code || '').trim();
    const stagedStlUrl = data.gallery?.stl_url ? resolveAssetUrl(data.gallery.stl_url) : null;
    const stagedSketchModel = (data.sketch_model || data.gallery?.sketch_model || '').trim();
    const stagedMaterialsText = deriveMaterialsText(data.materials, stagedSummary);

    applyGenerationState({
      basename: data.gallery?.basename || '',
      soundSamples: data.gallery?.sound_samples || [],
      itemPrompt: data.gallery?.prompt || '',
      plotImage: stagedPlotImage,
      sketchImage: stagedSketchImage,
      summary: stagedSummary,
      organogramCode: stagedOrganogramCode,
      geometryCode: stagedGeometryCode,
      stlUrl: stagedStlUrl,
      modulusData: stagedModulus,
      evaluatedAcousticSignature: acousticSignatureFromSimulation(stagedModulus),
      materialsText: stagedMaterialsText,
      responseModel: stagedResponseModel,
      responseElapsedMs: stagedElapsedMs,
      sketchModel: stagedSketchModel,
    });

    // Play sonic feedback for standard generation if modulus data exists
    if (stagedModulus?.results) {
      const freq = Number(stagedModulus.params?.freq) || 440;
      const amp = Number(stagedModulus.results.mic_response) || 0.5;
      playResponse(freq, amp);
    }

    // Append structured code sections in editor (left panel)
    const codeBundle = formatGeneratedCode(stagedOrganogramCode, stagedGeometryCode);
    if (showCode.value && codeBundle && editorRef.value) {
      editorRef.value.addToEditor(codeBundle, 'code');
    }
  } catch (err) {
    if (err?.cancelled) {
      processOutcome = 'cancelled';
    } else {
      console.error(err);
      error.value = err.message;
      processOutcome = 'error';
      processError = err.message;
    }
  } finally {
    finishProcess(processId, processOutcome, processError);
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
  border-right: none;
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
    linear-gradient(90deg, rgba(0, 0, 0, 0.94) 0%, rgba(0, 0, 0, 0.7) 45%, rgba(0, 0, 0, 0) 100%);
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
  flex: 1;
  height: auto;
  background: #000;
  display: flex;
  flex-direction: column;
  position: relative;
  overflow: hidden;
  min-height: 0;
}

.results-panel--background {
  background: transparent;
}

.results-panel--idle {
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.018), rgba(255, 255, 255, 0)),
    #000;
}

.workspace-preset-menu {
  position: relative;
}

.workspace-preset-trigger {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding-inline: 8px 6px;
}

.workspace-preset-trigger__label {
  font-size: 10px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: rgba(255, 255, 255, 0.76);
  white-space: nowrap;
}

.workspace-preset-trigger__chevron {
  width: 14px;
  height: 14px;
  color: rgba(255, 255, 255, 0.48);
  flex: 0 0 auto;
}

.workspace-preset-dropdown {
  position: absolute;
  top: calc(100% + 6px);
  left: 0;
  min-width: 228px;
  display: grid;
  gap: 1px;
  padding: 1px;
  background: rgba(119, 139, 169, 0.22);
  border: 1px solid rgba(119, 139, 169, 0.28);
  z-index: 140;
}

.workspace-preset-option {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  min-height: 44px;
  padding: 7px 10px;
  background: rgba(2, 6, 13, 0.96);
  border: none;
  color: rgba(255, 255, 255, 0.62);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.06em;
  cursor: pointer;
  text-align: left;
}

.workspace-preset-option:hover,
.workspace-preset-option.active {
  color: rgba(255, 255, 255, 0.94);
}

.workspace-preset-option__icon,
.workspace-preset-icon {
  width: 16px;
  height: 16px;
  flex: 0 0 auto;
}

.workspace-preset-option__copy {
  display: grid;
  gap: 3px;
}

.workspace-preset-option__copy strong {
  font-size: 11px;
  font-weight: 400;
  color: rgba(255, 255, 255, 0.9);
}

.workspace-preset-option__copy small {
  font-size: 10px;
  line-height: 1.35;
  letter-spacing: 0.03em;
  color: rgba(188, 202, 220, 0.58);
}

.workspace-stage {
  position: relative;
  flex: 1;
  min-height: 0;
  min-width: 0;
  background: transparent;
  overflow: hidden;
}

.workspace-stage--dragging {
  user-select: none;
  cursor: grabbing;
}

.results-panel--background .workspace-stage {
  background: transparent;
}

.workspace-stage--fullscreen .panel-placeholder {
  display: none;
}

.pane-handle {
  position: absolute;
  z-index: 6;
  touch-action: none;
}

.pane-handle--x {
  width: 11px;
  margin-left: -5px;
  cursor: col-resize;
}

.pane-handle--y {
  height: 11px;
  margin-top: -5px;
  cursor: row-resize;
}


.pane-handle::after {
  content: '';
  position: absolute;
  inset: 0;
  margin: auto;
  background: rgba(255, 255, 255, 0.0);
  transition: background-color 0.15s ease;
}

.pane-handle--x::after { width: 2px; }
.pane-handle--y::after { height: 2px; }
.pane-handle:hover::after,
.pane-handle:active::after { background: rgba(120, 190, 255, 0.7); }

.workspace-pane {
  position: absolute;
  left: var(--x);
  top: var(--y);
  width: var(--w);
  height: var(--h);
  box-sizing: border-box;
  min-width: 0;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: #000;
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.12);
}


.workspace-stage--fullscreen .workspace-pane {
  box-shadow: none;
}

.workspace-pane--moving {
  opacity: 0.5;
}

.workspace-pane__header--movable {
  cursor: grab;
  touch-action: none;
}

.pane-drop {
  position: absolute;
  z-index: 8;
  pointer-events: none;
  box-sizing: border-box;
  border: 1px solid rgba(120, 190, 255, 0.9);
  background: rgba(120, 190, 255, 0.14);
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.16em;
  color: #fff;
  transition: left 0.08s ease, top 0.08s ease, width 0.08s ease, height 0.08s ease;
}

.results-panel--background .workspace-pane {
  background: transparent;
}


.workspace-pane__header {
  min-height: 30px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 6px;
  padding: 2px 6px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.12);
  background: rgba(0, 0, 0, 0.78);
  flex: 0 0 auto;
}

.results-panel--background .workspace-pane__header {
  background: linear-gradient(180deg, rgba(0, 0, 0, 0.9), rgba(0, 0, 0, 0.32));
}

.workspace-pane__title {
  font-size: 10px;
  letter-spacing: 0.16em;
  color: rgba(255, 255, 255, 0.82);
  text-transform: uppercase;
  white-space: nowrap;
}

.workspace-pane__controls {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 2px;
  min-width: 0;
  flex-wrap: wrap;
  justify-content: flex-end;
}

.workspace-pane__select {
  min-width: 124px;
  height: 24px;
  background: transparent;
  border: 1px solid rgba(255, 255, 255, 0.14);
  color: rgba(255, 255, 255, 0.72);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.06em;
  padding: 0 8px;
}

.pane-action {
  height: 24px;
  min-width: 24px;
  padding: 0 7px;
  background: transparent;
  border: 1px solid rgba(255, 255, 255, 0.14);
  color: rgba(255, 255, 255, 0.72);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.06em;
  cursor: pointer;
}

.pane-action:hover {
  color: #fff;
  border-color: rgba(255, 255, 255, 0.4);
}

.pane-action:disabled {
  opacity: 0.4;
  cursor: default;
}

.sample-btn.playing {
  color: #000;
  background: rgba(255, 255, 255, 0.85);
  border-color: #fff;
}

.pane-timer {
  font-size: 10px;
  letter-spacing: 0.06em;
  color: rgba(255, 255, 255, 0.5);
}

.sound-modal-overlay {
  position: fixed;
  inset: 0;
  z-index: 4000;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(0, 0, 0, 0.9);
}

.sound-modal {
  width: min(400px, 92vw);
  padding: 20px;
  background: #000;
  border: 1px solid #222;
  font-family: 'IBM Plex Mono', monospace;
}

.sound-modal__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}

.sound-modal__header h4 {
  margin: 0;
  font-size: 10px;
  letter-spacing: 0.2em;
  color: #fff;
  font-weight: 400;
}

.sound-modal__header button {
  background: transparent;
  border: 0;
  color: #666;
  font-size: 18px;
  cursor: pointer;
}

.sound-modal__desc {
  margin: 0 0 10px;
  font-size: 10px;
  color: #666;
}

.sound-modal__textarea {
  width: 100%;
  height: 80px;
  box-sizing: border-box;
  margin-bottom: 14px;
  padding: 8px;
  background: #000;
  border: 1px solid #222;
  color: #eee;
  font: inherit;
  font-size: 11px;
  resize: none;
  outline: none;
}

.sound-modal__actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}

.cancel-btn {
  background: transparent;
  border: 1px solid #222;
  color: #777;
  font: inherit;
  font-size: 10px;
  padding: 4px 10px;
  cursor: pointer;
}

.confirm-btn {
  background: #fff;
  color: #000;
  border: 1px solid #fff;
  font: inherit;
  font-size: 10px;
  font-weight: 700;
  padding: 4px 14px;
  cursor: pointer;
}

.confirm-btn:disabled {
  opacity: 0.5;
}

.sota-pane {
  height: 100%;
  overflow: auto;
  padding: 14px 16px 24px;
  overscroll-behavior: contain;
}

.workspace-pane__select option {
  background: #000;
  color: #fff;
}

.workspace-pane__body {
  flex: 1;
  min-height: 0;
  min-width: 0;
  overflow: auto;
  position: relative;
}

.workspace-stage--fullscreen .workspace-pane__body {
  padding-bottom: 56px;
  box-sizing: border-box;
}

.sound-pane {
  min-height: 100%;
  padding: 14px 16px 72px;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.sound-pane__facts {
  margin-bottom: 2px;
}

.sound-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.sound-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 0;
  border-bottom: 1px solid #111;
}

.sound-row.playing .sound-row__prompt {
  color: #fff;
}

.sound-row__prompt {
  flex: 1;
  min-width: 0;
  color: rgba(255, 255, 255, 0.5);
  font-size: 11px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.sound-pane__generate {
  align-self: flex-start;
}

.workspace-pane__body .plot-image {
  width: 100%;
  height: 100%;
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  display: block;
}

.results-panel--background .workspace-pane__body .plot-image {
  padding: 18px 18px 72px;
  box-sizing: border-box;
}

.results-panel--background .stl-viewer-container,
.results-panel--background .modulus-results {
  padding-bottom: 56px;
  box-sizing: border-box;
}

.results-split {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  background: transparent;
  display: grid;
  grid-template-columns: minmax(320px, 1.02fr) minmax(300px, 0.98fr);
  grid-template-rows: minmax(250px, 0.94fr) minmax(320px, 1.06fr);
  grid-template-areas:
    'organogram text'
    'visualizer visualizer';
  gap: 0;
  padding: 0 0 88px;
  box-sizing: border-box;
  align-content: stretch;
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
  grid-area: organogram;
  justify-content: center;
  border-right: 1px solid rgba(255, 255, 255, 0.12);
}

.panel-visualizer {
  grid-area: visualizer;
  min-height: 380px;
  border-top: 1px solid rgba(255, 255, 255, 0.12);
}

.tab-header {
  margin-bottom: 12px;
}

.tabs {
  display: flex;
  gap: 16px;
}

.acoustic-mode-row {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 6px;
  margin-top: 18px;
}

.modulus-mode-toggle {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.acoustic-view-toggle {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  margin-left: 8px;
}

.acoustic-view-toggle--sidebar {
  margin: 12px 0 0;
}

.acoustic-view-btn {
  border: 1px solid rgba(132, 160, 188, 0.18);
  background: rgba(7, 12, 30, 0.28);
  color: rgba(204, 214, 225, 0.56);
  padding: 2px 7px;
  font-size: 10px;
  letter-spacing: 0.12em;
  cursor: pointer;
}

.acoustic-view-btn.active,
.acoustic-view-btn:hover {
  border-color: rgba(95, 195, 255, 0.68);
  color: rgba(238, 244, 250, 0.92);
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
  grid-area: text;
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

.acoustic-temporal-bar {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) minmax(140px, 220px);
  gap: 10px;
  align-items: center;
}

.acoustic-temporal-bar--sidebar {
  margin-top: 14px;
  grid-template-columns: 1fr;
}

.acoustic-temporal-group {
  display: inline-flex;
  align-items: center;
  gap: 8px;
}

.acoustic-temporal-readout {
  color: rgba(213, 223, 234, 0.72);
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.acoustic-temporal-slider {
  display: grid;
  gap: 4px;
  color: rgba(185, 198, 214, 0.7);
  font-size: 10px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.acoustic-temporal-slider--speed {
  min-width: 150px;
}

.modulus-surface-shell {
  min-height: 320px;
  border: 1px solid rgba(132, 160, 188, 0.18);
  background:
    repeating-linear-gradient(0deg, rgba(255, 255, 255, 0.035), rgba(255, 255, 255, 0.035) 1px, transparent 1px, transparent 18px),
    repeating-linear-gradient(90deg, rgba(255, 255, 255, 0.028), rgba(255, 255, 255, 0.028) 1px, transparent 1px, transparent 18px),
    radial-gradient(circle at top left, rgba(88, 117, 154, 0.24), rgba(6, 11, 19, 0.08) 60%),
    linear-gradient(180deg, rgba(6, 10, 18, 0.96), rgba(10, 15, 24, 0.58));
}

.modulus-volume-shell {
  min-height: 360px;
  border: 1px solid rgba(132, 160, 188, 0.18);
  background:
    repeating-linear-gradient(0deg, rgba(255, 255, 255, 0.032), rgba(255, 255, 255, 0.032) 1px, transparent 1px, transparent 20px),
    repeating-linear-gradient(90deg, rgba(255, 255, 255, 0.024), rgba(255, 255, 255, 0.024) 1px, transparent 1px, transparent 20px),
    radial-gradient(circle at 22% 18%, rgba(95, 126, 165, 0.2), rgba(8, 12, 19, 0.08) 48%),
    linear-gradient(180deg, rgba(6, 10, 18, 0.94), rgba(10, 15, 23, 0.52));
}

.modulus-volume-shell--iso {
  background:
    repeating-linear-gradient(0deg, rgba(255, 255, 255, 0.026), rgba(255, 255, 255, 0.026) 1px, transparent 1px, transparent 16px),
    repeating-linear-gradient(90deg, rgba(255, 255, 255, 0.02), rgba(255, 255, 255, 0.02) 1px, transparent 1px, transparent 16px),
    radial-gradient(circle at 70% 12%, rgba(84, 118, 164, 0.2), rgba(8, 12, 19, 0.08) 44%),
    linear-gradient(180deg, rgba(7, 11, 20, 0.95), rgba(11, 16, 25, 0.56));
}

.modulus-readout {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  font-size: 11px;
  color: rgba(204, 214, 225, 0.62);
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.modulus-readout span {
  border-top: 1px solid rgba(132, 160, 188, 0.18);
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

.background-stage__volume {
  width: min(92vw, 1040px);
  height: min(72vh, 780px);
}

/* Tools rail (left, see NavRail): black, hairline wireframe */
.app-container.has-rail {
  box-sizing: border-box;
  padding-left: 44px !important;
}

/* The gallery dock sits between the tools rail and the editor */
.app-container.gallery-open {
  padding-left: calc(44px + 300px) !important;
}
.app-container.gallery-open .right-column--background {
  left: calc(44px + 300px);
  width: auto !important;
}

/* With the gallery open the editor steps aside and the pods take the rest of the width. */
.app-container.gallery-open .left-column,
.app-container.gallery-open .divider {
  display: none;
}

.app-container.gallery-open:not(.workspace-background-mode) .right-column {
  flex: 1 1 auto;
  width: auto !important;
}
@media (max-width: 768px) {
  .app-container.gallery-open {
    padding-left: 44px !important;
  }
  .app-container.gallery-open .right-column--background {
    left: 0;
  }
}

.rail .workspace-preset-dropdown {
  top: 0;
  left: calc(100% + 6px);
}

.rail-flyout-wrap {
  position: relative;
}

.rail-flyout {
  position: absolute;
  left: calc(100% + 6px);
  bottom: 0;
  min-width: 220px;
  max-width: 320px;
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 6px;
  background: #000;
  border: 1px solid rgba(255, 255, 255, 0.22);
  z-index: 140;
}

.rail-flyout__title {
  margin-bottom: 4px;
  padding: 2px 6px 6px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.12);
  font-size: 10px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: rgba(255, 255, 255, 0.5);
  overflow-wrap: anywhere;
}

.rail-flyout__option {
  padding: 5px 6px;
  background: transparent;
  border: 0;
  color: rgba(255, 255, 255, 0.75);
  font: inherit;
  font-size: 11px;
  text-align: left;
  overflow-wrap: anywhere;
  cursor: pointer;
}

.rail-flyout__option:hover {
  background: rgba(255, 255, 255, 0.08);
  color: #fff;
}

.rail-flyout__option.active {
  color: #fff;
  box-shadow: inset 2px 0 0 #fff;
}

.rail-flyout__option:disabled {
  opacity: 0.4;
  cursor: default;
}

.rail-flyout__empty {
  margin: 0;
  padding: 4px 6px;
  font-size: 11px;
  color: rgba(255, 255, 255, 0.5);
}

/* keep pane header controls clear of the ? button */
.right-column .workspace-pane__header {
  padding-right: 44px;
}

.workspace-sidebar {
  position: fixed;
  top: 0;
  right: 0;
  bottom: 44px;
  width: 360px;
  display: flex;
  transform: translateX(100%);
  visibility: hidden;
  pointer-events: none;
  transition: transform 0.24s ease, visibility 0s linear 0.24s;
  z-index: 90;
  border-left: 1px solid rgba(255, 255, 255, 0.14);
  background: rgba(0, 0, 0, 0.18);
  backdrop-filter: blur(14px);
}

.workspace-sidebar--open {
  transform: translateX(0);
  visibility: visible;
  pointer-events: auto;
  transition: transform 0.24s ease;
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
  overflow: hidden;
  background: rgba(5, 8, 13, 0.9);
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
  position: sticky;
  top: 0;
  z-index: 2;
  background: rgba(5, 8, 13, 0.94);
}

.workspace-sidebar__close {
  display: none; /* the ? button is the toggle */
}

.workspace-sidebar__body {
  flex: 1;
  overflow: auto;
  padding: 14px 16px 68px;
  overscroll-behavior: contain;
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
  border: 1px solid rgba(132, 160, 188, 0.18);
  background:
    repeating-linear-gradient(0deg, rgba(255, 255, 255, 0.032), rgba(255, 255, 255, 0.032) 1px, transparent 1px, transparent 18px),
    repeating-linear-gradient(90deg, rgba(255, 255, 255, 0.024), rgba(255, 255, 255, 0.024) 1px, transparent 1px, transparent 18px),
    radial-gradient(circle at top left, rgba(90, 120, 160, 0.22), rgba(4, 8, 18, 0.06) 60%);
}

.acoustic-preview-volume {
  min-height: 300px;
}

.command-domain + .command-domain,
.acoustic-summary + .acoustic-summary,
.session-facts + .session-facts {
  margin-top: 18px;
}

.command-domain__header {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
  padding: 0;
  background: transparent;
  border: 0;
  color: inherit;
  font: inherit;
  cursor: pointer;
  text-align: left;
}

.command-domain__count {
  color: rgba(255, 255, 255, 0.35);
  font-size: 10px;
}

.command-domain__chevron {
  margin-left: auto;
  color: rgba(255, 255, 255, 0.55);
  font-size: 13px;
  line-height: 1;
}

.command-domain__header:hover .command-domain__chevron {
  color: #fff;
}

.command-domain__line {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 8px;
}

.command-domain__line code {
  flex: 1;
  min-width: 0;
  overflow-wrap: anywhere;
}

.command-domain__line .pane-action {
  flex: none;
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

.session-section {
  margin-top: 18px;
}

.session-section h4 {
  margin: 0 0 8px;
  font-size: 10px;
  letter-spacing: 0.14em;
  color: rgba(255, 255, 255, 0.82);
  font-weight: 400;
}

.session-empty {
  margin: 0;
  color: rgba(255, 255, 255, 0.45);
  font-size: 11px;
}

.stage-list {
  display: grid;
  gap: 6px;
}

.stage-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto 44px;
  align-items: center;
  gap: 10px;
  font-size: 11px;
  color: rgba(255, 255, 255, 0.6);
}

.stage-row__name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.stage-row__time {
  color: rgba(255, 255, 255, 0.86);
  text-align: right;
}

.stage-row--running .stage-row__name,
.stage-row--running .stage-row__time {
  color: #fff;
}

.stage-row__cpu,
.cpu-meter {
  display: block;
  height: 4px;
  background: rgba(255, 255, 255, 0.12);
}

.cpu-meter {
  margin-bottom: 10px;
  height: 6px;
}

.stage-row__cpu i,
.cpu-meter i {
  display: block;
  height: 100%;
  background: rgba(255, 255, 255, 0.78);
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
  .pane-handle { display: none; }

  .app-container {
    font-family: 'IBM Plex Mono', monospace;
    flex-direction: column;
  }
  
  .left-column, .right-column {
    width: 100% !important;
  }
  .left-column { height: 50vh !important; }
  .right-column { height: 50vh !important; }
  .workspace-stage {
    display: flex;
    flex-direction: column;
    overflow-y: auto;
  }
  .workspace-pane {
    position: relative;
    left: auto;
    top: auto;
    width: auto;
    height: auto;
    flex: 1 0 220px;
  }
  .workspace-pane__header {
    padding: 0 10px;
  }
  .workspace-pane__controls {
    gap: 6px;
    flex-wrap: wrap;
  }
  .workspace-pane__select {
    min-width: 108px;
  }
  .acoustic-temporal-bar {
    grid-template-columns: 1fr;
  }
  .acoustic-view-toggle {
    margin-left: 0;
    flex-wrap: wrap;
  }
  .results-split {
    grid-template-columns: minmax(0, 1fr);
    grid-template-rows: minmax(180px, 0.9fr) minmax(180px, 0.9fr) minmax(220px, 1fr);
    grid-template-areas:
      'organogram'
      'text'
      'visualizer';
    padding-bottom: 110px;
  }
  .panel-organogram,
  .panel-visualizer {
    border-right: none;
  }
  .panel-text,
  .panel-visualizer {
    border-top: 1px solid rgba(255, 255, 255, 0.12);
  }
  .workspace-sidebar {
    top: 0;
    width: calc(100vw - 44px);
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
