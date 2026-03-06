---
type: concept
tags:
  - dss/mechanics
  - dss/acoustics
person: Daniel Bernoulli
year: 1735
summary: A cantilever oscillation refers to the periodic motion of a structural element fixed at one end and free at the other, governed by Euler-Bernoulli beam theory. The system exhibits characteristic natural frequencies and mode shapes determined by material stiffness, geometry, and boundary conditions. This phenomenon is fundamental in structural dynamics, MEMS devices, and musical instrument design.
connect:
  - "[[Euler-Bernoulli beam theory]]"
  - "[[Modal analysis]]"
  - "[[Nonlinear dynamics]]"
created: 18-02-2025
modified: 
---

# Dynamics of Fixed-Free Elastic Systems

The vibrational behavior of cantilevers exemplifies the interplay between elasticity and inertia in constrained mechanical systems. According to <mark class="hltr-yellow">Euler-Bernoulli beam theory (Timoshenko 1955:112)</mark>, the governing equation for small-amplitude oscillations is:

$$EI\frac{\partial^4 w}{\partial x^4} + \rho A\frac{\partial^2 w}{\partial t^2} = 0$$

where $w(x,t)$ represents transverse displacement, $E$ is Young's modulus, and $I$ the area moment of inertia. The <mark class="hltr-green">natural frequencies emerge from boundary conditions</mark> - fixed (zero displacement/slope) at one end and free (zero moment/shear) at the other:

$$\omega_n = \frac{\lambda_n^2}{L^2}\sqrt{\frac{EI}{\rho A}}$$

with $\lambda_n$ being roots of the characteristic equation $1 + \cos\lambda\cosh\lambda = 0$. <mark class="hltr-blue">Mode shapes</mark> take the form:

$$W_n(x) = \left(\cosh\frac{\lambda_n x}{L} - \cos\frac{\lambda_n x}{L}\right) - \sigma_n\left(\sinh\frac{\lambda_n x}{L} - \sin\frac{\lambda_n x}{L}\right)$$

where $\sigma_n$ are modal constants. This framework was extended to <mark class="hltr-red">nonlinear regimes</mark> by (Nayfeh 1979:203), revealing complex phenomena like internal resonances.

# Microscale Metrology in Atomic Force Microscopy

Modern AFM cantilevers demonstrate this principle at microscopic scales, where thermal noise induces Brownian motion with measurable spectral peaks. A silicon nitride probe ($E=170$ GPa, $\rho=3100$ kg/m³) with dimensions $L=200$ μm, $w=50$ μm, $h=1$ μm exhibits fundamental frequency:

$$f_1 = \frac{1.875^2}{2\pi}\sqrt{\frac{170\times10^9 \times (50\times10^{-6})(1\times10^{-6})^3/12}{3100 \times (50\times10^{-6})(1\times10^{-6})}} \approx 17.3\text{kHz}$$

<mark class="hltr-orange">Laser Doppler vibrometry</mark> confirms these predictions within 2% error (Garcia 2018). The system's quality factor $Q=\omega_1/\Delta\omega$, often exceeding 100 in vacuum, enables nanoscale topography mapping.

```run-python
import numpy as np
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Slider
from bokeh.layouts import column
from bokeh.palettes import Viridis256

# Parameters
L = 1.0  # normalized length
x = np.linspace(0, L, 200)
modes = [1.875, 4.694, 7.855]  # First three λn values

def mode_shape(x, λn):
    σn = (np.cosh(λn) + np.cos(λn))/(np.sinh(λn) + np.sin(λn))
    return (np.cosh(λn*x) - np.cos(λn*x)) - σn*(np.sinh(λn*x) - np.sin(λn*x))

source = ColumnDataSource(data={'x': x})
palette = Viridis256[::85]

p = figure(width=600, height=400, toolbar_location=None, 
           background_fill_alpha=0, border_fill_alpha=0)
p.xaxis.axis_label = "Normalized Position (x/L)"
p.yaxis.axis_label = "Modal Amplitude"

for i, λn in enumerate(modes):
    p.line(x, mode_shape(x, λn), line_width=3, color=palette[i], 
          legend_label=f"Mode {i+1} (λ={λn:.3f})")

p.legend.location = "top_right"
show(column(p))
```

### Research Questions

1. **Fundamental**: How do non-classical boundary conditions (e.g., semi-rigid attachments or distributed elasticity) alter the modal characteristics of cantilever systems compared to ideal Euler-Bernoulli predictions?

2. **Applied**: Can coupled cantilever arrays exhibit topological insulator properties for vibration control in metamaterials, analogous to electronic systems in condensed matter physics?

3. **Speculative**: Might the stochastic resonance phenomena observed in nanoscale cantilevers inform new models of biological sensory systems at quantum-classical boundaries?

### In Music

```lily
\version "2.24.0"
\paper { tagline = ##f paper-height=#(* 9 cm) paper-width=#(* 20 cm) system-count=#2 }
\score {
    \new Staff {
        \tempo "Lento misterioso" 4 = 52
        \clef bass
        gis4--\pp^\markup{ \small "I mode" } ( aih8[\staccato b,\accent )] 
        e'16\mf\< ( disih' cis' b gis fih e dis\!
        cis8\f\> ) r r4 r16 aih\p ( gis fih e dis cis b,
        gis2~\ppp gis8 r r4
        
        \clef treble
        << 
            { cis''16\mf ( dis'' e'' fih'' gis'' aih'' b'' cis''' } \\
            { gis4..\accent~ gis8 fih e dis }
        >>
        cis'8--\> ( b aih gis fih e dis cis\! )
        b,\ppp [ r16 aih'] gis'8.. [ fih32] e2~
        e4 r r2
    }
}
```

The composition translates modal vibration patterns into musical structure through:
1. **Pitch organization**: Fundamental frequency (G#) with harmonic partials following mode shape ratios (λ₂/λ₁ ≈ 2.5 → augmented fourth)
2. **Articulation**: Staccato attacks represent nodal points while slurs emulate continuous deformation
3. **Dynamics**: Exponential decay envelope mirrors damping behavior ($e^{-\zeta\omega t}$)
4. **Texture**: Polyphonic splitting in measure 5 illustrates higher-mode participation

```bibtex
@book{timoshenko1955vibration,
  title={Vibration Problems in Engineering},
  author={Timoshenko, S.P.},
  year={1955},
  publisher={Van Nostrand}
}

@article{nayfeh1979nonlinear,
  title={Nonlinear oscillations},
  author={Nayfeh, Ali Hasan and Mook, Dean T},
  journal={Wiley-Interscience},
  year={1979}
}

@article{garcia2018thermal,
  title={Thermal noise limits on micromechanical measurements},
  author={Garcia, Ricardo and Proksch, Roger},
  journal={Beilstein Journal of Nanotechnology},
  volume={9},
  pages={852--863},
  year={2018}
}
```
example: ruler on a table, oscillates 

```run-python
import numpy as np
import base64
import scipy.io.wavfile as wav
import io

# Parámetros iniciales
length_cm = 20
young_GPa = 2.0
damping = 0.02

# Conversión de unidades
L = length_cm / 100  # metros
E = young_GPa * 1e9  # pascales
b = 0.025  # ancho de la regla
h = 0.002  # espesor
I = (b * h**3) / 12
rho = 1200  # densidad (kg/m³)
A = b * h
gamma = damping
A0 = 1.0
sr = 44100

# Frecuencia natural fundamental
f1 = (1.875**2) / (2 * np.pi * L**2) * np.sqrt(E * I / (rho * A))
duration = min(4.0, max(1.5, 2.0 / gamma))  # duración simulada

# Generación de onda
t = np.linspace(0, duration, int(sr * duration), endpoint=False)
envelope = A0 * np.exp(-gamma * t)
wave = envelope * np.sin(2 * np.pi * f1 * t)

# Normalizar y convertir a audio
wave *= 32767 / np.max(np.abs(wave))
wave = wave.astype(np.int16)
buffer = io.BytesIO()
wav.write(buffer, sr, wave)
b64_wav = base64.b64encode(buffer.getvalue()).decode('utf-8')

# HTML con sliders y audio dinámico
html = f"""
<h3>📏 Cantilever Ruler Simulation</h3>
<div>
  <label>Length (cm): <input id='lenSlider' type='range' min='5' max='40' value='{length_cm}' oninput='updateSound()'> <span id='lenVal'>{length_cm}</span></label><br>
  <label>Young Modulus (GPa): <input id='youngSlider' type='range' min='0.5' max='5' step='0.1' value='{young_GPa}' oninput='updateSound()'> <span id='youngVal'>{young_GPa}</span></label><br>
  <label>Damping Coeff: <input id='dampSlider' type='range' min='0.005' max='0.1' step='0.005' value='{damping}' oninput='updateSound()'> <span id='dampVal'>{damping}</span></label>
</div>
<br>
<audio id="cantileverSound" controls>
  <source id="soundSource" src="data:audio/wav;base64,{b64_wav}" type="audio/wav">
  Your browser does not support the audio element.
</audio>

<script>
function updateSound() {{
  const len = document.getElementById('lenSlider').value;
  const young = document.getElementById('youngSlider').value;
  const damp = document.getElementById('dampSlider').value;
  document.getElementById('lenVal').textContent = len;
  document.getElementById('youngVal').textContent = young;
  document.getElementById('dampVal').textContent = damp;
  location.reload();  // Reload required to regenerate sound via Pyodide
}}
</script>
"""

@html(html)
```
