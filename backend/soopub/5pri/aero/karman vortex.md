---
type: concept  
tags:  
  - dss/fluid_dynamics  
person: Theodore von Kármán  
year: 1911  
summary: A Kármán vortex street is a repeating pattern of swirling vortices caused by the unsteady separation of fluid flow around bluff bodies. These alternating vortices form downstream of an obstacle when the Reynolds number exceeds a critical value, creating oscillatory forces that can induce structural vibrations. The phenomenon bridges fluid dynamics, acoustics, and nonlinear dynamics, with applications in meteorology, engineering, and musical instrument design.  
connect:  
  - "[[Vortex shedding]]"  
  - "[[Aeolian tones]]"  
  - "[[Fluid-structure interaction]]"  

---

# Vortex-Induced Oscillations in Fluid-Structure Systems  

<mark class="hltr-green">1. Vortex shedding generates periodic forces that couple with structural resonances.  
2. The Strouhal number governs the dimensionless frequency of vortex formation.  
3. Nonlinear interactions can lead to lock-in phenomena and chaotic responses.</mark>  

## Hydrodynamic Foundations  

The Kármán vortex street arises when a fluid flows past a cylindrical or prismatic obstacle at intermediate Reynolds numbers ($40 < Re < 10^5$). The shedding frequency $f_s$ follows the Strouhal relation:  

$$St = \frac{f_s d}{U}$$  

where $d$ is the characteristic length of the obstacle and $U$ the free-stream velocity (Roshko 1954:12). <mark class="hltr-yellow">Von Kármán's stability analysis (1911)</mark> showed that the alternating vortex pattern remains stable only when the ratio of circulation strengths $\Gamma$ satisfies:  

$$\frac{\Gamma_1}{\Gamma_2} = \sqrt{3}$$  

<mark class="hltr-blue">This creates the distinctive staggered double row of vortices</mark>, with acoustic radiation occurring when the shedding frequency coincides with structural modes (Blevins 1990:147).  

## Aeroacoustic Coupling in Extended Structures  

Consider a ruler protruding from a table edge—an analog to vortex-induced vibrations:  

1. The free length $L$ determines the natural frequency via Euler-Bernoulli beam theory:  
   $$f_n = \frac{\beta_n^2}{2\pi}\sqrt{\frac{EI}{\rho A L^4}}$$  
   where $\beta_n$ are roots of the characteristic equation for cantilever boundary conditions.  

2. Airflow across the ruler's edge generates vortices at $f_s$, producing audible <mark class="hltr-purple">aeolian tones</mark> when $f_s \approx f_n$.  

3. Interactive Python visualization below demonstrates this lock-in effect:  




## Research Questions  

1. **Fundamental**: How does nonlinear mode coupling affect energy transfer between fluid and structure during lock-in? (Williamson & Govardhan 2004)  

2. **Applied**: Can Kármán vortices be harnessed for sustainable energy harvesting in urban wind environments? (Bernitsas et al. 2008)  

3. **Speculative**: Do vortex streets exhibit computational properties when interacting with arrays of smart materials?  


## References  
```bibtex
@article{karman1911,
  title={Über den Mechanismus des Widerstandes den ein bewegter Körper in einer Flüssigkeit erfährt},
  author={Kármán, Theodore von},
  journal={Nachrichten von der Gesellschaft der Wissenschaften zu Göttingen},
  pages={509--517},
  year={1911}
}

@book{blevins1990,
  title={Flow-Induced Vibration},
  author={Blevins, Robert D.},
  publisher={Van Nostrand Reinhold},
  year={1990}
}

@article{williamson2004,
  title={Vortex-induced vibrations},
  author={Williamson, Charles HK and Govardhan, Raghuraman},
  journal={Annual Review of Fluid Mechanics},
  volume={36},
  pages={413--455},
  year={2004}
}
```