the site is comprised of a backend in python and a frontend in nuxt bun vite. 
# Technologies Used

# frontend

to run
```bash
export NODE_OPTIONS="--max-old-space-size=4096" bun run dev
```


# backend
to run
```bash
source venv/bin/activate.fish
python app.py
```

# sketch 
1. Nueva Estética "Premium Studio" para Sketches:
He implementado tu propuesta de prompt para lograr ese look de catálogo de luthería de lujo:
## Archivos de Control:
 backend/prompt_sketch_base.txt
backend/prompt_sketch_negative.txt
### Personalización:
El backend ahora lee estos archivos en cada generación. Puedes editarlos en cualquier momento para ajustar la iluminación o el fondo sin tocar una sola línea de código Python.
### Resultado:
Fondo negro puro, suelo brillante con reflexiones sutiles y eliminación de ruido visual (manos, gente, luces de concierto).

# model training
### local pdf training
#### procesamiento del texto.
- tesseract
- PyMuPDF, PDFPlumber para contexto de imágenes.
- hugging face transformers`
- pares de preguntas y respuestas relacionadas con los conceptos
- BoW (bag of Words)
- TF-IDF
- Word2Vec, GloVe (embeddings preentrenados)
- BERT y GPT (modelos avanzados para converir texto en vectores densos)
el resultado es un conjunto de vectores que representan el texto.
#### procesamiento de imágenes
- CNNs (Convolutional Neural Networks)
	- TensorFlow y PyTorch (implementan RosNet, EfficientNet, MobileNet )
#### procesamiento de fórmulas
- MathPix
- MathBert
#### multimodales
- CLIP (modelo multimodal de OpenAI de txt e img)
- VisualBERT, ViLBERT: Extensiones de BERT.
- redes Siamés o Atenciones Cruzadas.
	- procesa texto e imágenes en ramas separadas y luego combina sus presentacione mediante mecanismos como:
		- atención cruzada.
		- similaridad coseno (para tareas de búsqueda multimodal)
### definición del modelo
- definir el modelo final para aceptar características combinadas de entrada y produzca salidas múltiples. 
	- definir el modelo final: MLP para clasificación o regresión.
	- cargar datos: frameworks PyTorch / TensorFlow .
	- entrenar: ajusta el modelo con una función de pérdida adecuada y optimizador (ej Adam)
	- guardar el modelo `torc.save / tensorflow.keras.models.save
	 
	`

# data
## captures
### running on openai
![|400](https://i.imgur.com/bdpwVsa.png)


![|400](https://i.imgur.com/H5I1O1G.png)
![|400](https://i.imgur.com/zY0CwrV.png)
![|400](https://i.imgur.com/3fzwqY7.png)
```
flute for ten hands, make a just intonation scale, put the ratios in numbers, make each interface with arrows, more than a mouthpiece
```
![|400](https://i.imgur.com/vtZAf6v.png) 
![|400](https://i.imgur.com/2cKVDqt.png)
![|400](https://i.imgur.com/g409Zk8.png)
![|400](https://i.imgur.com/1fTG0g1.png)
  
![|400](https://i.imgur.com/Ge818hm.png)

 Create a bassoon with metamaterial tone holes that control acoustic perfect transmission...

![|400](https://i.imgur.com/ky142nT.png)

one performer plays clarinet, other performer play trumpet, and a thir performer plays violin. A microphone is in the middle of the three, and 5 speakers are surrounding them. Graph the acoustical projection. 
![|400](https://i.imgur.com/PyP5qj1.png)


 microphone in the center surrounded by 5 speakers focusing to center , making feedback through a cymbal, a trumpet and a double bass
![|400](https://i.imgur.com/TntilCA.png)

a microphone inside a speaker inside a tuba, performer is located on the left. The three of them are organized with harmonic ratios, symbolize them with numbers.


![|400](https://i.imgur.com/e1x6RSy.png)





![|400](https://i.imgur.com/GcuCydT.png)
 Create a saxophone with keys that fold through fourth-dimensional space for alternate fingerings...
![|400](https://i.imgur.com/QuHxHjD.png)

 Create a hybrid violin-theremin that uses levitating strings tuned to the Pythagorean ratio 3:2...

![|400](https://i.imgur.com/5sg0R5P.png)

Imagine a clarinet with a metamaterial reed that generates negative group velocity waves...


That's a fascinating idea! Here's how you could visualize this invented speculative instrument using an organogram technique.

Firstly, the clarinet is an aerophone, so its resonant acoustical form can be represented by circles. Circles will represent the aerophone tubes. Two semi-circles, with the central part deleted, would represent the two genus of the clarinet: mouthpiece and the main body of the instrument.

As for the metamaterial reed that gives negative group velocity waves, it's a unique aspect and we could represent this by placing an inverted triangle inside the mouthpiece circle. The inverted triangle could indicate the direction of waves.

On the other hand we will use a teardrop shape to represent the negative group velocity reed. 

A small circle will be placed on the right side of the figure to represent the performer's position.

Let's proceed with the visualization.

![|400](https://i.imgur.com/hMOC8Xi.png)

 Envision the Biomorphic Trombone: slide that grows new positions for microtonal divisions...



In the Biomorphic Trombone, we would combine the traditional structure of a trombone (an aerophone) with a speculative morphing shape. This new trombone would feature a slide that can extend and retract to create new positions for microtonal divisions. 

Visually, we would represent the trombone with a circle for its resonating echo chamber (the bell), and a long cylinder for its slide. Arrows will be plotted on this cylindrical structure to denote the movement of the slide, while space between the arrows will indicate the microtones. 

As an aerophone, a circle is the starting point for the trombone's organogram. The slide of the trombone will be represented as a straight, vertical line coming from the circle. The new positions for microtonal divisions will be represented by smaller circles along the slide. 

Each circle will be assigned a number (in blue color) denoting a different new position for possible microtonal divisions, making the instrument's interface more explicit. 

This futuristic trombone model allows for a multiplicity of tonal possibilities, extending the capabilities beyond the traditional seven slide positions, and giving the player an immense range of microtonal expressivity. 

The acoustical space of the trombone is dynamically changing with the slide, and therefore, a spectral simulation of its sound would be projected as amorphous, shifting overtones based on the position of the slide. 

The use of material colors and numbers will also be included to show the parts of the trombone- brass (a vibrant golden yellow) for the main body and silver (light grey) for the sliding mechanism. 

Overall, this visualization will provide a unique perspective on the creation of new musical possibilities through speculative instrument design. Please ask for the python code for this plot if required.

![|400](https://i.imgur.com/IkldNDJ.png)

![|400](https://i.imgur.com/tpkPwHp.png)
Design the Hypersonic Harp: strings coated with superlensing metamaterials for subwavelength resolution...
![|400](https://i.imgur.com/KCw0pj6.png)  


![[Pasted image 20250414210723.png]]

### from here is running on deepseek


# meta
para nuxt config
```javascript
	export default defineNuxtConfig({
  app: {
    head: {
      title: 'Soog - Experiencias Inolvidables en Gastronomía y Estilo de Vida',
      meta: [
        { name: 'description', content: 'Descubre las mejores recomendaciones de gastronomía, eventos y estilo de vida en Soog.' },
        { name: 'keywords', content: 'Soog, gastronomía, estilo de vida, eventos, tendencias, experiencias únicas, recomendaciones' },
        { name: 'author', content: 'Luciano Azzigotti' },
        { name: 'viewport', content: 'width=device-width, initial-scale=1.0' },
        { name: 'robots', content: 'index, follow' },
        { name: 'theme-color', content: '#FF5733' },
        { property: 'og:title', content: 'Soog - Experiencias Únicas en Gastronomía y Estilo de Vida' },
        { property: 'og:description', content: 'Con Soog, accede a las mejores recomendaciones de gastronomía, estilo de vida y eventos.' },
        { property: 'og:image', content: 'https://soog.zztt.org/assets/images/soog-preview.png' },
        { property: 'og:url', content: 'https://soog.zztt.org' },
        { property: 'og:type', content: 'website' },
        { property: 'og:locale', content: 'es_ES' },
        { name: 'twitter:card', content: 'summary_large_image' },
        { name: 'twitter:title', content: 'Soog - Gastronomía y Estilo de Vida' },
        { name: 'twitter:description', content: 'Descubre tendencias, experiencias gastronómicas únicas y eventos destacados en Soog.' },
        { name: 'twitter:image', content: 'https://soog.zztt.org/assets/images/soog-twitter-preview.png' },
        { name: 'twitter:site', content: '@soog' },
      ],
      link: [
        { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' },
        { rel: 'apple-touch-icon', href: '/assets/images/apple-icon.png' },
        { rel: 'manifest', href: '/site.webmanifest' },
      ],
      script: [
        {
          type: 'application/ld+json',
          children: JSON.stringify({
            '@context': 'https://schema.org',
            '@type': 'Organization',
            name: 'Soog',
            url: 'https://soog.zztt.org',
            logo: 'https://soog.zztt.org/assets/images/logo.png',
            description: 'Soog es tu guía para descubrir las mejores experiencias de gastronomía, estilo de vida y eventos.',
            address: {
              '@type': 'PostalAddress',
              addressLocality: 'Buenos Aires',
              addressCountry: 'Argentina',
            },
            contactPoint: {
              '@type': 'ContactPoint',
              contactType: 'Technical Support',
              name: 'Luciano Azzigotti',
              telephone: '+41779573262',
            },
            sameAs: [
              'https://facebook.com/soog',
              'https://twitter.com/soog',
              'https://instagram.com/soog',
              'https://linkedin.com/company/soog',
            ],
          }),
        },
      ],
    },
  },
});
```

# roadmap
## planned
- query parser
	- using query parsing system to classify the user’s input. b
	- based on natural language proessing.
	- or specific keywords mapping each modality (text, vector, graph, 3d space)
	- vector : matplotlib d3 pltly
	- 3d:three js
- trainers
	- Skill-Based Training: Fine-tune submodels for specific tasks:
		- Natural language generation for text-based answers.
		- Code generation for 2D/3D visualizations.
		- Logic-based reasoning for schematic/systematic responses.
		- Dynamic Prompts: Design prompts to guide the model based on the question’s category:
		- Example: If a question involves “3D spaces,” prepend the prompt with: ‘You are a Three.js expert. Generate JavaScript code to visualize...’
	- parsing input: draw (vector) simulate (3D space) explain (text based response)
## studing
## backlogged


no drect signal .


interface - multiplexor - 

Phoenix - Arizona - Tuba Museum
lll


