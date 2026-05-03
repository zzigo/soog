⁄
In industry, versioning for deployed services follows several best practices depending on the nature of the project, the need for backwards compatibility, and how users interact with different versions. Here are some common strategies:
**1. Subdirectories (Versioned URLs)**
- Example:
- https://soog.zztt.org/v1/
- https://soog.zztt.org/v2/
- Pros:
- Simple to implement.
- Users can manually select a version.
- Good for frontend-heavy applications.
- Cons:
- Can become cluttered if too many versions exist.
- Harder to manage shared static assets across versions.
**2. Subdomains for Each Version**
- Example:
- https://v1.soog.zztt.org/
- https://v2.soog.zztt.org/
- Pros:
- Clean separation of versions.o
- Easier for A/B testing.
- Can use different backend configurations.
- Cons:
- Requires DNS configuration.
- Can be complex for frontend applications needing shared resources.
**3. API-Style Versioning (For Backend Services)**
- Example:
- https://soog.zztt.org/api/v1/
- https://soog.zztt.org/api/v2/
- Common for RESTful APIs and GraphQL.
- Pros:
- Maintains a stable base URL.
- Allows different versions of an API to be maintained in parallel.
- Cons:
- If your project is not an API, it might not be the best fit.
**4. Feature Flags / Environment-Based Testing**
- Instead of deploying separate versions, keep one production instance but enable features selectively:
- Example:
- https://soog.zztt.org/ → Default (Stable)
- https://soog.zztt.org/?feature=new_ui → Enables V2 features dynamically.
- Pros:
- No need for multiple deployments.
- Can enable/disable features dynamically.
- Cons:
- Complexity in code handling multiple versions simultaneously.
**5. Dockerized or Staging Deployments**
- Have separate deployments for each version:
- staging.soog.zztt.org → For internal testing.
- beta.soog.zztt.org → Public testing for select users.
- soog.zztt.org → Stable production.
- Pros:
- More control over production and test environments.
- Allows for gradual rollout.
- Cons:
- Requires DevOps management.
**Best Practice for Your Case?**
Since **SOOG** is a web-based project with UI + possible API interactions:
- If you want a clear separation for users → **Use versioned subdirectories (/v1, /v2) or subdomains (v1.soog.zztt.org).**
- If you are testing features internally → **Use feature flags or beta.soog.zztt.org for A/B testing.**
- If you’re maintaining a backend service → **Use /api/v1/, /api/v2/.**
🔹 **Recommended Approach for You:**
1. **For front-end versioning**: https://soog.zztt.org/v1/ and https://soog.zztt.org/v2/.
2. **For experimental features**: https://beta.soog.zztt.org/ with feature toggles.
3. **For long-term versioning**: Move stable versions to https://stable.soog.zztt.org/, and always have https://soog.zztt.org/ point to the latest stable release.


---

# typology

instruments properties
	register
affordance (N)
historical relations (N)


## functions
1. prediction can occur by linking “unknown” properties through neighborhood similarity.
2. Predictive morphogenesis for instruments
	1. given an instrumental sequence MOAIE, a structural morphology is suggested (3D model, affordance, resonance pattern)
3. “Instrument space” can be visualized as clusters or trajectories, showing mutations between types (e.g. flute ↔ clarinet ↔ hyperflute).

## implementations

 practical, end-to-end way to start with[[t-SNE]] on top of your current stack (ArangoDB + Nuxt + Vite + Bun + BERT/DeepSeek), and a roadmap toward a generative organological KB.

1. What to embed (feature design)
    
    1.1 Core categorical features (multi-hot or learned embeddings)
    

- excitation: strike, pluck, bow, breath, friction, rubbing, blowing, electronic actuation
    
- vibratory element: string, membrane, air column, plate, bar, shell, fluid, plasma, metamaterial
    
- resonator topology: tube(open/closed), box, sphere, plate, horn, coupled cavities
    
- interface/gesture: hand, breath, body, robotic, OSC, MIDI, biosignal
    
- material: wood species, metal type, ceramic, composite, skin/gut, piezoelectric, magnetostrictive
    
- agency: human, cyborg, autonomous, mixed
    
- environment: room type, outdoor, underwater, augmented space, feedback loop present
    
- function/role: solo, ensemble, drone, percussive bed, gesture display
    
    Encode each category as multi-hot initially, later replace with learned embeddings.
    

  

1.2 Numeric/physical features (standardize)

- typical fundamental range (Hz), partial density/inharmonicity estimate, Q factor proxy, loudness envelope features (attack/decay slopes), dynamic range, mass, size, wall thickness ratios, anisotropy proxy (0–1), coupling strength (feedback on/off + gain at loop), sustain time.
    

  

1.3 Graph/topological features from ArangoDB

- degree, betweenness, clustering coefficient
    
- node2vec/DeepWalk embeddings over your instrument-principle-material graph (captures “organological context”)
    
- motif counts: how many “material↔excitation↔resonator” triangles it participates in.
    

  

1.4 Text features (BERT you already run)

- description, lineage, technique notes → mean pooled sentence embedding (e.g., 384–768 dims). Concatenate or project down with PCA to ~50–100 dims before UMAP/t-SNE.
    

  

Concatenate: [categorical multi-hot] + [normalized numeric] + [graph embedding] + [text embedding reduced]. Save as X (n_instruments × d).

2. Data pipeline from ArangoDB
    
    2.1 Extract
    

  

- Create a view or AQL that returns one row per instrument with all needed properties and connected neighbors.
    
    Example AQL sketch (adjust names):
    
    FOR i IN instruments
    
    LET cats = {
    
    excitation: i.excitation, vibratory: i.vibrator, resonator: i.resonator,
    
    interface: i.interface, material: i.material, agency: i.agency, environment: i.environment, role: i.role
    
    }
    
    LET nums = {
    
    f0_min: i.f0_min, f0_max: i.f0_max, inharm: i.inharmonicity, q: i.q_factor,
    
    mass: i.mass, size: i.size, feedback_gain: i.feedback_gain
    
    }
    
    LET text = i.description
    
    RETURN { _key: i._key, cats, nums, text }
    

  

2.2 Featureization

- In Bun/Node: encode multi-hot from enumerations you control in config files.
    
- Text: call your BERT service to embed description strings.
    
- Graph: export edge list and run node2vec (Python is easiest), then map back by _key.
    

  

2.3 Store for reuse

- Write a compact feature table to Parquet/CSV and keep a version hash in Arango so the UI knows which embedding it’s showing.
    
- Optionally store “embedding2d: {umap_x, umap_y}” back on each instrument document for fast client rendering.
    

  

3. UMAP vs t-SNE: when and how
    

  

- Use UMAP by default for global + local structure and incremental transforms.
    
- Use t-SNE for highly local neighborhood inspection and pretty cluster islands.
    

  

Recommended starting hyperparams

- UMAP: n_neighbors=15–50, min_dist=0.05–0.3, metric=cosine for mixed/textual; try euclidean if you pare down to numeric only. Dim=2 or 3.
    
- t-SNE: perplexity=30–50 (smaller for <1k points), learning_rate=200–800, n_iter≥1000, early_exaggeration≈12.
    

  

Practical notes

- UMAP supports transform(new_points) if you keep the trained object; t-SNE generally doesn’t, so you need fit again or use openTSNE with transform approximations.
    
- Always PCA to 50–100 dims before t-SNE for stability; with UMAP this is optional but often helps when d is large (>300).
    

  

4. Where to compute
    
    Option A (recommended): Python sidecar for ML
    

  

- Use umap-learn and openTSNE/scikit-learn.
    
- Expose a tiny HTTP endpoint (FastAPI/Flask) that:
    
    1. Accepts a feature table hash
        
    2. Loads features from disk
        
    3. Computes UMAP/t-SNE
        
    4. Returns coords + cluster labels
        
    
- Or schedule embedding jobs offline and push results back into ArangoDB.
    

  

Option B: In-browser for small sets

- Use umap-js and tsne-js with Web Workers. Good for hundreds to a few thousand points; beyond that, do it offline and cache.
    

  

5. Frontend explorer (Nuxt/Vite, fullscreen)
    

  

- Use three.js or regl-scatterplot for 2D/3D scatter that handles 10k–100k points (instanced rendering).
    
- Hover = tooltip: show instrument name, families, materials, small waveform glyph (optional), and nearest-neighbor list.
    
- Lasso/brush selection → issue AQL queries for selected IDs.
    
- Side panel “Factors” with toggles/sliders: color by category, size by numeric (e.g., Q), filter by ranges.
    
- “Trajectories” mode: draw lines between an instrument and its k-NN in embedding or along historical lineage edges from the KG.
    

  

6. Clustering and labels
    

  

- After UMAP, run HDBSCAN (Python) or DBSCAN to assign cluster labels and outliers.
    
- Push labels back into each instrument document: {cluster_umap_v1: 7, outlier: true}
    
- Provide multiple label layers: “material cluster”, “gesture cluster”, “hybrid zone cluster”.
    

  

7. From maps to generation (Generative KB)
    
    7.1 Nearest-neighbor recombination
    

  

- For any target function (e.g., breath-driven drone), find neighbors in embedding and identify feature combinations not yet observed (e.g., breath + plate resonator + metamaterial horn). Propose a new “mutation” node.
    

  

7.2 Link prediction in your KG

- Train a simple graph completion model (start with heuristics: Common Neighbors, Adamic-Adar; then TransE/DistMult/RotatE later).
    
- Predict missing edges like instrument→has_resonator_type or instrument→affords_gesture. High-confidence missing links become design prompts.
    

  

7.3 Constraint-aware generator

- Define constraints as AQL rules (feasibility sets): incompatible combos, required couplings (e.g., if resonator=air column closed, interface cannot be bow unless mediator X).
    
- Sample candidates from the neighborhood in embedding space, then filter by constraints, then rank by novelty (low frequency of combo) × feasibility (constraint score).
    

  

7.4 Text-conditioned proposals

- Use BERT embeddings of design briefs (“self-resonant membrane with slow attack, breath control, metallic shimmer”) to retrieve nearest regions in embedding; propose concrete schemas with parameter defaults.
    

  

8. Minimal, runnable examples to start today
    
    8.1 Python (umap-learn) on a saved feature matrix X.npy
    

  

- Fit UMAP and save 2D coords; also save the model for future transform.
    
- Compute HDBSCAN clusters.
    
- Write JSON {key: instrument_id, x, y, cluster}.
    
    8.2 JS (umap-js) small demo
    
- Client-side UMAP for a few hundred points fetched from Arango REST.
    
- Render with regl-scatterplot; color by material, size by Q.
    

  

8.3 AQL helpers

- K-NN by precomputed embedding stored on each instrument:
    
    FOR i IN instruments
    
    FILTER HAS(i, “embedding2d”)
    
    RETURN { _key: i._key, x: i.embedding2d.x, y: i.embedding2d.y }
    
- Store neighbors list after an offline k-NN pass:
    
    UPDATE { _key: i._key, knn: neighborsArray } IN instruments
    

  

9. Evaluation loop
    

  

- Internal validity: neighborhood purity (how often nearest neighbors share key labels you trust).
    
- External validity: expert adjudication—pick 10 generated candidates, score feasibility and relevance.
    
- Diversity metric: coverage of rarely combined features.
    

  

10. Concrete first sprint (1–2 days of focused work)
    

  

11. Freeze a v0 feature schema with 8 categorical groups, 6 numeric, 1 text embedding (BERT→PCA50), and 1 graph embedding (node2vec 64d).
    
12. Export features, run UMAP (n_neighbors=30, min_dist=0.1, metric=cosine), store coords in ArangoDB.
    
13. Build a Nuxt fullscreen scatter with regl-scatterplot; hover tooltips; color by family; lasso selection → AQL fetch of details.
    
14. Compute k-NN (k=10) in the UMAP plane and write back a neighbors array; draw thin lines to reveal hybrid zones.
    
15. Implement a simple “mutation suggester”: for a selected cluster, list top 10 unseen triples (excitation, resonator, material) that appear among neighbors but not in the selected node—rank by frequency gap.
    

  

If you want, I can draft:

- a compact JSON schema for the feature table,
    
- a tiny FastAPI endpoint that returns UMAP coordinates from a posted feature CSV,
    
- and a Nuxt component that renders the embedding with lasso/hover and talks to ArangoDB.