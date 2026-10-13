# text_vlm_tesla — teljes workflow dokumentáció

Ez a dokumentum végigvezeti a **Tesla manőverek VLM-alapú szöveges leírás → embedding → koszinusz-redundancia → szűrt `.mat` export** folyamatát, az elejétől a végéig.

A pipeline belépési pontja:

```text
run_tesla_vlm_redundancy_pipeline.py
```

A módszer belső neve (kimeneti mappák / fájlnevek): **`text_vlm_tesla`**.

---

## 1. Cél

A cél nem az, hogy „szép szöveget” írjon a modell, hanem hogy:

1. minden manőverről **konzisztens, diszkriminatív természetes nyelvi leírás** készüljön (plotok + numerikus metadata alapján);
2. ezeket **sűrű embeddinggé** alakítsuk;
3. **manővercsoporton belül** koszinusz-hasonlósággal megtaláljuk a redundáns (túl hasonló) manővereket;
4. a megtartott manőverek `.mat` fájljait külön mappába exportáljuk **identifikáció / további modellezés** céljára.

Ez a VAE bottleneck-alapú (`OG_remake`) szűrés **embeddelt / szemantikai** megfelelője.

---

## 2. Előfeltételek

### 2.1 Adatok

| Szükséges | Alapértelmezett útvonal | Megjegyzés |
|-----------|-------------------------|------------|
| Strukturált plotok + `features.json` manőverenként | `Results/maneuver_structured_plots_tesla/<maneuver>/` | A text stage bemenete |
| Nyers / forrás `.mat` fájlok | `data_tesla/` | A mat stage innen másol |
| Promptok | `prompts/` | system + task A–D + fusion |

A plot mappában típikusan megtalálható (többek között):

- `features.json` — időbeli / jelstatisztikai metadata
- stacked / timecolor PNG-k (sebesség, gyorsulás, pálya, slip, erők, stb.)

### 2.2 Szoftver / szolgáltatások

- Python venv a projektben (`.venv`)
- Futó **Ollama** (`127.0.0.1:11434`)
- Modellek:
  - **VLM:** `qwen3-vl:32b` (text stage: A–D + fusion)
  - **Embedding:** `qwen3-embedding:latest` (embed stage)
- Python csomagok: `chromadb`, `numpy`, `scikit-learn`, `Pillow` (kép-JPEG kódoláshoz), stb.

### 2.3 Manővercsoportok

A koszinusz-szűrés **csak csoporton belül** történik (ugyanaz a 7 prefix, mint a VAE/PCA baseline-nél):

1. `allando_v_savvaltas`
2. `allando_v_chirp`
3. `allando_v_sin`
4. `valtozo_v_savvaltas_gas`
5. `valtozo_v_savvaltas_fek`
6. `valtozo_v_sin_gas`
7. `valtozo_v_sin_fek`

Egy tipikus Tesla futásban kb. **458** manőver; a csoportméretek pl.: 84 + 84 + 224 + 9 + 9 + 24 + 24.

---

## 3. Áttekintő folyamatábra

```text
Results/maneuver_structured_plots_tesla/<maneuver>/
        │  PNG plotok + features.json
        ▼
┌───────────────────────────────────────┐
│  STAGE: text                          │
│  qwen3-vl:32b                         │
│  A.json → B.json → C.json → D.json    │
│  → description.txt (fusion)           │
└───────────────────────────────────────┘
        │
        ▼
 texts/tesla_vlm/<maneuver>/
        │
        ▼
┌───────────────────────────────────────┐
│  STAGE: embed                         │
│  qwen3-embedding → Chroma upsert      │
└───────────────────────────────────────┘
        │
        ▼
 chroma_db/tesla_vlm  (collection: tesla_vlm_descriptions)
        │
        ▼
┌───────────────────────────────────────┐
│  STAGE: cosine                        │
│  csoporton belüli cosine mátrix       │
│  + manoeuvres_for_removing_90..99     │
└───────────────────────────────────────┘
        │
        ▼
 cosine_similarity_matrices/text_vlm_tesla/
        │
        ▼
┌───────────────────────────────────────┐
│  STAGE: mat                           │
│  kept = universe − removed            │
│  → data_filtered_mat/text_vlm_tesla_* │
└───────────────────────────────────────┘
```

A négy stage külön-külön is futtatható (`--stages`).

---

## 4. Stage: text — plotokból szakértői JSON, majd fúzió

### 4.1 Bemenet / kimenet

**Bemenet manőverenként:**

- `Results/maneuver_structured_plots_tesla/<maneuver>/features.json`
- task-specifikus PNG-k (lásd alább)

**Kimenet:**

```text
texts/tesla_vlm/<maneuver>/
  A.json
  B.json
  C.json
  D.json
  description.txt
texts/tesla_vlm/errors.jsonl   # hibák naplója (nem törlődik automatikusan)
```

Resume: ha a fájl már létezik és **nincs** `--overwrite`, a task / fusion **skippelődik**.

### 4.2 Négy specialista task (A–D)

Minden task:

1. betölti a **közös system promptot** (`prompts/systempromt.toml`);
2. betölti a **task promptot** (`prompts/task_a.toml` … `task_d.toml`);
3. a `features.json`-ból **csak a task jeleit / derived mezőit** adja át (a `maneuver` mező **nem** kerül az LLM-hez);
4. a plotokat JPEG-re kódolja (lefelé skálázva, max oldal ~1280 px), base64-ként küldi az Ollama `/api/chat`-nek;
5. `format=json` + `think=False` + system elején **`/no_think`** (a Qwen3-VL egyébként CoT-ot ír a `thinking` mezőbe);
6. validálja a JSON kulcsokat, max 3 próbálkozás.

#### Task A — control + globális válasz

| | |
|--|--|
| Prompt | `prompts/task_a.toml` |
| Plotok | `02_velocity_stacked.png`, `03_translational_accel_stacked.png`, `10b_longitudinal_control_response.png`, `10c_lateral_control_response.png` |
| Jelek | `vx`, `vy`, `ax`, `ay`, `az`, `gas`, `brake`, `steeringangel`, `yawrate` |
| Derived | `speed_kmh`, `accel_magnitude` |

#### Task B — pálya + rotációs / testdinamika

| | |
|--|--|
| Prompt | `prompts/task_b.toml` |
| Plotok | `02c_vy_vs_vx_timecolor.png`, `04_rotational_dynamics_stacked.png`, `05_trajectory_xy_timecolor.png` |
| Jelek | `vx`, `vy`, `yaw`, `yawrate`, `roll`, `rollrate`, `pitch`, `pitchrate` |
| Derived | `trajectory` |

#### Task C — gumi slip + keréksebesség

| | |
|--|--|
| Prompt | `prompts/task_c.toml` |
| Plotok | `06_sideslip_stacked.png`, `07_longslip_stacked.png`, `09_wheel_speeds_stacked.png` |
| Jelek | `sideslip{FL,FR,RL,RR}`, `longslip{…}`, keréksebességek `FL`…`RR` [rad/s] |
| Derived | front/rear slip mean-ek |

#### Task D — gumi hossz-/keresztirányú erők

| | |
|--|--|
| Prompt | `prompts/task_d.toml` |
| Plotok | `08_Fx_stacked.png`, `08b_Fy_stacked.png`, `08d_Fx_vs_Fy_timecolor.png`, `08e_sideslip_vs_Fy_timecolor.png` |
| Jelek | `Fx*`, `Fy*`, `sideslip*` |
| Derived | front/rear Fx/Fy mean-ek |

### 4.3 Közös JSON séma (A–D)

Minden task ugyanazt a mezőstruktúrát tölti, de **csak a saját hatókörében**:

```json
{
  "initial_state": "",
  "temporal_evolution": "",
  "main_events": "",
  "signal_relationships": "",
  "symmetry_asymmetry": "",
  "oscillatory_behavior": "",
  "final_state": "",
  "important_numeric_features": [],
  "technical_summary": ""
}
```

Fontos szabályok a system / task promptokban:

- a numerikus metadata a pontos értékek forrása; a plot a időbeli szerkezeté / alaké;
- **ne** inventáljon manővernevet, file stemet, run ID-t;
- aszimmetriát csak akkor állítson, ha a számok tényleg különböznek;
- mezők legyenek egymással konzisztensek.

### 4.4 Fusion → `description.txt`

A négy JSON-t **kép nélkül** összeolvasztja a modell:

- Prompt: `prompts/description_fusion.toml`
- Kimeneti forma: `{"description":"<egy folyó 200–300 szavas bekezdés>"}`
- Kötelező: **minden szám és mértékegység szóval** (nincs digit a leírásban)
- `num_predict` fusionnél max **1024** (hogy ne égjen el CoT-ra a 4096-os keret)
- CoT-szerű válasz (`We are given…`, `Steps:` stb.) → reject / hiba
- A pipeline kicsomagolja a `description` mezőt és `description.txt`-be írja

Ez a fájl lesz az embedding bemenete.

### 4.5 Ollama / Qwen3-VL sajátosságok (gyakorlati)

- A `think=False` **önmagában nem elég**; a system elején kell a `/no_think`.
- A válasz gyakran üres `content` + kitöltött `thinking` mező — a kód ilyenkor a `thinking`-et használja válaszként (ha az már a JSON / bekezdés).
- Alap `num_ctx=16384` (a modell saját 262k kontextusa nagyon lassú lenne).
- Timeout default: 1800 s / hívás.

---

## 5. Stage: embed — leírás → vektor → Chroma

### 5.1 Működés

Manőverenként:

1. beolvassa `texts/tesla_vlm/<maneuver>/description.txt`-et;
2. ha nincs `--overwrite` és az ID már létezik Chroma-ban → **skip**;
3. Ollama OpenAI-kompatibilis endpoint: `POST /v1/embeddings` modellel `qwen3-embedding:latest`;
4. `collection.upsert(id=maneuver, embedding=…, document=description, metadata={maneuver, group})`.

### 5.2 Tárolás

| | |
|--|--|
| Path | `chroma_db/tesla_vlm` |
| Collection | `tesla_vlm_descriptions` |
| Tér | cosine (`hnsw:space=cosine`) |
| Tipikus dim | 4096 |

**Fontos:** az upsert ID-nként történik. Egy-egy manőver javításakor **nem kell** az összeset újraembeddelni — elég:

```bash
.venv/bin/python -u run_tesla_vlm_redundancy_pipeline.py \
  --stages embed --overwrite \
  --maneuvers <javított_manőver_1> <javított_manőver_2> ...
```

---

## 6. Stage: cosine — csoporton belüli hasonlóság és redundancia

### 6.1 Működés

1. A `texts/tesla_vlm` alatt **összes** `description.txt`-es manővert veszi (nem csak a CLI `--maneuvers` listát).
2. Lekéri az embeddingeket Chromából.
3. 7 csoportba rendezi a prefix alapján.
4. Csoportonként koszinusz-hasonlósági mátrixot számol.
5. (Opcionálisan) heatmap + `.npy` + `.csv` + label JSON:
   - `cosine_similarity_matrices/text_vlm_tesla/text_vlm_tesla_cosine_similarity_matrix_<group>.*`
6. Küszöbönként (`90` … `99`) lefuttatja a meglévő `CosineSimilarity.detect_redundancy` / `remove_redundancy` logikát.
7. Kiírja:
   - `manoeuvres_for_removing_<thr>_text_vlm_tesla.json`

A JSON struktúra tipikusan csoport-index → eltávolítandó manővernevek listája.

### 6.2 Küszöb értelmezése

- Magasabb küszöb (pl. **98–99%**) → csak a **nagyon** hasonló párok számítanak redundánsnak → kevés eltávolítás.
- Alacsonyabb küszöb (pl. **90%**) → agresszívebb szűrés → kevesebb megtartott manőver.

A VLM embeddingeknél a hasonlóságok gyakran magasak, ezért:

| Küszöb | Tipikus removed (≈458-ból) | Tipikus kept |
|--------|----------------------------|--------------|
| 90% | ~423 | ~35 |
| 95% | ~193 | ~265 |
| 98% | ~1 | ~457 |
| 99% | 0 | mind |

**Identifikációra ajánlott (projekt döntés):** **`text_vlm_tesla` @ 95%**.

---

## 7. Stage: mat — megtartott `.mat` export

### 7.1 Működés

1. Indexeli a forrás `.mat` fájlokat: `data_tesla/`.
2. Küszöbönként (alapból **90, 95, 98**) betölti a `manoeuvres_for_removing_*_text_vlm_tesla.json`-t.
3. `kept = összes_forrás − removed`.
4. Újraépíti a kimeneti mappát, másolja (vagy linkeli) a megtartott fájlokat, és ír egy `filter_manifest.json`-t.

### 7.2 Kimenet

```text
data_filtered_mat/text_vlm_tesla_90/   # agresszív szűrés
data_filtered_mat/text_vlm_tesla_95/   # ajánlott identifikációhoz
data_filtered_mat/text_vlm_tesla_98/   # enyhe / szinte semmi szűrés
```

Minden mappában:

- a megtartott `*.mat` fájlok;
- `filter_manifest.json` (kept / removed listák, küszöb, forrás JSON).

---

## 8. Futtatási parancsok

### 8.1 Teljes pipeline (458 manőver)

```bash
cd ~/codes/autoencoder
# Ollama fusson, modellek legyenek letöltve

.venv/bin/python -u run_tesla_vlm_redundancy_pipeline.py \
  --stages text embed cosine mat \
  --vlm-model qwen3-vl:32b
```

Megjegyzések:

- **Ne** tedd rá az `--overwrite`-et az első / folytató futásra, ha resume-ot akarsz.
- A text stage a lassú rész (manőverenként tipikusan ~8–15 perc A–D+fusion).
- Screen / tmux ajánlott hosszú futáshoz.

### 8.2 Stage-enként (ajánlott hosszú futásnál)

```bash
# 1) Lassú rész
.venv/bin/python -u run_tesla_vlm_redundancy_pipeline.py --stages text --vlm-model qwen3-vl:32b

# 2) Gyors rész
.venv/bin/python -u run_tesla_vlm_redundancy_pipeline.py --stages embed cosine mat
```

### 8.3 Egy / néhány manőver javítása (resume-barát)

```bash
# Text újraírás csak a hibásokra
.venv/bin/python -u run_tesla_vlm_redundancy_pipeline.py \
  --stages text --overwrite --vlm-model qwen3-vl:32b \
  --maneuvers manouver_a manouver_b

# Csak ezek upsertje Chromába
.venv/bin/python -u run_tesla_vlm_redundancy_pipeline.py \
  --stages embed --overwrite \
  --maneuvers manouver_a manouver_b

# Teljes mátrix + mat újraszámolás (gyors; ne korlátozd --maneuvers-szel)
.venv/bin/python -u run_tesla_vlm_redundancy_pipeline.py \
  --stages cosine mat
```

### 8.4 Embedding vizualizáció (opcionális)

PCA / t-SNE 2D+3D, manővercsoportonként színezve:

```bash
.venv/bin/python -u plot_text_embed_maneuvers.py \
  --chroma-path chroma_db/tesla_vlm \
  --collection tesla_vlm_descriptions \
  --out-dir Results/text_vlm_tesla_viz \
  --overview-only \
  --methods pca tsne
```

Kimenet:

- `Results/text_vlm_tesla_viz/ALL_groups_pca_2d.png`
- `Results/text_vlm_tesla_viz/ALL_groups_pca_3d.png`
- `Results/text_vlm_tesla_viz/ALL_groups_tsne_2d.png`
- `Results/text_vlm_tesla_viz/ALL_groups_tsne_3d.png`

---

## 9. Fontos CLI kapcsolók

| Kapcsoló | Default | Szerep |
|----------|---------|--------|
| `--stages` | `text embed cosine mat` | Melyik szakasz(ok) |
| `--plots-root` | `Results/maneuver_structured_plots_tesla` | Plot + features |
| `--texts-root` | `texts/tesla_vlm` | A–D + description |
| `--chroma-path` | `chroma_db/tesla_vlm` | Embedding DB |
| `--collection` | `tesla_vlm_descriptions` | Chroma collection |
| `--matrices-dir` | `cosine_similarity_matrices/text_vlm_tesla` | Cosine kimenet |
| `--mat-source` | `data_tesla` | Forrás `.mat` |
| `--mat-out-root` | `data_filtered_mat` | Szűrt `.mat` gyökér |
| `--vlm-model` | `qwen3-vl:32b` | Vision-language modell |
| `--embed-model` | `qwen3-embedding:latest` | Embedding modell |
| `--overwrite` | off | Létező A–D / description / Chroma ID felülírása |
| `--maneuvers` | mind | Csak megadott manőverek (text/embed) |
| `--limit` | — | Első N manőver (smoke) |
| `--num-ctx` | 16384 | Ollama kontextusméret |
| `--num-predict` | 4096 | Max generált token (fusionnél ≤1024) |
| `--no-plot` | off | Cosine heatmapek kikapcsolása |

---

## 10. Minőségellenőrzés (gyakorlati checklist)

Egy kész korpusznál érdemes ellenőrizni:

1. **458/458** (vagy a plot-számmal egyező) mappa `description.txt`-tel.
2. A–D JSON mindenhol megvan.
3. A `description.txt`-ekben **nincs számjegy** (`\d`).
4. Nincs CoT-dump (nem „We are given… / Steps:” kezdetű szöveg).
5. Chroma `count()` = descriptionök száma.
6. `manoeuvres_for_removing_*.json` és `data_filtered_mat/text_vlm_tesla_{90,95,98}` friss.
7. Hibák: `texts/tesla_vlm/errors.jsonl` (történelmi sorok maradhatnak; a javított manővereknél a kimenet számít).

---

## 11. Kapcsolódó fájlok (gyors index)

| Fájl / mappa | Szerep |
|--------------|--------|
| `run_tesla_vlm_redundancy_pipeline.py` | Teljes orkesztrátor |
| `prompts/systempromt.toml` | Közös VLM system szabályok |
| `prompts/task_{a,b,c,d}.toml` | Specialista taskok |
| `prompts/description_fusion.toml` | A–D → egy bekezdés |
| `cosine_similarity.py` | Redundancia detektálás / eltávolítás |
| `build_filtered_mat_from_removing_json.py` | Mat indexelés / másolás segédek |
| `pca_baseline.py` | `GROUP_PREFIXES`, `THRESHOLDS` |
| `plot_text_embed_maneuvers.py` | PCA/t-SNE viz |
| `texts/tesla_vlm/` | Szöveges korpusz |
| `chroma_db/tesla_vlm/` | Embedding DB |
| `cosine_similarity_matrices/text_vlm_tesla/` | Mátrixok + removing JSON |
| `data_filtered_mat/text_vlm_tesla_*` | Szűrt identifikációs adathalmaz |

---

## 12. Döntési összefoglaló (projektállapot)

- A **VAE (`OG_remake`)** a dinamikai bottleneck-alapú szűrés referencia / „proposed” vonala.
- A **`text_vlm_tesla`** az **embeddelt, szemantikai** szűrés: plot+metadata → természetes nyelv → embedding.
- Identifikációra a VLM-vonalon a gyakorlatban elfogadott választás:
  - **`data_filtered_mat/text_vlm_tesla_95`** (~265 manőver).
- A 90%-os set nagyon kicsi (~35); a 98%-os gyakorlatilag alig szűr.

---

## 13. Egy mondatos végigvezetés

**Plotok + features → négy VLM-szakértő JSON (A–D) → szóval kiírt fúziós leírás → embedding Chroma-ba → csoporton belüli koszinusz-redundancia → a megtartott manőverek `.mat` fájljainak exportja identifikációhoz.**
