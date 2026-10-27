# text_vlm_tesla (v1) — teljes projekt-workflow

Ez a dokumentum a **v1 embeddinges manőver-redundancia módszert** írja le end-to-end: a projektbeli szerepétől a plot-előállításon és a VLM szöveggeneráláson át az embeddingig, a koszinusz-szűrésig és a szűrt `.mat` identifikációs adathalmazig.

| | |
|--|--|
| Belső név / kimeneti prefix | **`text_vlm_tesla`** |
| Belépési pont | [`run_tesla_vlm_redundancy_pipeline.py`](../run_tesla_vlm_redundancy_pipeline.py) |
| Verzió | **v1** (szöveg-only embedding; fusion bekezdés) |
| Kapcsolódó | v2: [`docs/text_vlm_tesla_v2_workflow.md`](text_vlm_tesla_v2_workflow.md) |

---

## 1. Projektbeli szerep és motiváció

### 1.1 Az identifikációs kontextus

A járműmodell-identifikációhoz (pl. Pacejka / ARX, laterális `vy` / yaw rate) olyan mérési / szimulációs manőverek kellenek, amelyek a becsülni kívánt dinamikáról **érdemi információt** hordoznak. A Tesla szimulációs sorozatban (~458 manőver) sok felvétel hasonló gerjesztést és választ ad; a teljes sorozat feldolgozása felesleges lehet, ugyanakkor egy ritka, erősen gerjesztett manőver kevés másikhoz hasonlíthat, mégis fontos.

A feladat: **utólagos manőver-szintű szűrés** — a kiválasztás egysége egy teljes manőver. A megtartott manőverek numerikus `.mat` adatai **változatlanul** mennek tovább; a nyelvi leírás és az embedding **csak a kiválasztáshoz** szolgál, nem helyettesíti a mérést.

### 1.2 Miért természetes nyelv + embedding?

Egy manőver több, összefüggő jelből áll (kormányzás, perdület, oldalgyorsulás, slip, abroncserő). A strukturált diagramok az időbeli lefutást, a `features.json` a pontos számokat adja. A vizuális nyelvi modell (VLM) e kettőből **összehasonlítható szakmai leírást** készít; a beágyazó modell ehhez sűrű vektort rendel. A koszinusz-hasonlóság a vektortérben jelöli a redundáns párokat.

A módszertani kulcskérdés: *a nyelvi / beágyazott térbeli hasonlóság mennyiben felel meg az identifikáció szempontjából fontos dinamikai hasonlóságnak?*

### 1.3 Hely a projekt módszercsaládjában

| Vonal | Röviden | Tipikus kimenet |
|-------|---------|-----------------|
| **VAE / OG_remake** | Dinamikai bottleneck → csoporton belüli hasonlóság | `data_filtered_mat/…` (referencia / proposed) |
| **PCA baseline** | Ugyanazok a `GROUP_PREFIXES` + küszöbök | Összehasonlító baseline |
| **text_vlm_tesla (v1)** | Plot + metadata → VLM szöveg → **szöveg-embedding** → cosine | `data_filtered_mat/text_vlm_tesla_{90,95,98}` |
| **text_vlm_tesla_v2** | ID-fókuszú promptok, nincs fusion, **multimodális** (szöveg+képek) embed | külön DB / mat (lásd v2 doc) |

A v1 a VAE-szűrés **embeddelt / szemantikai** megfelelője: ugyanaz a csoportlogika és küszöbrács, más reprezentáció.

---

## 2. End-to-end áttekintés

```text
data_tesla/*.mat  (vagy CSV a plot scripthez)
        │
        ▼
 plot_maneuver_structured_panels.py
        │
        ▼
 Results/maneuver_structured_plots_tesla/<maneuver>/
   ├── *.png  (strukturált panelek)
   └── features.json
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│  run_tesla_vlm_redundancy_pipeline.py                    │
│                                                          │
│  text  → A.json…D.json + description.txt (fusion)        │
│  embed → qwen3-embedding → chroma_db/tesla_vlm           │
│  cosine→ mátrixok + manoeuvres_for_removing_90…99        │
│  mat   → data_filtered_mat/text_vlm_tesla_{90,95,98}     │
└──────────────────────────────────────────────────────────┘
        │
        ▼
 identifikáció / további modellezés (megtartott .mat)
```

A négy pipeline-stage (`text`, `embed`, `cosine`, `mat`) külön-külön is futtatható (`--stages`).

---

## 3. Előfeltételek

### 3.1 Adatok

| Szükséges | Alapértelmezett útvonal | Megjegyzés |
|-----------|-------------------------|------------|
| Forrás mérések | `data_tesla/` | Mat stage innen másol |
| Strukturált plotok + metadata | `Results/maneuver_structured_plots_tesla/<maneuver>/` | Text stage bemenete |
| Promptok | `prompts/` | system + task A–D + fusion |

### 3.2 Szoftver / szolgáltatások

- Projekt `.venv` (`uv sync` / pip a projekt szerint)
- Futó **Ollama** (`127.0.0.1:11434`)
- Modellek:
  - **VLM (text):** default `qwen3-vl:32b` — gyakorlatban a CoT / sebesség miatt gyakran `qwen3-vl:8b` a megbízhatóbb választás (`--vlm-model`)
  - **Embedding:** `qwen3-embedding:latest` (OpenAI-kompatibilis `/v1/embeddings`)
- Python: `chromadb`, `numpy`, `scikit-learn`, `Pillow`, stb.

### 3.3 Manővercsoportok

A koszinusz-szűrés **csak csoporton belül** fut (`pca_baseline.GROUP_PREFIXES` — ugyanaz, mint a VAE/PCA vonalon):

1. `allando_v_savvaltas`
2. `allando_v_chirp`
3. `allando_v_sin`
4. `valtozo_v_savvaltas_gas`
5. `valtozo_v_savvaltas_fek`
6. `valtozo_v_sin_gas`
7. `valtozo_v_sin_fek`

Tipikus Tesla korpusz: **458** manőver (pl. 84 + 84 + 224 + 9 + 9 + 24 + 24).

---

## 4. Upstream: strukturált plotok

A VLM nem a nyers idősorra néz, hanem előre generált, szinkronizált panelekre.

**Script:** [`plot_maneuver_structured_panels.py`](../plot_maneuver_structured_panels.py)

- Bemenet: manőver CSV / adatkönyvtár
- Kimenet manőverenként: PNG panelek + `features.json` (determinisztikus jelstatisztika)
- **Nem** hív LLM-et
- Manővertípusonként időablak (pl. `allando_v_sin` első 15 s eldobása + renorm; `valtozo_v_sin_*` ≤20 s; sávváltás ≤5 s)

Példa:

```bash
.venv/bin/python plot_maneuver_structured_panels.py \
  --data-dir data_tesla \
  --maneuver allando_v_chirp_a1_v10
```

A text stage ezeket a PNG-ket és a `features.json` részhalmazát használja.

---

## 5. Stage: text — négy specialista + fusion

### 5.1 Bemenet / kimenet

**Bemenet:** `Results/maneuver_structured_plots_tesla/<maneuver>/` (PNG + `features.json`)

**Kimenet:**

```text
texts/tesla_vlm/<maneuver>/
  A.json
  B.json
  C.json
  D.json
  description.txt          # fusion eredménye — ez megy az embeddingbe
texts/tesla_vlm/errors.jsonl
```

Resume: létező fájl + **nincs** `--overwrite` → skip.

### 5.2 Közös VLM-hívás szabályai

Minden A–D (és a fusion) hívásnál:

1. system prompt: `prompts/systempromt.toml`, elején **`/no_think`**
2. `think=False` az Ollama payloadban (önmagában nem elég Qwen3-VL-nél)
3. plotok: JPEG-re skálázva (~max 1280 px), base64
4. `features.json` **task-specifikus részhalmaza** (a manővernév **nem** megy az LLM-hez)
5. A–D: `format=json`; fusion: JSON `description` mezővel
6. max 3 retry; CoT-szerű válasz reject

Gyakorlati defaultok a kódban: `num_ctx=16384`, `num_predict=16384` (fusionnél kisebb felső korlát, `DEFAULT_FUSION_NUM_PREDICT`).

### 5.3 Task A–D (összesen 14 panel)

#### Task A — control + globális válasz

| | |
|--|--|
| Prompt | `prompts/task_a.toml` |
| Plotok | `02_velocity_stacked.png`, `03_translational_accel_stacked.png`, `10b_longitudinal_control_response.png`, `10c_lateral_control_response.png` |
| Jelek | `vx`, `vy`, `ax`, `ay`, `az`, `gas`, `brake`, `steeringangel`, `yawrate` |
| Derived | `speed_kmh`, `accel_magnitude` |

#### Task B — pálya + rotációs dinamika

| | |
|--|--|
| Prompt | `prompts/task_b.toml` |
| Plotok | `02c_vy_vs_vx_timecolor.png`, `04_rotational_dynamics_stacked.png`, `05_trajectory_xy_timecolor.png` |
| Jelek | `vx`, `vy`, `yaw`, `yawrate`, `roll`, `rollrate`, `pitch`, `pitchrate` |
| Derived | `trajectory` |

#### Task C — slip + keréksebesség

| | |
|--|--|
| Prompt | `prompts/task_c.toml` |
| Plotok | `06_sideslip_stacked.png`, `07_longslip_stacked.png`, `09_wheel_speeds_stacked.png` |
| Jelek | sideslip / longslip / keréksebesség FL…RR |
| Derived | front/rear slip mean-ek |

#### Task D — Fx / Fy

| | |
|--|--|
| Prompt | `prompts/task_d.toml` |
| Plotok | `08_Fx_stacked.png`, `08b_Fy_stacked.png`, `08d_Fx_vs_Fy_timecolor.png`, `08e_sideslip_vs_Fy_timecolor.png` |
| Jelek | `Fx*`, `Fy*`, `sideslip*` |
| Derived | front/rear Fx/Fy mean-ek |

### 5.4 Közös JSON séma (A–D)

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

Szabályok (system / task promptok): a számok forrása a metadata; a ploté az időbeli alak; ne inventáljon manővernevet; aszimmetria csak ha a számok indokolják.

### 5.5 Fusion → `description.txt`

- Prompt: `prompts/description_fusion.toml`
- Kép **nélkül**, a négy JSON alapján
- Forma: `{"description":"<egy folyó ~200–300 szavas bekezdés>"}`
- Kötelező: **minden szám szóval** (a leírásban ne legyen digit)
- Kimenet: `description.txt` → ez az **egyedüli** embed bemenet v1-ben

---

## 6. Stage: embed — szöveg → vektor → Chroma

### 6.1 Működés

1. Beolvassa `texts/tesla_vlm/<maneuver>/description.txt`-et
2. Ha nincs `--overwrite` és az ID már a Chromában van → skip
3. `POST /v1/embeddings`, modell: `qwen3-embedding:latest`
4. Upsert: `id=maneuver`, embedding, `document=description`, metadata `{maneuver, group}`

### 6.2 Tárolás

| | |
|--|--|
| Path | `chroma_db/tesla_vlm` |
| Collection | `tesla_vlm_descriptions` |
| Metric | cosine (`hnsw:space=cosine`) |
| Dim | tipikusan **4096** |

**v1 sajátosság:** az embedding **csak szöveg**; a plotok a text stage-ben „beépülnek” a leírásba, de az embedder nem látja újra a képeket. (A v2 ezt multimodalissá teszi.)

Részleges újraembed:

```bash
.venv/bin/python -u run_tesla_vlm_redundancy_pipeline.py \
  --stages embed --overwrite \
  --maneuvers <manőver_1> <manőver_2>
```

---

## 7. Stage: cosine — csoporton belüli redundancia

### 7.1 Működés

1. Minden `description.txt`-es manőver (a cosine **nem** szűkíti a `--maneuvers` listára)
2. Embeddingek lekérése Chromából
3. 7 csoport prefix szerint
4. Csoportonként koszinusz-mátrix (`sklearn`)
5. Heatmap / `.npy` / `.csv` / label JSON a matrices mappába
6. Küszöbönként (`90`…`99`): `CosineSimilarity.detect_redundancy` → `remove_redundancy`
7. `manoeuvres_for_removing_<thr>_text_vlm_tesla.json` (csoport-index → eltávolítandó nevek)

Logika: ha két manőver hasonlósága ≥ `threshold/100`, redundáns párnak számít; a remove lépés csoportszinten kiválasztja az eltávolítandókat (lásd [`cosine_similarity.py`](../cosine_similarity.py)).

### 7.2 Küszöb értelmezése

| Küszöb | Jelleg | Tipikus kept (458-ból, kész v1 korpusz) |
|--------|--------|------------------------------------------|
| 90% | agresszív | ~35 |
| **95%** | **ajánlott ID-hez** | **~265** |
| 98% | enyhe | ~457 |
| 99% | gyakorlatilag semmi | mind |

A VLM-szövegek embeddingjei gyakran magas koszinuszt adnak → a 90% nagyon szigorú, a 98%+ alig szűr.

**Projekt döntés (v1):** identifikációra **`text_vlm_tesla` @ 95%** → `data_filtered_mat/text_vlm_tesla_95`.

### 7.3 Kimeneti mappa

```text
cosine_similarity_matrices/text_vlm_tesla/
  text_vlm_tesla_cosine_similarity_matrix_<group>.{png,npy,csv,…}
  manoeuvres_for_removing_{90…99}_text_vlm_tesla.json
```

---

## 8. Stage: mat — megtartott `.mat` export

1. Indexeli `data_tesla/` összes `.mat` fájlját (**univerzum**)
2. Küszöbönként (default **90, 95, 98**) betölti a removing JSON-t
3. `kept = univerzum − removed`
4. Újraépíti a kimeneti mappát; másol / linkel; `filter_manifest.json`

```text
data_filtered_mat/text_vlm_tesla_90/
data_filtered_mat/text_vlm_tesla_95/   ← ajánlott
data_filtered_mat/text_vlm_tesla_98/
```

**Fontos előfeltétel:** a cosine / removing lista csak az **embedelt** manőverek közül tud törölni. Ha a Chroma hiányos, a nem embedelt manőverek automatikusan „kept” maradnak (mert nincsenek a remove listán). Teljes, megbízható mat-szűréshez: **458/458** description + Chroma.

---

## 9. Futtatási parancsok

### 9.1 Teljes pipeline

```bash
cd ~/codes/autoencoder
source .venv/bin/activate
mkdir -p logs

screen -S text_vlm_v1
python -u run_tesla_vlm_redundancy_pipeline.py \
  --stages text embed cosine mat \
  --vlm-model qwen3-vl:8b \
  2>&1 | tee logs/text_vlm_tesla_full_$(date +%Y%m%d_%H%M%S).log
```

- **Ne** használd az `--overwrite`-et resume-hoz (kész A–D / description / Chroma ID megmarad).
- Text stage a lassú rész (manőverenként több VLM-hívás).
- A default CLI VLM `qwen3-vl:32b`; a thinking / idő miatt a gyakorlatban gyakran **`qwen3-vl:8b`** a stabilabb.

### 9.2 Stage-enként

```bash
python -u run_tesla_vlm_redundancy_pipeline.py --stages text --vlm-model qwen3-vl:8b
python -u run_tesla_vlm_redundancy_pipeline.py --stages embed cosine mat
```

### 9.3 Hibás manőverek javítása

```bash
python -u run_tesla_vlm_redundancy_pipeline.py \
  --stages text --overwrite --vlm-model qwen3-vl:8b \
  --maneuvers manouver_a manouver_b

python -u run_tesla_vlm_redundancy_pipeline.py \
  --stages embed --overwrite \
  --maneuvers manouver_a manouver_b

# Mátrix + mat mindig a teljes DB-re (ne korlátozd --maneuvers-szel)
python -u run_tesla_vlm_redundancy_pipeline.py --stages cosine mat
```

### 9.4 Embedding vizualizáció (opcionális)

```bash
python -u plot_text_embed_maneuvers.py \
  --chroma-path chroma_db/tesla_vlm \
  --collection tesla_vlm_descriptions \
  --out-dir Results/text_vlm_tesla_viz \
  --overview-only \
  --methods pca tsne
```

---

## 10. Fontos CLI kapcsolók

| Kapcsoló | Default | Szerep |
|----------|---------|--------|
| `--stages` | `text embed cosine mat` | Szakaszok |
| `--plots-root` | `Results/maneuver_structured_plots_tesla` | Plot + features |
| `--texts-root` | `texts/tesla_vlm` | A–D + description |
| `--prompts-dir` | `prompts` | Promptok |
| `--chroma-path` | `chroma_db/tesla_vlm` | Embedding DB |
| `--collection` | `tesla_vlm_descriptions` | Collection |
| `--matrices-dir` | `cosine_similarity_matrices/text_vlm_tesla` | Cosine kimenet |
| `--mat-source` | `data_tesla` | Forrás `.mat` |
| `--mat-out-root` | `data_filtered_mat` | Szűrt gyökér |
| `--vlm-model` | `qwen3-vl:32b` | Vision-language |
| `--embed-model` | `qwen3-embedding:latest` | Szöveg-embedding |
| `--overwrite` | off | Felülírás |
| `--maneuvers` | mind | Text/embed szűkítés |
| `--limit` | — | Első N (smoke) |
| `--num-ctx` | 16384 | Ollama kontextus |
| `--num-predict` | 16384 | Max generált token |
| `--no-plot` | off | Heatmapek ki |

---

## 11. Minőségellenőrző checklist

Kész v1 korpusznál:

1. Plot-mappák száma = `description.txt` száma (cél: **458/458**)
2. Minden manővernél megvan A–D JSON + `description.txt`
3. `description.txt`-ben **nincs számjegy** (`\d`) — fusion szabály
4. Nincs CoT-dump a leírásban
5. Chroma `count()` = descriptionök száma
6. Friss `manoeuvres_for_removing_*.json` és `data_filtered_mat/text_vlm_tesla_{90,95,98}`
7. `texts/tesla_vlm/errors.jsonl` — történelmi hibák maradhatnak; a kimeneti fájlok számítanak

**Aktuális kész állapot (referencia):** 458 description, 458 Chroma ID, mat kept ≈ 35 / 265 / 457 (90 / 95 / 98).

---

## 12. Ismert korlátok (v1)

1. **Szöveg-only embedding** — a plotinformáció csak annyira van jelen, amennyit a fusion bekezdés megőriz; numerikus részletek gyakran szóval íródnak, ami veszteséges lehet.
2. **Fusion tartalom** — longitudinal control, pálya-narratíva stb. bekerülhet a vektorba; laterális ID (Pacejka/ARX vy/r) szempontjából ez zaj lehet → ez motiválta a **v2**-t.
3. **Magas koszinuszok** — a küszöbérzékenység erős; 90% vs 95% vs 98% kept-számok erősen szórnak.
4. **Hiányos Chroma** — a mat stage az egész `data_tesla` univerzumot használja; nem embedelt manőver ≠ eltávolított.
5. **Qwen3-VL thinking** — `/no_think` + `think=False` kell; nagyobb VL modelleknél (pl. 32b) a thinking kikapcsolása nem mindig megbízható.

---

## 13. Fájlindex

| Fájl / mappa | Szerep |
|--------------|--------|
| `plot_maneuver_structured_panels.py` | Plot + `features.json` |
| `run_tesla_vlm_redundancy_pipeline.py` | v1 orkesztrátor |
| `prompts/systempromt.toml` | Közös VLM szabályok |
| `prompts/task_{a,b,c,d}.toml` | Specialista taskok |
| `prompts/description_fusion.toml` | A–D → egy bekezdés |
| `cosine_similarity.py` | Redundancia detektálás / remove |
| `build_filtered_mat_from_removing_json.py` | Mat index / másolás |
| `pca_baseline.py` | `GROUP_PREFIXES`, `THRESHOLDS` |
| `plot_text_embed_maneuvers.py` | PCA / t-SNE viz |
| `texts/tesla_vlm/` | Szöveges korpusz |
| `chroma_db/tesla_vlm/` | Embedding DB |
| `cosine_similarity_matrices/text_vlm_tesla/` | Mátrixok + removing JSON |
| `data_filtered_mat/text_vlm_tesla_*` | Szűrt ID-adathalmaz |

---

## 14. Egy mondatos összefoglaló

**Strukturált plotok + numerikus metadata → négy VLM-szakértő JSON (A–D) → számjegy nélküli fúziós bekezdés → szöveg-embedding Chroma-ba → csoporton belüli koszinusz-redundancia → a megtartott manőverek változatlan `.mat` exportja identifikációhoz (`text_vlm_tesla`, ajánlott küszöb 95%).**
