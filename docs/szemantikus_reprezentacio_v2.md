# Szemantikus reprezentáció — text_vlm_tesla_v2

Forrás a metodológia „Szemantikus reprezentáció” alfejezetéhez, ha a dolgozat a v1 után a **multimodális, laterális identifikációra hangolt** ágat is bevezeti.

Belső név: `text_vlm_tesla_v2`.  
Belépési pont: `run_tesla_vlm_redundancy_pipeline_v2.py`.

A v1 érintetlen marad (`prompts/`, `texts/tesla_vlm/`, `chroma_db/tesla_vlm/`). A v2 külön promptokat, szövegeket és Chroma-gyűjteményt használ.

---

## Cél

A v1 fusion-bekezdés + szöveg-only embedding a teljes manővernarratívát sűríti (pedálok, pálya, keréksebesség stb.). Laterális identifikációhoz (Pacejka \(v_y\) / yaw rate, ARX) ez zaj lehet. A v2:

1. a leírásokat **laterális** jelekre és kapcsolatokra szűkíti;
2. **nem** fésüli egyetlen VLM-fusion bekezdésbe a négy taskot;
3. az embeddingbe a szöveg mellé **ugyanazt a tíz panelt** is beadja.

Egy manőver = egy **4096** dimenziós, L2-normalizált vektor. A koszinusz-redundancia utána ugyanúgy csoporton belül fut.

---

## 1. Idősor → ábrák

Ugyanaz a plot-előállítás, mint a v1-nél: `plot_maneuver_structured_panels.py`,  
`Results/maneuver_structured_plots_tesla/<manőver>/` (PNG + `features.json`).

Az első 2500 minta (25 s ráfutás) kimarad; utána manővertípusonkénti ablak (csirip 83,04 s; állandó szinusz 68,04 s; sávváltás 5 s; változó szinusz 20 s). A VLM és az embedding ezeket az ablakokat látja.

---

## 2. Ábrák → szöveg (`qwen3-vl:8b`)

Modell: Ollama `qwen3-vl:8b`. Promptok: `prompts_v2/`.  
`num_ctx = 16 384`, thinking kikapcsolva (`/no_think`). Képek JPEG, max oldal 1280 px.

Négy task, mindegyik saját ábrákkal és jelkészlettel. A manővernév nem kerül a promptba. Kimenet: ugyanaz a JSON-séma, mint a v1-nél; a továbbvitt mező a **`technical_summary`** (kb. 80–100 szó / task).

| Task | Ábrák (összesen 10) | Fókusz |
|---|---|---|
| A | \(v_x,v_y\); transzlációs gyorsulás; laterális vezérlés→válasz | kormánygerjesztés és globális laterális válasz |
| B | időszínezett \(v_y\)–\(v_x\); yaw / yaw rate | laterális mozgás és yaw |
| C | kerék-oldalcsúszás; longslip | Pacejka-releváns slip |
| D | \(F_y\); oldalcsúszás–\(F_y\); \(F_x\) másodlagosan | laterális erő és slip–erő |

A v1-hez képest **kimarad** a hosszirányú pedál–\(v_x\) panel, a pálya (XY), a kerék-szögsebesség és az \(F_x\)–\(F_y\) szórásábra.

**Nincs fusion-hívás.** A `description.txt` a négy `technical_summary` egymás után, üres sorral elválasztva.

Kész korpusz: **458 / 458** leírás, mind **4 bekezdés**. Hossz: 244–361 szó, medián 288.

Szövegek: `texts/tesla_vlm_v2/<manőver>/` (`A.json`…`D.json`, `description.txt`).

---

## 3. Szöveg + tíz ábra → vektor

Modell: **`Qwen3-VL-Embedding-8B`** (helyi súlyok, bfloat16, CUDA).  
Ez **nem** az Ollama `qwen3-vl:8b`, és **nem** a v1 `qwen3-embedding` modellje.

Egy előrehajtás bemenete:

- a négy bekezdéses `description.txt`;
- ugyanaz a tíz PNG, A→D sorrendben.

Beállítások tipikusan: `max_length = 16 384`, képenként max. \(768\times 768\) pixel.  
Kimenet: **4096**-d, L2-normalizált.

Tárolás: `chroma_db/text_vlm_tesla_v2`, collection `text_vlm_tesla_v2`, koszinusz-tér.  
Állomány: **458 / 458**. Metaadat: `model=Qwen3-VL-Embedding-8B`, `dim=4096`, `n_panels=10`, manőver, csoport.

---

## 4. Redundancia és export

A csoporton belüli koszinusz és a frekvencia-alapú pipeline-szűrés ugyanaz, mint a v1-nél (`cosine` / `mat` stage).

Mátrixok: `cosine_similarity_matrices/text_vlm_tesla_v2/`.  
Frequency-export: `data_filtered_mat/text_vlm_tesla_v2_{küszöb}/`.

Megtartott darabszám (frequency, 458-ból):

| 90 | 91 | 92 | 93 | 94 | 95 | 96 | 97 | 98 | 99 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 19 | 27 | 40 | 68 | 118 | 221 | 371 | 457 | 458 | 458 |

A v1 frequency 95%-on 265, a v2 ugyanott 221: a multimodális / laterálisabb tér szigorúbb redundanciát jelöl. Magas küszöbön (97%+) a két ág összeér.

Látens tér / kept-görbék: `Results/text_vlm_tesla_v2_viz/`.

A **greedy set cover** szűrés külön eljárás; lásd: `docs/greedy_set_cover_redundancia.md`.

---

## 5. Egy mondat a dolgozatba

A v2 a manőver laterális dinamikáját négy szakértői összefoglalóval és tíz strukturált ábrával kódolja egy 4096 dimenziós multimodális embeddingbe; a koszinusz-redundancia ezután ugyanazon a hét manővercsoporton fut, mint a szöveges v1 ágon.
