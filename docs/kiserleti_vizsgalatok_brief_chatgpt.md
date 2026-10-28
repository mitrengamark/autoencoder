# Brief: „Kísérleti vizsgálatok és eredmények” — ChatGPT / fogalmazási segédlet

Cél: a meglévő fejezetbe beemelni **mind a négy vizsgált módszer** redundanciaszűrési eredményeit úgy, hogy a fejezet **még ne döntsön** a „legjobb” módszer + küszöb mellett. A döntés az **identifikációs** alfejezet / fejezet feladata (Pacejka / ARX).

Írj magyar tudományos nyelven, TDK stílusban. Ne találj ki számot. Ne említsd a ChatGPT-t vagy ezt a briefet. A `text_vlm_tesla` / `text_vlm_tesla_v2` neveket csak akkor használd, ha a dolgozat metodológiája már bevezette őket; egyébként: „szöveges VLM-reprezentáció” és „multimodális VLM-reprezentáció”.

---

## 0. Mi a négy módszer?

A négy vizsgált eljárás **nem** négy független embedding-modell, hanem két szemantikai reprezentáció és két redundanciaszűrő **kombinációja**:

| # | Dolgozatbeli megnevezés (javaslat) | Reprezentáció | Szűrő |
|---|---|---|---|
| 1 | szöveges VLM + frekvencia-alapú szűrés | `text_vlm_tesla` (v1) | frequency removal (pipeline) |
| 2 | szöveges VLM + mohó lefedőhalmaz | `text_vlm_tesla` (v1) | greedy set cover |
| 3 | multimodális VLM + frekvencia-alapú szűrés | `text_vlm_tesla_v2` | frequency removal (pipeline) |
| 4 | multimodális VLM + mohó lefedőhalmaz | `text_vlm_tesla_v2` | greedy set cover |

Fogalmazási irány: először a két reprezentációt említsd (ezek a metodológiában már megvannak), majd a két szűrőt, végül a négy kombináció eredményeit.

**Ne** keverd bele a VAE / PCA / `text_embed_*` / image-only ágakat ebbe a négyesbe. Azok más összehasonlítások.

---

## 1. Fejezet szerepe (1–2 bekezdés)

- Univerzum: **458** Tesla-manőver, hét manővercsoport.
- Két reprezentáció:
  - **v1 (`text_vlm_tesla`):** strukturált plotok + metadata → négy VLM-JSON → fusion bekezdés → szöveg-only embedding (`qwen3-embedding`, 4096-d).
  - **v2 (`text_vlm_tesla_v2`):** laterális ID-fókuszú leírások + tíz ábra → multimodális embedding (`Qwen3-VL-Embedding-8B`, 4096-d).
- Mindkét esetben csoporton belüli koszinuszmátrix, küszöbrács **90–99%**.
- Két szűrő ugyanazon a mátrixon:
  - **Frekvencia-alapú (pipeline):** a \(\tau\) feletti párokból addig esik ki a legtöbb párban szereplő manőver, amíg a bent maradók között nem marad ilyen pár. Nem garantálja, hogy minden kiesőhöz maradjon \(\tau\) feletti megtartott szomszéd.
  - **Greedy set cover:** a redundancia-szomszédságok mohó lefedése; minden eredeti manőver vagy bent marad, vagy van hozzá \(\tau\) feletti megtartott képviselő (100% szemantikus lefedettség a \(c_i(\tau)\) mutató szerint).
- Ez a fejezet a **kept-görbéket** és a lefedettséget mutatja. Identifikációs győztest itt ne hirdess.

Záró pont:

> A megtartott manőverek száma önmagában nem dönti el a kombináció alkalmasságát. A végső választás a későbbi identifikációs hibák (Pacejka \(v_y\), \(r\); ARX \(v_y\), \(r\)) alapján történik.

---

## 2. Közös kísérleti beállítás

- Adathalmaz: Tesla, 458 manőver, \(10805\times 40\), \(\Delta t=0{,}01\,\mathrm{s}\).
- Csoportok: állandó sávváltás 84, csirip 84, szinusz 224, változó sávváltás gáz/fék 9+9, változó szinusz gáz/fék 24+24.
- Redundáns pár: \(s_{ij}\ge\tau\).
- Kept-szám: \(\mathrm{kept}=458-|\mathrm{removed}|\), a küszöbönkénti removing-JSON / összehasonlító CSV alapján.
- Fő ábra (négy görbe):  
  `Results/text_vlm_tesla_v2_viz/kept_vs_threshold_text_vlm_tesla_v1_v2_both_filters.png`  
  CSV: `…_both_filters.csv`
- A csak v2-s, kétgörbés ábra (frequency vs greedy) megmaradt külön:  
  `Results/text_vlm_tesla_v2_viz/kept_vs_threshold_text_vlm_tesla_v2.png` — opcionális, ha a szűrők külön szemléltetése kell.

---

## 3. Fő táblázat — megtartott manőverek (458-ból)

| Kombináció | 90 | 91 | 92 | 93 | 94 | 95 | 96 | 97 | 98 | 99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 + frequency | 35 | 52 | 74 | 111 | 177 | 265 | 383 | 445 | 457 | 458 |
| v1 + greedy set cover | 16 | 18 | 32 | 55 | 107 | 199 | 346 | 443 | 457 | 458 |
| v2 + frequency | 19 | 27 | 40 | 68 | 118 | 221 | 371 | 457 | 458 | 458 |
| v2 + greedy set cover | 7 | 10 | 15 | 23 | 51 | 141 | 329 | 457 | 458 | 458 |

### Mit emelj ki szövegben

1. **Reprezentáció hatása:** ugyanazzal a szűrővel a v2 alacsonyabb keptet ad, mint a v1 (90–96%). Példa 95%-on: frequency 265 (v1) vs 221 (v2); greedy 199 (v1) vs 141 (v2). A multimodális / laterálisabb reprezentáció szigorúbb redundanciát jelöl.
2. **Szűrő hatása:** mindkét reprezentációnál a greedy set cover kevesebbet tart meg, mint a frekvencia-algoritmus, amíg a görbék össze nem érnek (~97–98%).
3. **Küszöbérzékenység:** 90% mind a négy kombinációnál nagyon agresszív (7–35 kept); 98–99% gyakorlatilag a teljes 458. A tipikus csoporton belüli koszinusz mediánja ~0,92, ezért a 90% sok párt redundánsnak ítél.
4. **Összeérés:** v1-nél 98%-tól mindkét szűrő 457–458; v2-nél 97%-tól 457–458. Magas küszöbön a szűrőválasztás elveszíti a jelentőségét.
5. Magas koszinusz ≠ fizikai egyenértékűség (közös prompt / terminológia / panelstruktúra).

---

## 4. A két reprezentáció (rövid, metodológiai visszautalás)

### 4.1 v1 — szöveges VLM

- 14 panel + metadata → A–D JSON → fusion bekezdés (számjegy nélkül) → szövegembedding.
- Mátrixok: `cosine_similarity_matrices/text_vlm_tesla/`.
- Frequency-export: `data_filtered_mat/text_vlm_tesla_{90…96,98}/` (95% → 265 fájl).
- Látens tér: `Results/text_vlm_tesla_viz/ALL_groups_tsne_2d.png`.
- Frequency 95%: állandó szinusz rács lefedett; csiripből 5 sebesség hiányzik; sávváltásból 110 és 130 km/h.
- Frequency 98%: egyetlen kieső `valtozo_v_sin_gas_a2_f3_pedal0_2`.

### 4.2 v2 — multimodális VLM

- Négy `technical_summary` + tíz ábra együttes embeddingje.
- Mátrixok: `cosine_similarity_matrices/text_vlm_tesla_v2/`.
- Frequency-export: `data_filtered_mat/text_vlm_tesla_v2_*`.
- Greedy-export: `data_filtered_mat/greedy_set_cover_text_vlm_tesla_v2_*` (és a v1 greedy: `data_filtered_mat/greedy_set_cover_*`).
- Viz: `Results/text_vlm_tesla_v2_viz/`.

---

## 5. A két szűrő és a szemantikus lefedettség

A metodológiában bevezetett \(c_i(\tau)=\max_{j\in\mathcal{K}} s_{ij}\) az eltávolított manőverekre:

| Szűrő | Lefedettség a mért korpuszon |
|---|---|
| Frequency (v1) | több küszöbnél (pl. 90, 92–95%) **99,7817%** → 458-ból 1 fedezetlen |
| Frequency (v2) | 92% és 96% ugyanígy **99,7817%** |
| Greedy set cover (v1 és v2) | **100%** minden vizsgált küszöbön |

A greedy tehát kisebb részhalmazt ad, de a \(c_i(\tau)\ge\tau\) feltételt minden kiesőre teljesíti. A frekvencia-algoritmus nem garantálja ezt. A \(c_i\) mutató **nem** igazolja a fizikai helyettesíthetőséget.

---

## 6. Ábrák a fejezetbe

| Ábra | Fájl | Szerep |
|---|---|---|
| **Fő ábra: négy görbe** | `Results/text_vlm_tesla_v2_viz/kept_vs_threshold_text_vlm_tesla_v1_v2_both_filters.png` | v1/v2 × frequency/greedy |
| v1 t-SNE | `Results/text_vlm_tesla_viz/ALL_groups_tsne_2d.png` | szöveges embedding tér |
| v2 t-SNE | `Results/text_vlm_tesla_v2_viz/ALL_groups_tsne_2d.png` | multimodális embedding tér |
| Opcionális: csak v2, két szűrő | `…/kept_vs_threshold_text_vlm_tesla_v2.png` | szűrők külön szemléltetése |
| Heatmap (kis csoport) | `cosine_similarity_matrices/text_vlm_tesla[_v2]/…_valtozo_v_savvaltas_gas.png` | páronkénti hasonlóság |

Javasolt ábraaláírás a fő ábrához:

> A megtartott manőverek száma a koszinusz-küszöb függvényében a szöveges (v1) és a multimodális (v2) reprezentáción, frekvencia-alapú, illetve mohó lefedőhalmaz (greedy set cover) szűréssel. Az univerzum 458 manőver.

---

## 7. Áthidalás az identifikációhoz

1. A négy kombináció ugyanazon a 90–99% rácson különböző méretű részhalmazokat ad.
2. A kisebb kept (pl. v2 + greedy @ 95% → 141) nem feltétlenül jobb identifikációra, mint a nagyobb (v1 + frequency @ 95% → 265).
3. Következő lépés: a megtartott `.mat` halmazokon Pacejka (\(v_y\), \(r\)) és ARX (\(v_y\), \(r\)) illesztés.
4. Végső ajánlás: **reprezentáció × szűrő × küszöb** hármas a négy hiba alapján.

Ne írd, hogy a `text_vlm_tesla_95` (frequency) „a legjobb”, amíg nincs mért identifikáció. Az csupán dokumentált frequency-export jelölt.

**Identifikációs placeholder tiltás:** a `docs/Untitled 6.md` `text_vlm_tesla_90/95` sorainak mind a nyolc 1,000-es cellája **nem** mért eredmény — ne közöld valódi hibaként.

---

## Fogalmazási tiltások

- Ne írd, hogy a négy módszer négy külön embedding-modell.
- Ne írd, hogy a magas koszinusz = azonos fizika.
- Ne rangsorolj identifikációs győztest ebben a fejezetben.
- Ne keverd a VAE/PCA/`text_embed_*` számokat ebbe a négyes táblába.
- A kept-számok forrása a fenti tábla / `kept_vs_threshold_text_vlm_tesla_v1_v2_both_filters.csv`.
