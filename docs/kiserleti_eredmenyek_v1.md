# Kísérleti eredmények (text_vlm_tesla)

Forrás a „Kísérleti vizsgálatok és eredmények” fejezethez.
A kész eredmény a redundanciaszűrés kimenete a 458 Tesla-manőveren: koszinuszmátrixok, küszöbönkénti bent maradó halmaz, és a hozzájuk másolt `.mat` fájlok.

Pacejka- vagy ARX-illesztés ehhez a szűréshez nincs a repóban. A `docs/Untitled 6.md` táblában a `text_vlm_tesla_90` és `text_vlm_tesla_95` mind a négy oszlopa pontosan 1,000. Ez nem mért hiba: a többi módszer sorában a számok változnak, itt mind a nyolc cella azonos, és nincs olyan futtatás, ami ezeket kiírná.

---

## 1. Mi készült el

| Eredmény | Hol van | Állapot |
|---|---|---|
| Hét csoporton belüli koszinuszmátrix (`.npy`, `.csv`, heatmap `.png`, címkék) | `cosine_similarity_matrices/text_vlm_tesla/` | 7 / 7 csoport |
| Eltávolító listák 90–99% | `manoeuvres_for_removing_<küszöb>_text_vlm_tesla.json` | megvan |
| Szűrt `.mat` export | `data_filtered_mat/text_vlm_tesla_{90…96,98}/` | a fájlszám egyezik a manifesttel |
| Embedding PCA és t-SNE | `Results/text_vlm_tesla_viz/` (`ALL_groups_pca_2d.png`, `pca_3d`, `tsne_2d`, `tsne_3d`) | megvan |
| Identifikációs hiba (Pacejka \(v_y\), Pacejka \(r\), ARX \(v_y\), ARX \(r\)) | — | nincs mért eredmény |

A 97%-os és a 99%-os küszöbhöz removing-JSON van, külön `.mat` mappa nincs. 99%-on a kiesők száma 0, tehát a halmaz a teljes 458.

---

## 2. Hasonlóság a csoportokon belül

Az átlón kívüli koszinuszok mediánja minden csoportban 0,91 és 0,93 között van. A legnagyobb pár 0,980 (változó sebesség, szinusz, gáz). Egyetlen pár sem éri el a 0,99-et.

Ebből következik a küszöbök viselkedése a 458-as korpuszon:

| Küszöb | Kieső | Bent marad | `.mat` mappa |
|---:|---:|---:|---|
| 90% | 423 | 35 | `text_vlm_tesla_90` (35 fájl) |
| 91% | 406 | 52 | van |
| 92% | 384 | 74 | van |
| 93% | 347 | 111 | van |
| 94% | 281 | 177 | van |
| 95% | 193 | 265 | `text_vlm_tesla_95` (265 fájl) |
| 96% | 75 | 383 | van |
| 97% | 13 | 445 | nincs exportálva |
| 98% | 1 | 457 | `text_vlm_tesla_98` (457 fájl) |
| 99% | 0 | 458 | nincs exportálva |

Az identifikációra dokumentált halmaz a 95%-os: 265 manőver, a 458-nak az 58%-a.

---

## 3. Mit hagy bent a 95%-os küszöb

| Csoport | Eredeti | Bent marad |
|---|---:|---:|
| állandó sávváltás | 84 | 47 |
| állandó csirip | 84 | 41 |
| állandó szinusz | 224 | 123 |
| változó sávváltás, gáz | 9 | 8 |
| változó sávváltás, fék | 9 | 8 |
| változó szinusz, gáz | 24 | 16 |
| változó szinusz, fék | 24 | 22 |
| összesen | 458 | 265 |

A paraméterrács lefedése ennél a halmaznál:

- Állandó szinusz: mind a 28 parancsolt sebesség (5–140 km/h, 5 km/h-s lépés) megmarad. Mindkét amplitúdó bent van (2°: 60, 8°: 63). Mind a négy frekvencia bent van (1, 3, 5, 7 rad/s: 30, 30, 35, 28 darab).
- Állandó sávváltás: 26 / 28 sebesség. Hiányzik a 110 és a 130 km/h. Mindhárom \(a_y\) szint bent van (alacsony 14, közepes 14, magas 19).
- Állandó csirip: 23 / 28 sebesség. Hiányzik a 20, 55, 90, 105 és 120 km/h. Mindhárom amplitúdó bent van (1°: 13, 3°: 12, 5°: 16).
- Változó sávváltás, gáz: 8 / 9. Kiesik az alacsony \(a_y\), 50% pedál.
- Változó sávváltás, fék: 8 / 9. Kiesik a magas \(a_y\), 20% pedál.
- Változó szinusz, gáz: 16 / 24. A 2°-os, 3 rad/s-os gerjesztés mindhárom pedálállással kiesik. A többi amplitúdó–frekvencia kombinációból legalább egy pedálállás bent marad.
- Változó szinusz, fék: 22 / 24. Kiesik a 2° / 1 rad/s / 50% és a 8° / 3 rad/s / 50%.

A 95%-os halmaz tehát a szinuszos rácsot végig lefedi, a csiripből és a sávváltásból néhány sebességet egészében elhagy, a kis változó sebességű csoportokból csak egyes pedálállásokat.

---

## 4. A két szélső exportált küszöb

**90% (35 manőver).** A csoportok összeesnek. Állandó csiripből 3 marad (5, 35 és 60 km/h; 3° amplitúdó nincs). Állandó szinuszból 12, de a 28 sebességből csak 11. A változó sávváltás gázzal csak a három magas \(a_y\)-szintet tartja meg. A változó szinusz fékezéssel csak 8°-os amplitúdót tart meg.

**98% (457 manőver).** Egyetlen kieső van: `valtozo_v_sin_gas_a2_f3_pedal0_2`. Ez az egyetlen pár, amely eléri a 0,980 hasonlóságot. A többi hat csoport teljes.

---

## 5. Ábra a fejezetbe: látens tér

A beszúrandó ábra a kétdimenziós t-SNE:

`Results/text_vlm_tesla_viz/ALL_groups_tsne_2d.png`

Mind a 458 embedding rajta van, csoportonként színezve, perplexity = 30. A csoportok külön tömbökbe rendeződnek: a kettős sávváltások (állandó és változó sebesség) alul, a csiripek felül, a fékezéssel futó szinuszos manőverek egy kisebb csomóban, a gázzal futó szinuszosak a nagy zöld szinuszfelhő szélén. A PCA-vetület (`ALL_groups_pca_2d.png`) ugyanezt a 458 pontot mutatja, de az első két komponens csak a szórás 21,6%-át fogja meg, és a csoportok jobban egymásba érnek. A 3D ábrák nyomtatásban nehezebben olvashatók.

A jelmagyarázat a fájlnevek angol prefixeivel van feliratozva. A dolgozat ábrájához ezt a képet érdemes a `PICTURES/` mappába másolni.

```latex
\begin{figure}[htb!]
    \centering
    \includegraphics[width=0.92\linewidth]{PICTURES/ALL_groups_tsne_2d.png}
    \caption{A 458 manőver szövegembeddingjének kétdimenziós t-SNE vetülete (perplexity = 30). A színek a hét manővercsoportot jelölik.}
    \label{fig:latens-tsne}
\end{figure}
```

A csoportonkénti heatmapek külön ábrák: `cosine_similarity_matrices/text_vlm_tesla/text_vlm_tesla_cosine_similarity_matrix_<csoport>.png`. A kis csoportokon (legfeljebb 24 manőver) a cellaérték ki van írva. Az állandó szinusz 224×224-es mátrixán nincs.
