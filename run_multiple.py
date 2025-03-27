import subprocess
import json
import numpy as np
import csv
from configobj import ConfigObj
from Analyse.manoeuvers_plot_together import plot_all_tsne_data
from Analyse.saliency_map import plot_saliency_map
from Reduction.detect_overlap import DetectOverlap
from Reduction.manoeuvres_filtering import ManoeuvresFiltering
from Config.load_config import num_manoeuvres, overlay_multiple_manoeuvres, filtering


# A config.ini fájl elérési útja
CONFIG_PATH = "Config/config.ini"

# A tesztelendő manőverek listája
velocity_maneuvers_list = [
    [
        "allando_v_savvaltas_alacsony_v5",
        "allando_v_savvaltas_kozepes_v5",
        "allando_v_savvaltas_magas_v5",
        "allando_v_sin_a2_f1_v5",
        "allando_v_sin_a8_f1_v5",
        "allando_v_sin_a2_f3_v5",
        "allando_v_sin_a8_f3_v5",
        "allando_v_sin_a2_f5_v5",
        "allando_v_sin_a8_f5_v5",
        "allando_v_sin_a2_f7_v5",
        "allando_v_sin_a8_f7_v5",
        "allando_v_chirp_a1_v5",
        "allando_v_chirp_a3_v5",
        "allando_v_chirp_a5_v5",
    ],
    # [
    #     "allando_v_savvaltas_alacsony_v10",
    #     "allando_v_savvaltas_kozepes_v10",
    #     "allando_v_savvaltas_magas_v10",
    #     "allando_v_sin_a2_f1_v10",
    #     "allando_v_sin_a8_f1_v10",
    #     "allando_v_sin_a2_f3_v10",
    #     "allando_v_sin_a8_f3_v10",
    #     "allando_v_sin_a2_f5_v10",
    #     "allando_v_sin_a8_f5_v10",
    #     "allando_v_sin_a2_f7_v10",
    #     "allando_v_sin_a8_f7_v10",
    #     "allando_v_chirp_a1_v10",
    #     "allando_v_chirp_a3_v10",
    #     "allando_v_chirp_a5_v10",
    # ],
    # [
    #     "allando_v_savvaltas_alacsony_v15",
    #     "allando_v_savvaltas_kozepes_v15",
    #     "allando_v_savvaltas_magas_v15",
    #     "allando_v_sin_a2_f1_v15",
    #     "allando_v_sin_a8_f1_v15",
    #     "allando_v_sin_a2_f3_v15",
    #     "allando_v_sin_a8_f3_v15",
    #     "allando_v_sin_a2_f5_v15",
    #     "allando_v_sin_a8_f5_v15",
    #     "allando_v_sin_a2_f7_v15",
    #     "allando_v_sin_a8_f7_v15",
    #     "allando_v_chirp_a1_v15",
    #     "allando_v_chirp_a3_v15",
    #     "allando_v_chirp_a5_v15",
    # ],
    # [
    #     "allando_v_savvaltas_alacsony_v20",
    #     "allando_v_savvaltas_kozepes_v20",
    #     "allando_v_savvaltas_magas_v20",
    #     "allando_v_sin_a2_f1_v20",
    #     "allando_v_sin_a8_f1_v20",
    #     "allando_v_sin_a2_f3_v20",
    #     "allando_v_sin_a8_f3_v20",
    #     "allando_v_sin_a2_f5_v20",
    #     "allando_v_sin_a8_f5_v20",
    #     "allando_v_sin_a2_f7_v20",
    #     "allando_v_sin_a8_f7_v20",
    #     "allando_v_chirp_a1_v20",
    #     "allando_v_chirp_a3_v20",
    #     "allando_v_chirp_a5_v20",
    # ],
    # [
    #     "allando_v_savvaltas_alacsony_v25",
    #     "allando_v_savvaltas_kozepes_v25",
    #     "allando_v_savvaltas_magas_v25",
    #     "allando_v_sin_a2_f1_v25",
    #     "allando_v_sin_a8_f1_v25",
    #     "allando_v_sin_a2_f3_v25",
    #     "allando_v_sin_a8_f3_v25",
    #     "allando_v_sin_a2_f5_v25",
    #     "allando_v_sin_a8_f5_v25",
    #     "allando_v_sin_a2_f7_v25",
    #     "allando_v_sin_a8_f7_v25",
    #     "allando_v_chirp_a1_v25",
    #     "allando_v_chirp_a3_v25",
    #     "allando_v_chirp_a5_v25",
    # ],
    # [
    #     "allando_v_savvaltas_alacsony_v30",
    #     "allando_v_savvaltas_kozepes_v30",
    #     "allando_v_savvaltas_magas_v30",
    #     "allando_v_sin_a2_f1_v30",
    #     "allando_v_sin_a8_f1_v30",
    #     "allando_v_sin_a2_f3_v30",
    #     "allando_v_sin_a8_f3_v30",
    #     "allando_v_sin_a2_f5_v30",
    #     "allando_v_sin_a8_f5_v30",
    #     "allando_v_sin_a2_f7_v30",
    #     "allando_v_sin_a8_f7_v30",
    #     "allando_v_chirp_a1_v30",
    #     "allando_v_chirp_a3_v30",
    #     "allando_v_chirp_a5_v30",
    # ],
    # [
    #     "allando_v_savvaltas_alacsony_v35",
    #     "allando_v_savvaltas_kozepes_v35",
    #     "allando_v_savvaltas_magas_v35",
    #     "allando_v_sin_a2_f1_v35",
    #     "allando_v_sin_a8_f1_v35",
    #     "allando_v_sin_a2_f3_v35",
    #     "allando_v_sin_a8_f3_v35",
    #     "allando_v_sin_a2_f5_v35",
    #     "allando_v_sin_a8_f5_v35",
    #     "allando_v_sin_a2_f7_v35",
    #     "allando_v_sin_a8_f7_v35",
    #     "allando_v_chirp_a1_v35",
    #     "allando_v_chirp_a3_v35",
    #     "allando_v_chirp_a5_v35",
    # ],
    # [
    #     "allando_v_savvaltas_alacsony_v40",
    #     "allando_v_savvaltas_kozepes_v40",
    #     "allando_v_savvaltas_magas_v40",
    #     "allando_v_sin_a2_f1_v40",
    #     "allando_v_sin_a8_f1_v40",
    #     "allando_v_sin_a2_f3_v40",
    #     "allando_v_sin_a8_f3_v40",
    #     "allando_v_sin_a2_f5_v40",
    #     "allando_v_sin_a8_f5_v40",
    #     "allando_v_sin_a2_f7_v40",
    #     "allando_v_sin_a8_f7_v40",
    #     "allando_v_chirp_a1_v40",
    #     "allando_v_chirp_a3_v40",
    #     "allando_v_chirp_a5_v40",
    # ],
    # [
    #     "allando_v_savvaltas_alacsony_v45",
    #     "allando_v_savvaltas_kozepes_v45",
    #     "allando_v_savvaltas_magas_v45",
    #     "allando_v_sin_a2_f1_v45",
    #     "allando_v_sin_a8_f1_v45",
    #     "allando_v_sin_a2_f3_v45",
    #     "allando_v_sin_a8_f3_v45",
    #     "allando_v_sin_a2_f5_v45",
    #     "allando_v_sin_a8_f5_v45",
    #     "allando_v_sin_a2_f7_v45",
    #     "allando_v_sin_a8_f7_v45",
    #     "allando_v_chirp_a1_v45",
    #     "allando_v_chirp_a3_v45",
    #     "allando_v_chirp_a5_v45",
    # ],
    # [
    #     "allando_v_savvaltas_alacsony_v50",
    #     "allando_v_savvaltas_kozepes_v50",
    #     "allando_v_savvaltas_magas_v50",
    #     "allando_v_sin_a2_f1_v50",
    #     "allando_v_sin_a8_f1_v50",
    #     "allando_v_sin_a2_f3_v50",
    #     "allando_v_sin_a8_f3_v50",
    #     "allando_v_sin_a2_f5_v50",
    #     "allando_v_sin_a8_f5_v50",
    #     "allando_v_sin_a2_f7_v50",
    #     "allando_v_sin_a8_f7_v50",
    #     "allando_v_chirp_a1_v50",
    #     "allando_v_chirp_a3_v50",
    #     "allando_v_chirp_a5_v50",
    # ],
    # [
    #     "allando_v_savvaltas_alacsony_v55",
    #     "allando_v_savvaltas_kozepes_v55",
    #     "allando_v_savvaltas_magas_v55",
    #     "allando_v_sin_a2_f1_v55",
    #     "allando_v_sin_a8_f1_v55",
    #     "allando_v_sin_a2_f3_v55",
    #     "allando_v_sin_a8_f3_v55",
    #     "allando_v_sin_a2_f5_v55",
    #     "allando_v_sin_a8_f5_v55",
    #     "allando_v_sin_a2_f7_v55",
    #     "allando_v_sin_a8_f7_v55",
    #     "allando_v_chirp_a1_v55",
    #     "allando_v_chirp_a3_v55",
    #     "allando_v_chirp_a5_v55",
    # ],
    # [
    #     "allando_v_savvaltas_alacsony_v60",
    #     "allando_v_savvaltas_kozepes_v60",
    #     "allando_v_savvaltas_magas_v60",
    #     "allando_v_sin_a2_f1_v60",
    #     "allando_v_sin_a8_f1_v60",
    #     "allando_v_sin_a2_f3_v60",
    #     "allando_v_sin_a8_f3_v60",
    #     "allando_v_sin_a2_f5_v60",
    #     "allando_v_sin_a8_f5_v60",
    #     "allando_v_sin_a2_f7_v60",
    #     "allando_v_sin_a8_f7_v60",
    #     "allando_v_chirp_a1_v60",
    #     "allando_v_chirp_a3_v60",
    #     "allando_v_chirp_a5_v60",
    # ],
    # [
    #     "allando_v_savvaltas_alacsony_v65",
    #     "allando_v_savvaltas_kozepes_v65",
    #     "allando_v_savvaltas_magas_v65",
    #     "allando_v_sin_a2_f1_v65",
    #     "allando_v_sin_a8_f1_v65",
    #     "allando_v_sin_a2_f3_v65",
    #     "allando_v_sin_a8_f3_v65",
    #     "allando_v_sin_a2_f5_v65",
    #     "allando_v_sin_a8_f5_v65",
    #     "allando_v_sin_a2_f7_v65",
    #     "allando_v_sin_a8_f7_v65",
    #     "allando_v_chirp_a1_v65",
    #     "allando_v_chirp_a3_v65",
    #     "allando_v_chirp_a5_v65",
    # ],
    # [
    #     "allando_v_savvaltas_alacsony_v70",
    #     "allando_v_savvaltas_kozepes_v70",
    #     "allando_v_savvaltas_magas_v70",
    #     "allando_v_sin_a2_f1_v70",
    #     "allando_v_sin_a8_f1_v70",
    #     "allando_v_sin_a2_f3_v70",
    #     "allando_v_sin_a8_f3_v70",
    #     "allando_v_sin_a2_f5_v70",
    #     "allando_v_sin_a8_f5_v70",
    #     "allando_v_sin_a2_f7_v70",
    #     "allando_v_sin_a8_f7_v70",
    #     "allando_v_chirp_a1_v70",
    #     "allando_v_chirp_a3_v70",
    #     "allando_v_chirp_a5_v70",
    # ],
    # [
    #     "allando_v_savvaltas_alacsony_v75",
    #     "allando_v_savvaltas_kozepes_v75",
    #     "allando_v_savvaltas_magas_v75",
    #     "allando_v_sin_a2_f1_v75",
    #     "allando_v_sin_a8_f1_v75",
    #     "allando_v_sin_a2_f3_v75",
    #     "allando_v_sin_a8_f3_v75",
    #     "allando_v_sin_a2_f5_v75",
    #     "allando_v_sin_a8_f5_v75",
    #     "allando_v_sin_a2_f7_v75",
    #     "allando_v_sin_a8_f7_v75",
    #     "allando_v_chirp_a1_v75",
    #     "allando_v_chirp_a3_v75",
    #     "allando_v_chirp_a5_v75",
    # ],
    # [
    #     "allando_v_savvaltas_alacsony_v80",
    #     "allando_v_savvaltas_kozepes_v80",
    #     "allando_v_savvaltas_magas_v80",
    #     "allando_v_sin_a2_f1_v80",
    #     "allando_v_sin_a8_f1_v80",
    #     "allando_v_sin_a2_f3_v80",
    #     "allando_v_sin_a8_f3_v80",
    #     "allando_v_sin_a2_f5_v80",
    #     "allando_v_sin_a8_f5_v80",
    #     "allando_v_sin_a2_f7_v80",
    #     "allando_v_sin_a8_f7_v80",
    #     "allando_v_chirp_a1_v80",
    #     "allando_v_chirp_a3_v80",
    #     "allando_v_chirp_a5_v80",
    # ],
    # [
    #     "allando_v_savvaltas_alacsony_v85",
    #     "allando_v_savvaltas_kozepes_v85",
    #     "allando_v_savvaltas_magas_v85",
    #     "allando_v_sin_a2_f1_v85",
    #     "allando_v_sin_a8_f1_v85",
    #     "allando_v_sin_a2_f3_v85",
    #     "allando_v_sin_a8_f3_v85",
    #     "allando_v_sin_a2_f5_v85",
    #     "allando_v_sin_a8_f5_v85",
    #     "allando_v_sin_a2_f7_v85",
    #     "allando_v_sin_a8_f7_v85",
    #     "allando_v_chirp_a1_v85",
    #     "allando_v_chirp_a3_v85",
    #     "allando_v_chirp_a5_v85",
    # ],
    # [
    #     "allando_v_savvaltas_alacsony_v90",
    #     "allando_v_savvaltas_kozepes_v90",
    #     "allando_v_savvaltas_magas_v90",
    #     "allando_v_sin_a2_f1_v90",
    #     "allando_v_sin_a8_f1_v90",
    #     "allando_v_sin_a2_f3_v90",
    #     "allando_v_sin_a8_f3_v90",
    #     "allando_v_sin_a2_f5_v90",
    #     "allando_v_sin_a8_f5_v90",
    #     "allando_v_sin_a2_f7_v90",
    #     "allando_v_sin_a8_f7_v90",
    #     "allando_v_chirp_a1_v90",
    #     "allando_v_chirp_a3_v90",
    #     "allando_v_chirp_a5_v90",
    # ],
    # [
    #     "allando_v_savvaltas_alacsony_v95",
    #     "allando_v_savvaltas_kozepes_v95",
    #     "allando_v_savvaltas_magas_v95",
    #     "allando_v_sin_a2_f1_v95",
    #     "allando_v_sin_a8_f1_v95",
    #     "allando_v_sin_a2_f3_v95",
    #     "allando_v_sin_a8_f3_v95",
    #     "allando_v_sin_a2_f5_v95",
    #     "allando_v_sin_a8_f5_v95",
    #     "allando_v_sin_a2_f7_v95",
    #     "allando_v_sin_a8_f7_v95",
    #     "allando_v_chirp_a1_v95",
    #     "allando_v_chirp_a3_v95",
    #     "allando_v_chirp_a5_v95",
    # ],
    # [
    #     "allando_v_savvaltas_alacsony_v100",
    #     "allando_v_savvaltas_kozepes_v100",
    #     "allando_v_savvaltas_magas_v100",
    #     "allando_v_sin_a2_f1_v100",
    #     "allando_v_sin_a8_f1_v100",
    #     "allando_v_sin_a2_f3_v100",
    #     "allando_v_sin_a8_f3_v100",
    #     "allando_v_sin_a2_f5_v100",
    #     "allando_v_sin_a8_f5_v100",
    #     "allando_v_sin_a2_f7_v100",
    #     "allando_v_sin_a8_f7_v100",
    #     "allando_v_chirp_a1_v100",
    #     "allando_v_chirp_a3_v100",
    #     "allando_v_chirp_a5_v100",
    # ],
    # [
    #     "allando_v_savvaltas_alacsony_v105",
    #     "allando_v_savvaltas_kozepes_v105",
    #     "allando_v_savvaltas_magas_v105",
    #     "allando_v_sin_a2_f1_v105",
    #     "allando_v_sin_a8_f1_v105",
    #     "allando_v_sin_a2_f3_v105",
    #     "allando_v_sin_a8_f3_v105",
    #     "allando_v_sin_a2_f5_v105",
    #     "allando_v_sin_a8_f5_v105",
    #     "allando_v_sin_a2_f7_v105",
    #     "allando_v_sin_a8_f7_v105",
    #     "allando_v_chirp_a1_v105",
    #     "allando_v_chirp_a3_v105",
    #     "allando_v_chirp_a5_v105",
    # ],
    # [
    #     "allando_v_savvaltas_alacsony_v110",
    #     "allando_v_savvaltas_kozepes_v110",
    #     "allando_v_savvaltas_magas_v110",
    #     "allando_v_sin_a2_f1_v110",
    #     "allando_v_sin_a8_f1_v110",
    #     "allando_v_sin_a2_f3_v110",
    #     "allando_v_sin_a8_f3_v110",
    #     "allando_v_sin_a2_f5_v110",
    #     "allando_v_sin_a8_f5_v110",
    #     "allando_v_sin_a2_f7_v110",
    #     "allando_v_sin_a8_f7_v110",
    #     "allando_v_chirp_a1_v110",
    #     "allando_v_chirp_a3_v110",
    #     "allando_v_chirp_a5_v110",
    # ],
    # [
    #     "allando_v_savvaltas_alacsony_v115",
    #     "allando_v_savvaltas_kozepes_v115",
    #     "allando_v_savvaltas_magas_v115",
    #     "allando_v_sin_a2_f1_v115",
    #     "allando_v_sin_a8_f1_v115",
    #     "allando_v_sin_a2_f3_v115",
    #     "allando_v_sin_a8_f3_v115",
    #     "allando_v_sin_a2_f5_v115",
    #     "allando_v_sin_a8_f5_v115",
    #     "allando_v_sin_a2_f7_v115",
    #     "allando_v_sin_a8_f7_v115",
    #     "allando_v_chirp_a1_v115",
    #     "allando_v_chirp_a3_v115",
    #     "allando_v_chirp_a5_v115",
    # ],
    # [
    #     "allando_v_savvaltas_alacsony_v120",
    #     "allando_v_savvaltas_kozepes_v120",
    #     "allando_v_savvaltas_magas_v120",
    #     "allando_v_sin_a2_f1_v120",
    #     "allando_v_sin_a8_f1_v120",
    #     "allando_v_sin_a2_f3_v120",
    #     "allando_v_sin_a8_f3_v120",
    #     "allando_v_sin_a2_f5_v120",
    #     "allando_v_sin_a8_f5_v120",
    #     "allando_v_sin_a2_f7_v120",
    #     "allando_v_sin_a8_f7_v120",
    #     "allando_v_chirp_a1_v120",
    #     "allando_v_chirp_a3_v120",
    #     "allando_v_chirp_a5_v120",
    # ],
    # [
    #     "allando_v_savvaltas_alacsony_v125",
    #     "allando_v_savvaltas_kozepes_v125",
    #     "allando_v_savvaltas_magas_v125",
    #     "allando_v_sin_a2_f1_v125",
    #     "allando_v_sin_a8_f1_v125",
    #     "allando_v_sin_a2_f3_v125",
    #     "allando_v_sin_a8_f3_v125",
    #     "allando_v_sin_a2_f5_v125",
    #     "allando_v_sin_a8_f5_v125",
    #     "allando_v_sin_a2_f7_v125",
    #     "allando_v_sin_a8_f7_v125",
    #     "allando_v_chirp_a1_v125",
    #     "allando_v_chirp_a3_v125",
    #     "allando_v_chirp_a5_v125",
    # ],
    # [
    #     "allando_v_savvaltas_alacsony_v130",
    #     "allando_v_savvaltas_kozepes_v130",
    #     "allando_v_savvaltas_magas_v130",
    #     "allando_v_sin_a2_f1_v130",
    #     "allando_v_sin_a8_f1_v130",
    #     "allando_v_sin_a2_f3_v130",
    #     "allando_v_sin_a8_f3_v130",
    #     "allando_v_sin_a2_f5_v130",
    #     "allando_v_sin_a8_f5_v130",
    #     "allando_v_sin_a2_f7_v130",
    #     "allando_v_sin_a8_f7_v130",
    #     "allando_v_chirp_a1_v130",
    #     "allando_v_chirp_a3_v130",
    #     "allando_v_chirp_a5_v130",
    # ],
    # [
    #     "allando_v_savvaltas_alacsony_v135",
    #     "allando_v_savvaltas_kozepes_v135",
    #     "allando_v_savvaltas_magas_v135",
    #     "allando_v_sin_a2_f1_v135",
    #     "allando_v_sin_a8_f1_v135",
    #     "allando_v_sin_a2_f3_v135",
    #     "allando_v_sin_a8_f3_v135",
    #     "allando_v_sin_a2_f5_v135",
    #     "allando_v_sin_a8_f5_v135",
    #     "allando_v_sin_a2_f7_v135",
    #     "allando_v_sin_a8_f7_v135",
    #     "allando_v_chirp_a1_v135",
    #     "allando_v_chirp_a3_v135",
    #     "allando_v_chirp_a5_v135",
    # ],
    # [
    #     "allando_v_savvaltas_alacsony_v140",
    #     "allando_v_savvaltas_kozepes_v140",
    #     "allando_v_savvaltas_magas_v140",
    #     "allando_v_sin_a2_f1_v140",
    #     "allando_v_sin_a8_f1_v140",
    #     "allando_v_sin_a2_f3_v140",
    #     "allando_v_sin_a8_f3_v140",
    #     "allando_v_sin_a2_f5_v140",
    #     "allando_v_sin_a8_f5_v140",
    #     "allando_v_sin_a2_f7_v140",
    #     "allando_v_sin_a8_f7_v140",
    #     "allando_v_chirp_a1_v140",
    #     "allando_v_chirp_a3_v140",
    #     "allando_v_chirp_a5_v140",
    # ],
]

basic_maneuvers_list = [
    [
        "allando_v_savvaltas_alacsony_v5",
        "allando_v_savvaltas_kozepes_v5",
        "allando_v_savvaltas_magas_v5",
        "allando_v_savvaltas_alacsony_v10",
        "allando_v_savvaltas_kozepes_v10",
        "allando_v_savvaltas_magas_v10",
        "allando_v_savvaltas_alacsony_v15",
        "allando_v_savvaltas_kozepes_v15",
        "allando_v_savvaltas_magas_v15",
        "allando_v_savvaltas_alacsony_v20",
        "allando_v_savvaltas_kozepes_v20",
        "allando_v_savvaltas_magas_v20",
        "allando_v_savvaltas_alacsony_v25",
        "allando_v_savvaltas_kozepes_v25",
        "allando_v_savvaltas_magas_v25",
        "allando_v_savvaltas_alacsony_v30",
        "allando_v_savvaltas_kozepes_v30",
        "allando_v_savvaltas_magas_v30",
        "allando_v_savvaltas_alacsony_v35",
        "allando_v_savvaltas_kozepes_v35",
        "allando_v_savvaltas_magas_v35",
        "allando_v_savvaltas_alacsony_v40",
        "allando_v_savvaltas_kozepes_v40",
        "allando_v_savvaltas_magas_v40",
        "allando_v_savvaltas_alacsony_v45",
        "allando_v_savvaltas_kozepes_v45",
        "allando_v_savvaltas_magas_v45",
        "allando_v_savvaltas_alacsony_v50",
        "allando_v_savvaltas_kozepes_v50",
        "allando_v_savvaltas_magas_v50",
        "allando_v_savvaltas_alacsony_v55",
        "allando_v_savvaltas_kozepes_v55",
        "allando_v_savvaltas_magas_v55",
        "allando_v_savvaltas_alacsony_v60",
        "allando_v_savvaltas_kozepes_v60",
        "allando_v_savvaltas_magas_v60",
        "allando_v_savvaltas_alacsony_v65",
        "allando_v_savvaltas_kozepes_v65",
        "allando_v_savvaltas_magas_v65",
        "allando_v_savvaltas_alacsony_v70",
        "allando_v_savvaltas_kozepes_v70",
        "allando_v_savvaltas_magas_v70",
        "allando_v_savvaltas_alacsony_v75",
        "allando_v_savvaltas_kozepes_v75",
        "allando_v_savvaltas_magas_v75",
        "allando_v_savvaltas_alacsony_v80",
        "allando_v_savvaltas_kozepes_v80",
        "allando_v_savvaltas_magas_v80",
        "allando_v_savvaltas_alacsony_v85",
        "allando_v_savvaltas_kozepes_v85",
        "allando_v_savvaltas_magas_v85",
        "allando_v_savvaltas_alacsony_v90",
        "allando_v_savvaltas_kozepes_v90",
        "allando_v_savvaltas_magas_v90",
        "allando_v_savvaltas_alacsony_v95",
        "allando_v_savvaltas_kozepes_v95",
        "allando_v_savvaltas_magas_v95",
        "allando_v_savvaltas_alacsony_v100",
        "allando_v_savvaltas_kozepes_v100",
        "allando_v_savvaltas_magas_v100",
        "allando_v_savvaltas_alacsony_v105",
        "allando_v_savvaltas_kozepes_v105",
        "allando_v_savvaltas_magas_v105",
        "allando_v_savvaltas_alacsony_v110",
        "allando_v_savvaltas_kozepes_v110",
        "allando_v_savvaltas_magas_v110",
        "allando_v_savvaltas_alacsony_v115",
        "allando_v_savvaltas_kozepes_v115",
        "allando_v_savvaltas_magas_v115",
        "allando_v_savvaltas_alacsony_v120",
        "allando_v_savvaltas_kozepes_v120",
        "allando_v_savvaltas_magas_v120",
        "allando_v_savvaltas_alacsony_v125",
        "allando_v_savvaltas_kozepes_v125",
        "allando_v_savvaltas_magas_v125",
        "allando_v_savvaltas_alacsony_v130",
        "allando_v_savvaltas_kozepes_v130",
        "allando_v_savvaltas_magas_v130",
        "allando_v_savvaltas_alacsony_v135",
        "allando_v_savvaltas_kozepes_v135",
        "allando_v_savvaltas_magas_v135",
        "allando_v_savvaltas_alacsony_v140",
        "allando_v_savvaltas_kozepes_v140",
        "allando_v_savvaltas_magas_v140",
    ],
    [
        "allando_v_sin_a2_f1_v5",
        "allando_v_sin_a8_f1_v5",
        "allando_v_sin_a2_f3_v5",
        "allando_v_sin_a8_f3_v5",
        "allando_v_sin_a2_f5_v5",
        "allando_v_sin_a8_f5_v5",
        "allando_v_sin_a2_f7_v5",
        "allando_v_sin_a8_f7_v5",
        "allando_v_sin_a2_f1_v10",
        "allando_v_sin_a8_f1_v10",
        "allando_v_sin_a2_f3_v10",
        "allando_v_sin_a8_f3_v10",
        "allando_v_sin_a2_f5_v10",
        "allando_v_sin_a8_f5_v10",
        "allando_v_sin_a2_f7_v10",
        "allando_v_sin_a8_f7_v10",
        "allando_v_sin_a2_f1_v15",
        "allando_v_sin_a8_f1_v15",
        "allando_v_sin_a2_f3_v15",
        "allando_v_sin_a8_f3_v15",
        "allando_v_sin_a2_f5_v15",
        "allando_v_sin_a8_f5_v15",
        "allando_v_sin_a2_f7_v15",
        "allando_v_sin_a8_f7_v15",
        "allando_v_sin_a2_f1_v20",
        "allando_v_sin_a8_f1_v20",
        "allando_v_sin_a2_f3_v20",
        "allando_v_sin_a8_f3_v20",
        "allando_v_sin_a2_f5_v20",
        "allando_v_sin_a8_f5_v20",
        "allando_v_sin_a2_f7_v20",
        "allando_v_sin_a8_f7_v20",
        "allando_v_sin_a2_f1_v25",
        "allando_v_sin_a8_f1_v25",
        "allando_v_sin_a2_f3_v25",
        "allando_v_sin_a8_f3_v25",
        "allando_v_sin_a2_f5_v25",
        "allando_v_sin_a8_f5_v25",
        "allando_v_sin_a2_f7_v25",
        "allando_v_sin_a8_f7_v25",
        "allando_v_sin_a2_f1_v30",
        "allando_v_sin_a8_f1_v30",
        "allando_v_sin_a2_f3_v30",
        "allando_v_sin_a8_f3_v30",
        "allando_v_sin_a2_f5_v30",
        "allando_v_sin_a8_f5_v30",
        "allando_v_sin_a2_f7_v30",
        "allando_v_sin_a8_f7_v30",
        "allando_v_sin_a2_f1_v35",
        "allando_v_sin_a8_f1_v35",
        "allando_v_sin_a2_f3_v35",
        "allando_v_sin_a8_f3_v35",
        "allando_v_sin_a2_f5_v35",
        "allando_v_sin_a8_f5_v35",
        "allando_v_sin_a2_f7_v35",
        "allando_v_sin_a8_f7_v35",
        "allando_v_sin_a2_f1_v40",
        "allando_v_sin_a8_f1_v40",
        "allando_v_sin_a2_f3_v40",
        "allando_v_sin_a8_f3_v40",
        "allando_v_sin_a2_f5_v40",
        "allando_v_sin_a8_f5_v40",
        "allando_v_sin_a2_f7_v40",
        "allando_v_sin_a8_f7_v40",
        "allando_v_sin_a2_f1_v45",
        "allando_v_sin_a8_f1_v45",
        "allando_v_sin_a2_f3_v45",
        "allando_v_sin_a8_f3_v45",
        "allando_v_sin_a2_f5_v45",
        "allando_v_sin_a8_f5_v45",
        "allando_v_sin_a2_f7_v45",
        "allando_v_sin_a8_f7_v45",
        "allando_v_sin_a2_f1_v50",
        "allando_v_sin_a8_f1_v50",
        "allando_v_sin_a2_f3_v50",
        "allando_v_sin_a8_f3_v50",
        "allando_v_sin_a2_f5_v50",
        "allando_v_sin_a8_f5_v50",
        "allando_v_sin_a2_f7_v50",
        "allando_v_sin_a8_f7_v50",
        "allando_v_sin_a2_f1_v55",
        "allando_v_sin_a8_f1_v55",
        "allando_v_sin_a2_f3_v55",
        "allando_v_sin_a8_f3_v55",
        "allando_v_sin_a2_f5_v55",
        "allando_v_sin_a8_f5_v55",
        "allando_v_sin_a2_f7_v55",
        "allando_v_sin_a8_f7_v55",
        "allando_v_sin_a2_f1_v60",
        "allando_v_sin_a8_f1_v60",
        "allando_v_sin_a2_f3_v60",
        "allando_v_sin_a8_f3_v60",
        "allando_v_sin_a2_f5_v60",
        "allando_v_sin_a8_f5_v60",
        "allando_v_sin_a2_f7_v60",
        "allando_v_sin_a8_f7_v60",
        "allando_v_sin_a2_f1_v65",
        "allando_v_sin_a8_f1_v65",
        "allando_v_sin_a2_f3_v65",
        "allando_v_sin_a8_f3_v65",
        "allando_v_sin_a2_f5_v65",
        "allando_v_sin_a8_f5_v65",
        "allando_v_sin_a2_f7_v65",
        "allando_v_sin_a8_f7_v65",
        "allando_v_sin_a2_f1_v70",
        "allando_v_sin_a8_f1_v70",
        "allando_v_sin_a2_f3_v70",
        "allando_v_sin_a8_f3_v70",
        "allando_v_sin_a2_f5_v70",
        "allando_v_sin_a8_f5_v70",
        "allando_v_sin_a2_f7_v70",
        "allando_v_sin_a8_f7_v70",
        "allando_v_sin_a2_f1_v75",
        "allando_v_sin_a8_f1_v75",
        "allando_v_sin_a2_f3_v75",
        "allando_v_sin_a8_f3_v75",
        "allando_v_sin_a2_f5_v75",
        "allando_v_sin_a8_f5_v75",
        "allando_v_sin_a2_f7_v75",
        "allando_v_sin_a8_f7_v75",
        "allando_v_sin_a2_f1_v80",
        "allando_v_sin_a8_f1_v80",
        "allando_v_sin_a2_f3_v80",
        "allando_v_sin_a8_f3_v80",
        "allando_v_sin_a2_f5_v80",
        "allando_v_sin_a8_f5_v80",
        "allando_v_sin_a2_f7_v80",
        "allando_v_sin_a8_f7_v80",
        "allando_v_sin_a2_f1_v85",
        "allando_v_sin_a8_f1_v85",
        "allando_v_sin_a2_f3_v85",
        "allando_v_sin_a8_f3_v85",
        "allando_v_sin_a2_f5_v85",
        "allando_v_sin_a8_f5_v85",
        "allando_v_sin_a2_f7_v85",
        "allando_v_sin_a8_f7_v85",
        "allando_v_sin_a2_f1_v90",
        "allando_v_sin_a8_f1_v90",
        "allando_v_sin_a2_f3_v90",
        "allando_v_sin_a8_f3_v90",
        "allando_v_sin_a2_f5_v90",
        "allando_v_sin_a8_f5_v90",
        "allando_v_sin_a2_f7_v90",
        "allando_v_sin_a8_f7_v90",
        "allando_v_sin_a2_f1_v95",
        "allando_v_sin_a8_f1_v95",
        "allando_v_sin_a2_f3_v95",
        "allando_v_sin_a8_f3_v95",
        "allando_v_sin_a2_f5_v95",
        "allando_v_sin_a8_f5_v95",
        "allando_v_sin_a2_f7_v95",
        "allando_v_sin_a8_f7_v95",
        "allando_v_sin_a2_f1_v100",
        "allando_v_sin_a8_f1_v100",
        "allando_v_sin_a2_f3_v100",
        "allando_v_sin_a8_f3_v100",
        "allando_v_sin_a2_f5_v100",
        "allando_v_sin_a8_f5_v100",
        "allando_v_sin_a2_f7_v100",
        "allando_v_sin_a8_f7_v100",
        "allando_v_sin_a2_f1_v105",
        "allando_v_sin_a8_f1_v105",
        "allando_v_sin_a2_f3_v105",
        "allando_v_sin_a8_f3_v105",
        "allando_v_sin_a2_f5_v105",
        "allando_v_sin_a8_f5_v105",
        "allando_v_sin_a2_f7_v105",
        "allando_v_sin_a8_f7_v105",
        "allando_v_sin_a2_f1_v110",
        "allando_v_sin_a8_f1_v110",
        "allando_v_sin_a2_f3_v110",
        "allando_v_sin_a8_f3_v110",
        "allando_v_sin_a2_f5_v110",
        "allando_v_sin_a8_f5_v110",
        "allando_v_sin_a2_f7_v110",
        "allando_v_sin_a8_f7_v110",
        "allando_v_sin_a2_f1_v115",
        "allando_v_sin_a8_f1_v115",
        "allando_v_sin_a2_f3_v115",
        "allando_v_sin_a8_f3_v115",
        "allando_v_sin_a2_f5_v115",
        "allando_v_sin_a8_f5_v115",
        "allando_v_sin_a2_f7_v115",
        "allando_v_sin_a8_f7_v115",
        "allando_v_sin_a2_f1_v120",
        "allando_v_sin_a8_f1_v120",
        "allando_v_sin_a2_f3_v120",
        "allando_v_sin_a8_f3_v120",
        "allando_v_sin_a2_f5_v120",
        "allando_v_sin_a8_f5_v120",
        "allando_v_sin_a2_f7_v120",
        "allando_v_sin_a8_f7_v120",
        "allando_v_sin_a2_f1_v125",
        "allando_v_sin_a8_f1_v125",
        "allando_v_sin_a2_f3_v125",
        "allando_v_sin_a8_f3_v125",
        "allando_v_sin_a2_f5_v125",
        "allando_v_sin_a8_f5_v125",
        "allando_v_sin_a2_f7_v125",
        "allando_v_sin_a8_f7_v125",
        "allando_v_sin_a2_f1_v130",
        "allando_v_sin_a8_f1_v130",
        "allando_v_sin_a2_f3_v130",
        "allando_v_sin_a8_f3_v130",
        "allando_v_sin_a2_f5_v130",
        "allando_v_sin_a8_f5_v130",
        "allando_v_sin_a2_f7_v130",
        "allando_v_sin_a8_f7_v130",
        "allando_v_sin_a2_f1_v135",
        "allando_v_sin_a8_f1_v135",
        "allando_v_sin_a2_f3_v135",
        "allando_v_sin_a8_f3_v135",
        "allando_v_sin_a2_f5_v135",
        "allando_v_sin_a8_f5_v135",
        "allando_v_sin_a2_f7_v135",
        "allando_v_sin_a8_f7_v135",
        "allando_v_sin_a2_f1_v140",
        "allando_v_sin_a8_f1_v140",
        "allando_v_sin_a2_f3_v140",
        "allando_v_sin_a8_f3_v140",
        "allando_v_sin_a2_f5_v140",
        "allando_v_sin_a8_f5_v140",
        "allando_v_sin_a2_f7_v140",
        "allando_v_sin_a8_f7_v140",
    ],
    [
        "allando_v_chirp_a1_v5",
        "allando_v_chirp_a3_v5",
        "allando_v_chirp_a5_v5",
        "allando_v_chirp_a1_v10",
        "allando_v_chirp_a3_v10",
        "allando_v_chirp_a5_v10",
        "allando_v_chirp_a1_v15",
        "allando_v_chirp_a3_v15",
        "allando_v_chirp_a5_v15",
        "allando_v_chirp_a1_v20",
        "allando_v_chirp_a3_v20",
        "allando_v_chirp_a5_v20",
        "allando_v_chirp_a1_v25",
        "allando_v_chirp_a3_v25",
        "allando_v_chirp_a5_v25",
        "allando_v_chirp_a1_v30",
        "allando_v_chirp_a3_v30",
        "allando_v_chirp_a5_v30",
        "allando_v_chirp_a1_v35",
        "allando_v_chirp_a3_v35",
        "allando_v_chirp_a5_v35",
        "allando_v_chirp_a1_v40",
        "allando_v_chirp_a3_v40",
        "allando_v_chirp_a5_v40",
        "allando_v_chirp_a1_v45",
        "allando_v_chirp_a3_v45",
        "allando_v_chirp_a5_v45",
        "allando_v_chirp_a1_v50",
        "allando_v_chirp_a3_v50",
        "allando_v_chirp_a5_v50",
        "allando_v_chirp_a1_v55",
        "allando_v_chirp_a3_v55",
        "allando_v_chirp_a5_v55",
        "allando_v_chirp_a1_v60",
        "allando_v_chirp_a3_v60",
        "allando_v_chirp_a5_v60",
        "allando_v_chirp_a1_v65",
        "allando_v_chirp_a3_v65",
        "allando_v_chirp_a5_v65",
        "allando_v_chirp_a1_v70",
        "allando_v_chirp_a3_v70",
        "allando_v_chirp_a5_v70",
        "allando_v_chirp_a1_v75",
        "allando_v_chirp_a3_v75",
        "allando_v_chirp_a5_v75",
        "allando_v_chirp_a1_v80",
        "allando_v_chirp_a3_v80",
        "allando_v_chirp_a5_v80",
        "allando_v_chirp_a1_v85",
        "allando_v_chirp_a3_v85",
        "allando_v_chirp_a5_v85",
        "allando_v_chirp_a1_v90",
        "allando_v_chirp_a3_v90",
        "allando_v_chirp_a5_v90",
        "allando_v_chirp_a1_v95",
        "allando_v_chirp_a3_v95",
        "allando_v_chirp_a5_v95",
        "allando_v_chirp_a1_v100",
        "allando_v_chirp_a3_v100",
        "allando_v_chirp_a5_v100",
        "allando_v_chirp_a1_v105",
        "allando_v_chirp_a3_v105",
        "allando_v_chirp_a5_v105",
        "allando_v_chirp_a1_v110",
        "allando_v_chirp_a3_v110",
        "allando_v_chirp_a5_v110",
        "allando_v_chirp_a1_v115",
        "allando_v_chirp_a3_v115",
        "allando_v_chirp_a5_v115",
        "allando_v_chirp_a1_v120",
        "allando_v_chirp_a3_v120",
        "allando_v_chirp_a5_v120",
        "allando_v_chirp_a1_v125",
        "allando_v_chirp_a3_v125",
        "allando_v_chirp_a5_v125",
        "allando_v_chirp_a1_v130",
        "allando_v_chirp_a3_v130",
        "allando_v_chirp_a5_v130",
        "allando_v_chirp_a1_v135",
        "allando_v_chirp_a3_v135",
        "allando_v_chirp_a5_v135",
        "allando_v_chirp_a1_v140",
        "allando_v_chirp_a3_v140",
        "allando_v_chirp_a5_v140",
    ],
    [
        "valtozo_v_savvaltas_gas_alacsony_pedal0_2",
        "valtozo_v_savvaltas_gas_kozepes_pedal0_2",
        "valtozo_v_savvaltas_gas_magas_pedal0_2",
        "valtozo_v_savvaltas_gas_alacsony_pedal0_5",
        "valtozo_v_savvaltas_gas_kozepes_pedal0_5",
        "valtozo_v_savvaltas_gas_magas_pedal0_5",
        "valtozo_v_savvaltas_gas_alacsony_pedal1_0",
        "valtozo_v_savvaltas_gas_kozepes_pedal1_0",
        "valtozo_v_savvaltas_gas_magas_pedal1_0",
    ],
    [
        "valtozo_v_savvaltas_fek_alacsony_pedal0_2",
        "valtozo_v_savvaltas_fek_kozepes_pedal0_2",
        "valtozo_v_savvaltas_fek_magas_pedal0_2",
        "valtozo_v_savvaltas_fek_alacsony_pedal0_5",
        "valtozo_v_savvaltas_fek_kozepes_pedal0_5",
        "valtozo_v_savvaltas_fek_magas_pedal0_5",
        "valtozo_v_savvaltas_fek_alacsony_pedal1_0",
        "valtozo_v_savvaltas_fek_kozepes_pedal1_0",
        "valtozo_v_savvaltas_fek_magas_pedal1_0",
    ],
    [
        "valtozo_v_sin_gas_a2_f1_pedal0_2",
        "valtozo_v_sin_gas_a2_f1_pedal0_5",
        "valtozo_v_sin_gas_a2_f1_pedal1_0",
        "valtozo_v_sin_gas_a2_f3_pedal0_2",
        "valtozo_v_sin_gas_a2_f3_pedal0_5",
        "valtozo_v_sin_gas_a2_f3_pedal1_0",
        "valtozo_v_sin_gas_a2_f5_pedal0_2",
        "valtozo_v_sin_gas_a2_f5_pedal0_5",
        "valtozo_v_sin_gas_a2_f5_pedal1_0",
        "valtozo_v_sin_gas_a2_f7_pedal0_2",
        "valtozo_v_sin_gas_a2_f7_pedal0_5",
        "valtozo_v_sin_gas_a2_f7_pedal1_0",
        "valtozo_v_sin_gas_a8_f1_pedal0_2",
        "valtozo_v_sin_gas_a8_f1_pedal0_5",
        "valtozo_v_sin_gas_a8_f1_pedal1_0",
        "valtozo_v_sin_gas_a8_f3_pedal0_2",
        "valtozo_v_sin_gas_a8_f3_pedal0_5",
        "valtozo_v_sin_gas_a8_f3_pedal1_0",
        "valtozo_v_sin_gas_a8_f5_pedal0_2",
        "valtozo_v_sin_gas_a8_f5_pedal0_5",
        "valtozo_v_sin_gas_a8_f5_pedal1_0",
        "valtozo_v_sin_gas_a8_f7_pedal0_2",
        "valtozo_v_sin_gas_a8_f7_pedal0_5",
        "valtozo_v_sin_gas_a8_f7_pedal1_0",
    ],
    [
        "valtozo_v_sin_fek_a2_f1_pedal0_2",
        "valtozo_v_sin_fek_a2_f1_pedal0_5",
        "valtozo_v_sin_fek_a2_f1_pedal1_0",
        "valtozo_v_sin_fek_a2_f3_pedal0_2",
        "valtozo_v_sin_fek_a2_f3_pedal0_5",
        "valtozo_v_sin_fek_a2_f3_pedal1_0",
        "valtozo_v_sin_fek_a2_f5_pedal0_2",
        "valtozo_v_sin_fek_a2_f5_pedal0_5",
        "valtozo_v_sin_fek_a2_f5_pedal1_0",
        "valtozo_v_sin_fek_a2_f7_pedal0_2",
        "valtozo_v_sin_fek_a2_f7_pedal0_5",
        "valtozo_v_sin_fek_a2_f7_pedal1_0",
        "valtozo_v_sin_fek_a8_f1_pedal0_2",
        "valtozo_v_sin_fek_a8_f1_pedal0_5",
        "valtozo_v_sin_fek_a8_f1_pedal1_0",
        "valtozo_v_sin_fek_a8_f3_pedal0_2",
        "valtozo_v_sin_fek_a8_f3_pedal0_5",
        "valtozo_v_sin_fek_a8_f3_pedal1_0",
        "valtozo_v_sin_fek_a8_f5_pedal0_2",
        "valtozo_v_sin_fek_a8_f5_pedal0_5",
        "valtozo_v_sin_fek_a8_f5_pedal1_0",
        "valtozo_v_sin_fek_a8_f7_pedal0_2",
        "valtozo_v_sin_fek_a8_f7_pedal0_5",
        "valtozo_v_sin_fek_a8_f7_pedal1_0",
    ],
]

basic_folder_names = [
    "allando_v_savvaltas",
    "allando_v_sin",
    "allando_v_chirp",
    "valtozo_v_savvaltas_gas",
    "valtozo_v_savvaltas_fek",
    "valtozo_v_sin_gas",
    "valtozo_v_sin_fek",
]

velocity_folder_names = [
    "allando_v5",
    # "allando_v10",
    # "allando_v15",
    # "allando_v20",
    # "allando_v25",
    # "allando_v30",
    # "allando_v35",
    # "allando_v40",
    # "allando_v45",
    # "allando_v50",
    # "allando_v55",
    # "allando_v60",
    # "allando_v65",
    # "allando_v70",
    # "allando_v75",
    # "allando_v80",
    # "allando_v85",
    # "allando_v90",
    # "allando_v95",
    # "allando_v100",
    # "allando_v105",
    # "allando_v110",
    # "allando_v115",
    # "allando_v120",
    # "allando_v125",
    # "allando_v130",
    # "allando_v135",
    # "allando_v140",
]

maneuvers_list = basic_maneuvers_list
folder_names = basic_folder_names

redundantpairs_list = []
current_run = 0

if num_manoeuvres > 1:
    # Manővercsoportonként beállítható paraméterek
    batch_sizes = [
        9
    ]  # [84, 224, 84, 9, 9, 24, 24]  # Hány manővert dolgozzunk fel egyszerre?
    steps = batch_sizes  # Hány lépésenként lépjünk tovább az adott típusban?
    total_runs = sum(
        len(maneuvers) // steps[idx] for idx, maneuvers in enumerate(maneuvers_list)
    )
else:
    total_runs = sum(len(maneuvers) for maneuvers in maneuvers_list)

# Fő ciklus, ami végigmegy a manővercsoportokon
for group_idx, maneuvers in enumerate(maneuvers_list):
    # Tároló az összegyűjtött TSNE adatokhoz
    all_tsne_data = []
    all_tsne_label = []
    all_bottleneck_outputs = []
    all_bottleneck_labels = []

    if num_manoeuvres > 1:
        batch_size = batch_sizes[group_idx]
        step = steps[group_idx]

        print(
            f"\n=== {group_idx+1}. Manővercsoport ({batch_size} db / lépésköz: {step}) ==="
        )

        # Iterálás a manővereken adott batch_size és lépésköz szerint
        for i in range(0, len(maneuvers), step):
            selected_maneuvers = maneuvers[i : i + batch_size]
            folder_name = f"{folder_names[group_idx]}_all_reduced"

            # Ha a kiválasztott csoport kisebb, mint batch_size, akkor nem futtatjuk
            if not selected_maneuvers:
                continue

            current_run += 1

            print(
                f"\n Futtatás [{current_run}/{total_runs}]\n Iteráció: {i//step+1} | Manőverek: {selected_maneuvers}"
            )

            # 1 Betöltjük a jelenlegi konfigurációt
            config = ConfigObj(CONFIG_PATH, encoding="utf-8")

            # 2 Frissítjük a kiválasztott manővereket
            config["Data"]["selected_manoeuvres"] = selected_maneuvers
            config["Plot"]["folder_name"] = folder_name

            # 3 Visszaírjuk a módosított konfigurációt
            config.write()

            # 4 Elindítjuk a run.py-t
            print(f"Indítom a run.py-t...")
            process = subprocess.Popen(["python", "run.py"])

            # 5 Megvárjuk a futás végét
            process.wait()
            print(f"run.py futtatás {current_run}/{total_runs} befejeződött!")

            # 6 JSON fájl beolvasása
            try:
                output_file = "tsne_output.json"
                output_file_2 = "bottleneck_output.json"
                with open(output_file, "r") as f:
                    tsne_data = json.load(f)

                latent_data = np.array(tsne_data["latent_data"])
                latent_label = np.array(tsne_data["labels"])

                all_tsne_data.append(latent_data)
                all_tsne_label.append(latent_label)

                with open(output_file_2, "r") as f:
                    bottleneck_output_data = json.load(f)

                bottleneck_data = np.array(bottleneck_output_data["bottleneck_outputs"])
                bottleneck_label = np.array(bottleneck_output_data["labels"])
                label_mapping = bottleneck_output_data["label_mapping"]

                all_bottleneck_outputs.append(bottleneck_data)
                all_bottleneck_labels.append(bottleneck_label)

                print(f"TSNE adatok sikeresen beolvasva: {output_file}")
                print(f"Bottleneck adatok sikeresen beolvasva: {output_file_2}")

            except Exception as e:
                print(f"Hiba történt a TSNE adatok beolvasásakor: {e}")
    else:
        print(f"\n=== {group_idx+1}. Manővercsoport ===")

        # Iterálás minden egyes manőveren külön
        for i, maneuver in enumerate(maneuvers):
            folder_name = f"{folder_names[group_idx]}_allinone"
            current_run += 1
            print(f"\n Futtatás [{current_run}/{total_runs}]\n Manőver: {maneuver}")

            # 1 Betöltjük a jelenlegi konfigurációt
            config = ConfigObj(CONFIG_PATH, encoding="utf-8")

            # 2 Frissítjük a kiválasztott manővert
            config["Data"]["selected_manoeuvres"] = [
                maneuver
            ]  # Egyetlen manőver listában
            config["Plot"]["folder_name"] = f"{maneuver}"

            # 3 Visszaírjuk a módosított konfigurációt
            config.write()

            # 4 Elindítjuk a run.py-t és várunk az eredményre
            print(f"Indítom a run.py-t a(z) {maneuver} manőverrel...")
            process = subprocess.Popen(
                ["python", "run.py"]
            )  # , stdout=subprocess.PIPE, text=True)

            # 5 Megvárjuk a futás végét
            process.wait()
            print(f"run.py futtatás {current_run}/{total_runs} befejeződött!")

            try:
                with open("saliency_output.json", "r") as f:
                    saliency_data = json.load(f)
                    saliency = np.array(saliency_data["saliency"])
                    features = saliency_data["features"]
                    if "all_saliency_values" not in globals():
                        all_saliency_values = []
                    all_saliency_values.append(saliency)
            except Exception as e:
                print(f"❌ Hiba a saliency beolvasásakor: {e}")

            # # 6 JSON fájl beolvasása
            # try:
            #     output_file = "tsne_output.json"
            #     output_file_2 = "bottleneck_output.json"
            #     with open(output_file, "r") as f:
            #         tsne_data = json.load(f)

            #     latent_data = np.array(tsne_data["latent_data"])
            #     latent_label = np.array(tsne_data["labels"])

            #     all_tsne_data.append(latent_data)
            #     all_tsne_label.append(latent_label)

            #     with open(output_file_2, "r") as f:
            #         bottleneck_output_data = json.load(f)

            #     bottleneck_data = np.array(bottleneck_output_data["bottleneck_outputs"])
            #     bottleneck_label = np.array(bottleneck_output_data["labels"])
            #     label_mapping = bottleneck_output_data["label_mapping"]

            #     all_bottleneck_outputs.append(bottleneck_data)
            #     all_bottleneck_labels.append(bottleneck_label)

            #     print(f"TSNE adatok sikeresen beolvasva: {output_file}")
            #     print(f"Bottleneck adatok sikeresen beolvasva: {output_file_2}")

            # except Exception as e:
            #     print(f"Hiba történt a TSNE adatok beolvasásakor: {e}")

        # # Konvertálás listává, mert NumPy tömb nem menthető közvetlenül JSON-ben
        # tsne_data_list = [latent.tolist() for latent in all_tsne_data]
        # tsne_label_list = [label.tolist() for label in all_tsne_label]

        # # Mentés JSON fájlba
        # output_file = f"{folder_names[group_idx]}_tsne_data.json"
        # with open(output_file, "w") as f:
        #     json.dump({"tsne_data": tsne_data_list, "labels": tsne_label_list}, f)

        # print(f"TSNE adatok elmentve: {output_file}")

#     if overlay_multiple_manoeuvres == 1:
#         detect = DetectOverlap(
#             tsne_data_list=all_tsne_data,
#             labels_list=all_tsne_label,
#             folder_name=folder_name,
#         )
#         detect.detect_overlap_by_grid()
#         if num_manoeuvres == 1:
#             print(
#                 f"\nPlotolás indul a(z) {folder_names[group_idx]} manővercsoportra..."
#             )
#             plot_all_tsne_data(all_tsne_data, all_tsne_label, folder_name)

#     if filtering == 1:
#         mf = ManoeuvresFiltering(
#             bottleneck_data=all_bottleneck_outputs,
#             labels=all_bottleneck_labels,
#             label_mapping=label_mapping,
#         )
#         redundant_pairs = mf.filter_by_distance()
#         redundantpairs_list.append(redundant_pairs)

# print("\n=== Futtatások befejeződtek! ===")

# csv_output = "redundantpairs_list.csv"
# with open(csv_output, "w", newline="") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["Run Index", "Redundant Pairs"])  # opcionális fejléc

#     for idx, pairs in enumerate(redundantpairs_list):
#         writer.writerow([idx, pairs])

# print(f"Redundáns párok CSV fájlba mentve: {csv_output}")

if "all_saliency_values" in globals():
    avg_saliency = np.mean(all_saliency_values, axis=0)
    plot_saliency_map(avg_saliency, features)