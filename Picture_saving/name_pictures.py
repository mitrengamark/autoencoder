def fig_names(description):
    if "allando_v_sin" in description:
        save_path = f"Results/allando_v_sin/{description}.png"
    elif "allando_v_chirp" in description:
        save_path = f"Results/allando_v_chirp/{description}.png"
    elif "allando_v_savvaltas" in description:
        save_path = f"Results/allando_v_savvaltas/{description}.png"
    elif "valtozo_v_savvaltas_gas" in description:
        save_path = f"Results/valtozo_v_savvaltas_gas/{description}.png"
    elif "valtozo_v_savvaltas_fek" in description:
        save_path = f"Results/valtozo_v_savvaltas_fek/{description}.png"
    elif "valtozo_v_sin_gas" in description:
        save_path = f"Results/valtozo_v_sin_gas/{description}.png"
    elif "valtozo_v_sin_fek" in description:
        save_path = f"Results/valtozo_v_sin_fek/{description}.png"

    return save_path
