import neptune


def init_neptune(project_name, api_token, parameters=None):
    """
    Inicializálja a Neptune.ai run-t és beállítja az alap paramétereket.

    :param project_name: Neptune projekt neve (pl. "workspace/project").
    :param api_token: Neptune API token.
    :param parameters: Opcionális szótár, amely tartalmazza a futtatás paramétereit.
    :return: Neptune run objektum.
    """
    run = neptune.init_run(
        project=project_name,
        api_token=api_token,
    )
    if parameters:
        run["parameters"] = parameters
    return run
