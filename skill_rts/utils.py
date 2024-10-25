def parse_task(text: str) -> list:
    from skill_rts import logger
    import ast
    import re

    task_list = []
    params_list = []
    text = text.split("START OF TASK")[1].split("END OF TASK")[0]
    text_list = text.split("\n")
    for task_with_params in text_list:
        task_beg = task_with_params.find("[")
        task_end = task_with_params.find("]")
        param_beg = task_with_params.find("(")
        param_end = task_with_params.rfind(")")
        if task_beg + 1 and task_end + 1:
            task = task_with_params[task_beg : task_end + 1]
        else:
            task = None
        params = re.sub(
            r"(?<!\')(\b[a-zA-Z_]+\b)(?!\')",
            r"'\1'",
            task_with_params[param_beg : param_end + 1],
        )
        params = re.sub(r"'(\d+)'", r"\1", params)
        if param_beg + 1 and param_end + 1:
            params = ast.literal_eval(params)
            if task is not None:
                task_list.append(task)
                params_list.append(params)
    logger.info("Parsed Tasks from LLM's Respond:")
    for task, params in zip(task_list, params_list):
        logger.info(f"{task}{params}")
    return list(zip(task_list, params_list))