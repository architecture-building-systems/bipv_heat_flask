DISPLAY_TYPE_DICT = {
        "spec": "BIPV-Special",
        "reg": "BIPV",
        "dark": "Dark",
        "hpl": "HPL"
    }

def get_idx_code(exp_code):
    return exp_code.split("_")[0].split("-")[1]

def name_experiment_type(exp_code):
    raw_type = exp_code.split("_")[1]
    return DISPLAY_TYPE_DICT.get(raw_type.lower(), raw_type)

def map_type_to_display_type(raw_type):
    """Map raw experiment type to display type for filtering"""
    return DISPLAY_TYPE_DICT.get(raw_type.lower(), raw_type)

def make_display_name(exp_code):
    return f"{get_idx_code(exp_code)}-{name_experiment_type(exp_code)}"