def init():
    global _global_dict
    _global_dict = {}


def set_value(key: str, value: ...) -> None:
    _global_dict[key] = value


def get_value(key: str) -> ...:
    try:
        return _global_dict[key]
    except KeyError:
        raise KeyError("KeyError: %s" % key)
