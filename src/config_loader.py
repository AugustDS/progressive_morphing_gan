import ast
import configparser


class Config:
    """
    Helper class for loading configs from .ini file
    """

    def __init__(self, path, section= 'DEFAULT'):
        self.cfg = configparser.ConfigParser()
        self.cfg.read(path)
        self.section = section if section is not None else "DEFAULT"

    def get(self, option, section=None, star=None, raw=False, vars=None,
            fallback=configparser._UNSET):
        section = section if section is not None else self.section
        value = self.cfg.get(section, option, raw=raw, vars=vars,
                             fallback=fallback)
        if value == "" or value is None:
            print("Warning. Value is invalid")
            return None
        try:
            return ast.literal_eval(value)
        except Exception:
            return value
