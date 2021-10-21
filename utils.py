# -*- coding: utf-8 -*-
from argparse import Namespace


class Config:
    def __init__(self, initial_data: dict) -> None:
        for key in initial_data:
            if hasattr(self, key):
                setattr(self, key, initial_data[key])

    def namespace(self) -> Namespace:
        return Namespace(
            **{
                name: getattr(self, name)
                for name in dir(self)
                if not callable(getattr(self, name)) and not name.startswith("__")
            }
        )
