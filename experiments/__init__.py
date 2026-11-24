from importlib import import_module


def get_experiment(name):
    return import_module(f"experiments.{name}")
