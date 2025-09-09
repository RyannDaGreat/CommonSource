"""Auto-detecting __init__.py with lazy-loaded modules"""

from pathlib import Path
import lazy_loader

# Auto-detect Python modules in this directory
_current_dir = Path(__file__).parent
__all__=[]

for _file_path in _current_dir.iterdir():
    if (
        _file_path.is_file()
        and _file_path.suffix == ".py"
        and _file_path.name != "__init__.py"
        and not _file_path.name.startswith(".")
    ):
        _module_name = _file_path.stem
        globals()[_module_name] = lazy_loader.load(__name__ + '.' + _module_name)
        __all__.append(_module_name)

