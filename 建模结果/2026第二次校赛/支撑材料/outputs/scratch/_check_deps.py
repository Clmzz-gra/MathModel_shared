# 检查批次校正库可用性（一次性）
import importlib

for m in ['numpy', 'pandas', 'sklearn', 'scipy', 'matplotlib']:
    try:
        mod = importlib.import_module(m)
        print(f'{m}: OK {getattr(mod, "__version__", "?")}')
    except ImportError:
        print(f'{m}: MISSING')

for m in ['pycombat', 'combat', 'harmonypy', 'sva', 'statsmodels']:
    try:
        importlib.import_module(m)
        print(f'{m}: INSTALLED')
    except ImportError:
        print(f'{m}: not installed')
