### Welcome!

To install on windows:
pip install --editable .[cuda]

To install on mac:
pip install --editable ".[mlx]"


If you want to build the package, run the following:
(If built before and you want to rebuild, first remove the dist/ folder)
pip install build      
python -m build

test