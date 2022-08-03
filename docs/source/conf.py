# Configuration file for the Sphinx documentation builder.
#
import sys, os 
sys.path.insert(0, os.path.abspath('./../../StatOD/'))
print(os.path.abspath('./../../'))

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'StatOD'
copyright = '2022, John M'
author = 'John M'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages', # Produces .nojekyll file for GH pages
    'sphinx_gallery.gen_gallery',
    ]
templates_path = ['_templates']
exclude_patterns = []


sphinx_gallery_conf = {
    'examples_dirs':[ '../../Examples', '../../Tutorials'],   # path to your example scripts
    'gallery_dirs': ['auto_examples', 'tutorials'],  # path to where to save gallery generated output,
    'run_stale_examples': True, # Change to True when working on the underlying library. Forces examples to be rerun
    'line_numbers': True,
    'filename_pattern': '/', # Execute all of the examples
    # 'sphinx_gallery_thumbnail_number' : -1,
}



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"

html_static_path = ['_static']
