from setuptools import setup, find_packages

setup(
  name = 'tranception-pytorch',
  packages = find_packages(exclude=[]),
  version = '0.0.3',
  license='MIT',
  description = 'Tranception - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  long_description_content_type = 'text/markdown',
  url = 'https://github.com/lucidrains/tranception-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'attention mechanism',
    'protein fitness'
  ],
  install_requires=[
    'einops>=0.4',
    'einops-exts',
    'torch>=1.6',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
