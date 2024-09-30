# synoptic-SOM
[![DOI](https://zenodo.org/badge/500991505.svg)](https://doi.org/10.5281/zenodo.13864055)

## Overview
This reposity contains a python implementation of the Self-Organizing Map (Kohonen 1990) for applications in synoptic climatology, developed for the 
paper "Investigating Climate Drivers of Lightning in Alaska with a SOM-RF". Also included are the weights of the sea level pressure (SLP) and 500hPa 
geopoential height (Z) SOM nodes, and a csv table of the daily synoptic types derived from the SOM for June-July during the 2004-2022 period.

## Computational Requirements

### Software Requirments
This library is entirely contained within Python and CUDA (implicitly via the cupy library). The environment to execute 
the synoptic-SOM can be set up in conda with the provided requirements.txt. Additionally, some helper functions are included
in the geospatial_utils.py file.

### Memory and Runtime
This toolkit was last run on 9-30-2024 on a Windows 11 PC with an AMD Ryzen 9 3900x, 32GB of RAM, and an RTX2080. Time to train a 4x5 SOM on daily pressure fields for June-July of the 2004-2022 period is about 1 hour. Current implementation requires an Nvidia GPU with CUDA enabled.

### Example
See "Training SOMs.ipynb" for an example of a scripted application of this toolkit.

## License
MIT License

Copyright (c) 2024 Joshua Hostler

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## References
Alaska Interagency Coordination Center. (n.d.). Predictive services - maps/imagery/geospatial. AICC - Predictive Services - maps/imagery/geospatial. https://fire.ak.blm.gov/predsvcs/maps.php

Kohonen, T., 1990: The self-organizing map. Proceedings of the IEEE, 78, 1464–1480, https://doi.org/10.1109/5.58325.

## Acknowledgments
This material is supported by the National Science Foundation under award #OIA-1757348 and by the State of Alaska
