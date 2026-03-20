![Welly banner](https://www.dropbox.com/s/a8jg7zomi4wgolb/welly_banner.png?raw=1)

[![Run tests](https://github.com/agilescientific/welly/actions/workflows/run-tests.yml/badge.svg)](https://github.com/agilescientific/welly/actions/workflows/run-tests.yml)
[![Build docs](https://github.com/agilescientific/welly/actions/workflows/build-docs.yml/badge.svg)](https://github.com/agilescientific/welly/actions/workflows/build-docs.yml)
[![PyPI version](https://img.shields.io/pypi/v/welly.svg)](https://pypi.python.org/pypi/welly/)
[![PyPI versions](https://img.shields.io/pypi/pyversions/welly.svg)](https://pypi.org/project/welly//)
[![PyPI license](https://img.shields.io/pypi/l/welly.svg)](https://pypi.org/project/welly/)

**`welly` facilitates the loading, processing, and analysis of subsurface wells and well data, such as striplogs, formation tops, well log curves, and synthetic seismograms.**


## Installation

    pip install welly

For DLIS file support, install with the optional dependency:

    pip install welly[dlis]

For developers, there are `pip` options for installing `test`, `docs` or `dev` (docs plus test) dependencies.


## Quick start

```python
from welly import Well, Project

w = Well.from_las('my_wells/my_well.las')  # Load a single well from LAS.
w = Well.from_dlis('my_wells/my_well.dlis')  # Load from DLIS (requires dlisio).
p = Project.from_las('my_wells/*.las')     # Load lots of wells.

gr = w.data['GR']  # One log...
gr.plot()          # ...with some superpowers!
```

For borehole image data (FMI, UBI, etc.):

```python
import welly

images = welly.load_images('fmi_data.dlis')  # Load 2D image data
fmi = images['FMI_DYN']
fmi.plot()                                    # Quick visualization
fmi.to_pdf('fmi_output.pdf')                  # Multi-page PDF export

# De-rotate to align tool pads vertically
if fmi.can_derotate:
    fmi_derot = fmi.derotate()
    fmi.plot_with_derotated()                 # Side-by-side comparison
```

For petrophysical calculations:

```python
from welly import petro

vsh = petro.vshale_larionov(gr, gr_clean=30, gr_shale=120)
phie = petro.porosity_effective(phid, vsh, phi_shale=0.05)
sw = petro.archie(rt, phie, rw=0.04, a=1, m=2, n=2)
perm = petro.perm_timur(phie, sw)
pay = petro.net_pay_flag(vsh, phie, sw)
```

Next, check out the tutorial notebooks.


## Documentation

[The `welly` documentation](https://code.agilescientific.com/welly) is a work in progress.


## Questions or suggestions?

[![mattermost](https://img.shields.io/badge/mattermost-software_underground-green)](https://mattermost.softwareunderground.org/)

**If you'd like to chat about `welly` with us or other users, look for the **#welly-and-lasio** channel in the [Software Underground's Mattermost](https://mattermost.softwareunderground.org/).**

To report bugs or suggest new features/improvements to the code, please [open an issue](https://github.com/agilescientific/welly/issues).


## Contributing

Please see [`CONTRIBUTING.md`](CONTRIBUTING.md).


## Philosophy

The [`lasio`](https://github.com/kinverarity1/lasio) project provides a very nice way to read and write [CWLS](http://www.cwls.org/) Log ASCII Standard files. The result is an object that contains all the LAS data — it's more or less analogous to the LAS file.

Sometimes we want a higher-level object, for example to contain methods that have nothing to do with LAS files. We may want to handle other well data, such as deviation surveys, tops (aka picks), engineering data, striplogs, synthetics, and so on. This is where `welly` comes in.

`welly` uses `lasio` for data I/O, but hides much of it from the user. We recommend you look at both projects before deciding if you need the 'well-level' functionality that `welly` provides.


## Supported File Formats

- **LAS** (Log ASCII Standard) - via `lasio`
- **DLIS** (Digital Log Interchange Standard) - via `dlisio` (optional, install with `pip install welly[dlis]`)
  - 1D curves via `Well.from_dlis()`
  - 2D borehole images (FMI, UBI, etc.) via `welly.load_images()`
  - Image de-rotation via `ImageCurve.derotate()` (requires orientation curve)
  - Memory-efficient single-channel loading via `welly.load_single_image()`


## Petrophysics

The `welly.petro` module provides petrophysical calculations:

- Shale volume (linear, Larionov, Steiber, Clavier, SP, neutron-density)
- Porosity (density, neutron, sonic Wyllie/Raymer, neutron-density combination, effective)
- Water saturation (Archie, Simandoux, Indonesia, Fertl, Waxman-Smits, Dual-Water)
- Permeability (Timur, Coates, Tixier, Morris-Biggs, Wyllie-Rose, Kozeny-Carman)
- Net pay flagging and summaries
- Fluid contact detection (OWC, GOC, FWL, gradient intersection)
- `PetroInterpreter` for full interpretation workflows with alias support
