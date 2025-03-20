# HoloForge

This framework can be used to generate large datasets of simulated data. Originally, the framework was intended to be used to generate labelled data to train a deep learning model that shall estimate the background illumination of a forward propagation to improve the reconstruction quality, however, at the moment the labels cannot readily be used for that application. Still, it can be used to generate holograms, where the experimental setup (including the phantom) are randomly constructed. This implies, that this framework can also be used to generate random phantoms.

## Table of Contents

1. [Setup Project](#setup-project)
2. [How to Use this Framework](#how-to-use-this-framework)
3. [Customization](#customization)
4. [Default Configuration](#default-configuration)
5. [Output Structure](#output-structure)
6. [Labelling](#labelling)
7. [Extend the Framework](#extend-the-framework)
   - [Add New Parameters](#add-new-parameters)
   - [Add New Shapes](#add-new-shapes) 
8. [Henke Interface](#henke-interface)
9. [Suggestions for Improvements](#suggestions-for-improvements)
   - [Adapter Pattern for Configuration Files](#adapter-pattern-for-configuration-files)

## Setup Project

To setup the project run the following commands:

### Create environment on Maxwell
```bash
$ module load maxwell mamba
$ . mamba-init
$ mamba create -p <path_to_env> python=3.11 
```

### Activate environment
```bash
$ mamba activate <path_to_env>
```

### Option 1: From pip package
```bash
pip install -e <path_to_project>
```

### Option 2: From pip package
```bash
$ pip install holoforge --index-url https://gitlab.desy.de/api/v4/projects/9756/packages/pypi/simple
```

## How to use this Framework

The entire data generation is encapsulated in the class `DataGenerator` in `holoforge.generators` which offers the method `generate_data`, that can be used to generate data. Another more convenient way is through the command line by running the script

```{bash}
holoforge/generate_data.py
```

The script expects three positional and two optional arguments:

| Argument               | Description                                                      | Position |
|------------------------|------------------------------------------------------------------|----------|
| `num_samples`          | Number of data samples that should be generated.                 | 1        |
| `output`               | Output folder where the generated data is stored.                | 2        |
| `--config CONFIG`      | Path to the custom configuration file.                           | optional |
| `--override`           | Override the output folder if it already exists.                 | optional |


When a custom configuration file path is given, the script first looks in the directory `custom_configs`. Hence, the path to a configuration file can be relative to the folder `custom_configs`. If it was not found there, the relative path to the current working directoy is checked.

Since the command `python holoforge/generate_data.py` is, even without the argument list, quite long, there exists a wrapper `generate_data.sh` in the `project`'s root directory, that passes the parameters to the actual Python script.

As an example: Generate 100 datasamples with a custom configuration for an experimental setup that has an X-ray beam of 11 keV and store them at `/gpfs/petra3/scratch/$USER/output`:

```{bash}
./generate_data.sh 100 /gpfs/petra3/scratch/$USER/output --config custom_configs/test_config.json
```

## Customization

The framework allows for a high degree of flexibility in the data generation process. Both the `DataGenerator` and the `ValueGenerator` can be configured through a configuration file. At the moment, it must be a `json`-file. In the following, such a file will be referred to as *configuration file*, *configuration*, or simply *config*. There exists a configuration containing the default settings that are stored in `holoforge/configs/default.json`, which contains every adjustable parameter and their default values. The default configuration has two objects, `params_data_generator` and `params_value_generator`, that encapsulate the parameters for the `DataGenerator` and `ValueGenerator`, respectively. Generically, such a file looks like this:

```json
{
    "params_data_generator": {
        "<parameter_1>": "value_1",
        ...
        "<parameter_n>": "value_n"
    },
    "params_value_generator": {
        "<parameter_1>": "value_1",
        ...
        "<parameter_m>": "value_m"
    }
}

```

It is pointed out that the default configuration should not be changed and in most cases, it should suffice. Nonetheless, to generate the data with different parameters, custom configuration files come into play. Even though their naming can be arbitrary, conventionally, those should be stored in the folder `custom_configs`. Hence, to customize the parameters, it is not necessary to copy each parameter from the default configuration into the custom configuration. Instead, only the parameters that should be changed must be listed in the custom configuration. Below is an example for a custom config, where two parameters of each generator are edited; for instance, the dataset name is changed to `my_dataset`. It is noteworthy that those are arguably the most changed parameters.

```json
{
    "params_data_generator": {
        "dataset_name": "my_dataset",
        "seed": 42
    },
    "params_value_generator": {
        "materials": ["Mg"],
        "A0_range": [0.9, 1.1]
    }
}

```

As it might suggest, the naming of all parameters in a custom configuration must coincide with the naming in the default configuration; meaning, that on the one hand, the key params_data_generator must be present if the `DataGenerator`'s parameters should be changed (analogously for `params_value_generator`) and on the other hand, the parameter's name must be both present and coincide with the name in the default configuration. It is remarked that the parameter's types need to be considered. The path to a custom configuration can be passed as a command-line argument using the `--config` option (see CLI arguments for the script `holoforge/generate_data.py`). The configuration file is read by the `ConfigParser` in the module `holoforge.utils.config_parser`.

## Default Configuration

The default configuration is stored in `holoforge/configs/default.json`.

### Parameters for `DataGenerator`

- `format` (default: `"tiff"`, dtype: `str`): Dataformat of the saved holograms.
- `dataset_name` (default: `"train"`, dtype: `str`): Name of the dataset.
- `downsample_dim` (default: `null`, dtype: `int`): If given, the hologram is downsampled to this size. Note: This must be smaller than or equal to `hologram_size` from the `ValueGenerator` options.
- `store_phantoms` (default: `true`, dtype: `bool`): If True, store the phantom that was used in the forward propagation when simulating a hologram.
- `subfolder_holograms` (default: `"holograms"`, dtype: `str`): Name of the subfolder, where the holograms are stored.
- `subfolder_phantoms` (default: `"phantoms"`, dtype: `str`): Name of the subfolder, where the phantoms are stored.
- `label_filename` (default: `"labels.csv"`, dtype: `str`): Name of the file, in which the labels are stored. Note: It is always a csv file.
- `metadata_filename` (default: `"metadata.csv"`, dtype: `str`): Name of the file, in which the metadata are stored. Note: It is always a csv file.
- `save_point` (default: `10`, dtype: `int`): In the generating loop, each `save_point` iterations, the labels for the already generated samples are stored as fallback. If the program crashes while genering the data, the labels-file and metadata-files are not closed properly. This option caps the amount of unusable samples.
- `seed` (default: `null`, dtype: `float`): Seed for the RNGs.

The `DataGenerator` has two specical values, where the value is another object, namely `header_labels` and `header_metadata`, that correspond to the `labels.csv` and `metadata.csv`. The keys correspond to the columns and their values name that column. The intention behind those parameters is that the user does not need to know the column names, when reading in the labels or the metadata, but can access the column names through the stored configuration file. For the csv-file to have no header, the values must represent indices of the columns, i.e. $0, 1, ..., N$. Note: The indices must be in increasing order.


```json
"header_labels": {
    "holo_path": "holo_path",
    "a0": "a0",
    "mx": "mx",
    "my": "my",
    "sqx": "sqx",
    "sqy": "sqy",
    "phantom_path": "phantom_path"
    }
```

```json
"header_metadata": {
    "holo_path": "holo_path",
    "phantom_path": "phantom_path",
    "z01": "z01",
    "z02": "z02",
    "energy": "energy",
    "Fr": "Fr",
    "det_px_size": "det_px_size"
    }
```


### Parameters for `ValueGenerator`

The `ValueGenerator` is responsible for sampling random values to create random experiments for each simulation. A huge focus lies on the generation of the phantoms. A phantom is constructed, by stacking smaller, simple phantoms.

- `materials` (default: `["Mg", "Cu", "Fe", "Ag", "Au"`, dtype: `[str]`): List of materials (chemical formulas), the phantoms can be made of.
- `A0_range` (default: `[1, 1]`, dtype: `(int, int)`): Constant offset of the background illumination.
- `beam_center_std` (default: `50`, dtype: `float`): When creating a beam, the origin is in the center of the beam of the array. This option impacts the actual origin. If it is > 0 the origin probably is not in the beam's center. 
- `beam_decimals` (default: `5`, dtype: `int`): Number of decimals the sampled values for the beam are rounded to.
- `beam_size_px` (default: `4096`, dtype: `int`): Size of the beam (square).
- `beam_linear_max` (default: `0`, dtype: `float`): Add a linear component to the beam. Note:  It is not advised to have `beam_linear_max` to be greater than 2e-4.
- `beam_square_max` (default: `0`, dtype: `float`): Add a squared component to the beam. Note: It is not advised to have `beam_square_max` to be greater than 2e-7.
- `relative_noise` (default: `true`, dtype: `bool`): If `true`, the noise added to the beam, is relative to its `a_0` value.
- `z01_range` (default: `[80, 80]`, dtype: `(int, int)`): Minimum and maximum distance in from focal stop at $z=0m$ to object at $z=z_{01}$ in cm.
- `z02_range` (default: `[20, 20]`, dtype: `(int, int)`): Minimum and maximum distance in from focal stop at $z=0m$ to detector at $z=z_{02}$ in m.
- `thickness_range` (default: `[1, 20]`, dtype: `(int, int)`): Minimum and maximum thickness of a simple phantom in µm.
- `object_placement`: Dictionary, containing settings about the placement of the shapes/objects for constructing a phantom.
  - `positioning` (default: `center`, dtype: `str`): The way on how to place the objects on the "canvas". Either `center` or `fov`/`detector`. If it is set to `center`, the objects are placed on and around the center of the "canvas". If set to `fov` (or `detector`; name alias), the objects are placed randomly on the "canvas".
  - `num_objects_range` (default: `[4, 8]`, dtype: `(int, int)`): Minimum and maximum number of simple phantoms for a phantom.
- `noise_beam_active`: Activate additive noise for the illumination, before propagation. (Not implemented yet, issue 20)
- `noise_beam_config`: Activate additive noise for the illumination, before propagation. (Not implemented yet, issue 20)
- `noise_hologram_active`: Activate additive noise for the hologram.
- `noise_hologram_config`: Dictionary, containing the configuration for the hologram noise (see below).
  - `type` (default: `poisson`, dtype: `str`): Type of noise, can be `gaussian` or `poisson`.
  - `intensity` (default: `0.1`, dtype: `float`): Noise intensity.
  - `poisson_lambda` (default: `1.0`, dtype: `float`): Lambda value for Poisson distributed noise. 
  - `gaussian_sigma` (default: `1.0`, dtype: `float`): Standard deviation for Gaussian distributed noise. 
- `shapes_config`: Dictionary containing configurations for the shapes (see below).
    - `shapes` (default: `["rectangle", "polygon", "ellipse", "ball", "cylinder", "cylinder_round_tip"]`, dtype: `List[str]`): List of possible shapes of the objects.
    - `size_range` (default: `[1, 1024]`, dtype: `(int, int)`): Minimum and maximum size of shapes in pixel.
    - `radius_range` (default: `[1, 512]`, dtype: `(int, int)`): Minimum and maximum radius of circular shapes in pixel.
    - `polygon_max_corners` (default: `4`, dtype: `int`): Maximum number of corners of a polygon. Note: A min. of 3 corners is required.
    - `rotate` (default: `true`, dtype: `bool`): If True, simple phantoms are rotated after creation to induce more diversity.
    - `ellipse_max_cut` (default: `0`, dtype: `int`): Only applies for elliptical shapes. Relative amount an ellipse is cut at one side, e.g. an ellipse with radius $r_1 = 100$ and $r_2 = 100$ and and `ellipse_max_cut = 0.1`. Then each side of the ellipse can be truncated of at most 10 pixel. Note: The value must be $\in [0,1)$.
- `holo_size` (default: `2048`, dtype: `int`): Size of the generated hologram (square).
- `phantom_size` (default: `2048`, dtype: `int`): Size of the phantom (square).
- `det_px_size` (default: `6500`, dtype: `int`): Detector pixel size. Not to be confused with the image pixels, e.g. of a hologram.


## Output Structure

The output of the data generation will be written to the specified location. At least when working on the DESY HPC-cluster *Maxwell*, it is recommended to use `/gpfs/petra3/scratch/$USER/output` as the output's root directory. Regardless of the actual output path, in the following, `output` will be used as its alias. The `DataGenerator` creates a directory tree with the following output structure:

```
output/data
└── 11000
└── train
├── holo_path
│ ├── hologram_000000.tiff
│ ├── \vdots
│ └── hologram_000099.tiff
├── phantom_path
│ ├── phantom_000000.pkl
│ ├── \vdots
│ └── phantom_000099.pkl
├── config.json
├── labels.csv
└── metadata.csv
```

Even though most of the folder names and file names can be specified in the configuration file, the folder name `data` is hardcoded, thus, cannot be influenced. The dataset is called `train`. So, each dataset is stored at `output/data/<energy>/<dataset_name>`. As the names suggest, the generated holograms and phantoms are stored in `holo_path` and `phantom_path`, respectively. The configuration used for this dataset is saved in `config.json`. Each hologram is computed using a different experimental setup constructed by the `ValueGenerator`, which includes, inter alia, the distances $z_{01}$ and $z_{02}$, that are needed for the reconstruction of the original object from the hologram. In `metadata.csv` this kind of information is preserved for every sample for later use.

## Labelling

Since this framework's main purpose is to create data for the training of a deep learning model, that will estimate the background illumination, it is crucial to store its parameters for the corresponding hologram, i.e. the *label*. Apart from the parameters, both the filename of the phantom and the filename of the resulting hologram are recorded in the aforementioned `labels.csv`.

Even though this framework already allows for more complex X-ray beams, the example below shows the labels for a constant valued beam. The header and the first five labels are listed.

```{csv}
holo_path,a0,mx,my,sqx,sqy,phantom_path
hologram_000000.tiff,0.93467,0.0,0.0,0.0,0.0,phantom_000000.pkl
hologram_000001.tiff,0.97117,0.0,0.0,0.0,0.0,phantom_000001.pkl
hologram_000002.tiff,1.07984,0.0,0.0,0.0,0.0,phantom_000002.pkl
hologram_000003.tiff,1.05493,0.0,0.0,0.0,0.0,phantom_000003.pkl
hologram_000004.tiff,1.08105,0.0,0.0,0.0,0.0,phantom_000004.pkl
```

## Extend the Framework

### Add new Parameters
Adding new parameters to the configuration is a great way to make the data generation process more flexible. In order to accomplish this, the framework needs to be updated at two locations. Obviously, the new parameter has to be added to the default configuration `configs/default.json`. Assume, we want to add a new parameter corresponding to the `DataGenerator`, then this parameter has to be included in the corresponding object in the configuration file, i.e. to `params_data_generator`. The `ConfigParser` has a method `_check_config`, which asserts the type of the parameters, and even though it is not mandatory to do so, it is recommended to contribute to the list of assertions and validate the properties of the new field. Now, the new parameter can be used in the code. The `ValueGenerator` is updated similarly.

In short:
1. Add new parameter to the default config
2. Add a type check to `_check_config` in the `ConfigParser`

### Add new Shapes
Clearly, it is desirable to generate holograms with objects that are real-world-like. By the composition of simple phantoms to a phantom, we hope to not just add randomness into the data generation, but also to mimic more complex objects. Consequently, it would be advantageous to define new shapes.

In order to add a shape to the framework, the code must be adjusted at four locations. Since every shape is implemented as their own class, that all inherit from `Shape`; one module each in the package `objects.shapes`. Now assume, we want to add an imperfect and more realistic shape, e.g. by having noisy edges. The corresponding class has to create a `torch.Tensor` for the shape S, which is then passed to the constructor of its parent. Note, the array does not necessarily need to be standardized already, since it will be standardized in the constructor of `Shape`. After defining the new shape, it needs a builder, which creates a random version of the shape. The shape-builders are implemented in the module `objects.shape_builder`. Each builder has the name `<shape-name>Builder`, e.g. `CylinderRoundTipBuilder` for the shape `CylinderRoundTip`. The module `objects.shape_sampler` implements a dictionary `SHAPE_BUILDER_DICT`, in which all existing builders are listed. After defining the new shape and corresponding builder, the builder needs to be registered in `SHAPE_BUILDER_DICT`. The key should be the shape's class name in snake_case, e.g. `cylinder_round_tip`. In the last step, the default config must be updated. The field `params_value_generator.shapes_config.shapes` must be extended by the new shape's class name in snake_case (same name as in `SHAPE_BUILDER_DICT`).

In short:
1. Define the new shape (e.g. `NewShape`)
2. Define a builder (e.g. `NewShapeBuilder`)
3. Register the builder for the `ShapeSampler` (e.g. add `"new_shape": NewShapeBuilder`)
4. Update the default config (e.g. add `"new_shape"` to `params_value_generator.shapes_config.shapes`)

## Henke Interface

The framework offers an interface to automatically crawl the Henke-website, which is implemented as the the class `HenkeInterface` in the module `utils.material_properties.henke_interface`. It utilizes the third-party library `selenium` to automatically run a query, i.e. to either fetch $(\delta, \beta)$ or the transmission $t$. Such a query takes several seconds. For instance, in order to create a single phantom that is composed of five simple phantoms, ten queries need to be started, just to fetch all physical properties. This actually becomes the bottleneck for more complex phantoms. For this reason, the interface `utils.material_properties.property_db_builder` offers routines to build local databases. A database has two subfolders, one for the $\delta$ and $\beta$ values, called `delta_beta` and one for the transmissions, called `transmission`. For each (queried) chemical element, e.g. $Mg$, there is corresponding CSV-file named `<subfolder>/<formula>-<density>.csv`. Examples for a database file containing the ($\delta$, $\beta$)-values and the transmissions are given in \refListing{code:local_database:delta_beta} and \refListing{code:local_database:transmission}, respectively.


## Suggestions for Improvements

### Adapter Pattern for Configuration Files

Currently (04/2024), a configuration File must be `json`-file. However, in the future a config file should be generic. In the implementation an adapter pattern should be used, where there exists a Parser for each accepted format. The `DataGenerator` and the `ValueGenerator` then use an adapter independent of the file format of the config. Also, the variable names `json_path`, `json_config`, etc. are used, they must be renamed to be generic, e.g. `config_path`.

