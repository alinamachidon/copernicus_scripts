# Sentinel DL

## Env

```bash
conda create -n sentinel_env python=3.10
conda activate sentinel_env

pip install -r requirements
```

## SentinelHub config

In order to download data from SentinelHub, you need to setup the API credentials used with `sentinelhub` library.

Use the provided `sh_setup.py`, which saves this into `sentinelhub` library config file under `sentinel-dl` profile. For more information, refer to [configuration docs](https://sentinelhub-py.readthedocs.io/en/latest/configure.html).

Use the script with the following params:
```bash
python sh_setup.py --id <insert ClientID here> --secret <insert ClinetSecret here>
```

This config will be later used when you use the `eolearn` functionalities to download the data.

## Data downloading

To download data for the entire Slovenian region, run `main.py`:

```bash
python main.py --date 29.10.2024
```

This will download Sentinel2 L1C patches with a single temporal dimension and all 13 bands for the entire country of Slovenia
Patches will be saved as EOPatch inside `./patches/<date>`. The report of execution will be saved inside `./log`.

With default resolution and patch size, there is about 900 patches, totaling in about 5GB for a single run.

---

Sentinel makes a recording every 5 days but due to overlap some regions of Slovenia make 2 recordings in 5 days. 
To get a single best image for each patch, the code assembles a **single image** from tiles recorded in last month ending on a given `date`.
With default settings, this is done by taking the least cloudy sample and mosaicking that into a single image.
The `date` is also the only required argument.

Use `python main.py -h` to get help on other arguments: [date, save_dir, resolution, patch_size, maxcc, mosaicking_order, num_workers].

## Simple visualisation

To visualise the downloaded data, you can use the provided `visualise.ipynb` which visualises a region around Ljubljana.

## Resolution (resolution), resolution (size) and resolution (bbox size).
Sounds as confusing as it is...

In EOlearn API, `resolution` refers to sentinel sampling resolution in meters (default 10m). 
Size refers to resolution of patch in pixels (default 512). 
Then bbox also contains "edge size" in meters of earth covered (default is resolution * patch_size -> 5120 x 5120 m).

It's important to be careful if you are changing the defaults.

Side note (or warning): `SentinelHubInputTask` can't take simultaneous `size` and `resolution`, 
so in order to get let's say region of 2560m with resolution of 10m we need bbox with 2560m side and pass resolution=10. 

The problem is if you want to have a region of 5120m with resolution of 20m and patch of 512x512pixels, 
because this will default to 256x256 (as that is the size if you sample 5120m with spatial resolution of 20m).

Now one can try this: specify 5120m sided bbox and set size to 512px (but you can't also specify resolution). 
This then yields 512x512 res just fine, but it doesn't say which sampling resolution it uses (probably 10). 

> This is not a problem with default options: resolution of 10m, and patch size of 512. Each patch is therefore 512x512 and covers 5120 x 5120 m.

## Developing

### Format and code style

Use ruff formater and Google docstring format for development.

To format run:
```bash
ruff format .
```

### Docs and guides

Some nice guides are available here:
[eolearn sentinelhub io](https://github.com/sentinel-hub/eo-learn/blob/master/examples/io/SentinelHubIO.ipynb).
[eolearn landcover](https://github.com/sentinel-hub/eo-learn/blob/master/examples/land-cover-map/SI_LULC_pipeline.ipynb).

also useful are [eoleran docs](https://eo-learn.readthedocs.io/en/latest/) and [sentinelhub docs](https://docs.sentinel-hub.com/api/latest/).

### Experimental

There is also some experimental code inside `experimental.ipynb`, `resolution_exp.ipynb` and `time.ipynb`.