# Quick start
## 1. Environment Setup
Setup the environment by running 

```bash
conda env create -f env.yml
```

If you have problem running this command, just run

```bash
conda create -n nn4syspec python==3.11
```

Then install the missing packages when running the following scripts.

## 2. Specification Generation
Note: You can replace the `python3` with `python` in the following command if needed.

To generate specifications for ABR, run:
```bash
python3 src/optim_specs_abr.py
```

To generate specifications for CC, run:
```bash
python3 src/optim_specs_cc.py
```

Both the above scripts take a set of optional parameters. To see the list of parameters, run:
```bash
python3 src/optim_specs_abr.py --help
```

To get the support and confidence values for the generated specifications, run (```<spec_file>``` is path of where the specifications are stored):

For ABR:
```bash
python3 src/test_optim_specs_abr.py <spec_file>
```

For CC:
```bash
python3 src/test_optim_specs_cc.py <spec_file>
```

To verify specifications, convert them to VNN-LIB format using the following command:
For ABR:
```bash
python3 src/specs_2_vnnlib_abr.py <name_spec_file>
```

For CC:
```bash
python3 src/specs_2_vnnlib_cc.py <name_spec_file>
```

To verify the specifications, install alpha-beta CROWN https://github.com/Verified-Intelligence/alpha-beta-CROWN, and run the script:
For ABR:
```bash
python3 src/run_abcrown_abr.py <model_name> --abcrown <alpha-beta-crown path>
```

For CC:
```bash
python3 src/run_abcrown_cc.py <model_name> --abcrown <alpha-beta-crown path>
```

We provide the network traces and training files for our Aurora implementation in ```Aurora/```.
