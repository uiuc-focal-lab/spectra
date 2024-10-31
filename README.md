Setup the environment by running 

```conda env create -f env.yml```

To generate specifications for ABR, run:
```python3 src/optim_specs_abr.py```

To generate specifications for CC, run:
```python3 src/optim_specs_cc.py```

Both the above scripts take a set of optional parameters. To see the list of parameters, run:
```python3 src/optim_specs_abr.py --help```

To get the support and confidence values for the generated specifications for ABR, run:
```python3 src/test_optim_specs_abr.py <spec_file>```

To get the support and confidence values for the generated specifications for CC, run:
```python3 src/test_optim_specs_cc.py <spec_file>```

To verify specifications, convert them to VNN-LIB format using the following command:
For ABR:
```python3 src/specs_2_vnnlib_abr.py <name_spec_file>```

For CC:
```python3 src/specs_2_vnnlib_cc.py <name_spec_file>```

To verify the specifications, install alpha-beta CROWN https://github.com/Verified-Intelligence/alpha-beta-CROWN, and run the script:
For ABR:
```python3 src/run_abcrown_abr.py <model_name>```

For CC:
```python3 src/run_abcrown_cc.py <model_name>```

We provide the network traces and training files for our Aurora implementation in ```Aurora/```.