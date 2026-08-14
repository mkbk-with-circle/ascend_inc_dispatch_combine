# Example of SHMEM Operator Integration with PyTorch
This directory provides an example of integrating selected SHMEM operators with PyTorch. This example is intended for demonstration purposes only and are not recommended for production environments.
## Build and Run
Run the following commands in the root directory of SHMEM:
```sh
# Build example operators and their PyTorch extensions.
bash scripts/build.sh -python_example
source install/set_env.sh
cd examples/python_extension/torch_test
python xxx.py # Run the 8-device example by default.
python xxx.py --pes 2 # Run the 2-device example (--pes specifies the number of devices).
```
Note: Replace xxx with an actual file name.
## Parameter Description
### --tool
- `--tool 0`: Run directly without using the performance profiling tool. (This is the default value.)
- `--tool 1`: The msprof profiling tool is used.

**Note**: A memory check tool (for example, msSanitizer) and a profiling tool (for example, msprof) cannot be used simultaneously.

### --pes
- `--pes <N>`: the number of devices to be used. For example, `--pes 2` indicates that two devices are to be used.

## Memory Check Tool Usage
To use the memory check tool (msSanitizer), perform the following operations:
1. **Add build options during the build**:
   ```sh
   bash scripts/build.sh -python_example -mssanitizer
   ```

2. **Run the example using msSanitizer**:
   ```sh
   mssanitizer -- python xxx.py --pes 2
   ```
   Note: After adding the build options, you cannot directly run the example via `python xx.py`. You must run it via `mssanitizer -- python xx.py`.
