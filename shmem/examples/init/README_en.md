### Instructions

#### 1. Building a Project

**To run non-default workflows, you must manually install MPI and import the corresponding environment variables.** [Installation Reference](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/hccltool/HCCLpertest_16_0002.html)
This test case is implemented based on MPICH. If an error is reported during the installation of Open MPI or any other MPI, you may need to adjust the MPI-related parameters in the script.

```bash
# MPICH is installed in the default path. Replace the path with the actual path as needed.
export PATH=/usr/local/mpich/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/mpich/lib:$LD_LIBRARY_PATH
```

Run the `build.sh` script in the root (`shmem/`) directory.

```bash
bash scripts/build.sh
source install/set_env.sh
```

##### 2. Running the init Case

Case directory
```bash
cd examples/init
```

The case accepts two parameters. The first parameter specifies the workflow, and the second parameter specifies the number of PEs (which cannot exceed the number of devices). If not specified, `mode = default` and `pesize = 2` are used by default.
Execute the default workflow with 2 PEs.
```bash
bash run.sh -mode default -pesize 2
```

Execute the MPI workflow with 2 PEs.
```bash
bash run.sh -mode mpi -pesize 2
```

Execute the UID workflow with 2 PEs.
```bash
bash run.sh -mode uid -pesize 2
```

Execute the uid_multi workflow with 4 PEs.
```bash
# Replace enpxxx with the NIC obtained by running the ip addr command.
export SHMEM_UID_SOCK_IFNAME=enpxxxxxxx:inet4

bash run.sh -mode uid_multi -pesize 4

unset SHMEM_UID_SOCK_IFNAME
```

Execute the uid_default workflow with 2 PEs.
```bash
bash run.sh -mode uid_default -pesize 2
```

##### 3. Cross-Server Running
Execute the default workflow on two servers, with 2 PEs on each server. Run the following commands on both servers:
```bash
# Server where PE0 is located
bash run.sh -mode default -pesize 4 -fpe 0 -gnpus 2 -ipport ${IP address:port number of the server}
# Any other server
bash run.sh -mode default -pesize 4 -fpe 2 -gnpus 2 -ipport ${IP address:port number of the server where PE0 is located}
```

Execute the MPI/UID workflow on two servers, with 2 PEs on each server.
The two workflows depend on the MPI capability. For cross-server execution, you need to configure the hostfile file. The following uses the MPICH configuration file as an example.
```sh
# Replace the IP addresses with the actual ones and ensure that the server where PE0 is located is placed at the very top. After the colon (:), configure the number of PEs that can be started on each server. The number of PEs must be less than the number of devices. It is recommended that the number of PEs started on each server be the same.
0.0.0.1:2
0.0.0.2:2
```
**mpirun requires that the locations of the executable files on multiple servers be the same. Compile the samples on all servers in the same mode first. If the modes are inconsistent, an error will be reported or the system will be suspended.**
```sh
bash run.sh -build -mode mpi # For the UID workflow, change the mode parameter value to uid.
```
Then, run the following commands on PE0:
MPI workflow
```bash
# The gnpu parameter specifies the number of PEs on each server. When the number of devices started on all servers is the same, set this parameter to the number of PEs on a single server to ensure that execution always starts from device 0 on every server.
bash run.sh -mode mpi -gnpus 2
```
UID workflow
```bash
bash run.sh -mode uid -gnpus 2 -ipport ${IP address:port number of the server where PE0 is located}
```
**If you want to execute a single-device test case again in MPI or UID mode, you are advised to delete the hostfile file.**
