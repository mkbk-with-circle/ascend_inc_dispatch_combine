# Security Statements

## Security Hardening
### Security Hardening Precautions

The security hardening measures listed in this document are basic security hardening suggestions. You need to review the network security hardening measures of the entire system based on your services. You should perform related configurations according to the security policies of your organization, including but not limited to software versions, password complexity requirements, security configurations (protocols, cipher suites, and key lengths), permission configurations, and firewall settings. If necessary, refer to excellent hardening solutions in the industry and suggestions from security experts.
### Communication Matrix

| Component              | TCPStore                                         |
| ------------------ | -------------------------------------------------- |
| Source Device            | TCP client                                        |
| Source IP              | Device IP address                                         |
| Source Port            | Automatically allocated by the OS. The range is determined by the OS configuration.|
| Destination Device          | TCP server                                        |
| Destination IP            | Device IP address                                        |
| Destination Port (Listening)  | User-specified port, ranging from 1025 to 65535                        |
| Protocol              | TCP                                                |
| Port Description          | TCP message interface between the server and client                    |
| Listening Port Configurable (Yes/No)| Yes                                                |
| Authentication Mode          | Digital certificate                                      |
| Encryption Mode          | TLS 1.3                                            |
| Plane          | Management plane                                            |
| Version              | All                                          |
| Special Scenario          | None                                                |

Note:
You can use the `aclshmemx_set_conf_store_tls` API to configure the TLS key certificate for TLS secure connections. You are advised to enable TLS encryption to ensure communication security. After the system is started, you are advised to delete sensitive files such as local key certificates. When this API is called, the input file path cannot contain semicolons (;), commas (,), or colons (:).
You can configure the certificate check period and certificate expiration warning time through the environment variables `ACCLINK_CHECK_PERIOD_HOURS` and `ACCLINK_CERT_CHECK_AHEAD_DAYS`.

API example:
```c
// Disable TLS.
aclshmemx_set_conf_store_tls(false, nullptr, 0);

// Enable TLS.

char *tls_info ="                               \
    tlsCaPath: /etc/ssl/certs/;                 \
    tlsCert: /etc/ssl/certs/server.crt;         \
    tlsPk: /etc/ssl/private/server.key;         \
    tlsPkPwd: /etc/ssl/private/key_pwd.txt;     \
    tlsCrlPath: /etc/ssl/crl/;                  \
    tlsCrlFile: server_crl1.pem,server_crl2.pem;\
    tlsCaFile: ca.pem1,ca.pem2;                 \
    packagePath: /etc/lib"
int32_t ret = aclshmemx_set_conf_store_tls(true, tls_info, strlen(tls_info));
```
| Environment Variable         | Description           |
| -------------- | ------------- |
| SHMEM_LOG_LEVEL| SHMEM log level    |
| SHMEM_HOME_PATH| SHMEM installation path    |
| VERSION        | Default version of the compiled .whl package|
| ASCEND_RT_VISIBLE_DEVICES | Devices that are visible to the current process. One or more device IDs can be specified at a time. By using this environment variable, you can adjust the devices without modifying the application.|
| SHMEM_CYCLE_PROF_PE  |  PEs on which profiling is to be performed. The value range of `pe_id` is [0, PEs-1]. To cancel profiling, run the `unset SHMEM_CYCLE_PROF_PE` command.|


## Recommended Running Users

The **root** user is the administrator of the Linux OS and has the permission to access all Linux system resources. The use of the **root** user to log in to the system and perform operations on the system will incur many potential security risks. Generally, you are advised to set `PermitRootLogin` in the `/etc/ssh/sshd_config` file to `no`. After the setting, the **root** user cannot log in to the system using SSH. This improves system security. If you want to use the **root** permission to perform management operations, you can log in to the system as a common user and run the `su` or `sudo` command to switch to the **root** user. This can avoid logging in to the system directly with the **root** user, thereby reducing the risk of system attacks. To ensure security and minimize permissions, you are not advised to use administrator accounts such as **root**.

## File Permission Control

- You are advised to set the system `umask` value to `0027` or higher on hosts (both physical and virtual hosts) and containers. This ensures that new folders have a default maximum permission of `750` and new files have a default maximum permission of `640`.
- You are advised to take security measures such as permission control on sensitive content, including personal privacy data, business assets, source files, and various files saved during operator development. Refer to [A-Recommended Maximum Permissions for Files and Folders in Different Scenarios](#a-recommended-maximum-permissions-for-files-and-folders-in-different-scenarios) to learn about the permissions for installation directories and public data files in this project.
- Permission control is required during installation and use. You are advised to set permissions by referring to [A-Recommended Maximum Permissions for Files and Folders in Different Scenarios](#a-recommended-maximum-permissions-for-files-and-folders-in-different-scenarios).

## Build Security Statement

When you are building and installing this project from the source code, some intermediate files will be generated. After the build is complete, you are advised to perform permission control on the intermediate files to ensure file security.

## Runtime Security Statement

- You are advised to write an operator calling script based on the operating environment resources. If the operator calling script does not match the resource status, for example, the space used for generating input data and benchmark computing results exceeds the memory capacity limit, or the data stored locally in the script exceeds the disk space, an error may occur and the process may exit unexpectedly.
- When an operator encounters an exception at runtime, it will exit the process and print an error message. It is advised to locate the specific error cause based on the error message, including methods such as setting the operator to execute synchronously and viewing log files.
- When calling an operator via [PyTorch](https://gitcode.com/ascend/pytorch), a runtime error may occur due to version mismatch. For details, see [PyTorch Security Statement](https://gitcode.com/Ascend/pytorch/blob/master/SECURITYNOTE.md).

## Performing Security Hardening on the Memory Address Randomization Mechanism

After address space layout randomization (ASLR) is enabled, the vulnerability attack defense capability is enhanced. You are advised to enable this function by setting the value in `/proc/sys/kernel/randomize_va_space` to `2`.

## Public Network Address Statement
The following table lists the public network addresses contained in the code of this project.

| Type |        Open-Source Code Address       | File Name                 | Public IP/Public URL/Domain Name/Email/Archive File Address                                          | Description                                     |
| :---: | :------------------------: | :---------------------- | :------------------------------------------------------------------------------------------ | :-------------------------------------------- |
| Dependency |           N/A          | build.sh                | https://gitcode.com/cann/catlass.git                                                            | Download the catlass source code from GitCode as a build dependency.         |
| Dependency | https://github.com/doxygen | build.sh                | https://github.com/doxygen/doxygen/releases/download/Release_1_9_6/doxygen-1.9.6.src.tar.gz | Download the Doxygen-1.9.6 source code from GitHub as a build dependency.  |
| Dependency |           N/A          | build.sh                | https://gitcode.com/GitHub_Trending/go/googletest                                                    | Download the GoogleTest source code from GitCode as a dependency of the unit test framework.  |
| Document |           N/A          | shmemi_device_barrier.h | https://www.inf.ed.ac.uk/teaching/courses/ppls/BarrierPaper.pdf                             | Parallel programming language and system                           |
## Vulnerability Handling Mechanism
[Vulnerability Management](https://gitcode.com/cann/community/blob/master/security/security.md)

## Appendix
<a name="file-permission-table"></a>

### A-Recommended Maximum Permissions for Files and Folders in Different Scenarios

| Type          | Maximum Linux Permission|
| -------------- | ---------------  |
| User's home directory                       |   750 (rwxr-x---)           |
| Program files (including scripts and library files)      |   550 (r-xr-x---)            |
| Program file directory                     |   550 (r-xr-x---)           |
| Configuration file                         |  640 (rw-r-----)            |
| Configuration file directory                     |   750 (rwxr-x---)           |
| Log files (recorded or archived)       |  440 (r--r-----)            |
| Log files (being recorded)               |    640 (rw-r-----)          |
| Log file directory                     |   750 (rwxr-x---)           |
| Debug files                        |  640 (rw-r-----)        |
| Debug file directory                    |   750 (rwxr-x---) |
| Temporary file directory                     |   750 (rwxr-x---)  |
| Maintenance and upgrade file directory                 |   770 (rwxrwx---)   |
| Service data files                     |   640 (rw-r-----)   |
| Service data file directory                 |   750 (rwxr-x---)     |
| Key components, private keys, certificates, and ciphertext file directory   |  700 (rwx------)     |
| Key components, private keys, certificates, and ciphertext files       | 600 (rw-------)     |
| APIs and scripts for encryption and decryption           |   500 (r-x------)       |
