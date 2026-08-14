## SHMEM Python API Reference
### shmem.core APIs
##### External APIs
1. Obtain the current library version. The ACLSHMEM library version is returned.

    ```python
    def get_version() -> str
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |Return Value|[out]|A string containing the version information, in the format of `"libaclshmem_version=X.Y"`|

1. Generate a unique ID for UID initialization. The API should be called by a single process (for example, rank 0) and broadcast to other processes.

    ```python
    def get_unique_id(empty: bool=False) -> UniqueID
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |empty|[in]|Reserved. It has no actual meaning.|
    |Return Value|[out]|Handle representing a unique ID. If the generation fails, `AclshmemError` is thrown.|

1. Initialize the ACLSHMEM runtime using a unique ID. This is a collective operation and must be called by all PEs.

    ```python
    def init(device: int=None, uid: UniqueID=None, rank: int=None, nranks: int=None, mpi_comm=None, initializer_method: str="", mem_size: int=None) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |device|[in]|Reserved. It has no actual meaning.|
    |uid|[in]|(Required) Unique identifier for initialization.|
    |rank|[in]|(Required) Rank (0-based) of the current process in the ACLSHMEM job.|
    |nranks|[in]|(Required) Total number of processes involved in the ACLSHMEM job.|
    |mpi_comm|[in]|Reserved. It has no actual meaning.|
    |initializer_method|[in]|Initialization method. The value must be `"uid"`.|
    |mem_size|[in]|(Required) Size of the symmetric memory allocated to each PE, in bytes.|
    |Return Value|-|No return value. `AclshmemInvalid` is thrown due to missing parameters, and `AclshmemError` is thrown due to an initialization failure.|

1. Destroy the ACLSHMEM runtime and resource allocations. Each process should call this API once after completing all ACLSHMEM operations.

    ```python
    def finalize() -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |Return Value|-|No return value. If the destruction fails, `AclshmemError` is thrown.|

1. Allocate an NPU buffer supported by ACLSHMEM. This is a collective operation and must be called by all PEs synchronously.

    ```python
    def buffer(size, release=False, except_on_del=True) -> Buffer
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |size|[in]|Size of the buffer to be allocated, in bytes|
    |release|[in]|Reserved. It has no actual meaning.|
    |except_on_del|[in]|Reserved. It has no actual meaning.|
    |Return Value|[out]|Original memory buffer represented by the address and byte length If the allocation fails, `AclshmemError` is thrown.|

1. Free the buffer allocated by `buffer()`. This is a collective operation.

    ```python
    def free(buf: Buffer) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |buf|[in]|Buffer to be freed|
    |Return Value|-|No return value.|

1. Obtain the symmetric address of the target buffer on a specified PE. This address can be used to directly reference the data on the PE.

    ```python
    def get_peer_buffer(buf: Buffer, pe: int) -> Buffer
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |buf|[in]|Symmetric address of the remotely accessible data|
    |pe|[in]|PE ID|
    |Return Value|[out]|Remote symmetric address buffer on the specified PE. If the address fails to be obtained, `AclshmemError` is thrown.|

1. Copy contiguous data from the local PE to the symmetric memory address of a specified PE and update the remote signal variable after the operation is complete. Currently, only the Memory Transfer Engine (MTE) is supported. This is a synchronous (blocking) API.

    ```python
    def put_signal(dst: Buffer, src: Buffer, signal_var: Buffer, signal_val: int, signal_operation: SignalOp, remote_pe: int=-1, stream=None) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |dst|[in]|Symmetric address of the target data on the remote PE|
    |src|[in]|Address of the source data in the local memory|
    |signal_var|[in]|Symmetric address of the signal word to be updated on the remote PE|
    |signal_val|[in]|Value of the signal variable to be updated|
    |signal_operation|[in]|Signal variable update operation. Supported operation: `SignalOp.SIGNAL_SET` or `SignalOp.SIGNAL_ADD`|
    |remote_pe|[in]|Remote PE ID. The default value is `-1`.|
    |stream|[in]|Reserved. Ignore this parameter. The default stream is used at the bottom layer.|
    |Return Value|-|No return value.|

1. Copy contiguous data from the local PE to the symmetric memory address of the remote PE through a specified stream. The caller needs to synchronize the stream to ensure that the transfer is complete. Currently, only the MTE is supported. This is a non-blocking API.

    ```python
    def put(dst: Buffer, src: Buffer, remote_pe: int=-1, stream: int=None) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |dst|[in]|Symmetric address of the target data on the remote PE|
    |src|[in]|Address of the source data in the local memory|
    |remote_pe|[in]|Remote PE ID. The default value is `-1`.|
    |stream|[in]|ACL stream object, which is used for sorting. Pass `0` or `None` to use the default stream.|
    |Return Value|-|No return value.|

1. Copy contiguous data from the symmetric memory of the remote PE to the local buffer through a specified stream. The caller needs to synchronize the stream to ensure that the transfer is complete. Currently, only the MTE is supported. This is a non-blocking API.

    ```python
    def get(dst: Buffer, src: Buffer, remote_pe: int=-1, stream: int=None) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |dst|[in]|Address of the target data in the local memory|
    |src|[in]|Symmetric address of the source data on the remote PE|
    |remote_pe|[in]|Remote PE ID. The default value is `-1`.|
    |stream|[in]|ACL stream object, which is used for sorting. Pass `0` or `None` to use the default stream.|
    |Return Value|-|No return value.|

1. Perform atomic operations on remote signal variables on a given PE. The operations are performed in a given stream. The caller needs to synchronize the stream to observe the result. Currently, only the MTE is supported. This is a non-blocking API.

    ```python
    def signal_op(signal_var: Buffer, signal_val: int, signal_operation: SignalOp, remote_pe: int=-1, stream: int=None) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |signal_var|[in]|Local address of the signal variable that can be accessed on the target PE|
    |signal_val|[in]|Value used for atomic operations|
    |signal_operation|[in]|Operation performed on a remote signal. Supported operation: `SignalOp.SIGNAL_SET` or `SignalOp.SIGNAL_ADD`|
    |remote_pe|[in]|ID of the PE where the remote signal variable to be updated is located. The default value is `-1`.|
    |stream|[in]|ACL stream object, which is used for sorting. Passing `None` will cause an `AclshmemInvalid` exception.|
    |Return Value|-|No return value. If `stream` is set to `None`, `AclshmemInvalid` is thrown.|

1. Wait until the symmetric signal variable meets the specified comparison condition. The wait operation is performed on the given stream, and the calling result is returned immediately on the host. After the stream is synchronized, the condition `signal_var` `cmp` `signal_val` must be true. Currently, only the MTE is supported.

    ```python
    def signal_wait(signal_var: Buffer, signal_val: int, signal_operation: ComparisonType, stream: int) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |signal_var|[in]|Local address of the source signal variable|
    |signal_val|[in]|Value to be compared with the value pointed to by signal_var|
    |signal_operation|[in]|Comparison operator. Supported comparison operator: `ComparisonType.CMP_EQ`, `CMP_NE`, `CMP_GT`, `CMP_GE`, `CMP_LT`, or `CMP_LE`|
    |stream|[in]|ACL stream object, which is used for sorting. Passing `None` will cause an `AclshmemInvalid` exception.|
    |Return Value|-|No return value. If `stream` is set to `None`, `AclshmemInvalid` is thrown.|

1. Ensure that all previously issued symmetric data operations are completed on the given stream. The quiet operation is enqueued into the specified stream, and the calling result is returned immediately on the host. The caller needs to synchronize the stream to observe the completion. Currently, only the MTE is supported.

    ```python
    def quiet(stream: int) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |stream|[in]|ACL stream for performing the quiet operation. A valid stream must be passed. Passing `None` will cause an `AclshmemInvalid` exception.|
    |Return Value|-|No return value. If `stream` is set to `None`, `AclshmemInvalid` is thrown.|

1. Obtain the local PE ID.

    ```python
    def my_pe() -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |Return Value|[out]|Local PE ID|

1. Obtain the PE ID of the current process in a specified team.

    ```python
    def team_my_pe(team) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |team|[in]|Target team ID|
    |Return Value|[out]|PE ID in the specified team. If the team is invalid, `-1` is returned.|

1. Obtain the total number of PEs running in the program (world team dimension).

    ```python
    def n_pes() -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |Return Value|[out]|Total number of PEs|

1. Obtain the total number of PEs in a specified team.

    ```python
    def team_n_pes(team) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |team|[in]|Target team ID|
    |Return Value|[out]|The number of PEs in the specified team. If the team is invalid, `-1` is returned.|

1. Query the current initialization status of the shared memory module.

    ```python
    def init_status() -> InitStatus
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |Return Value|[out]|The initialization status is returned. The enumerated values are as follows: `NOT_INITIALIZED`, `SHM_CREATED`, `INITIALIZED`, and `INVALID`.|

##### Class
1. UniqueId class: unique identifier handle for UID initialization

    ```python
    class UniqueId:
        def __init__(self):
    ```

    |Attribute|Direction|Meaning|
    |-|-|-|
    |version|[out]|Version|
    |my_rank|[out]|PE ID of the current process|
    |n_pes|[out]|Total number of PEs of all processes|
    |internal|[out]|Internal information about the UID (bytes)|

1. InitStatus enumeration class: initialization statuses of the shared memory module

    ```python
    class InitStatus(Enum):
        NOT_INITIALIZED
        SHM_CREATED
        INITIALIZED
        INVALID
    ```

    |Enumerated Value|Meaning|
    |-|-|
    |NOT_INITIALIZED|Not initialized|
    |SHM_CREATED|Shared memory created|
    |INITIALIZED|Initialization completed|
    |INVALID|Invalid|

1. SignalOp enumeration class: atomic operation types of signal variables

    ```python
    class SignalOp(Enum):
        SIGNAL_SET
        SIGNAL_ADD
    ```

    |Enumerated Value|Meaning|
    |-|-|
    |SIGNAL_SET|Atomic setting: writes a given value to a remote signal.|
    |SIGNAL_ADD|Atomic addition: adds a given value to the existing value of a remote signal.|

1. ComparisonType enumeration class: signal waiting comparison operation types

    ```python
    class ComparisonType(Enum):
        CMP_EQ
        CMP_NE
        CMP_GT
        CMP_GE
        CMP_LT
        CMP_LE
    ```

    |Enumerated Value|Meaning|
    |-|-|
    |CMP_EQ|Equal to (==)|
    |CMP_NE|Not equal to (!=)|
    |CMP_GT|Greater than (>)|
    |CMP_GE|Greater than or equal to (>=)|
    |CMP_LT|Less than (<)|
    |CMP_LE|Less than or equal to (<=)|

### shmem._pyshmem APIs
##### External APIs
1. Initialize the shared memory module. This is a collective operation.

    ```python
    def aclshmem_init(mype, npes, mem_size) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |mype|[in]|Local PE index. The value ranges from `0` to `npes-1`.|
    |npes|[in]|Total number of PEs|
    |mem_size|[in]|Size of the memory allocated to each PE, in bytes|
    |Return Value|[out]|Result code. Success: 0. Failure: error code.|

1. Destroy the shared memory module and all resource allocations.

    ```python
    def aclshmem_finalize(instance_id=0) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |instance_id|[in]|Instance ID. The default value is `0`.|
    |Return Value|-|No return value.|

1. Query the current initialization status of the shared memory module.

    ```python
    def aclshmemx_init_status() -> InitStatus
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |Return Value|[out]|Enumeration value of the initialization status. `INITIALIZED` indicates that the initialization is complete.|

1. Set the TLS private key and password, and register the decryption callback function.

    ```python
    def set_conf_store_tls_key(tls_pk, tls_pk_pw, py_decrypt_func:Callable[[str], str]) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |tls_pk|[in]|TLS private key content|
    |tls_pk_pw|[in]|TLS private key password|
    |py_decrypt_func|[in]|Decryption callback function. It receives `(str cipher_text)` and returns `(str plain_text)`.|
    |Return Value|[out]|Result code. Success: 0. Failure: error code.|

1. Allocate symmetric memory. This is a collective operation with an implicit barrier.

    ```python
    def aclshmem_malloc(size) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |size|[in]|Size of the allocated memory, in bytes|
    |Return Value|[out]|If the operation is successful, a pointer (int) to the allocated memory is returned. If `size` is `0` or the allocation fails, `0` is returned, and an exception is thrown.|

1. Allocate zero-initialized symmetric memory. This is a collective operation with an implicit barrier.

    ```python
    def aclshmem_calloc(nmemb, size) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |nmemb|[in]|The number of elements|
    |size|[in]|Size of each element, in bytes|
    |Return Value|[out]|If the operation is successful, a pointer (int) to the allocated memory is returned. If `nmemb` or `size` is `0`, `0` is returned, and an exception is thrown.|

1. Allocate symmetric memory with a specified alignment mode. This is a collective operation with an implicit barrier.

    ```python
    def aclshmem_align(alignment, size) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |alignment|[in]|Memory alignment requirement (The value must be a power of 2.)|
    |size|[in]|The number of bytes to be allocated|
    |Return Value|[out]|If the operation is successful, a pointer (int) to the allocated memory is returned. If the allocation fails, an exception is thrown.|

1. Free the memory allocated by the symmetric memory allocation function. This is a collective operation with an implicit barrier.

    ```python
    def aclshmem_free(ptr) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |ptr|[in]|Pointer to the memory to be freed|
    |Return Value|-|No return value.|

1. Obtain the remote reference address of the symmetric address on a specified PE. This address can be used to access the data on the PE directly.

    ```python
    def aclshmem_ptr(ptr, peer) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |ptr|[in]|Symmetric address of the remotely accessible data|
    |peer|[in]|PE ID|
    |Return Value|[out]|If the operation is successful, the remote symmetric address (int) is returned. If the input address is invalid, `0` is returned.|

1. Obtain the PE ID (world team dimension).

    ```python
    def my_pe() -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |Return Value|[out]|Local PE ID|

1. Obtain the PE ID in a specified team.

    ```python
    def team_my_pe(team_id) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |team_id|[in]|Handle of the team|
    |Return Value|[out]|ID of the PE in the specified team. If an error occurs, `-1` is returned.|

1. Obtain the total number of PEs (world team dimension).

    ```python
    def pe_count() -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |Return Value|[out]|Total number of PEs|

1. Obtain the number of PEs in a specified team.

    ```python
    def team_n_pes(team_id) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |team_id|[in]|Handle of the team|
    |Return Value|[out]|The number of PEs in the specified team. If an error occurs, `-1` is returned.|

1. Split child teams from the existing parent team by stride. This is a collective operation.

    ```python
    def team_split_strided(parent, start, stride, size) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |parent|[in]|Parent team ID|
    |start|[in]|Start PE ID of a child team|
    |stride|[in]|Stride between PE IDs|
    |size|[in]|The number of PEs in a child team|
    |Return Value|[out]|Success: new team ID. Failure: `-1`|

1. Split a team from the parent team based on the two-dimensional Cartesian space. This is a collective operation.

    ```python
    def team_split_2d(parent, x_range) -> tuple
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |parent|[in]|Handle of the parent team|
    |x_range|[in]|The number of elements in the first dimension|
    |Return Value|[out]|If the operation is successful, the (x_team_id, y_team_id) tuple is returned. If the operation fails, an exception is thrown.|

1. Obtain the team configuration passed during team creation.

    ```python
    def aclshmem_team_get_config(team) -> TeamConfig
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |team|[in]|team ID|
    |Return Value|[out]|If the operation is successful, the `TeamConfig` object is returned. If the operation fails, an exception is thrown.|

1. Copy contiguous data from the local PE to the symmetric address of a specified PE. This is a synchronous (blocking) API.

    ```python
    def aclshmem_putmem(dst, src, elem_size, pe) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |dst|[in]|Pointer to the symmetric address of the remote PE|
    |src|[in]|Pointer to the local source data memory|
    |elem_size|[in]|Total number of bytes of elements in the target and source addresses|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

1. Copy contiguous data from the symmetric memory of a specified PE to the local PE. This is a synchronous (blocking) API.

    ```python
    def aclshmem_getmem(dst, src, elem_size, pe) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |dst|[in]|Pointer to the local target memory|
    |src|[in]|Pointer to the symmetric address of the remote PE|
    |elem_size|[in]|Total number of bytes of elements in the target and source addresses|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

1. Copy the data arranged by the sst stride in the local memory to the corresponding position in the symmetric memory of a specified PE by the dst stride. This is a synchronous (blocking) API.

    ```python
    def aclshmem_{TYPE}_iput(dest, source, dst, sst, nelems, pe)
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `double`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |dest|[in]|Symmetric memory pointer to the remote target data|
    |source|[in]|Pointer to the local source data|
    |dst|[in]|Stride between consecutive elements in the target address|
    |sst|[in]|Stride between consecutive elements in the source address|
    |nelems|[in]|The number of consecutive element blocks|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

1. Copy the data arranged by the sst stride in the symmetric memory of a remote PE to the corresponding position of the local memory by the dst stride. This is a synchronous (blocking) API.

    ```python
    def aclshmem_{TYPE}_iget(dest, source, dst, sst, nelems, pe)
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `double`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |dest|[in]|Pointer to the local target data|
    |source|[in]|Symmetric memory pointer to the remote source data|
    |dst|[in]|Stride between consecutive elements in the target address|
    |sst|[in]|Stride between consecutive elements in the source address|
    |nelems|[in]|The number of consecutive element blocks|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

1. Copy contiguous data from the local PE to the symmetric memory address of a specified PE. This is a synchronous (blocking) API.

    ```python
    def aclshmem_put{BITS}(dst, src, elem_size, pe)
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |BITS|-|Data bit width. Supported option: `8`, `16`, `32`, `64`, or `128`|
    |dst|[in]|Pointer to the symmetric address of the remote PE|
    |src|[in]|Pointer to the local source data memory|
    |elem_size|[in]|Total number of bytes of elements in the target and source addresses|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

1. Copy contiguous data from the symmetric memory of a specified PE to the local PE. This is a synchronous (blocking) API.

    ```python
    def aclshmem_get{BITS}(dst, src, elem_size, pe)
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |BITS|-|Data bit width. Supported option: `8`, `16`, `32`, `64`, or `128`|
    |dst|[in]|Pointer to the local target memory|
    |src|[in]|Pointer to the symmetric address of the remote PE|
    |elem_size|[in]|Total number of bytes of elements in the target and source addresses|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

1. Copy the data arranged by the sst stride in the local memory to the corresponding position (bit width version) of a specified PE by the dst stride. This is a synchronous (blocking) API.

    ```python
    def aclshmem_iput{BITS}(dest, source, dst, sst, nelems, pe)
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |BITS|-|Data bit width. Supported option: `8`, `16`, `32`, `64`, or `128`|
    |dest|[in]|Symmetric memory pointer to the remote target data|
    |source|[in]|Pointer to the local source data|
    |dst|[in]|Stride between consecutive elements in the target address|
    |sst|[in]|Stride between consecutive elements in the source address|
    |nelems|[in]|The number of consecutive element blocks|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

1. Copy the data arranged by the sst stride in the symmetric memory of a remote PE to the corresponding position of the local memory by the dst stride (bit width version). This is a synchronous (blocking) API.

    ```python
    def aclshmem_iget{BITS}(dest, source, dst, sst, nelems, pe)
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |BITS|-|Data bit width. Supported option: `8`, `16`, `32`, `64`, or `128`|
    |dest|[in]|Pointer to the local target data|
    |source|[in]|Symmetric memory pointer to the remote source data|
    |dst|[in]|Stride between consecutive elements in the target address|
    |sst|[in]|Stride between consecutive elements in the source address|
    |nelems|[in]|The number of consecutive element blocks|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

1. Return the major and minor version numbers of the library.

    ```python
    def aclshmem_info_get_version() -> tuple
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |Return Value|[out]|(major, minor) tuple|

1. Return the vendor-defined name string.

    ```python
    def aclshmem_info_get_name() -> str
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |Return Value|[out]|Vendor-defined name string|

1. Convert a given PE ID in one team to the corresponding PE ID in another team.

    ```python
    def team_translate_pe(src_team, src_pe, dest_team) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |src_team|[in]|Source team ID|
    |src_pe|[in]|Source PE ID|
    |dest_team|[in]|Target team ID|
    |Return Value|[out]|If the operation is successful, the ID of the corresponding PE in the target team is returned. If an error occurs, `-1` is returned.|

1. Destroy a team.

    ```python
    def team_destroy(team) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |team|[in]|ID of the team to be destroyed|
    |Return Value|-|No return value.|

1. Obtain the runtime FFTS configuration.

    ```python
    def get_ffts_config() -> str
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |Return Value|[out]|FFTS configuration string|

1. Copy contiguous data from the local PE to the symmetric address of a specified PE. This is an asynchronous (non-blocking) API.

    ```python
    def aclshmem_putmem_nbi(dst, src, elem_size, pe) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |dst|[in]|Pointer to the symmetric address of the remote PE|
    |src|[in]|Pointer to the local source data memory|
    |elem_size|[in]|Total number of bytes of elements in the target and source addresses|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

1. Copy contiguous data from the symmetric memory of a specified PE to the local PE. This is an asynchronous (non-blocking) API.

    ```python
    def aclshmem_getmem_nbi(dst, src, elem_size, pe) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |dst|[in]|Pointer to the local target memory|
    |src|[in]|Pointer to the symmetric address of the remote PE|
    |elem_size|[in]|Total number of bytes of elements in the target and source addresses|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

1. Copy contiguous data from the local PE to the symmetric memory address of a specified PE (bit width version). This is an asynchronous (non-blocking) API.

    ```python
    def aclshmem_put{BITS}_nbi(dst, src, elem_size, pe)
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |BITS|-|Data bit width. Supported option: `8`, `16`, `32`, `64`, or `128`|
    |dst|[in]|Pointer to the symmetric address of the remote PE|
    |src|[in]|Pointer to the local source data memory|
    |elem_size|[in]|Total number of bytes of elements in the target and source addresses|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

1. Copy contiguous data from the symmetric memory of a specified PE to the local PE (bit width version). This is an asynchronous (non-blocking) API.

    ```python
    def aclshmem_get{BITS}_nbi(dst, src, elem_size, pe)
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |BITS|-|Data bit width. Supported option: `8`, `16`, `32`, `64`, or `128`|
    |dst|[in]|Pointer to the local target memory|
    |src|[in]|Pointer to the symmetric address of the remote PE|
    |elem_size|[in]|Total number of bytes of elements in the target and source addresses|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

1. Copy contiguous data from the local PE to the symmetric address of a specified PE and update the remote signal variable after the operation is complete. This is an asynchronous (non-blocking) API.

    ```python
    def aclshmemx_putmem_signal_nbi(dst, src, elem_size, sig, signal, sig_op, pe) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |dst|[in]|Pointer to the symmetric address of the remote PE|
    |src|[in]|Pointer to the local source data memory|
    |elem_size|[in]|Total number of bytes of elements in the target and source addresses|
    |sig|[in]|Symmetric address of the signal word to be updated|
    |signal|[in]|Value of the signal to be updated|
    |sig_op|[in]|Signal update operation. Supported operation: `ACLSHMEM_SIGNAL_SET` or `ACLSHMEM_SIGNAL_ADD`|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

1. Copy contiguous data from the local PE to the symmetric address of a specified PE and update the remote signal variable. This is a synchronous (blocking) API.

    ```python
    def aclshmemx_putmem_signal(dst, src, elem_size, sig, signal, sig_op, pe) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |dst|[in]|Pointer to the symmetric address of the remote PE|
    |src|[in]|Pointer to the local source data memory|
    |elem_size|[in]|Total number of bytes of elements in the target and source addresses|
    |sig|[in]|Symmetric address of the signal word to be updated|
    |signal|[in]|Value of the signal to be updated|
    |sig_op|[in]|Signal update operation. Supported operation: `ACLSHMEM_SIGNAL_SET` or `ACLSHMEM_SIGNAL_ADD`|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

1. Copy contiguous data from the local PE to the symmetric address of a specified PE and update the remote signal (bit width version). This is an asynchronous (non-blocking) API.

    ```python
    def aclshmemx_put{BITS}_signal_nbi(dst, src, elem_size, sig, signal, sig_op, pe)
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |BITS|-|Data bit width. Supported option: `8`, `16`, `32`, `64`, or `128`|
    |dst|[in]|Pointer to the symmetric address of the remote PE|
    |src|[in]|Pointer to the local source data memory|
    |elem_size|[in]|Total number of bytes of elements in the target and source addresses|
    |sig|[in]|Symmetric address of the signal word to be updated|
    |signal|[in]|Value of the signal to be updated|
    |sig_op|[in]|Signal update operation. Supported operation: `ACLSHMEM_SIGNAL_SET` or `ACLSHMEM_SIGNAL_ADD`|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

1. Copy contiguous data from the local PE to the symmetric address of a specified PE and update the remote signal (bit width version). This is a synchronous (blocking) API.

    ```python
    def aclshmemx_put{BITS}_signal(dst, src, elem_size, sig, signal, sig_op, pe)
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |BITS|-|Data bit width. Supported option: `8`, `16`, `32`, `64`, or `128`|
    |dst|[in]|Pointer to the symmetric address of the remote PE|
    |src|[in]|Pointer to the local source data memory|
    |elem_size|[in]|Total number of bytes of elements in the target and source addresses|
    |sig|[in]|Symmetric address of the signal word to be updated|
    |signal|[in]|Value of the signal to be updated|
    |sig_op|[in]|Signal update operation. Supported operation: `ACLSHMEM_SIGNAL_SET` or `ACLSHMEM_SIGNAL_ADD`|
    |pe|[in]|Remote PE ID|
    |Return Value|-|No return value.|

1. All PEs call exit() through broadcast to exit the process.

    ```python
    def aclshmem_global_exit(status) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |status|[in]|Status value passed to exit()|
    |Return Value|-|No return value.|

1. Ensure that all previously issued symmetric data operations are completed on the default stream. The quiet operation is enqueued into the default stream, and the calling result is returned immediately on the host. The caller needs to synchronize the default stream to observe the completion.

    ```python
    def aclshmem_quiet() -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |Return Value|-|No return value.|

1. Obtain the PE ID in a specified team.

    ```python
    def my_pe(team) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |team|[in]|team ID|
    |Return Value|[out]|ID of the PE in the specified team. If an error occurs, `-1` is returned.|

1. Obtain the number of PEs in a specified team.

    ```python
    def pe_count(team) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |team|[in]|team ID|
    |Return Value|[out]|The number of PEs in the specified team. If an error occurs, `-1` is returned.|

1. Wait until the signal variable meets the comparison condition `*sig_addr cmp cmp_val`. This is a blocking API.

    ```python
    def aclshmem_signal_wait_until(sig_addr, cmp, cmp_val) -> int
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |sig_addr|[in]|Local address of the source signal variable|
    |cmp|[in]|Comparison operator. Supported comparison operator: `CMP_EQ`, `CMP_NE`, `CMP_GT`, `CMP_GE`, `CMP_LT`, or `CMP_LE`|
    |cmp_val|[in]|Comparison value|
    |Return Value|[out]|Value of `sig_addr` when the condition is met|

1. Wait until a single element meets the comparison condition `ivar cmp cmp_val`. This is a blocking API.

    ```python
    def aclshmem_{TYPE}_wait_until(ivar, cmp, cmp_val) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |ivar|[in]|Pointer to the signal variable in the symmetric memory|
    |cmp|[in]|Comparison operator|
    |cmp_val|[in]|Comparison value|
    |Return Value|-|No return value.|

1. Wait until the signal variable is not equal to the given value. This is a blocking API.

    ```python
    def aclshmem_{TYPE}_wait(ivar, cmp_val) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |ivar|[in]|Pointer to the signal variable in the symmetric memory|
    |cmp_val|[in]|Comparison value|
    |Return Value|-|No return value.|

1. Wait until all elements in the array meet the comparison condition `ivars[i] cmp cmp_val`. This is a blocking API.

    ```python
    def aclshmem_{TYPE}_wait_until_all(ivars_ptr, nelems, status_ptr, cmp, cmp_val) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |ivars_ptr|[in]|Array with a length of `nelems` in symmetric memory|
    |nelems|[in]|The number of array elements|
    |status_ptr|[in]|Optional local mask array. The value `0` indicates that the mask is not used.|
    |cmp|[in]|Comparison operator|
    |cmp_val|[in]|Comparison value|
    |Return Value|-|No return value.|

1. Wait until at least one element in the array meets the comparison condition `ivars[i] cmp cmp_val`. This is a blocking API.

    ```python
    def aclshmem_{TYPE}_wait_until_any(ivars_ptr, nelems, status_ptr, cmp, cmp_val, res_out_ptr) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |ivars_ptr|[in]|Array with a length of `nelems` in symmetric memory|
    |nelems|[in]|The number of array elements|
    |status_ptr|[in]|Optional local mask array. The value `0` indicates that the mask is not used.|
    |cmp|[in]|Comparison operator|
    |cmp_val|[in]|Comparison value|
    |res_out_ptr|[out]|Index values of elements that meet the comparison condition|
    |Return Value|-|No return value.|

1. Wait until at least one element in the array meets the comparison condition and return the indexes of all elements that meet the condition. This is a blocking API.

    ```python
    def aclshmem_{TYPE}_wait_until_some(ivars_ptr, nelems, indices_ptr, status_ptr, cmp, cmp_val, res_out_ptr) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |ivars_ptr|[in]|Array with a length of `nelems` in symmetric memory|
    |nelems|[in]|The number of array elements|
    |indices_ptr|[out]|Array of index values of elements that meet the condition|
    |status_ptr|[in]|Optional local mask array. The value `0` indicates that the mask is not used.|
    |cmp|[in]|Comparison operator|
    |cmp_val|[in]|Comparison value|
    |res_out_ptr|[out]|The number of elements that meet the comparison condition|
    |Return Value|-|No return value.|

1. Wait until all elements in the array meet the vector comparison condition `ivars[i] cmp cmp_values[i]`. This is a blocking API.

    ```python
    def aclshmem_{TYPE}_wait_until_all_vector(ivars_ptr, nelems, status_ptr, cmp, cmp_values_ptr) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |ivars_ptr|[in]|Array with a length of `nelems` in symmetric memory|
    |nelems|[in]|The number of array elements|
    |status_ptr|[in]|Optional local mask array. The value `0` indicates that the mask is not used.|
    |cmp|[in]|Comparison operator|
    |cmp_values_ptr|[in]|Comparison value array|
    |Return Value|-|No return value.|

1. Wait until at least one element in the array meets the vector comparison condition `ivars[i] cmp cmp_values[i]`. This is a blocking API.

    ```python
    def aclshmem_{TYPE}_wait_until_any_vector(ivars_ptr, nelems, status_ptr, cmp, cmp_values_ptr, res_out_ptr) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |ivars_ptr|[in]|Array with a length of `nelems` in symmetric memory|
    |nelems|[in]|The number of array elements|
    |status_ptr|[in]|Optional local mask array. The value `0` indicates that the mask is not used.|
    |cmp|[in]|Comparison operator|
    |cmp_values_ptr|[in]|Comparison value array|
    |res_out_ptr|[out]|Index values of elements that meet the comparison condition|
    |Return Value|-|No return value.|

1. Wait until at least one element in the array meets the vector comparison condition and return the index values of all elements that meet the condition. This is a blocking API.

    ```python
    def aclshmem_{TYPE}_wait_until_some_vector(ivars_ptr, nelems, indices_ptr, status_ptr, cmp, cmp_values_ptr, res_out_ptr) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |ivars_ptr|[in]|Array with a length of `nelems` in symmetric memory|
    |nelems|[in]|The number of array elements|
    |indices_ptr|[out]|Array of index values of elements that meet the condition|
    |status_ptr|[in]|Optional local mask array. The value `0` indicates that the mask is not used.|
    |cmp|[in]|Comparison operator|
    |cmp_values_ptr|[in]|Comparison value array|
    |res_out_ptr|[out]|The number of elements that meet the comparison condition|
    |Return Value|-|No return value.|

1. Check whether a single element meets the comparison condition `ivar cmp cmp_value`. This is a non-blocking query API.

    ```python
    def aclshmem_{TYPE}_test(ivar, cmp, cmp_value, res_out_ptr) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |ivar|[in]|Pointer to the signal variable in the symmetric memory|
    |cmp|[in]|Comparison operator|
    |cmp_value|[in]|Comparison value|
    |res_out_ptr|[out]|If the condition is met, `1` is returned. Otherwise, `0` is returned.|
    |Return Value|-|No return value. The result is returned through `res_out_ptr`.|

1. Check whether at least one element in the array meets the comparison condition `ivars[i] cmp cmp_value`. This is a non-blocking query API.

    ```python
    def aclshmem_{TYPE}_test_any(ivars_ptr, nelems, status_ptr, cmp, cmp_value, res_out_ptr) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |ivars_ptr|[in]|Array with a length of `nelems` in symmetric memory|
    |nelems|[in]|The number of array elements|
    |status_ptr|[in]|Optional local mask array. The value `0` indicates that the mask is not used.|
    |cmp|[in]|Comparison operator|
    |cmp_value|[in]|Comparison value|
    |res_out_ptr|[out]|Index of the element that meets the condition. If no element meets the condition or the test dataset is empty, `SIZE_MAX` is returned.|
    |Return Value|-|No return value. The result is returned through `res_out_ptr`.|

1. Check whether at least one element in the array meets the comparison condition and return the index values of all elements that meet the condition. This is a non-blocking query API.

    ```python
    def aclshmem_{TYPE}_test_some(ivars_ptr, nelems, indices_ptr, status_ptr, cmp, cmp_value, res_out_ptr) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |ivars_ptr|[in]|Array with a length of `nelems` in symmetric memory|
    |nelems|[in]|The number of array elements|
    |indices_ptr|[out]|Array of index values of elements that meet the condition|
    |status_ptr|[in]|Optional local mask array. The value `0` indicates that the mask is not used.|
    |cmp|[in]|Comparison operator|
    |cmp_value|[in]|Comparison value|
    |res_out_ptr|[out]|The number of elements that meet the condition. If the test dataset is empty, `0` is returned.|
    |Return Value|-|No return value. The result is returned through `res_out_ptr` and `indices_ptr`.|

1. Check whether all elements in the array meet the vector comparison condition `ivars[i] cmp cmp_values[i]`. This is a non-blocking query API.

    ```python
    def aclshmem_{TYPE}_test_all_vector(ivars_ptr, nelems, status_ptr, cmp, cmp_values_ptr, res_out_ptr) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |ivars_ptr|[in]|Array with a length of `nelems` in symmetric memory|
    |nelems|[in]|The number of array elements|
    |status_ptr|[in]|Optional local mask array. The value `0` indicates that the mask is not used.|
    |cmp|[in]|Comparison operator|
    |cmp_values_ptr|[in]|Comparison value array|
    |res_out_ptr|[out]|If all elements meet the condition or `nelems` is `0`, `1` is returned. Otherwise, `0` is returned.|
    |Return Value|-|No return value. The result is returned through `res_out_ptr`.|

1. Check whether at least one element in the array meets the vector comparison condition `ivars[i] cmp cmp_values[i]`. This is a non-blocking query API.

    ```python
    def aclshmem_{TYPE}_test_any_vector(ivars_ptr, nelems, status_ptr, cmp, cmp_values_ptr, res_out_ptr) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |ivars_ptr|[in]|Array with a length of `nelems` in symmetric memory|
    |nelems|[in]|The number of array elements|
    |status_ptr|[in]|Optional local mask array. The value `0` indicates that the mask is not used.|
    |cmp|[in]|Comparison operator|
    |cmp_values_ptr|[in]|Comparison value array|
    |res_out_ptr|[out]|Index of the first element that meets the condition. If no element meets the condition or the test dataset is empty, `SIZE_MAX` is returned.|
    |Return Value|-|No return value. The result is returned through `res_out_ptr`.|

1. Check whether at least one element in the array meets the vector comparison condition and return the indexes of all elements that meet the condition. This is a non-blocking query API.

    ```python
    def aclshmem_{TYPE}_test_some_vector(ivars_ptr, nelems, indices_ptr, status_ptr, cmp, cmp_values_ptr, res_out_ptr) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |TYPE|-|Data type. Supported type: `float`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, or `char`|
    |ivars_ptr|[in]|Array with a length of `nelems` in symmetric memory|
    |nelems|[in]|The number of array elements|
    |indices_ptr|[out]|Array of index values of elements that meet the condition|
    |status_ptr|[in]|Optional local mask array. The value `0` indicates that the mask is not used.|
    |cmp|[in]|Comparison operator|
    |cmp_values_ptr|[in]|Comparison value array|
    |res_out_ptr|[out]|The number of elements that meet the condition. If the test dataset is empty, `0` is returned.|
    |Return Value|-|No return value. The result is returned through `res_out_ptr` and `indices_ptr`.|

1. Copy contiguous data from the local PE to the symmetric address of a specified PE through a specified stream. This is a non-blocking API.

    ```python
    def aclshmemx_putmem_on_stream(dst, src, elem_size, pe, stream) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |dst|[in]|Pointer to the symmetric address of the remote PE|
    |src|[in]|Pointer to the local source data memory|
    |elem_size|[in]|Total number of bytes of elements in the target and source addresses|
    |pe|[in]|Remote PE ID|
    |stream|[in]|ACL stream object. Pass `0` or `None` to use the default stream.|
    |Return Value|-|No return value.|

1. Copy contiguous data from the symmetric memory of a specified PE to the local PE through a specified stream. This is a non-blocking API.

    ```python
    def aclshmemx_getmem_on_stream(dst, src, elem_size, pe, stream) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |dst|[in]|Pointer to the local target memory|
    |src|[in]|Pointer to the symmetric address of the remote PE|
    |elem_size|[in]|Total number of bytes of elements in the target and source addresses|
    |pe|[in]|Remote PE ID|
    |stream|[in]|ACL stream object. Pass `0` or `None` to use the default stream.|
    |Return Value|-|No return value.|

1. Perform atomic operations on a remote signal variable in a specified stream. Currently, only the MTE is supported. This is a non-blocking API.

    ```python
    def aclshmemx_signal_op_on_stream(sig, signal, sig_op, pe, stream) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |sig|[in]|Local address of the signal variable that can be accessed on the target PE|
    |signal|[in]|Value used for atomic operations|
    |sig_op|[in]|Operation performed on a remote signal. Supported operation: `ACLSHMEM_SIGNAL_SET` or `ACLSHMEM_SIGNAL_ADD`|
    |pe|[in]|Remote PE ID|
    |stream|[in]|ACL stream object. Pass `0` or `None` to use the default stream.|
    |Return Value|-|No return value.|

1. This is a non-blocking API. Wait until the signal variable meets the comparison condition in a specified stream. The call is returned immediately on the host and waits for execution in the stream. The caller needs to synchronize the stream to ensure that the wait is complete. Currently, only the MTE is supported.

    ```python
    def aclshmemx_signal_wait_until_on_stream(sig, cmp, cmp_val, stream) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |sig|[in]|Local address of the source signal variable|
    |cmp|[in]|Comparison operator. Supported comparison operator: `ACLSHMEM_CMP_EQ`, `CMP_NE`, `CMP_GT`, `CMP_GE`, `CMP_LT`, or `CMP_LE`|
    |cmp_val|[in]|Value to be compared with the value pointed to by `sig`|
    |stream|[in]|ACL stream object, which is used for sorting A valid stream must be passed.|
    |Return Value|-|No return value.|

1. Ensure that all previously issued symmetric data operations are completed on the given stream. The quiet operation is enqueued into the specified stream, and the calling result is returned immediately on the host. The caller needs to synchronize the stream to observe the completion. Currently, only the MTE is supported.

    ```python
    def aclshmemx_quiet_on_stream(stream) -> None
    ```

    |Parameter/Return Value|Direction|Meaning|
    |-|-|-|
    |stream|[in]|ACL stream for performing the quiet operation. Pass `0` or `None` to use the default stream.|
    |Return Value|-|No return value.|

##### Class
1. OpEngineType enumeration class: data transfer engine types

    ```python
    class OpEngineType(Enum):
        MTE
        SDMA
        ROCE
        UDMA
    ```

    |Enumerated Value|Meaning|
    |-|-|
    |MTE|Memory Transfer Engine|
    |SDMA|System DMA|
    |ROCE|RDMA over Converged Ethernet|
    |UDMA|Unified DMA|

1. OptionalAttr class: optional attribute configurations for initialization

    ```python
    class OptionalAttr:
        def __init__(self):
    ```

    |Attribute|Direction|Meaning|
    |-|-|-|
    |version|[in]|Configuration version|
    |data_op_engine_type|[in]|Data transfer engine type (an enumerated value of `OpEngineType`)|
    |shm_init_timeout|[in]|Timeout period of the init operation|
    |shm_create_timeout|[in]|Timeout period of the create operation|
    |control_operation_timeout|[in]|Timeout period of the control operation|
    |sockFd|[in]|Socket file descriptor. The default value is `-1`.|

1. InitAttr class: attribute configurations for initialization

    ```python
    class InitAttr:
        def __init__(self):
    ```

    |Attribute|Direction|Meaning|
    |-|-|-|
    |my_rank|[in]|PE ID of the current process|
    |n_ranks|[in]|Total number of PEs|
    |ip_port|[in]|IP address and port number of the communication server|
    |local_mem_size|[in]|Size of the symmetric memory allocated to the current PE, in bytes|
    |option_attr|[in]|`OptionalAttr` optional attribute configurations|

1. TeamConfig class: team configurations

    ```python
    class TeamConfig:
    ```

    |Attribute|Direction|Meaning|
    |-|-|-|
    |num_contexts|[in]|The number of contexts that can run simultaneously in a team|

1. UniqueId class: unique identifier handle for UID initialization

    ```python
    class UniqueId:
        def __init__(self):
    ```

    |Attribute|Direction|Meaning|
    |-|-|-|
    |version|[out]|Version|
    |my_pe|[out]|PE ID of the current process|
    |n_pes|[out]|Total number of PEs of all processes|
    |internal|[out]|Internal information about the UID (bytes)|

1. InitStatus enumeration class: initialization statuses of the shared memory module

    ```python
    class InitStatus(Enum):
        NOT_INITIALIZED
        SHM_CREATED
        INITIALIZED
        INVALID
    ```

    |Enumerated Value|Meaning|
    |-|-|
    |NOT_INITIALIZED|Not initialized|
    |SHM_CREATED|Shared memory created|
    |INITIALIZED|Initialization completed|
    |INVALID|Invalid|

1. SignalOp enumeration class: atomic operation types of signal variables

    ```python
    class SignalOp(Enum):
        SIGNAL_SET
        SIGNAL_ADD
    ```

    |Enumerated Value|Meaning|
    |-|-|
    |SIGNAL_SET|Atomic setting: writes a given value to a remote signal.|
    |SIGNAL_ADD|Atomic addition: adds a given value to the existing value of a remote signal.|

1. CmpOp enumeration class: signal comparison operation types

    ```python
    class CmpOp(Enum):
        CMP_EQ
        CMP_NE
        CMP_GT
        CMP_GE
        CMP_LT
        CMP_LE
    ```

    |Enumerated Value|Meaning|
    |-|-|
    |CMP_EQ|Equal to (==)|
    |CMP_NE|Not equal to (!=)|
    |CMP_GT|Greater than (>)|
    |CMP_GE|Greater than or equal to (>=)|
    |CMP_LT|Less than (<)|
    |CMP_LE|Less than or equal to (<=)|
