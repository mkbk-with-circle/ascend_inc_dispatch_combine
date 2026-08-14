DEVICE API
=================================

shmem_device_amo.h
---------------------------------

.. doxygenfile:: shmem_device_amo.h
    :project: ACLSHMEM_CPP_API

shmem_device_cc.h
---------------------------------

.. doxygenfile:: shmem_device_cc.h
    :project: ACLSHMEM_CPP_API

shmem_device_mo.h
---------------------------------

.. doxygenfile:: shmem_device_mo.h
    :project: ACLSHMEM_CPP_API

shmem_device_p2p_sync.h
---------------------------------

.. doxygenfile:: shmem_device_p2p_sync.h
    :project: ACLSHMEM_CPP_API

shmem_device_rma.h
---------------------------------

.. doxygenfile:: gm2gm/shmem_device_rma.h
    :project: ACLSHMEM_CPP_API

.. doxygenfile:: ub2gm/shmem_device_rma.h
    :project: ACLSHMEM_CPP_API

.. doxygenfile:: ub2gm/engine/shmem_device_mte.h
    :project: ACLSHMEM_CPP_API

.. doxygenfile:: gm2gm/engine/shmem_device_mte.h
    :project: ACLSHMEM_CPP_API

.. doxygenfile:: gm2gm/engine/shmem_device_rdma.h
    :project: ACLSHMEM_CPP_API

.. doxygenfile:: gm2gm/engine/shmem_device_sdma.h
    :project: ACLSHMEM_CPP_API

UDMA put interfaces transfer at most 256 MB
(256 * 1024 * 1024 bytes) per asynchronous put request. The
``elem_size`` parameter is an element count, so the transferred byte
size is ``elem_size * sizeof(T)``. For larger data, split the operation
into multiple UDMA put requests whose byte size does not exceed 256 MB,
then use the matching completion or synchronization interface, such as
``aclshmemx_udma_quiet(pe)`` or the documented signal protocol, before
reading destination data or reusing buffers whose contents must remain
stable.

.. doxygenfile:: gm2gm/engine/shmem_device_udma.h
    :project: ACLSHMEM_CPP_API

shmem_device_so.h
---------------------------------

.. doxygenfile:: shmem_device_so.h
    :project: ACLSHMEM_CPP_API

shmem_device_team.h
---------------------------------

.. doxygenfile:: shmem_device_team.h
    :project: ACLSHMEM_CPP_API
