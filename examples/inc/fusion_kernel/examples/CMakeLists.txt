# Public C API examples.  Keep these targets separate from the qualification
# drivers: they are intentionally small enough for an integrator to read end to
# end.
add_executable(inc_fusion_plan_smoke inc_fusion_plan_smoke.cpp)
inc_host_target(inc_fusion_plan_smoke)
target_link_libraries(inc_fusion_plan_smoke PRIVATE inc_fusion_api)

add_executable(inc_fusion_runtime_skeleton inc_fusion_runtime_skeleton.cpp)
inc_host_target(inc_fusion_runtime_skeleton)
target_link_libraries(inc_fusion_runtime_skeleton PRIVATE inc_fusion_api)
