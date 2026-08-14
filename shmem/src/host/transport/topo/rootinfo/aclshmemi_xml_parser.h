/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLSHMEMI_XML_PARSER_HPP
#define ACLSHMEMI_XML_PARSER_HPP

#include <string>
#include <vector>

namespace shm {
namespace topo {

struct aclshmemi_xml_attr_t {
    std::string name;
    std::string value;
};

struct aclshmemi_xml_tag_t {
    std::string name;
    std::vector<aclshmemi_xml_attr_t> attrs;
    int parent_index{-1};
    int depth{0};
    bool self_closing{false};

    const std::string* find_attr(const std::string& attr_name) const;
};

class aclshmemi_xml_parser_t {
public:
    static bool parse_file(const std::string& path, std::vector<aclshmemi_xml_tag_t>& tags, std::string& error_message);
    static bool parse(const std::string& xml, std::vector<aclshmemi_xml_tag_t>& tags, std::string& error_message);
};

} // namespace topo
} // namespace shm

#endif // ACLSHMEMI_XML_PARSER_HPP
