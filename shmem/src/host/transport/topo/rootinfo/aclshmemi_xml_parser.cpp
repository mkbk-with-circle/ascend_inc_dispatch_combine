/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclshmemi_xml_parser.h"

#include <cctype>
#include <fstream>
#include <limits>
#include <utility>

namespace shm {
namespace topo {

namespace {

constexpr size_t MAX_XML_FILE_SIZE = 1024 * 1024;
constexpr size_t MAX_TAG_ENTRIES = 1024;
constexpr size_t MAX_TAG_NAME_LEN = 32;
constexpr size_t MAX_ATTRS_PER_TAG = 8;
constexpr size_t MAX_ATTR_NAME_LEN = 48;
constexpr size_t MAX_ATTR_VALUE_LEN = 64;

bool is_name_start(char ch)
{
    const auto value = static_cast<unsigned char>(ch);
    return std::isalpha(value) != 0 || ch == '_' || ch == ':';
}

bool is_name_char(char ch)
{
    const auto value = static_cast<unsigned char>(ch);
    return is_name_start(ch) || std::isdigit(value) != 0 || ch == '-' || ch == '.';
}

void skip_whitespace(const std::string& xml, size_t& pos)
{
    while (pos < xml.size() && std::isspace(static_cast<unsigned char>(xml[pos])) != 0) {
        ++pos;
    }
}

bool parse_name(
    const std::string& xml, size_t& pos, size_t max_length, const char* field_name, std::string& out,
    std::string& error_message)
{
    if (pos >= xml.size() || !is_name_start(xml[pos])) {
        error_message = std::string("invalid ") + field_name + " at byte " + std::to_string(pos);
        return false;
    }

    const size_t start = pos++;
    while (pos < xml.size() && is_name_char(xml[pos])) {
        ++pos;
    }
    const size_t length = pos - start;
    if (length > max_length) {
        error_message = std::string(field_name) + " is too long at byte " + std::to_string(start);
        return false;
    }
    out = xml.substr(start, length);
    return true;
}

bool skip_markup_declaration(const std::string& xml, size_t& pos, std::string& error_message)
{
    bool in_quote = false;
    char quote = '\0';
    int subset_depth = 0;
    while (pos < xml.size()) {
        const char ch = xml[pos++];
        if (in_quote) {
            if (ch == quote) {
                in_quote = false;
            }
            continue;
        }
        if (ch == '\'' || ch == '"') {
            in_quote = true;
            quote = ch;
        } else if (ch == '[') {
            ++subset_depth;
        } else if (ch == ']') {
            if (subset_depth > 0) {
                --subset_depth;
            }
        } else if (ch == '>' && subset_depth == 0) {
            return true;
        }
    }
    error_message = "unterminated markup declaration";
    return false;
}

bool parse_close_tag(
    const std::string& xml, size_t& pos, const std::vector<aclshmemi_xml_tag_t>& tags, std::vector<int>& open_tags,
    std::string& error_message)
{
    pos += 2;
    skip_whitespace(xml, pos);
    std::string name;
    if (!parse_name(xml, pos, MAX_TAG_NAME_LEN, "closing tag name", name, error_message)) {
        return false;
    }
    skip_whitespace(xml, pos);
    if (pos >= xml.size() || xml[pos] != '>') {
        error_message = "invalid closing tag at byte " + std::to_string(pos);
        return false;
    }
    ++pos;

    if (open_tags.empty()) {
        error_message = "closing tag </" + name + "> has no matching opening tag";
        return false;
    }
    const auto& open_tag = tags[static_cast<size_t>(open_tags.back())];
    if (open_tag.name != name) {
        error_message = "closing tag </" + name + "> does not match <" + open_tag.name + ">";
        return false;
    }
    open_tags.pop_back();
    return true;
}

bool parse_attrs_and_end(const std::string& xml, size_t& pos, aclshmemi_xml_tag_t& tag, std::string& error_message)
{
    while (pos < xml.size()) {
        skip_whitespace(xml, pos);
        if (pos >= xml.size()) {
            break;
        }
        if (xml[pos] == '>') {
            ++pos;
            return true;
        }
        if (xml[pos] == '/' && pos + 1 < xml.size() && xml[pos + 1] == '>') {
            pos += 2;
            tag.self_closing = true;
            return true;
        }
        if (tag.attrs.size() >= MAX_ATTRS_PER_TAG) {
            error_message = "too many attributes in <" + tag.name + ">";
            return false;
        }

        aclshmemi_xml_attr_t attr;
        if (!parse_name(xml, pos, MAX_ATTR_NAME_LEN, "attribute name", attr.name, error_message)) {
            return false;
        }
        for (const auto& existing : tag.attrs) {
            if (existing.name == attr.name) {
                error_message = "duplicate attribute '" + attr.name + "' in <" + tag.name + ">";
                return false;
            }
        }

        skip_whitespace(xml, pos);
        if (pos >= xml.size() || xml[pos] != '=') {
            error_message = "missing '=' after attribute '" + attr.name + "'";
            return false;
        }
        ++pos;
        skip_whitespace(xml, pos);
        if (pos >= xml.size() || (xml[pos] != '\'' && xml[pos] != '"')) {
            error_message = "attribute '" + attr.name + "' must use quotes";
            return false;
        }

        const char quote = xml[pos++];
        const size_t value_start = pos;
        while (pos < xml.size() && xml[pos] != quote) {
            if (xml[pos] == '<') {
                error_message = "invalid '<' in attribute '" + attr.name + "'";
                return false;
            }
            ++pos;
        }
        if (pos >= xml.size()) {
            error_message = "unterminated value for attribute '" + attr.name + "'";
            return false;
        }
        if (pos - value_start > MAX_ATTR_VALUE_LEN) {
            error_message = "value of attribute '" + attr.name + "' is too long";
            return false;
        }
        attr.value = xml.substr(value_start, pos - value_start);
        ++pos;
        tag.attrs.push_back(std::move(attr));
    }

    error_message = "unterminated opening tag <" + tag.name + ">";
    return false;
}

bool parse_open_tag(
    const std::string& xml, size_t& pos, std::vector<aclshmemi_xml_tag_t>& tags, std::vector<int>& open_tags,
    std::string& error_message)
{
    ++pos;
    aclshmemi_xml_tag_t tag;
    if (!parse_name(xml, pos, MAX_TAG_NAME_LEN, "tag name", tag.name, error_message)) {
        return false;
    }
    tag.parent_index = open_tags.empty() ? -1 : open_tags.back();
    tag.depth = static_cast<int>(open_tags.size());
    if (!parse_attrs_and_end(xml, pos, tag, error_message)) {
        return false;
    }
    if (tags.size() >= MAX_TAG_ENTRIES) {
        error_message = "XML contains too many tags";
        return false;
    }

    tags.push_back(std::move(tag));
    if (!tags.back().self_closing) {
        if (tags.size() - 1 > static_cast<size_t>(std::numeric_limits<int>::max())) {
            error_message = "XML tag index overflow";
            return false;
        }
        open_tags.push_back(static_cast<int>(tags.size() - 1));
    }
    return true;
}

bool contains_non_whitespace(const std::string& xml, size_t begin, size_t end)
{
    for (size_t i = begin; i < end; ++i) {
        if (std::isspace(static_cast<unsigned char>(xml[i])) == 0) {
            return true;
        }
    }
    return false;
}

} // namespace

const std::string* aclshmemi_xml_tag_t::find_attr(const std::string& attr_name) const
{
    for (const auto& attr : attrs) {
        if (attr.name == attr_name) {
            return &attr.value;
        }
    }
    return nullptr;
}

bool aclshmemi_xml_parser_t::parse_file(
    const std::string& path, std::vector<aclshmemi_xml_tag_t>& tags, std::string& error_message)
{
    tags.clear();
    error_message.clear();

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        error_message = "cannot open XML file: " + path;
        return false;
    }
    const std::streampos end = file.tellg();
    if (end <= 0) {
        error_message = "XML file is empty or unreadable: " + path;
        return false;
    }
    const auto file_size = static_cast<std::streamoff>(end);
    if (file_size > static_cast<std::streamoff>(MAX_XML_FILE_SIZE)) {
        error_message = "XML file exceeds size limit: " + path;
        return false;
    }
    file.seekg(0, std::ios::beg);

    std::string xml(static_cast<size_t>(file_size), '\0');
    if (!file.read(&xml[0], static_cast<std::streamsize>(xml.size()))) {
        error_message = "failed to read XML file: " + path;
        return false;
    }
    return parse(xml, tags, error_message);
}

bool aclshmemi_xml_parser_t::parse(
    const std::string& xml, std::vector<aclshmemi_xml_tag_t>& tags, std::string& error_message)
{
    tags.clear();
    error_message.clear();
    if (xml.empty()) {
        error_message = "XML content is empty";
        return false;
    }

    std::vector<int> open_tags;
    size_t pos = 0;
    if (xml.size() >= 3 && static_cast<unsigned char>(xml[0]) == 0xEF && static_cast<unsigned char>(xml[1]) == 0xBB &&
        static_cast<unsigned char>(xml[2]) == 0xBF) {
        pos = 3;
    }
    int root_count = 0;
    while (pos < xml.size()) {
        const size_t open = xml.find('<', pos);
        if (open == std::string::npos) {
            if (open_tags.empty() && contains_non_whitespace(xml, pos, xml.size())) {
                error_message = "non-whitespace text exists outside the root element";
                return false;
            }
            break;
        }
        if (open_tags.empty() && contains_non_whitespace(xml, pos, open)) {
            error_message = "non-whitespace text exists outside the root element";
            return false;
        }
        pos = open;

        if (xml.compare(pos, 4, "<!--") == 0) {
            const size_t end = xml.find("-->", pos + 4);
            if (end == std::string::npos) {
                error_message = "unterminated XML comment";
                return false;
            }
            pos = end + 3;
            continue;
        }
        if (xml.compare(pos, 2, "<?") == 0) {
            const size_t end = xml.find("?>", pos + 2);
            if (end == std::string::npos) {
                error_message = "unterminated processing instruction";
                return false;
            }
            pos = end + 2;
            continue;
        }
        if (xml.compare(pos, 9, "<![CDATA[") == 0) {
            if (open_tags.empty()) {
                error_message = "CDATA exists outside the root element";
                return false;
            }
            const size_t end = xml.find("]]>", pos + 9);
            if (end == std::string::npos) {
                error_message = "unterminated CDATA section";
                return false;
            }
            pos = end + 3;
            continue;
        }
        if (xml.compare(pos, 2, "<!") == 0) {
            pos += 2;
            if (!skip_markup_declaration(xml, pos, error_message)) {
                return false;
            }
            continue;
        }
        if (xml.compare(pos, 2, "</") == 0) {
            if (!parse_close_tag(xml, pos, tags, open_tags, error_message)) {
                return false;
            }
            continue;
        }

        const bool is_root = open_tags.empty();
        if (!parse_open_tag(xml, pos, tags, open_tags, error_message)) {
            return false;
        }
        if (is_root) {
            ++root_count;
            if (root_count > 1) {
                error_message = "XML contains multiple root elements";
                return false;
            }
        }
    }

    if (!open_tags.empty()) {
        error_message = "unclosed tag <" + tags[static_cast<size_t>(open_tags.back())].name + ">";
        return false;
    }
    if (root_count != 1 || tags.empty()) {
        error_message = "XML does not contain one root element";
        return false;
    }
    return true;
}

} // namespace topo
} // namespace shm
