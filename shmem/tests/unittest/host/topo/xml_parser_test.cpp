/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "aclshmemi_xml_parser.h"

namespace shm {
namespace topo {

TEST(XmlParserTest, ParsesTopologyHierarchyAndQuotedAttributes)
{
    const std::string xml = "<?xml version=\"1.0\"?>\n"
                            "<!-- topologyd output -->\n"
                            "<system version=\"1.0\"><cpu numaid='0'><ub><nic><net name=\"hrn5_0\"/></nic>"
                            "<npu chipphyid=\"0\"/></ub></cpu></system>";

    std::vector<aclshmemi_xml_tag_t> tags;
    std::string error;
    ASSERT_TRUE(aclshmemi_xml_parser_t::parse(xml, tags, error)) << error;
    ASSERT_EQ(tags.size(), 6U);
    EXPECT_EQ(tags[0].name, "system");
    EXPECT_EQ(tags[0].parent_index, -1);
    EXPECT_EQ(tags[1].name, "cpu");
    EXPECT_EQ(tags[1].parent_index, 0);
    EXPECT_EQ(tags[2].name, "ub");
    EXPECT_EQ(tags[2].depth, 2);
    EXPECT_TRUE(tags[4].self_closing);
    ASSERT_NE(tags[4].find_attr("name"), nullptr);
    EXPECT_EQ(*tags[4].find_attr("name"), "hrn5_0");
    ASSERT_NE(tags[5].find_attr("chipphyid"), nullptr);
    EXPECT_EQ(*tags[5].find_attr("chipphyid"), "0");
}

TEST(XmlParserTest, RejectsMismatchedClosingTag)
{
    const std::string xml = "<system><ub></pci></system>";
    std::vector<aclshmemi_xml_tag_t> tags;
    std::string error;
    EXPECT_FALSE(aclshmemi_xml_parser_t::parse(xml, tags, error));
    EXPECT_NE(error.find("does not match"), std::string::npos);
}

TEST(XmlParserTest, RejectsMissingOuterPciGroupClosingTag)
{
    const std::string xml = "<system><cpu><pci><pci busid=\"0000:03:00.0\"/>"
                            "<pci busid=\"0000:05:00.0\"><nic><net name=\"ens100f0\"/></nic></pci>"
                            "</cpu></system>";
    std::vector<aclshmemi_xml_tag_t> tags;
    std::string error;
    EXPECT_FALSE(aclshmemi_xml_parser_t::parse(xml, tags, error));
    EXPECT_EQ(error, "closing tag </cpu> does not match <pci>");
}

TEST(XmlParserTest, RejectsUnquotedAttribute)
{
    const std::string xml = "<system><ub><npu chipphyid=0/></ub></system>";
    std::vector<aclshmemi_xml_tag_t> tags;
    std::string error;
    EXPECT_FALSE(aclshmemi_xml_parser_t::parse(xml, tags, error));
    EXPECT_NE(error.find("must use quotes"), std::string::npos);
}

TEST(XmlParserTest, RejectsMultipleRootElements)
{
    const std::string xml = "<system/><system/>";
    std::vector<aclshmemi_xml_tag_t> tags;
    std::string error;
    EXPECT_FALSE(aclshmemi_xml_parser_t::parse(xml, tags, error));
    EXPECT_NE(error.find("multiple root"), std::string::npos);
}

TEST(XmlParserTest, ReportsMissingFile)
{
    std::vector<aclshmemi_xml_tag_t> tags;
    std::string error;
    EXPECT_FALSE(aclshmemi_xml_parser_t::parse_file("/tmp/aclshmem_xml_parser_missing_file", tags, error));
    EXPECT_NE(error.find("cannot open"), std::string::npos);
}

} // namespace topo
} // namespace shm
