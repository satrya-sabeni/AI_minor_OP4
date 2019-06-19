import json
import xml.etree.cElementTree as ET
import os

def json2xml(json_obj, line_padding = ""):
    result_list = list()

    json_obj_type = type(json_obj)

    if json_obj_type is list:
        for sub_elem in json_obj:
            result_list.append(json2xml(sub_elem, line_padding))

        return "\n".join(result_list)

    if json_obj_type is dict:
        for tag_name in json_obj:
            sub_obj = json_obj[tag_name]
            result_list.append("%s<%s>" % (line_padding, tag_name))
            result_list.append(json2xml(sub_obj, "\t" + line_padding))
            result_list.append("%s</%s>" % (line_padding, tag_name))

        return "\n".join(result_list)

    return "%s%s" % (line_padding, json_obj)


def GetJSON():
    path = "../JSON/via_project_12Jun2019_15h22m.json"
    file = open(path, "rb").read(); 
    return file

def Format():
    js = json.loads(GetJSON())
    for k, v in js['_via_img_metadata'].items():

        xml = json2xml(v)
        filename ="ResultXML/" + k+".xml"
        f = open(filename, "w+")
        f.write(str(xml))

Format()
