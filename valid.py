import xml.etree.ElementTree as ET


tree = ET.parse('trace.xml')
root = tree.getroot()


for child in root:
    print(child.tag, child.attrib)


for elem in root.iter('specific_element'):
    print(elem.text)
