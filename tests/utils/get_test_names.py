from lxml import etree

tree = etree.parse('test-results/junit.xml')

root = tree.getroot()

for test_case in root.getchildren()[0].getchildren():
    test_case.set('name', f'{test_case.get("file")}::{test_case.get("name")}')


with open('test-results/junit.xml', 'w') as f:
    f.write(etree.tostring(tree, pretty_print=True, encoding=str))
