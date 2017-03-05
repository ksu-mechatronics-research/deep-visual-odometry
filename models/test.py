#!/usr/local/lib/python3.5/dist-packages
import os
import sys
import json

PATH = os.path.dirname(os.path.abspath(__file__))

test_dict = {}

test_list = []
for i in range(5):
    test_list.append([96*i, 11, "prelu", 3, i])
test_list2 = []
for i in range(3):
    test_list2.append([4096*i, "prelu"])
test_dict['convolution'] = test_list
test_dict['dense'] = test_list2

with open(os.path.join(PATH, "test.json"), 'w') as f:
    json.dump(test_dict, f, indent=4)

print(test_dict)