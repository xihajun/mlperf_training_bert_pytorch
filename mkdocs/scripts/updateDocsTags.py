import re
import os
import json

# read all .md files
readmefiles = []
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".md"):
             readmefiles.append(os.path.join(root, file))

# load variable json
with open('mkdocsDefs.json') as f:
  var_dic = json.load(f)


def replaceTagVal4DiffQuotes(tag, replaceByVal, content):

    matchPattern41quote = r"(<a vartag='{}')(>.*?<)(/a>)".format(tag)
    content = re.sub(matchPattern41quote, replaceByVal, content)
    matchPattern42quote = r"(<a vartag=\"{}\")(>.*?<)(/a>)".format(tag)
    content = re.sub(matchPattern42quote, replaceByVal, content)
    return content

# match old pattern (<variable-*.?-tag>)(`*.?`)
# example: (<variable-VERSION-tag>)(`1.1`)
# new pattern (<a vartag=\"{}\")(>.*?<)(/a>)

def updateOldPattern2New(content):
    matchOldPattern = r"(<variable-)(.*?)(-tag>)(`.*?`)"
    replaceNewPattern = r"<a vartag='\2'>\4</a>"
    content = re.sub(matchOldPattern, replaceNewPattern, content)
    return content

for filename in readmefiles:
    with open(filename,"r") as f:
      content = f.read()
    
    # update readme variables
    for tag, value in var_dic.items():

      content = updateOldPattern2New(content)

      replaceByVal = r"\1>{}<\3".format(value)
      content = replaceTagVal4DiffQuotes(tag, replaceByVal, content)
      
    with open(filename,"w") as f:
      f.write(content)