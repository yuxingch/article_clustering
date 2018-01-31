import os
import re
import csv
from nltk.corpus import stopwords
sw = set(stopwords.words('english'))

def main():
    #   import file 
    file_name = "hoodline_challenge.csv"
    input_csv = open(file_name, 'rb')
    contents = []
    reader = csv.DictReader(input_csv)
    for row in reader:
        clean_row = re.sub('[^a-zA-Z]+', ' ', row['content'])
        contents.append(clean_row.lower())
    temp = ','.join(contents)
        
    sw.update(('www','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','https','http','dir','href','br','li')) #remove stop words
    vocabulary = filter(lambda w: not w in sw, temp.split())
    output = ' '.join(vocabulary)
    output_file = open('2_contents.txt', 'w')
    output_file.write(output)
    output_file.close()


if __name__ == "__main__":
    main()