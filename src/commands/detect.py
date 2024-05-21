import sys

def detect (datafile):
  pass

if __name__ == '__main__':
  input = open(sys.argv[1]).read()
  output = detect(input)
  open(sys.argv[2], 'w').write(output).close()