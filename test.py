import time
import sys

try:
	wait = int(sys.argv[1])
except:
	wait = 0
	
print(wait)

print(f"Start Program - wait: {wait}")
time.sleep(wait)
print(f"Stop Program -  wait: {wait}")
