import subprocess
from collections import Counter

def compute_index(filename):
    """The atom for which the VP index should be calculated must be on the last line of the file."""
    result = subprocess.run(f'voronoi {filename}', shell=True, check=True, stdout=subprocess.PIPE).stdout
    index = result.strip().split()
    index = [int(x) for x in index]
    index = Counter(index)
    index = tuple(index[i] for i in range(3,13))
    return index

def main():
    import sys
    index = compute_index(sys.argv[1])
    print(index)

if __name__ == '__main__':
    main()
