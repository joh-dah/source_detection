import os
import glob

if __name__ == "__main__":
    dir = os.path.join("data", "val", "karate")
    print(dir)
    size = len(glob.glob(dir + "/*.pt"))
    print(size)
