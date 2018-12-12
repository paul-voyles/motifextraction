with open("run_all.sh", "w") as f:
    for i in range(158):
        print(f"python align_clusters.py {i} &", file=f)
        if i % 4 == 0 and i > 0:
            print("wait", file=f)
    print("wait", file=f)

    for i in range(158):
        print(f"python extract_errors.py ../data/results/{i}.xyz.json 3.6 &", file=f)
        if i % 4 == 0 and i > 0:
            print("wait", file=f)
    print("wait", file=f)
