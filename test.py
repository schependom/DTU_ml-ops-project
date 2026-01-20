import multiprocessing

cores = multiprocessing.cpu_count()
print(f"Number of cores: {cores}, Number of threads: {2 * cores}")
