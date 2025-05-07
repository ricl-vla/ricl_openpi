import time
import os

memory_hog = []
chunk_size_mb = 10000  # Allocate 100MB at a time
chunk_size_bytes = chunk_size_mb * 1024 * 1024
total_allocated_gb = 0

print(f"Starting RAM fill process. PID: {os.getpid()}")
print(f"Allocating in chunks of {chunk_size_mb} MB...")

try:
    while True:
        # try:
        # Create a large byte array chunk
        chunk = bytearray(chunk_size_bytes)
        memory_hog.append(chunk)
        total_allocated_gb = (len(memory_hog) * chunk_size_bytes) / (1024**3)
        print(f"Total RAM allocated: {total_allocated_gb:.2f} GB")
        # Optional: Add a small delay to allow monitoring
        # time.sleep(0.1)
        # except MemoryError:
        #     print("-" * 30)
        #     print(f"!!! MemoryError caught! Likely reached RAM limit. !!!")
        #     print(f"Final allocation approx: {total_allocated_gb:.2f} GB")
        #     print("Holding memory... Press Ctrl+C to attempt release and exit.")
        #     print("-" * 30)
        #     # Keep the allocated memory alive until interrupted
        #     while True:
        #         time.sleep(60)
except KeyboardInterrupt:
    print("\nCtrl+C received. Releasing memory (may take time)...")
    # Explicitly clear the list to signal garbage collection
    memory_hog.clear()
    print("Memory released (hopefully). Exiting.")
    # Note: Python's GC might take time, or the OS might have started swapping heavily.