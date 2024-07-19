// A simple hash table implementation using the GPU. The main function tests the
// hash table with ~13 million of randomly generated keys.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda-error.h"

#include "lock.h"

// Size of elements in bytes.
const size_t ELEMENTS_SIZE = 100 * 1024 * 1024;

// Number of elements to store in the hash table.
const size_t ELEMENTS = ELEMENTS_SIZE / sizeof(unsigned int);

// Size of the bucket.
const size_t BUCKET_SIZE = 1024;

// ht_entry is a single entry of the hash table.
struct ht_entry {
    // Key of the entry.
    unsigned int key;

    // Value of the entry (memory managed by the caller).
    void *value;

    // Next entry in the same bucket. May be null if it is the last entry.
    ht_entry *next;
};

// ht_table is an hash table.
struct ht_table {
    // Size of the bucket array.
    size_t size;

    // Number of elements this hash table can store.
    size_t max_elements;

    // Bucket array. Each bucket is a pointer to the first entry.
    ht_entry **entries;

    // Pool of free pre-allocated entries
    ht_entry *pool;

    // First free entry to be used.
    ht_entry *first_free;
};

// Returns a new allocated hash table on the device with the given bucket size.
// It can store at maximium max_elements.
//
ht_table *ht_new(size_t bucket_size, size_t max_elements) {
    // Allocate the bucket array.
    ht_entry **dev_entries;
    HANDLE_ERROR(cudaMalloc((void **)&dev_entries, sizeof(ht_entry *) * bucket_size));
    HANDLE_ERROR(cudaMemset(dev_entries, 0, sizeof(ht_entry *) * bucket_size));

    // Allocate the pool of free entries.
    ht_entry *dev_pool;
    HANDLE_ERROR(cudaMalloc((void **)&dev_pool, sizeof(ht_entry) * max_elements));

    // Allocate ht_table.
    ht_table *dev_table;
    HANDLE_ERROR(cudaMalloc((void **)&dev_table, sizeof(ht_table)));

    // Local table on host, used to set the fields and copy them to the device.
    ht_table table;
    table.size = bucket_size;
    table.max_elements = max_elements;
    table.entries = dev_entries;
    table.pool = dev_pool;
    table.first_free = dev_pool;
    HANDLE_ERROR(cudaMemcpy(dev_table, &table, sizeof(ht_table), cudaMemcpyHostToDevice));

    return dev_table;
}

// Free the hash table.
void ht_free(ht_table *table) {
    // I need to copy the table structure to get the entries and pool addresses.
    ht_table host_table;
    HANDLE_ERROR(cudaMemcpy(&host_table, table, sizeof(ht_table), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(host_table.entries));
    HANDLE_ERROR(cudaFree(host_table.pool));
}

// Returns the hash of the given key with the given bucket size.
__device__ __host__ __forceinline__ size_t ht_hash(unsigned int key, size_t bucket_size) {
    return key % bucket_size;
}

// Returns a copy of dev_table to the host memory in host_table.
// host_table.entries and host_table.pool must be freed manually.
void ht_copy_to_host(ht_table *dev_table, ht_table *host_table) {
    // I need to copy the table structure to get the entries and pool addresses.
    // dev_table is a pointer on the device.
    HANDLE_ERROR(cudaMemcpy(host_table, dev_table, sizeof(ht_table), cudaMemcpyDeviceToHost));

    // Allocate entries and pool on host.
    ht_entry **host_entries = (ht_entry **)calloc(host_table->size, sizeof(ht_entry *));
    HANDLE_NULL(host_entries);
    ht_entry *host_pool = (ht_entry *)malloc(host_table->max_elements * sizeof(ht_entry));
    HANDLE_NULL(host_pool);

    // Copy the device pool and entries to the host.
    HANDLE_ERROR(cudaMemcpy(host_entries, host_table->entries,
                sizeof(ht_entry *) * host_table->size, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(host_pool, host_table->pool,
                sizeof(ht_entry) * host_table->max_elements, cudaMemcpyDeviceToHost));

    // The pointers in host_table.entries and host_table.pool are addresses on
    // the GPU, not in the CPU. But the relative offsets are still valid. So we
    // need to change every pointer manually and copy.
    //
    // The relative offset are converted to absolute using this formula:
    //      dev_entry_address - dev_pool_address + host_pool_address
    //
    // Note here host_table.pool is the device address, not the host address.
    for (size_t bucket = 0; bucket < host_table->size; bucket++)
        if (host_entries[bucket] != NULL)
            host_entries[bucket] = (ht_entry *)((size_t)host_entries[bucket] - (size_t)host_table->pool + (size_t)host_pool);

    for (size_t element = 0; element < host_table->max_elements; element++)
        if (host_pool[element].next != NULL)
            host_pool[element].next = (ht_entry *)((size_t)host_pool[element].next - (size_t)host_table->pool + (size_t)host_pool);

    host_table->entries = host_entries;
    host_table->pool = host_pool;
}

// Checks if the hash table has correctly hashed all elements and if there are
// the expected number of elements in the buckets.
void ht_check(ht_table *table, size_t expected_elements) {
    size_t elements = 0;

    ht_table host_table;
    ht_copy_to_host(table, &host_table);

    for (int bucket = 0; bucket < host_table.size; bucket++) {
        // Check al entries on a single bucket.
        ht_entry *entry = host_table.entries[bucket];
        while (entry != NULL) {
            elements++;

            // Check if this key has been hashed and located in the right
            // bucket.
            unsigned int key_hash = ht_hash(entry->key, host_table.size);
            if (key_hash != bucket) {
                printf("Key %d hashed to %d, but was located to %d\n",
                        entry->key, key_hash, bucket);
            }

            entry = entry->next;
        }
    }

    if (elements != expected_elements) {
        printf("Found %ld elements in hash table, expected %ld\n", elements,
                expected_elements);
    }

    free(host_table.entries);
    free(host_table.pool);
}

// Add all keys with associated values to the given table.
//
// Each lock protects a bucket in the table. Each thread add one ore more
// elements, the total is ELEMENTS.
__global__ void ht_add(ht_table *table, unsigned int *keys, void **values, mutex_t *locks) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x; // Total numer of threads.

    // Each loop add one element on the table.
    while (tid < ELEMENTS) {
        unsigned int key = keys[tid];
        size_t hash = ht_hash(key, table->size);

        for (int i = 0; i < warpSize; i++) {
            if ((tid % warpSize) == i) {
                ht_entry *entry = &(table->pool[tid]);
                entry->key = key;
                // entry->value = values[tid];

                mutexLock(&locks[hash]);
                entry->next = table->entries[hash];
                table->entries[hash] = entry;
                mutexUnlock(&locks[hash]);
            }
        }

        tid += offset;
    }
}

int main() {
    // Allocate memory and fill with generated random keys.
    unsigned int *keys = (unsigned int *)malloc(ELEMENTS_SIZE);
    HANDLE_NULL(keys);
    for (int i = 0; i < ELEMENTS; i++) {
        keys[i] = rand();
    }

    // Create events and start the first.
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start));

    // Allocate keys on the device.
    unsigned int *dev_keys;
    HANDLE_ERROR(cudaMalloc((void **)&dev_keys, ELEMENTS_SIZE));
    HANDLE_ERROR(cudaMemcpy(dev_keys, keys, ELEMENTS_SIZE, cudaMemcpyHostToDevice));

    // Allocate locks on the device.
    mutex_t *dev_locks;
    HANDLE_ERROR(cudaMalloc((void **)&dev_locks, BUCKET_SIZE * sizeof(mutex_t)));
    HANDLE_ERROR(cudaMemset(dev_locks, 0, BUCKET_SIZE * sizeof(mutex_t)));

    // Create table on the device.
    ht_table *dev_table = ht_new(BUCKET_SIZE, ELEMENTS);

    // 60 block with 256 threads for each block. Arbitrary values.
    // We do not allocate values because we set only empty keys.
    ht_add<<<60, 256>>>(dev_table, dev_keys, NULL, dev_locks);

    float elapsed;
    HANDLE_ERROR(cudaEventRecord(stop));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    HANDLE_ERROR(cudaEventElapsedTime(&elapsed, start, stop));
    printf("Time to hash: %3.1f ms\n", elapsed);

    // Check table for errors.
    ht_check(dev_table, ELEMENTS);

    // Free allocated memory.
    ht_free(dev_table);
    free(keys);
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));
    HANDLE_ERROR(cudaFree(dev_keys));
    HANDLE_ERROR(cudaFree(dev_locks));

    return EXIT_SUCCESS;
}
