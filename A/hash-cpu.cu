// A simple hash table implementation using only the CPU. The main function
// tests the hash table with ~13 million of randomly generated keys.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda-error.h"

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

    // Bucket array. Each bucket is a pointer to the first entry.
    ht_entry **entries;

    // Pool of free pre-allocated entries
    ht_entry *pool;

    // First free entry to be used.
    ht_entry *first_free;
};

// Returns a new allocated hash table with the given bucket size. It can store
// at maximium max_elements.
ht_table *ht_new(size_t bucket_size, size_t max_elements) {
    // Allocate ht_table.
    ht_table *table = (ht_table *)malloc(sizeof(ht_table));
    HANDLE_NULL(table);

    // Allocate the bucket array.
    table->size = bucket_size;
    table->entries = (ht_entry **)calloc(bucket_size, sizeof(ht_entry));
    HANDLE_NULL(table->entries);

    // Allocate the pool of free entries.
    table->pool = (ht_entry *)malloc(sizeof(ht_entry) * max_elements);
    HANDLE_NULL(table->pool);
    table->first_free = table->pool;

    return table;
}

// Free the hash table.
void ht_free(ht_table *table) {
    free(table->entries);
    free(table->pool);
}

// Returns the hash of the given key with the given bucket size.
inline size_t ht_hash(unsigned int key, size_t bucket_size) {
    return key % bucket_size;
}

// Add the given pair of key/value in the given hash table.
void ht_add(ht_table *table, unsigned int key, void *value) {
    // Calculate the hash of the given key.
    size_t key_hash = ht_hash(key, table->size);


    // Get a free entry from the pool.
    ht_entry *entry = table->first_free++; // first_free now points to a new entry.
    entry->key = key;
    entry->value = value;

    // Get the bucket from the table and add the entry in the bucket, moving the
    // old first entry to the next.
    ht_entry *bucket = table->entries[key_hash];
    entry->next = bucket;
    table->entries[key_hash] = entry;
}

// Checks if the hash table has correctly hashed all elements and if there are
// the expected number of elements in the buckets.
void ht_check(ht_table *table, size_t expected_elements) {
    size_t elements = 0;

    for (int bucket = 0; bucket < table->size; bucket++) {
        // Check al entries on a single bucket.
        ht_entry *entry = table->entries[bucket];
        while (entry != NULL) {
            elements++;

            // Check if this key has been hashed and located in the right
            // bucket.
            unsigned int key_hash = ht_hash(entry->key, table->size);
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
}

int main() {
    unsigned int *keys = (unsigned int *)malloc(ELEMENTS_SIZE);
    HANDLE_NULL(keys);

    // Generate random keys and add them to the hash table, tracking the time of
    // the operations.
    clock_t start, stop;
    start = clock();

    ht_table *table = ht_new(BUCKET_SIZE, ELEMENTS);
    HANDLE_NULL(table);

    for (int i = 0; i < ELEMENTS; i++) {
        keys[i] = rand();
        ht_add(table, keys[i], (void *)NULL); // Add keys without a value.
    }

    stop = clock();

    float elapsed = (float)(stop - start) / (float)CLOCKS_PER_SEC * 1000.0f;
    printf("Time to hash: %3.1f ms\n", elapsed);

    // Check table for errors.
    ht_check(table, ELEMENTS);

    // Free allocated memory.
    ht_free(table);
    free(keys);

    return EXIT_SUCCESS;
}
