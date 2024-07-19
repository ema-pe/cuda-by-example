#ifndef __LOCK_H__
#define __LOCK_H__

// This lock must be passed to the kernel with the volatile modifier, to avoid
// compiler optimisations that break the visibility of mutex changes.
typedef int mutex_t;

mutex_t *mutexCreate() {
        mutex_t *m;
        HANDLE_ERROR(cudaMalloc((void **)&m, sizeof(int)));
        HANDLE_ERROR(cudaMemset(m, 0, sizeof(int)));
        return m;
}

void mutexDestroy(mutex_t *m) {
        HANDLE_ERROR(cudaFree(m));
}

__device__ __forceinline__ void mutexLock(mutex_t *m) {
        // The thread trying to lock the mutex will spin until it can
        // change the atomic value, thanks to atomicCAS.
        while (atomicCAS(m, 0, 1) != 0);

        // The fence ensures that other threads read the updated value
        // of the mutex. This is because CUDA uses a weakly-ordered
        // memory model.
        //
        // I cannot use __syncthreads() because it needs to be called by
        // all threads in the same block. Only one thread calls the
        // fence instruction. Also __syncthreads() only synchronises the
        // threads in the same block.
        __threadfence();
}

__device__ __forceinline__ void mutexUnlock(mutex_t *m) {
        // Before calling the atomic instruction, we make sure to read
        // the correct value of the mutex.
        __threadfence();

        atomicExch(m, 0);
}

#endif // __LOCK_H__
