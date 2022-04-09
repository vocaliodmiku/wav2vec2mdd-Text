#pragma once

#include <cstddef>
#include <cstdint>

namespace Simpleviterbi {

template <size_t Alignment = 256>
class Workspace {
    public:
        explicit Workspace(void* workspacePtr)
            : workspacePtr_(reinterpret_cast<uintptr_t>(workspacePtr)), offset_(0) {
                align();
            }
        
        template <class T>
        T* request(size_t s0, size_t s1 = 1, size_t s2 = 1, size_t s3 = 1) {
            align();
            auto p = reinterpret_cast<T*>(workspacePtr_ + offset_);
            offset_ += sizeof(T) * s0 * s1 * s2 * s3;
            return p;
        }

        template <class T>
        void request(T** p, size_t s0, size_t s1 = 1, size_t s2 = 1, size_t s3 = 1) {
            *p = request<T>(s0, s1, s2, s3);
        }

        size_t requiredSize() const {
            return offset_ + Alignment - 1;
        }

    private:
        void align() {
            offset_ += Alignment - 1 - (workspacePtr_ + offset_ + Alignment - 1) % Alignment;
        }
        const uintptr_t workspacePtr_;
        size_t offset_;
};

template <class Float>
struct WorkspacePtrs {
    explicit WorkspacePtrs(void* workspace, int B, int T, int N) {
        Workspace<> ws(workspace);
        ws.request(&alpha, B, 2, N);
        ws.request(&beta, B, T, N);
        requiredSize = ws.requiredSize();
    }

    Float* alpha;
    int* beta;
    size_t requiredSize;
};

template <class Float>
struct ViterbiPath {
    static size_t getWorkspaceSize(int B, int T, int N);
    static void compute(
        int B,
        int T,
        int N,
        const Float* input,
        const Float* trans,
        int* path,
        void* workspace);
};

}