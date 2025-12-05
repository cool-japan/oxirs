// OpenCL kernel for triple pattern matching
// This kernel matches triples against a pattern with wildcards

__kernel void triple_pattern_match(
    __global const ulong* subjects,
    __global const ulong* predicates,
    __global const ulong* objects,
    const ulong pattern_subject,
    const ulong pattern_predicate,
    const ulong pattern_object,
    __global uchar* matches
) {
    const int gid = get_global_id(0);

    // Wildcard is represented as ULONG_MAX
    const ulong WILDCARD = ULONG_MAX;

    // Check if triple matches pattern
    uchar match = 1;

    // Check subject
    if (pattern_subject != WILDCARD && subjects[gid] != pattern_subject) {
        match = 0;
    }

    // Check predicate
    if (pattern_predicate != WILDCARD && predicates[gid] != pattern_predicate) {
        match = 0;
    }

    // Check object
    if (pattern_object != WILDCARD && objects[gid] != pattern_object) {
        match = 0;
    }

    matches[gid] = match;
}

// Kernel for counting matches (returns partial counts)
__kernel void count_pattern_matches(
    __global const ulong* subjects,
    __global const ulong* predicates,
    __global const ulong* objects,
    const ulong pattern_subject,
    const ulong pattern_predicate,
    const ulong pattern_object,
    __global ulong* partial_counts,
    const int count_per_workitem
) {
    const int gid = get_global_id(0);
    const int start = gid * count_per_workitem;
    const int end = start + count_per_workitem;

    const ulong WILDCARD = ULONG_MAX;

    ulong count = 0;
    for (int i = start; i < end; i++) {
        uchar match = 1;

        if (pattern_subject != WILDCARD && subjects[i] != pattern_subject) {
            match = 0;
        }
        if (pattern_predicate != WILDCARD && predicates[i] != pattern_predicate) {
            match = 0;
        }
        if (pattern_object != WILDCARD && objects[i] != pattern_object) {
            match = 0;
        }

        count += match;
    }

    partial_counts[gid] = count;
}

// Kernel for join operations
__kernel void join_triples_subject(
    __global const ulong* left_subjects,
    __global const ulong* left_predicates,
    __global const ulong* left_objects,
    const int left_count,
    __global const ulong* right_subjects,
    __global const ulong* right_predicates,
    __global const ulong* right_objects,
    const int right_count,
    __global int* match_indices
) {
    const int left_idx = get_global_id(0);

    if (left_idx >= left_count) return;

    const ulong left_subject = left_subjects[left_idx];

    int match_count = 0;
    for (int right_idx = 0; right_idx < right_count; right_idx++) {
        if (left_subject == right_subjects[right_idx]) {
            // Store match pair indices
            int out_idx = atomic_inc(&match_indices[0]);
            match_indices[out_idx * 2 + 1] = left_idx;
            match_indices[out_idx * 2 + 2] = right_idx;
            match_count++;
        }
    }
}
