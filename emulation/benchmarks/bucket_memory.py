import random
import heapq
import math
from collections import defaultdict
from typing import List, Dict, Any, Optional


class BucketMemory:
    """
    Efficient class-balanced memory supporting insert/delete/batch replacement/quota replacement and multiple retrieval strategies.
    - buckets: class -> list[sample]
    - Each sample must contain the field 'klass'.
    """

    def __init__(self, memory_size: int):
        if memory_size <= 0:
            raise ValueError("memory_size must be positive.")
        self.memory_size = memory_size
        self.buckets: Dict[Any, List[dict]] = defaultdict(list)
        self.class_counts: Dict[Any, int] = defaultdict(int)
        self.heap: List = []  # (-count, class)
        self.total_count = 0

    def __len__(self) -> int:
        return self.total_count

    # ===========================
    # Basic insert / delete / replace
    # ===========================

    def _insert(self, sample: dict, klass: Any):
        self.buckets[klass].append(sample)
        self.class_counts[klass] += 1
        self.total_count += 1
        heapq.heappush(self.heap, (-self.class_counts[klass], klass))

    def random_delete(self, klass: Any) -> Optional[dict]:
        """Randomly delete one sample from a class using O(1) swap-delete."""
        bucket = self.buckets.get(klass, [])
        if not bucket:
            return None
        idx = random.randrange(len(bucket))
        deleted = bucket[idx]
        bucket[idx] = bucket[-1]
        bucket.pop()
        self.class_counts[klass] -= 1
        self.total_count -= 1
        # Drop the key when a class becomes empty to keep the mapping tidy.
        if self.class_counts[klass] == 0:
            del self.buckets[klass]
            del self.class_counts[klass]
        return deleted

    def _replace_one(self, sample: dict, klass: Any):
        """Replace one sample from the class with the largest population."""
        while self.heap:
            neg_count, top_class = heapq.heappop(self.heap)
            if top_class in self.class_counts and self.class_counts[top_class] == -neg_count and self.class_counts[top_class] > 0:
                break
        else:
            # No replaceable class found (very rare).
            return
        self.random_delete(top_class)
        self._insert(sample, klass)

    # ===========================
    # Insert / batch replace / quota replace
    # ===========================

    def add_sample(self, sample: dict):
        klass = sample["klass"]
        if self.total_count < self.memory_size:
            self._insert(sample, klass)
        else:
            self._replace_one(sample, klass)

    def replace_batch(self, samples: List[dict]):
        """Insert or replace samples one by one."""
        if not samples:
            return
        for sample in samples:
            self.add_sample(sample)

    def replace_batch_quota(self, samples: List[dict]):
        """
        Quota-based batch replacement:
        - Compute total_after = min(total_count + len(samples), memory_size)
        - Build class set as existing_classes U new_sample_classes
        - Compute quota = total_after / num_classes
        - Delete overflow samples from oversized classes, then insert or replace with new samples
        """
        if not samples:
            return

        # Fast path: enough headroom to insert everything without balancing.
        if self.total_count + len(samples) <= self.memory_size:
            for sample in samples:
                self._insert(sample, sample["klass"])
            return

        # Use available headroom first, then handle the remainder with balancing.
        headroom = max(0, self.memory_size - self.total_count)
        if headroom > 0:
            for sample in samples[:headroom]:
                self._insert(sample, sample["klass"])
            samples = samples[headroom:]
            if not samples:
                return

        new_by_class: Dict[Any, List[dict]] = defaultdict(list)
        for sample in samples:
            new_by_class[sample["klass"]].append(sample)

        all_classes = set(self.class_counts.keys()) | set(new_by_class.keys())
        if not all_classes:
            return

        num_classes = len(all_classes)
        total_after = self.memory_size  # memory is saturated at this point
        quota = total_after / num_classes
        lower_quota = int(math.floor(quota))
        upper_quota = int(math.ceil(quota))

        future_counts: Dict[Any, int] = {
            cls: self.class_counts.get(cls, 0) + len(new_by_class.get(cls, []))
            for cls in all_classes
        }

        target_counts: Dict[Any, int] = {
            cls: min(lower_quota, future_counts[cls])
            for cls in all_classes
        }
        max_allocation: Dict[Any, int] = {
            cls: min(future_counts[cls], upper_quota)
            for cls in all_classes
        }

        remaining = total_after - sum(target_counts.values())
        if remaining > 0:
            # Prefer classes that still have spare samples beyond the lower quota.
            sorted_classes = sorted(
                all_classes,
                key=lambda cls: (future_counts[cls] - target_counts[cls], future_counts[cls], cls),
                reverse=True,
            )
            for cls in sorted_classes:
                if remaining <= 0:
                    break
                capacity = max_allocation[cls] - target_counts[cls]
                if capacity <= 0:
                    continue
                add = min(capacity, remaining)
                target_counts[cls] += add
                remaining -= add

        # If we still have room to distribute (rare when upper_quota limits), allow spilling beyond upper_quota.
        if remaining > 0:
            sorted_classes = sorted(
                all_classes,
                key=lambda cls: (future_counts[cls] - target_counts[cls], future_counts[cls], cls),
                reverse=True,
            )
            for cls in sorted_classes:
                if remaining <= 0:
                    break
                capacity = future_counts[cls] - target_counts[cls]
                if capacity <= 0:
                    continue
                add = min(capacity, remaining)
                target_counts[cls] += add
                remaining -= add

        # Ensure total matches the budget to avoid rounding drift.
        if remaining != 0:
            # Fallback: quota rounding could leave a one-off gap; keep the invariants by spilling into any class.
            sorted_classes = sorted(all_classes, key=lambda cls: future_counts[cls], reverse=True)
            for cls in sorted_classes:
                if remaining <= 0:
                    break
                additional = min(future_counts[cls] - target_counts[cls], remaining)
                if additional <= 0:
                    continue
                target_counts[cls] += additional
                remaining -= additional
            remaining = 0

        for cls in all_classes:
            target = target_counts[cls]
            incoming_samples = new_by_class.get(cls, [])
            incoming_keep = min(len(incoming_samples), target)
            existing_needed = target - incoming_keep

            current_count = self.class_counts.get(cls, 0)
            delete_count = max(0, current_count - existing_needed)
            for _ in range(delete_count):
                if self.random_delete(cls) is None:
                    break

            # Insert the newest samples first to favour recency.
            if incoming_keep:
                for sample in incoming_samples[-incoming_keep:]:
                    self._insert(sample, cls)

            # Replace in-class samples with any remaining incoming data to inject freshness without growing the class.
            if target > 0 and len(incoming_samples) > incoming_keep:
                for sample in incoming_samples[:-incoming_keep]:
                    if self.random_delete(cls) is None:
                        break
                    self._insert(sample, cls)

    # ===========================
    # Retrieval methods
    # ===========================

    def retrieval(self, size: int) -> List[dict]:
        """
        Global random retrieval without replacement (up to `size` samples).
        Complexity: O(N) due to constructing `all_samples`, then `random.sample`.
        """
        if size <= 0 or self.total_count == 0:
            return []
        if size >= self.total_count:
            # Return all samples (shallow copy).
            return [s for bucket in self.buckets.values() for s in bucket]
        all_samples = [s for bucket in self.buckets.values() for s in bucket]
        return random.sample(all_samples, size)

    def balanced_retrieval(self, size: int) -> List[dict]:
        """
        Class-balanced retrieval (class choice with replacement; per-class sample without replacement).
        Works well when approximate class balance is desired.
        Implementation details:
          - Build `valid_classes` with at least one sample each
          - Sample `size` classes uniformly with replacement from `valid_classes`
          - For each chosen class, randomly select one sample from its bucket
        """
        if size <= 0 or self.total_count == 0 or not self.class_counts:
            return []

        valid_classes = [c for c, cnt in self.class_counts.items() if cnt > 0]
        if not valid_classes:
            return []

        # Return everything if the request count exceeds current population.
        if size >= self.total_count:
            return [s for bucket in self.buckets.values() for s in bucket]

        result: List[dict] = []
        for _ in range(size):
            cls = random.choice(valid_classes)  # Equal probability per class (can be weighted if needed).
            bucket = self.buckets[cls]
            # Class-level sampling is without replacement only per call. We reuse samples if the class has size 1.
            # For strict intra-class no-replacement semantics you would track used indices separately.
            sample = random.choice(bucket)
            result.append(sample)
        return result

    def quota_balanced_retrieval(self, size: int) -> List[dict]:
        """
        Quota-style balanced retrieval (closer to global balance):
        - Split requested `size` across classes (quota = size / num_classes)
        - Take all samples from classes with population below quota
        - For classes above quota, take `floor(quota)` first and use a round-robin pool for remaining demand
        Retrieval aims for no replacement; if insufficient samples remain, fall back to what's available.
        """
        if size <= 0 or self.total_count == 0 or not self.class_counts:
            return []

        total_available = self.total_count
        if size >= total_available:
            return [s for bucket in self.buckets.values() for s in bucket]

        class_to_indices = {cls: list(range(len(self.buckets[cls]))) for cls in self.buckets if len(self.buckets[cls]) > 0}
        classes = list(class_to_indices.keys())
        num_classes = len(classes)
        if num_classes == 0:
            return []

        target_float = size / num_classes
        lower_quota = int(math.floor(target_float))

        selected_samples: List[dict] = []
        extra_pools: List[List[int]] = []

        # First pass: take everything from under-quota classes; take `lower_quota` from larger classes.
        for cls in classes:
            idx_list = class_to_indices[cls]
            count = len(idx_list)
            if count <= target_float:
                # Class population below quota: take every sample.
                selected_samples.extend([self.buckets[cls][i] for i in idx_list])
            else:
                # Take `lower_quota` first (may be zero).
                if lower_quota > 0:
                    chosen = random.sample(idx_list, min(lower_quota, count))
                    selected_samples.extend([self.buckets[cls][i] for i in chosen])
                    chosen_set = set(chosen)
                    remaining = [i for i in idx_list if i not in chosen_set]
                else:
                    remaining = idx_list.copy()
                if remaining:
                    extra_pools.append((cls, remaining))

        # If the request is already satisfied, shuffle and truncate.
        if len(selected_samples) >= size:
            random.shuffle(selected_samples)
            return selected_samples[:size]

        # Second pass: round-robin through remaining pools until filled.
        remaining_needed = size - len(selected_samples)
        pool_idx = 0
        # convert extra_pools to modifiable lists of indices
        extra_pools_lists = [(cls, lst.copy()) for cls, lst in extra_pools]
        while remaining_needed > 0 and extra_pools_lists:
            cls, idxs = extra_pools_lists[pool_idx]
            # Randomly pick an index and remove it from the pool.
            pick_pos = random.randrange(len(idxs))
            pick_idx = idxs.pop(pick_pos)
            selected_samples.append(self.buckets[cls][pick_idx])
            remaining_needed -= 1
            if not idxs:
                extra_pools_lists.pop(pool_idx)
                if not extra_pools_lists:
                    break
                pool_idx %= len(extra_pools_lists)
            else:
                pool_idx = (pool_idx + 1) % len(extra_pools_lists)

        # Safety fallback (should rarely trigger).
        if len(selected_samples) < size:
            all_samples = [s for bucket in self.buckets.values() for s in bucket]
            random.shuffle(all_samples)
            return all_samples[:size]

        random.shuffle(selected_samples)
        return selected_samples[:size]

    # ===========================
    # Utilities
    # ===========================

    def sample(self, k: int = 1) -> List[dict]:
        """Alias for global random retrieval without replacement."""
        return self.retrieval(k)

    def summary(self):
        print(f"Memory Usage: {self.total_count}/{self.memory_size}")
        for klass, count in sorted(self.class_counts.items(), key=lambda x: -x[1]):
            print(f"  Class {klass}: {count} samples")


# ===========================
# Simple demonstration
# ===========================
if __name__ == "__main__":
    mem = BucketMemory(memory_size=20)
    # init
    samples = [{"klass": "A", "x": i} for i in range(8)] + \
              [{"klass": "B", "x": i} for i in range(4)] + \
              [{"klass": "C", "x": i} for i in range(3)]
    mem.replace_batch(samples)
    print("=== summary ===")
    mem.summary()

    print("\nretrieval(5):")
    print(mem.retrieval(5))

    print("\nbalanced_retrieval(6):")
    print(mem.balanced_retrieval(6))

    print("\nquota_balanced_retrieval(9):")
    print(mem.quota_balanced_retrieval(9))
