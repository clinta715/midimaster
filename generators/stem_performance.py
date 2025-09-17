"""
Performance Optimization System for Multi-Stem MIDI Generation

This module provides comprehensive performance optimizations including:
- Memory management and pooling
- Computational optimizations
- Caching and reuse strategies
- Parallel processing management
- Resource monitoring and limits
"""

from typing import Dict, List, Any, Optional, Set, Tuple, Deque
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
import psutil
import os
from collections import defaultdict, deque
import weakref
import gc

from generators.stem_manager import StemRole, StemData, StemConfig
from generators.pattern_orchestrator import PatternOrchestrator


class OptimizationLevel(Enum):
    """Performance optimization levels."""
    MINIMAL = "minimal"      # Basic optimizations
    STANDARD = "standard"    # Balanced performance
    AGGRESSIVE = "aggressive"  # Maximum performance
    ULTRA = "ultra"         # Extreme optimizations (may affect quality)


class CacheStrategy(Enum):
    """Caching strategies for stem data."""
    NONE = "none"           # No caching
    MEMORY = "memory"       # In-memory caching
    DISK = "disk"          # Disk-based caching
    HYBRID = "hybrid"      # Memory + disk caching


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring and optimization."""
    total_processing_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    cache_hit_rate: float = 0.0
    stems_generated_per_second: float = 0.0
    average_stem_size_kb: float = 0.0
    thread_utilization: float = 0.0
    memory_fragmentation_ratio: float = 0.0


@dataclass
class CacheEntry:
    """Cache entry for stem data."""
    stem_data: StemData
    access_time: float
    size_bytes: int
    frequency: int = 0
    expires_at: Optional[float] = None


@dataclass
class MemoryPool:
    """Memory pool for efficient allocation."""
    pool_size: int
    block_size: int
    free_blocks: Deque[bytes] = field(default_factory=lambda: deque())
    used_blocks: Set[bytes] = field(default_factory=set)

    def allocate(self) -> Optional[bytes]:
        """Allocate a memory block from the pool."""
        if self.free_blocks:
            block = self.free_blocks.popleft()
            self.used_blocks.add(block)
            return block
        return None

    def deallocate(self, block: bytes) -> None:
        """Return a memory block to the pool."""
        if block in self.used_blocks:
            self.used_blocks.remove(block)
            if len(self.free_blocks) < self.pool_size:
                self.free_blocks.append(block)

    def get_utilization(self) -> float:
        """Get pool utilization ratio."""
        return len(self.used_blocks) / self.pool_size if self.pool_size > 0 else 0.0


class StemPerformanceOptimizer:
    """
    Performance optimization system for multi-stem generation.

    Features:
    - Memory pooling and management
    - Intelligent caching system
    - Parallel processing optimization
    - Resource monitoring and limiting
    - Computational optimizations
    """

    def __init__(self,
                 optimization_level: OptimizationLevel = OptimizationLevel.STANDARD,
                 cache_strategy: CacheStrategy = CacheStrategy.MEMORY,
                 max_memory_mb: int = 512,
                 cache_size_mb: int = 256,
                 thread_pool_size: int = 4):
        """
        Initialize the performance optimizer.

        Args:
            optimization_level: Level of optimization to apply
            cache_strategy: Caching strategy to use
            max_memory_mb: Maximum memory usage in MB
            cache_size_mb: Maximum cache size in MB
            thread_pool_size: Size of thread pool for parallel processing
        """
        self.optimization_level = optimization_level
        self.cache_strategy = cache_strategy
        self.max_memory_mb = max_memory_mb
        self.cache_size_mb = cache_size_mb
        self.thread_pool_size = thread_pool_size

        # Initialize components
        self._memory_pools = self._initialize_memory_pools()
        self._cache = self._initialize_cache()
        self._metrics = PerformanceMetrics()
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitoring = False

        # Performance tracking
        self._processing_times: Deque[float] = deque(maxlen=100)
        self._memory_usage_history: Deque[float] = deque(maxlen=50)

        # Start monitoring if enabled
        if optimization_level.value in ['standard', 'aggressive', 'ultra']:
            self.start_monitoring()

    def _initialize_memory_pools(self) -> Dict[str, MemoryPool]:
        """Initialize memory pools for different data types."""
        pools = {}

        # Pool sizes based on optimization level
        if self.optimization_level == OptimizationLevel.AGGRESSIVE:
            pool_configs = {
                'midi_data': (100, 1024),      # 100 blocks of 1KB each
                'pattern_data': (50, 4096),    # 50 blocks of 4KB each
                'cache_entries': (200, 512)   # 200 blocks of 512B each
            }
        elif self.optimization_level == OptimizationLevel.ULTRA:
            pool_configs = {
                'midi_data': (200, 1024),
                'pattern_data': (100, 4096),
                'cache_entries': (500, 512)
            }
        else:
            pool_configs = {
                'midi_data': (50, 1024),
                'pattern_data': (25, 4096),
                'cache_entries': (100, 512)
            }

        for pool_name, (pool_size, block_size) in pool_configs.items():
            pools[pool_name] = MemoryPool(
                pool_size=pool_size,
                block_size=block_size
            )

        return pools

    def _initialize_cache(self) -> Dict[str, CacheEntry]:
        """Initialize the caching system."""
        if self.cache_strategy == CacheStrategy.NONE:
            return {}

        cache = {}

        # Set cache expiration times based on strategy
        if self.cache_strategy == CacheStrategy.MEMORY:
            # Keep in memory for 10 minutes
            self._cache_expiry = 600.0
        elif self.cache_strategy == CacheStrategy.DISK:
            # Keep on disk for 1 hour
            self._cache_expiry = 3600.0
        else:  # HYBRID
            # Keep in memory for 5 minutes, on disk for 30 minutes
            self._cache_expiry = 300.0

        return cache

    def optimize_stem_generation(self,
                               stem_configs: Dict[StemRole, StemConfig],
                               generation_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize stem generation parameters and setup.

        Args:
            stem_configs: Stem configurations to optimize
            generation_context: Generation context information

        Returns:
            Optimized generation parameters
        """
        optimized_params = {
            'use_caching': self._should_use_caching(generation_context),
            'parallel_processing': self._should_use_parallel(generation_context),
            'memory_preallocation': self._calculate_memory_preallocation(stem_configs),
            'batch_processing': self._optimize_batch_processing(stem_configs),
            'quality_tradeoffs': self._calculate_quality_tradeoffs(generation_context)
        }

        return optimized_params

    def _should_use_caching(self, context: Dict[str, Any]) -> bool:
        """Determine if caching should be used."""
        if self.cache_strategy == CacheStrategy.NONE:
            return False

        # Use caching for repeated patterns or similar stems
        has_repeats = context.get('has_repeated_patterns', False)
        complexity_level = context.get('complexity_level', 'medium')

        if self.optimization_level == OptimizationLevel.ULTRA:
            return True  # Always cache in ultra mode
        elif self.optimization_level == OptimizationLevel.AGGRESSIVE:
            return has_repeats or complexity_level in ['high', 'very_high']
        else:
            return has_repeats and complexity_level == 'very_high'

    def _should_use_parallel(self, context: Dict[str, Any]) -> bool:
        """Determine if parallel processing should be used."""
        stem_count = context.get('stem_count', 1)
        system_cores = context.get('available_cores', 4)

        if self.optimization_level == OptimizationLevel.MINIMAL:
            return False
        elif stem_count < 3:
            return False  # Not worth parallelizing for few stems
        elif system_cores < 2:
            return False  # Not enough cores
        else:
            return True

    def _calculate_memory_preallocation(self, stem_configs: Dict[StemRole, StemConfig]) -> Dict[str, int]:
        """Calculate memory preallocation requirements."""
        # Estimate memory needs based on stem types and count
        base_memory_per_stem = 50  # KB per stem
        complexity_multiplier = {
            'low': 0.8,
            'medium': 1.0,
            'high': 1.5,
            'very_high': 2.0
        }

        total_stems = len(stem_configs)
        avg_complexity = 'medium'  # Would be calculated from stem configs

        estimated_memory_kb = (
            total_stems * base_memory_per_stem *
            complexity_multiplier.get(avg_complexity, 1.0)
        )

        return {
            'midi_buffer_kb': int(estimated_memory_kb * 0.4),
            'pattern_buffer_kb': int(estimated_memory_kb * 0.3),
            'cache_buffer_kb': int(estimated_memory_kb * 0.3)
        }

    def _optimize_batch_processing(self, stem_configs: Dict[StemRole, StemConfig]) -> Dict[str, Any]:
        """Optimize batch processing parameters."""
        stem_count = len(stem_configs)

        if self.optimization_level == OptimizationLevel.ULTRA:
            batch_size = min(stem_count, 8)
            prefetch_count = 2
        elif self.optimization_level == OptimizationLevel.AGGRESSIVE:
            batch_size = min(stem_count, 4)
            prefetch_count = 1
        else:
            batch_size = min(stem_count, 2)
            prefetch_count = 0

        return {
            'batch_size': batch_size,
            'prefetch_count': prefetch_count,
            'use_streaming': stem_count > 6
        }

    def _calculate_quality_tradeoffs(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate quality tradeoffs for performance optimization."""
        base_quality = 1.0

        if self.optimization_level == OptimizationLevel.ULTRA:
            # 20% quality reduction for maximum performance
            quality_reduction = 0.2
        elif self.optimization_level == OptimizationLevel.AGGRESSIVE:
            # 10% quality reduction for good performance
            quality_reduction = 0.1
        else:
            quality_reduction = 0.0

        return {
            'pattern_complexity': max(0.1, base_quality - quality_reduction),
            'velocity_resolution': max(0.5, base_quality - quality_reduction * 0.5),
            'timing_precision': max(0.8, base_quality - quality_reduction * 0.2)
        }

    def cache_stem_data(self,
                       cache_key: str,
                       stem_data: StemData,
                       expiry_seconds: Optional[float] = None) -> None:
        """
        Cache stem data for future reuse.

        Args:
            cache_key: Unique key for the cached data
            stem_data: Stem data to cache
            expiry_seconds: Optional expiry time in seconds
        """
        if self.cache_strategy == CacheStrategy.NONE:
            return

        # Calculate data size (simplified)
        size_bytes = self._estimate_data_size(stem_data)

        # Check cache size limits
        if self._get_cache_size_mb() + (size_bytes / 1024 / 1024) > self.cache_size_mb:
            self._evict_cache_entries(size_bytes)

        # Create cache entry
        expires_at = time.time() + (expiry_seconds or self._cache_expiry)

        cache_entry = CacheEntry(
            stem_data=stem_data,
            access_time=time.time(),
            size_bytes=size_bytes,
            expires_at=expires_at
        )

        with self._lock:
            self._cache[cache_key] = cache_entry

    def get_cached_stem_data(self, cache_key: str) -> Optional[StemData]:
        """
        Retrieve cached stem data.

        Args:
            cache_key: Cache key to look up

        Returns:
            Cached stem data or None if not found/expired
        """
        if self.cache_strategy == CacheStrategy.NONE:
            return None

        with self._lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]

                # Check if expired
                if entry.expires_at and time.time() > entry.expires_at:
                    del self._cache[cache_key]
                    return None

                # Update access time and frequency
                entry.access_time = time.time()
                entry.frequency += 1

                # Update cache hit rate
                self._update_cache_metrics(hit=True)

                return entry.stem_data

        self._update_cache_metrics(hit=False)
        return None

    def _estimate_data_size(self, stem_data: StemData) -> int:
        """Estimate the memory size of stem data."""
        # Rough estimation based on MIDI messages and patterns
        base_size = 1024  # Base overhead

        if stem_data.midi_messages:
            base_size += len(stem_data.midi_messages) * 32  # ~32 bytes per message

        if stem_data.pattern and hasattr(stem_data.pattern, 'notes'):
            base_size += len(stem_data.pattern.notes) * 64  # ~64 bytes per note

        return base_size

    def _get_cache_size_mb(self) -> float:
        """Get current cache size in MB."""
        total_size = sum(entry.size_bytes for entry in self._cache.values())
        return total_size / 1024 / 1024

    def _evict_cache_entries(self, required_space: int) -> None:
        """Evict cache entries using LRU strategy."""
        if not self._cache:
            return

        # Sort by access time (oldest first) and frequency (lowest first)
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: (x[1].access_time, -x[1].frequency)
        )

        freed_space = 0
        entries_to_remove = []

        for cache_key, entry in sorted_entries:
            entries_to_remove.append(cache_key)
            freed_space += entry.size_bytes

            if freed_space >= required_space:
                break

        # Remove entries
        for cache_key in entries_to_remove:
            del self._cache[cache_key]

    def _update_cache_metrics(self, hit: bool) -> None:
        """Update cache hit rate metrics."""
        # Simplified cache metrics update
        current_rate = self._metrics.cache_hit_rate
        if hit:
            self._metrics.cache_hit_rate = (current_rate * 0.9) + 0.1  # Moving average
        else:
            self._metrics.cache_hit_rate = current_rate * 0.9

    def allocate_memory_block(self, pool_name: str) -> Optional[bytes]:
        """Allocate a memory block from the specified pool."""
        if pool_name in self._memory_pools:
            return self._memory_pools[pool_name].allocate()
        return None

    def deallocate_memory_block(self, pool_name: str, block: bytes) -> None:
        """Deallocate a memory block back to the specified pool."""
        if pool_name in self._memory_pools:
            self._memory_pools[pool_name].deallocate(block)

    def start_monitoring(self) -> None:
        """Start performance monitoring thread."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_performance, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)

    def _monitor_performance(self) -> None:
        """Monitor system performance metrics."""
        while self._monitoring:
            try:
                # Update memory usage
                process = psutil.Process(os.getpid())
                memory_mb = process.memory_info().rss / 1024 / 1024
                self._metrics.memory_usage_mb = memory_mb
                self._memory_usage_history.append(memory_mb)

                # Update CPU usage
                cpu_percent = process.cpu_percent(interval=0.1)
                self._metrics.cpu_usage_percent = cpu_percent

                # Calculate memory fragmentation
                if self._memory_pools:
                    total_utilization = sum(
                        pool.get_utilization() for pool in self._memory_pools.values()
                    )
                    self._metrics.memory_fragmentation_ratio = total_utilization / len(self._memory_pools)

                # Sleep before next measurement
                time.sleep(1.0)

            except Exception as e:
                # Log error but continue monitoring
                print(f"Performance monitoring error: {e}")
                time.sleep(5.0)

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        # Calculate derived metrics
        if self._processing_times:
            self._metrics.stems_generated_per_second = (
                len(self._processing_times) / sum(self._processing_times)
            )

        if self._memory_usage_history:
            avg_memory = sum(self._memory_usage_history) / len(self._memory_usage_history)
            self._metrics.average_stem_size_kb = avg_memory * 1024  # Convert to KB

        return self._metrics

    def cleanup_resources(self) -> None:
        """Clean up all resources and reset optimizer state."""
        self.stop_monitoring()

        # Clear caches
        with self._lock:
            self._cache.clear()

        # Reset memory pools
        for pool in self._memory_pools.values():
            pool.free_blocks.clear()
            pool.used_blocks.clear()

        # Reset metrics
        self._metrics = PerformanceMetrics()
        self._processing_times.clear()
        self._memory_usage_history.clear()

        # Force garbage collection
        gc.collect()

    def get_optimization_suggestions(self) -> List[str]:
        """Get optimization suggestions based on current performance."""
        suggestions = []

        metrics = self.get_performance_metrics()

        if metrics.memory_usage_mb > self.max_memory_mb * 0.9:
            suggestions.append("High memory usage detected. Consider reducing cache size or stem count.")

        if metrics.cpu_usage_percent > 80:
            suggestions.append("High CPU usage. Consider reducing parallel processing or optimization level.")

        if metrics.cache_hit_rate < 0.3:
            suggestions.append("Low cache hit rate. Caching may not be effective for current workload.")

        if metrics.memory_fragmentation_ratio > 0.8:
            suggestions.append("High memory fragmentation. Consider memory pool optimization.")

        if not suggestions:
            suggestions.append("Performance looks good. Current optimization settings are effective.")

        return suggestions

    def record_processing_time(self, processing_time: float) -> None:
        """Record a processing time measurement."""
        self._processing_times.append(processing_time)

    def should_garbage_collect(self) -> bool:
        """Determine if garbage collection should be triggered."""
        if self.optimization_level in [OptimizationLevel.AGGRESSIVE, OptimizationLevel.ULTRA]:
            # More aggressive garbage collection in high optimization modes
            return len(self._processing_times) % 10 == 0  # Every 10 operations
        return False