/* Author: Paulo R. A. Gilz
 *
 * Stable C ABI for the aoeViewer core.
 *
 * Everything the Python front end needs crosses this boundary. It is plain C on
 * purpose: one shared library works with any CPython 3.x through ctypes, with no
 * build-time dependency on Python itself, on pybind11, or on a matching C++ ABI.
 * That matters on locked-down machines where installing a binding library is not
 * possible.
 *
 * Conventions
 * -----------
 *  * Every function takes the session as its first argument. The session is not
 *    internally synchronised: call it from one thread at a time. It does use all
 *    cores internally while parsing.
 *  * Returned `const char*` point at memory owned by the session and are valid
 *    until the next call that changes the thing they came from. Copy them
 *    immediately (ctypes does this by itself when the restype is c_char_p).
 *  * Returned `const float*` alias the core's column buffers directly -- no copy.
 *    They stay valid until that file is reloaded, removed, or the session is
 *    cleared or destroyed.
 *  * Index arguments out of range are errors, never crashes: functions return
 *    0 / NULL / -1.
 */

#ifndef AOE_C_API_H
#define AOE_C_API_H

#include <stddef.h>

#if defined(_WIN32)
#if defined(AOE_BUILD_SHARED)
#define AOE_API __declspec(dllexport)
#else
#define AOE_API
#endif
#else
#if defined(AOE_BUILD_SHARED)
#define AOE_API __attribute__((visibility("default")))
#else
#define AOE_API
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct aoe_session aoe_session;

/* ------------------------------------------------------------ lifetime */

AOE_API const char* aoe_version(void);
AOE_API int aoe_worker_count(void);

AOE_API aoe_session* aoe_session_create(void);
AOE_API void aoe_session_destroy(aoe_session* session);

/* Message describing the most recent failure, or "" if the last call
 * succeeded. Valid until the next call on this session. */
AOE_API const char* aoe_last_error(aoe_session* session);

/* ---------------------------------------------------------------- files */

/* Returns the new file's index, or -1 on failure (see aoe_last_error). */
AOE_API int aoe_add_file(aoe_session* session, const char* path);
AOE_API void aoe_remove_file(aoe_session* session, int file);
AOE_API void aoe_clear_files(aoe_session* session);

AOE_API int aoe_file_count(aoe_session* session);
AOE_API const char* aoe_file_name(aoe_session* session, int file);
AOE_API const char* aoe_file_path(aoe_session* session, int file);

/* Number of files whose modification time changed on disk; the indices are
 * written into `out` (up to `max` of them). Pass NULL/0 to just count. */
AOE_API int aoe_stale_files(aoe_session* session, int* out, int max);
AOE_API int aoe_reload_file(aoe_session* session, int file);

/* Display names. Renaming affects only what is shown: the identity keys that
 * styles and workspaces are built on are untouched, so a renamed run keeps its
 * colour. */
AOE_API void aoe_set_file_name(aoe_session* session, int file, const char* name);
AOE_API void aoe_set_sim_name(aoe_session* session, int file, int sim, const char* name);

AOE_API int aoe_sim_count(aoe_session* session, int file);
AOE_API const char* aoe_sim_id(aoe_session* session, int file, int sim);
/* The display name, which defaults to the id until renamed. */
AOE_API const char* aoe_sim_name(aoe_session* session, int file, int sim);
AOE_API const char* aoe_sim_key(aoe_session* session, int file, int sim);

AOE_API int aoe_header_count(aoe_session* session, int file);
AOE_API const char* aoe_header(aoe_session* session, int file, int index);

// The unit declared for a column in *PARAMETRES, or "" when the file's dialect
// carries none. Parallel to aoe_header(); never NULL.
AOE_API const char* aoe_header_unit(aoe_session* session, int file, int index);

// The unit of a parameter by name, searched across every open file, following
// aliases. "" when unknown or when the parameter is computed.
AOE_API const char* aoe_param_unit(aoe_session* session, const char* name);

/* ----------------------------------------------------------- parameters */

/* Union of physical headers, computed parameters and aliases, sorted. */
AOE_API int aoe_param_count(aoe_session* session);
AOE_API const char* aoe_param(aoe_session* session, int index);

AOE_API void aoe_set_alias(aoe_session* session, const char* alias, const char* target);
AOE_API void aoe_remove_alias(aoe_session* session, const char* alias);
AOE_API const char* aoe_resolve(aoe_session* session, const char* name);

/* 1 on success, 0 on failure (see aoe_last_error). */
AOE_API int aoe_set_computed(aoe_session* session, const char* name, const char* expression);

/* Removes `name` and everything defined in terms of it. Returns how many were
 * removed; their names are readable with aoe_removed_name(). */
AOE_API int aoe_remove_computed(aoe_session* session, const char* name);
AOE_API const char* aoe_removed_name(aoe_session* session, int index);

AOE_API int aoe_is_computed(aoe_session* session, const char* name);
AOE_API const char* aoe_computed_source(aoe_session* session, const char* name);

/* Parses `expression` and checks its variables against the open files, without
 * evaluating anything. 1 = valid. Use for live feedback in the formula editor. */
AOE_API int aoe_validate_formula(aoe_session* session, const char* expression);

/* ----------------------------------------------------------------- data */

/* Zero-copy view of one channel. Returns NULL when unavailable.
 *
 * The pointer is only valid until the next eviction. Anything that keeps it --
 * a plotted curve does, for as long as it is on screen -- must use
 * aoe_channel_pinned() instead. */
AOE_API const float* aoe_channel(aoe_session* session, int file, int sim,
                                 const char* name, size_t* out_count);

/* A reference to a channel's sample block. Holding one keeps the samples alive
 * no matter what the memory cap does; release it when the view is dropped. */
typedef struct aoe_pin aoe_pin;

/* Zero-copy view of one channel, plus a pin that keeps it valid. *out_pin is
 * set to NULL when there is nothing to return; otherwise the caller owns it and
 * must pass it to aoe_pin_release(). */
AOE_API const float* aoe_channel_pinned(aoe_session* session, int file, int sim,
                                        const char* name, size_t* out_count,
                                        aoe_pin** out_pin);

AOE_API void aoe_pin_release(aoe_pin* pin);

/* Summary computed when the block was parsed, so these cost nothing.
 * aoe_pin_sorted: 1 when the samples are non-decreasing -- the precondition of
 * every binary search over an x channel. aoe_pin_bounds: 1 when finite samples
 * exist, writing the extent; this is what autoscaling needs, and answering it
 * from the block avoids rescanning a million samples per curve per replot. */
AOE_API int aoe_pin_sorted(aoe_pin* pin);
AOE_API int aoe_pin_bounds(aoe_pin* pin, float* out_min, float* out_max);
AOE_API size_t aoe_pin_count(aoe_pin* pin);

/* Loads every physical column the named parameters need, for the given
 * simulations, in one parallel pass. Pass sims=NULL/n_sims=0 for all. */
AOE_API void aoe_preload(aoe_session* session, int file, const int* sims, int n_sims,
                         const char* const* names, int n_names);

/* --------------------------------------------------------------- plotting */

/* Min/max decimation. Writes at most 2*max_points values into out_x/out_y and
 * returns how many were written. When x_min < x_max only that x range is
 * considered (binary-searched); otherwise the whole series is used. */
AOE_API size_t aoe_decimate(const float* x, const float* y, size_t count,
                            float x_min, float x_max, size_t max_points,
                            float* out_x, float* out_y);

/* As above, but told whether x is non-decreasing.
 *
 * The range is located with a binary search, which needs sorted x. A phase-plane
 * tile -- MACH against ALTITUDE on a climb-and-descend, and phase planes are
 * half of what this tool is for -- has x rising and then falling, so the search
 * brackets only the first crossing and the entire descending half of the
 * trajectory silently disappears. Pass x_sorted=0 and the range is found by a
 * full scan instead, which keeps both branches. x_sorted=-1 works it out. */
AOE_API size_t aoe_decimate_ex(const float* x, const float* y, size_t count,
                               float x_min, float x_max, size_t max_points,
                               int x_sorted, float* out_x, float* out_y);

/* 1 when the samples are non-decreasing across the finite ones. */
AOE_API int aoe_series_sorted(const float* values, size_t count);

/* 1 if finite samples were found. */
AOE_API int aoe_series_bounds(const float* values, size_t count, float* out_min,
                              float* out_max);

AOE_API size_t aoe_nearest_index(const float* x, size_t count, float target);

/* Statistics of y over the samples whose x falls in [x_min, x_max], or over the
 * whole series when x is NULL or x_min >= x_max. Writes 7 doubles into `out`:
 * min, max, mean, stddev, first, last, sample count. Returns 1 on success. */
AOE_API int aoe_series_stats(const float* x, const float* y, size_t count,
                             float x_min, float x_max, double* out);

/* As above, with the same x_sorted contract as aoe_decimate_ex: statistics of
 * "what I am looking at" are wrong in exactly the same way on a phase plane. */
AOE_API int aoe_series_stats_ex(const float* x, const float* y, size_t count,
                                float x_min, float x_max, int x_sorted, double* out);

/* ---------------------------------------------------------------- memory */

/* Bytes of sample data cached across every open file. */
AOE_API size_t aoe_resident_bytes(aoe_session* session);

/* Frees least-recently-used cached columns until at most keep_bytes remain,
 * returning the resident total afterwards. Always safe: evicted data is re-read
 * on demand, so this trades time for memory. */
AOE_API size_t aoe_trim_memory(aoe_session* session, size_t keep_bytes);

/* ----------------------------------------------------------------- cache */

/* A .aoec sidecar stores a file's numbers the way this program reads them:
 * parameter-major float32, with the run offsets and each block's minimum and
 * maximum in a header. With one present, opening does not scan the file and
 * loading a column does not parse anything.
 *
 * It is never required and never written behind the caller's back. Everything
 * still works with no cache; it is simply slower. A sidecar is only used when
 * the source file's length, modification time and a hash of its contents all
 * still match what was recorded, so a file that changed is re-read rather than
 * served from stale data. */

/* 0 when the AOE_CACHE environment variable switches caching off. */
AOE_API int aoe_cache_enabled(void);

/* 1 when a valid sidecar is attached to this file and serving its reads. */
AOE_API int aoe_cache_active(aoe_session* session, int file);

/* Path of the attached sidecar, or "" when there is none. Valid until the next
 * call that returns a string from this session. */
AOE_API const char* aoe_cache_path(aoe_session* session, int file);

/* Parses the file once and writes its sidecar, then attaches it. Returns 1 on
 * success, 0 with aoe_last_error() set otherwise.
 *
 * This costs a full parse, so call it off the UI thread -- typically once,
 * after a file has been opened the slow way. Memory stays bounded to one
 * parameter across every run however large the file is. */
AOE_API int aoe_build_cache(aoe_session* session, int file);

/* Detaches and deletes this file's sidecar. Returns 1 when none remains. */
AOE_API int aoe_discard_cache(aoe_session* session, int file);

/* ------------------------------------------------------------------ iso */

/* Value of an ISO parameter for each simulation of a file (NaN when absent).
 * Writes up to `max` values, returns the number written. */
AOE_API int aoe_iso_values(aoe_session* session, int file, const char* name,
                           float* out, int max);

/* Distinct ISO values across all open files, ascending. Pass out=NULL to learn
 * the count first. */
AOE_API int aoe_iso_distinct(aoe_session* session, const char* name, float* out, int max);

/* Evenly spaced grouping of `values`. Bounds are written as pairs into
 * out_bounds (2 floats per group); returns the number of groups. */
AOE_API int aoe_iso_auto_group(const float* values, int count, int group_count,
                               float* out_bounds, int max_groups);

/* Index of the group containing `value`, or -1. `bounds` is 2 floats per group. */
AOE_API int aoe_iso_group_index(float value, const float* bounds, int group_count);

/* Formats a value the way the ISO lists show it (trailing zeros trimmed).
 * Valid until the next call on this session. */
AOE_API const char* aoe_iso_format(aoe_session* session, float value);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* AOE_C_API_H */
