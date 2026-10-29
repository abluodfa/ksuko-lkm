/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * drm_guard - Widevine DRM "deviceUniqueId" interception.
 *
 * Ported from the SKP "device identity manager" 1.0.1 / DRMID 1.4.0 kernel-side
 * logic (kernel_context.h, binder_hook_builder.cpp): the request is recognised by
 * (descriptor android.hardware.drm.IDrmPlugin) + (code 11 == getPropertyByteArray)
 * + (property name "deviceUniqueId"), the reply by (status == 0) + (byte-array
 * length exactly 32), scoped to Widevine HAL processes only.
 *
 * What is NOT ported: the SKRoot kernel_module_kit hook API and the asmjit
 * generated AArch64 trampoline.  Here the hook is a kretprobe on binder_ioctl, so
 * the matching logic is plain C and nothing is written into kernel text.
 *
 * M1 is observation only: dry-run counters, no byte is ever modified.
 */
#ifndef __SUSFS_DRM_GUARD_H
#define __SUSFS_DRM_GUARD_H

#include <linux/types.h>

#ifdef __aarch64__
int drm_guard_init(void);
void drm_guard_exit(void);
/* target uid 白名单（config.conf 的 APP= 列表）。taint_spoof 复用同一份：
 * 要骗的 app 和要隐藏路径的 app 是同一批，没必要维护两套配置。
 * 名单为空时 drm_uid_allowed() 恒为 false。 */
bool drm_uid_allowed(void);
u32 drm_uid_count(void);
#else
static inline int drm_guard_init(void) { return 0; }
static inline void drm_guard_exit(void) { }
static inline bool drm_uid_allowed(void) { return false; }
static inline u32 drm_uid_count(void) { return 0; }
#endif

#endif /* __SUSFS_DRM_GUARD_H */
