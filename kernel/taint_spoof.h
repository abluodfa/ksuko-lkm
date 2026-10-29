/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * taint_spoof - 只对 target uid 伪装 /proc/sys/kernel/tainted。
 *
 * 与 v16.85 及之前的全局清零不同：这里只在 current_uid() 命中
 * drm_guard 的 target uid 白名单时把 get_taint() 的返回值改成 0，
 * 其余进程（含 root shell）读到的仍是真实值。
 */
#ifndef __SUSFS_TAINT_SPOOF_H
#define __SUSFS_TAINT_SPOOF_H

#include <linux/types.h>

#ifdef __aarch64__
/* expose_proc 为 false 时只装 hook、不建 /proc/susfs_taint：节点存在本身
 * 就是模块存在的证据，与其它控制节点用同一个开关（susfs_control_node_allowed）。 */
int susfs_taint_spoof_init(bool expose_proc);
void susfs_taint_spoof_exit(void);
#else
static inline int susfs_taint_spoof_init(bool expose_proc) { return 0; }
static inline void susfs_taint_spoof_exit(void) { }
#endif

#endif /* __SUSFS_TAINT_SPOOF_H */
