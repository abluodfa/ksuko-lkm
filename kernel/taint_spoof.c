/* SPDX-License-Identifier: GPL-2.0-only */
/*
 * taint_spoof.c - 只对 target uid 伪装 /proc/sys/kernel/tainted。
 *
 * 为什么按 uid 而不是全局清零：
 *   v16.85 及之前的版本把 get_taint() 的返回值无条件改成 0，root 自己也
 *   看不到真实值。后果是「hook 在工作」和「内核本来就没 taint」无法区分，
 *   模块一卸载真实值立刻跳出来，反而更像异常；排查时也没有任何自证手段。
 *   这里只在 current_uid() 命中 drm_guard 的 target uid 白名单时清零，
 *   其余进程（含 root shell）读到的仍是真实值。
 *
 * 为什么 hook get_taint()：
 *   proc_dointvec_taint()（/proc/sys/kernel/tainted 的 read 路径）和
 *   print_tainted()（oops 头）都只从它取数，一个点覆盖全部读取路径。
 *   get_taint() 是 EXPORT_SYMBOL_GPL 且未内联，kretprobe 可注册。
 *
 * 白名单来源：
 *   复用 drm_guard 的 g_uids（config.conf 的 APP= 列表，开机由
 *   susfs_boot.sh 下发）。名单为空时 drm_uid_allowed() 恒为 false，
 *   即不伪装——fail-open，不会因为配置缺失把真实值藏起来。
 */
#define pr_fmt(fmt) "taint_spoof: " fmt

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/kprobes.h>
#include <linux/ptrace.h>
#include <linux/sched.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/atomic.h>

#include "susfs_kprobe.h"
#include "drm_guard.h"
#include "ksu_allow_gate.h"	/* the control-node gate is su's own */
#include "taint_spoof.h"

#ifdef __aarch64__

/* 两个计数器只用于自证 hook 是否在工作：spoofed 增长说明有 target uid
 * 在读 tainted，passthru 增长说明有名单外的进程在读。两者都不增长
 * 说明根本没人读，不能据此判断 hook 失效。 */
static atomic64_t g_spoofed = ATOMIC64_INIT(0);
static atomic64_t g_passthru = ATOMIC64_INIT(0);

static int taint_ret_handler(struct kretprobe_instance *ri, struct pt_regs *regs)
{
	if (drm_uid_allowed()) {
		regs->regs[0] = 0;
		atomic64_inc(&g_spoofed);
	} else {
		atomic64_inc(&g_passthru);
	}
	return 0;
}

static struct kretprobe g_taint_krp = {
	.handler = taint_ret_handler,
	.maxactive = 128,
};

/* 只读节点，不提供 real= 字段：本文件自己的 read 也会经过 get_taint()，
 * 名单内的进程读到的 real 恒为 0，写出来只会误导。真实值从 root shell
 * 直接 cat /proc/sys/kernel/tainted 看。 */
static int taint_proc_show(struct seq_file *m, void *v)
{
	seq_printf(m, "mode=per_uid\n");
	seq_printf(m, "uids=%u\n", drm_uid_count());
	seq_printf(m, "spoofed=%llu\n",
		   (unsigned long long)atomic64_read(&g_spoofed));
	seq_printf(m, "passthru=%llu\n",
		   (unsigned long long)atomic64_read(&g_passthru));
	return 0;
}

static int taint_proc_open(struct inode *inode, struct file *file)
{
	/* self_hide 之外的第二道门：节点进了 susfs_self_hide_paths，
	 * sus_path_relax_mode() 会把 mode 放宽成 0777，可见性只能靠这里答。
	 * ENOENT 而不是 EACCES——EACCES 等于承认节点存在。 */
	if (!susfs_control_visible())
		return -ENOENT;
	return single_open(file, taint_proc_show, NULL);
}

static const struct proc_ops taint_proc_ops = {
	.proc_open = taint_proc_open,
	.proc_read = seq_read,
	.proc_lseek = seq_lseek,
	.proc_release = single_release,
};

static struct proc_dir_entry *g_proc;

int susfs_taint_spoof_init(bool expose_proc)
{
	int ret;

	/* Hot function: register_kretprobe() rewrites its prologue with a
	 * breakpoint, so it must not already be in use by another tool. */
	g_taint_krp.kp.symbol_name = "get_taint";
	g_taint_krp.kp.addr = NULL;
	ret = register_kretprobe(&g_taint_krp);
	if (ret) {
		susfs_krp_forget_addr(&g_taint_krp);
		pr_warn("get_taint is not probeable (%d) - taint stays visible\n", ret);
		return ret;
	}

	if (expose_proc) {
		/* 0444 只是出生时的模式。注册进 susfs_self_hide_paths 之后，
		 * sus_path_relax_mode()（sus_path.c:169）会把 inode mode 强制改成 0777，
		 * 让 DAC 不先答 EACCES、改由 LSM 钩子对非 root 答 ENOENT；卸载时
		 * sus_path_restore_mode() 还原。所以 stat 看到 777 是设计结果，
		 * 不是 proc_create 的 mode 没生效。 */
		g_proc = proc_create("susfs_taint", 0444, NULL, &taint_proc_ops);
		if (!g_proc)
			pr_warn("proc_create(susfs_taint) failed - counters unavailable\n");
	}

	pr_info("get_taint hooked: taint reads back as 0 for target uids only\n");
	return 0;
}

void susfs_taint_spoof_exit(void)
{
	if (g_proc) {
		proc_remove(g_proc);
		g_proc = NULL;
	}
	unregister_kretprobe(&g_taint_krp);
	susfs_krp_forget_addr(&g_taint_krp);
}

#endif /* __aarch64__ */
